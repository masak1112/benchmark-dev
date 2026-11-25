"""Many-task launcher for Aurora: one worker per XPU tile, a task list, per-task state files.

Run under `mpiexec -n <nodes*12> --ppn 12 python -m s2storm.aurora.taskfarm --tasks T.jsonl --state DIR`.
Each rank pins itself to one tile (ZE_AFFINITY_MASK = local rank) and makes one pass over the task list: claim a task that has
no `done` file, run its command with its own log, then write `<id>.done` (exit 0) or append to `<id>.failed`.

Claims (heartbeat, safe with several jobs at once; 2026-11-10, replaces "any other job's claim is stale"):
  * claim = `<id>.claim`, created with O_EXCL, content {job, host, pid, time}. While the task runs the owner touches it every
    `--heartbeat_s` (60 s). A claim is stale only when its mtime is older than `--stale_s` (900 s), whoever owns it: a live task
    in another job is never taken over; a task of a killed/expired job is taken over 15 min after its last heartbeat.
  * Takeover is serialised by a short lock `<id>.takeover` (O_EXCL): the taker re-checks the claim's mtime under the lock, moves it
    to `<id>.stale.<job>.<time>` and creates its own claim. Without the lock two takers could both see the stale claim and the
    slower one would move away the fresh claim of the faster one (the race in the AIFS Pool). A lock older than 120 s is broken.
  * After claiming, `done` is checked again (a task may finish between the listing and the claim).
  * The owner checks at every heartbeat that the claim is still its own; if another worker owns it the task is killed and
    recorded as `lost` (no two writers on one output). If the claim vanished it is re-created (O_EXCL), else the task is killed.
  * A failed task (exit != 0) appends a line to `<id>.failed` and releases its claim; it is retried (by any rank or job) until it
    has failed `--max_tries` times (default 2), then it is left alone.
A task is only started if the job has more than `--min_left` seconds of walltime left (S2STORM_DEADLINE, epoch seconds).

Task list: JSON lines {"id": "...", "cmd": "...", "est_s": 600}; `cmd` runs through bash with TASK_ID, TILE, OUT set.
`--dry_run S` sleeps S seconds instead of running (heartbeats still run). For CPU tests: RANK/WORLD_SIZE/LOCAL_RANK and
PBS_JOBID may be set by hand (tests/test_taskfarm_claims.py).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import signal
import time
from pathlib import Path

# set by mpiexec/PALS for the farm rank; a child that inherits them tries to join the MPI job
# (MPI_Init -> "PMIx_Fence returned -25" abort, then a hang; job 8997352), so they are removed
MPI_ENV_PREFIXES = ('PALS_', 'PMI_', 'PMIX_', 'PMI', 'MPI_LOCAL', 'OMPI_')
STALE_S, HEARTBEAT_S, LOCK_STALE_S = 900.0, 60.0, 120.0


def rank_info():
    env = os.environ
    rank = int(env.get('PALS_RANKID', env.get('PMI_RANK', env.get('RANK', 0))))
    size = env.get('PALS_SIZE', env.get('PMI_SIZE', env.get('WORLD_SIZE')))
    if size is None:   # Aurora PALS sets no size variable
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()
    size = int(size)
    local = int(env.get('PALS_LOCAL_RANKID', env.get('PMI_LOCAL_RANK', env.get('LOCAL_RANK', 0))))
    return rank, size, local


def read_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def n_failed(state: Path, task_id: str) -> int:
    try:
        return sum(1 for x in (state / f'{task_id}.failed').read_text().splitlines() if x.strip())
    except OSError:
        return 0


def _owner(path: Path):
    try:
        return json.loads(path.read_text() or '{}')
    except (OSError, json.JSONDecodeError):
        return None


class Claimer:
    def __init__(self, state: Path, job: str, stale_s=STALE_S, max_tries=2, rank=0):
        self.state, self.job, self.stale_s, self.max_tries = Path(state), job, float(stale_s), int(max_tries)
        self.me = {'job': job, 'host': socket.gethostname(), 'pid': os.getpid(), 'rank': rank}
        self.log = []

    def _create(self, claim: Path) -> bool:
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, json.dumps(self.me | {'time': time.time()}).encode())
        os.close(fd)
        return True

    def _takeover(self, tid: str) -> bool:
        claim, lock = self.state / f'{tid}.claim', self.state / f'{tid}.takeover'
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                lock.unlink()
        except OSError:
            pass
        if not self._create(lock):
            return False
        try:
            try:
                age = time.time() - claim.stat().st_mtime
            except FileNotFoundError:
                age = None                      # released meanwhile
            if age is not None:
                if age < self.stale_s:
                    return False                # refreshed or re-claimed meanwhile
                old = _owner(claim) or {}
                claim.rename(self.state / f'{tid}.stale.{old.get("job", "unknown")}.{int(time.time())}')
                self.log.append(('takeover', tid, old.get('job'), round(age)))
            return self._create(claim)
        finally:
            lock.unlink(missing_ok=True)

    def try_claim(self, tid: str) -> bool:
        if (self.state / f'{tid}.done').exists() or n_failed(self.state, tid) >= self.max_tries:
            return False
        claim = self.state / f'{tid}.claim'
        got = self._create(claim)
        if not got:
            try:
                if time.time() - claim.stat().st_mtime < self.stale_s:
                    return False                # live claim (any job, incl. ours): leave it
            except FileNotFoundError:
                pass
            got = self._takeover(tid)
        if got and (self.state / f'{tid}.done').exists():
            self.release(tid)
            return False
        return got

    def heartbeat(self, tid: str) -> bool:
        """refresh our claim; False if another worker owns it now (then the caller must stop the task)"""
        claim = self.state / f'{tid}.claim'
        o = _owner(claim)
        if o is None and not claim.exists():
            return self._create(claim)
        if o is None:                            # unreadable (being written): try again next beat
            return True
        if (o.get('job'), o.get('host'), o.get('pid')) != (self.me['job'], self.me['host'], self.me['pid']):
            return False
        try:
            os.utime(claim)
        except OSError:
            pass
        return True

    def release(self, tid: str):
        claim = self.state / f'{tid}.claim'
        o = _owner(claim) or {}
        if (o.get('job'), o.get('host'), o.get('pid')) == (self.me['job'], self.me['host'], self.me['pid']):
            claim.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--min_left', type=float, default=None,
                    help='seconds of walltime needed to start a task (default: task est_s * 1.3 + 60)')
    ap.add_argument('--timeout_factor', type=float, default=2.0, help='kill a task after est_s * this')
    ap.add_argument('--stale_s', type=float, default=STALE_S, help='claim is stale after this many s without heartbeat')
    ap.add_argument('--heartbeat_s', type=float, default=HEARTBEAT_S)
    ap.add_argument('--max_tries', type=int, default=2, help='give up on a task after this many failures')
    ap.add_argument('--dry_run', type=float, default=None, metavar='SECONDS')
    args = ap.parse_args(argv)

    rank, size, local = rank_info()
    job = os.environ.get('PBS_JOBID', 'nojob').split('.')[0]
    deadline = float(os.environ.get('S2STORM_DEADLINE', 'inf'))
    tile = local % 12
    args.state.mkdir(parents=True, exist_ok=True)
    (args.state / 'logs').mkdir(exist_ok=True)
    tasks = read_tasks(args.tasks)
    cl = Claimer(args.state, job, args.stale_s, args.max_tries, rank)
    # start each rank at a different offset so claims rarely collide
    order = tasks[rank % max(len(tasks), 1):] + tasks[:rank % max(len(tasks), 1)]
    summary = {'rank': rank, 'size': size, 'tile': tile, 'host': socket.gethostname(), 'job': job,
               'done': [], 'failed': [], 'lost': [], 'skipped_time': []}
    for task in order:
        tid = str(task['id'])
        need = args.min_left if args.min_left is not None else float(task.get('est_s', 600)) * 1.3 + 60
        if deadline - time.time() < need:
            summary['skipped_time'].append(tid)
            continue
        if not cl.try_claim(tid):
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith(MPI_ENV_PREFIXES)}
        env |= {'ZE_AFFINITY_MASK': str(tile), 'TASK_ID': tid, 'TILE': str(tile), 'OUT': str(task.get('out', ''))}
        log = args.state / 'logs' / f'{tid}.log'
        t0 = time.time()
        with open(log, 'a') as f:
            if args.dry_run is not None:
                f.write(f'[dry run] rank {rank} tile {tile} host {summary["host"]} job {job}: {task["cmd"]}\n')
                f.flush()
                cmd = ['sleep', str(args.dry_run)]
            else:
                cmd = ['bash', '-c', task['cmd']]
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            kill_at = min(t0 + float(task.get('est_s', 600)) * args.timeout_factor, deadline - 30)
            code, why = None, ''
            while code is None:
                try:
                    code = proc.wait(timeout=max(min(args.heartbeat_s, kill_at - time.time()), 0.05))
                except subprocess.TimeoutExpired:
                    if time.time() >= kill_at:
                        why = 'timeout or job deadline'
                    elif not cl.heartbeat(tid):
                        why = 'claim lost to another worker'
                    if why:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        code = 124 if why.startswith('timeout') else 125
                        f.write(f'[taskfarm] killed after {time.time() - t0:.0f} s ({why})\n')
        record = {'id': tid, 'rank': rank, 'tile': tile, 'host': summary['host'], 'job': job,
                  'exit': code, 'seconds': round(time.time() - t0, 1), 'time': time.time()}
        if code == 125:                         # someone else owns it: write nothing, don't touch their claim
            summary['lost'].append(tid)
            continue
        if code == 0:
            (args.state / f'{tid}.done').write_text(json.dumps(record) + '\n')
        else:
            with open(args.state / f'{tid}.failed', 'a') as g:
                g.write(json.dumps(record) + '\n')
        cl.release(tid)
        summary['done' if code == 0 else 'failed'].append(tid)
    summary['claim_log'] = cl.log
    (args.state / f'rank{rank:05d}.{job}.json').write_text(json.dumps(summary) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
