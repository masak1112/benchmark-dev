"""CPU tests of the heartbeat claims in s2storm.aurora.taskfarm (several jobs at once must never run one task twice).

    PYTHONPATH=. python -m pytest -q tests/test_taskfarm_claims.py
Set TASKFARM_TEST_DIR=<dir on Lustre> to run the file operations on the shared filesystem instead of /tmp.
"""
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2storm.aurora.taskfarm import Claimer  # noqa: E402

FAST = ['--heartbeat_s', '0.2', '--stale_s', '2']


@pytest.fixture
def base(tmp_path):
    d = os.environ.get('TASKFARM_TEST_DIR')
    if d:
        p = Path(d) / f'tf_{uuid.uuid4().hex[:8]}'
        p.mkdir(parents=True)
        return p
    return tmp_path


def write_tasks(base, n, cmd, est_s=60, prefix='t'):
    f = base / 'tasks.jsonl'
    f.write_text(''.join(json.dumps({'id': f'{prefix}{i:03d}', 'cmd': cmd, 'est_s': est_s}) + '\n' for i in range(n)))
    return f


def farm(base, job, rank, size, extra=(), state=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PALS_', 'PMI', 'PMIX_'))}
    env |= {'PBS_JOBID': f'{job}.aurora', 'RANK': str(rank), 'WORLD_SIZE': str(size), 'LOCAL_RANK': str(rank),
            'PYTHONPATH': str(ROOT), 'RUNLOG': str(base / 'runs.log')}
    return subprocess.Popen([sys.executable, '-m', 's2storm.aurora.taskfarm', '--tasks', str(base / 'tasks.jsonl'),
                             '--state', str(state or base / 'state'), *FAST, *extra], env=env, start_new_session=True)


def runs(base):
    p = base / 'runs.log'
    return [x.split() for x in p.read_text().splitlines()] if p.exists() else []


CMD = 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; sleep 0.6; echo "$TASK_ID $PBS_JOBID end" >> $RUNLOG'


def test_two_jobs_run_each_task_once(base):
    write_tasks(base, 24, CMD)
    ps = [farm(base, job, r, 4) for job in ('1001', '1002') for r in range(4)]
    assert all(p.wait(120) == 0 for p in ps)
    starts = [r[0] for r in runs(base) if r[2] == 'start']
    assert sorted(starts) == sorted(f't{i:03d}' for i in range(24))      # each exactly once
    assert len({r[1] for r in runs(base)}) == 2                          # both jobs did work
    assert len(list((base / 'state').glob('*.done'))) == 24
    assert not list((base / 'state').glob('*.claim'))


def test_live_claim_of_other_job_is_not_taken(base):
    """old bug: a second job treated the first job's claims as stale and ran the same members"""
    write_tasks(base, 1, 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; sleep 5')
    a = farm(base, '2001', 0, 1)
    time.sleep(1.5)
    b = farm(base, '2002', 0, 1)
    assert b.wait(60) == 0 and a.wait(60) == 0
    assert [r[1] for r in runs(base)] == ['2001.aurora']


def test_killed_job_is_taken_over_after_stale(base):
    write_tasks(base, 1, 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; sleep 30', est_s=100)
    a = farm(base, '3001', 0, 1)
    time.sleep(1.5)
    os.killpg(a.pid, signal.SIGKILL)                                     # node dies: claim stays, heartbeat stops
    a.wait()
    write_tasks(base, 1, 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; sleep 0.3', est_s=100)
    b = farm(base, '3002', 0, 1)                                         # immediately: claim still fresh -> skip
    assert b.wait(60) == 0
    assert not (base / 'state' / 't000.done').exists()
    time.sleep(2.5)
    c = farm(base, '3003', 0, 1)                                         # after stale_s: takeover
    assert c.wait(60) == 0
    assert json.loads((base / 'state' / 't000.done').read_text())['job'] == '3003'
    assert [r[1] for r in runs(base)] == ['3001.aurora', '3003.aurora']
    assert len(list((base / 'state').glob('t000.stale.3001.*'))) == 1


def _race(state, i, q):
    os.environ['PBS_JOBID'] = f'r{i}'
    q.put(Claimer(state, f'r{i}', stale_s=2).try_claim('x'))


def test_takeover_race_has_one_winner(base):
    state = base / 'state'
    state.mkdir()
    ctx = mp.get_context('fork')
    for trial in range(15):
        for f in state.glob('x.*'):
            f.unlink()
        c = state / 'x.claim'
        c.write_text(json.dumps({'job': 'dead'}))
        os.utime(c, (time.time() - 100, time.time() - 100))
        q = ctx.Queue()
        ps = [ctx.Process(target=_race, args=(state, i, q)) for i in range(12)]
        [p.start() for p in ps]
        [p.join() for p in ps]
        wins = [q.get() for _ in ps]
        assert sum(wins) == 1, (trial, wins)
        assert json.loads(c.read_text())['job'].startswith('r')


def test_failed_task_retried_up_to_max_tries(base):
    write_tasks(base, 1, 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; exit 3')
    for job in ('4001', '4002', '4003'):
        assert farm(base, job, 0, 1).wait(60) == 0
    assert [r[1] for r in runs(base)] == ['4001.aurora', '4002.aurora']     # 2 tries, third job leaves it
    assert len((base / 'state' / 't000.failed').read_text().splitlines()) == 2
    assert not (base / 'state' / 't000.claim').exists()


def test_lost_claim_kills_task_without_done(base):
    write_tasks(base, 1, 'echo "$TASK_ID $PBS_JOBID start" >> $RUNLOG; sleep 20; echo "$TASK_ID $PBS_JOBID end" >> $RUNLOG')
    a = farm(base, '5001', 0, 1)
    time.sleep(1.5)
    (base / 'state' / 't000.claim').write_text(json.dumps({'job': 'other', 'host': 'h', 'pid': 1}))
    assert a.wait(30) == 0
    assert not (base / 'state' / 't000.done').exists()
    assert json.loads((base / 'state' / 't000.claim').read_text())['job'] == 'other'   # untouched
    s = json.loads(next((base / 'state').glob('rank*.5001.json')).read_text())
    assert s['lost'] == ['t000'] and [r[2] for r in runs(base)] == ['start']


def test_done_between_listing_and_claim(base):
    state = base / 'state'
    state.mkdir()
    (state / 'y.done').write_text('{}')
    assert not Claimer(state, 'j').try_claim('y')
    assert not (state / 'y.claim').exists()
