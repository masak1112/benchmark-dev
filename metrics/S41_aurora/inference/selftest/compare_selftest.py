"""Compare a self-test member file with the production one: every data variable (27 fields + zm_u, zm_t), NaN-aware.
PASS = bitwise identical values (same seed, same device type; observed for UC-S2S on Aurora, job 9077592).
    python compare_selftest.py NEW_h25.nc PRODUCTION_h25.nc
"""
import sys

import numpy as np
import xarray as xr


def main(new, old):
    a, b = xr.open_dataset(new), xr.open_dataset(old)
    bad = []
    for v in b.data_vars:
        if v not in a:
            bad.append(f"{v}: missing in new"); continue
        x, y = a[v].values, b[v].values
        if x.shape != y.shape:
            bad.append(f"{v}: shape {x.shape} vs {y.shape}"); continue
        same = np.array_equal(x, y, equal_nan=True)
        d = np.nanmax(np.abs(x.astype(np.float64) - y)) if not same else 0.0
        print(f"{v:8s} bitwise={same} max|diff|={d:.3g}")
        if not same:
            bad.append(v)
    t_ok = np.array_equal(a.time.values, b.time.values)
    print("time equal:", t_ok)
    print("PASS" if not bad and t_ok else f"FAIL: {bad}")
    return 0 if not bad and t_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
