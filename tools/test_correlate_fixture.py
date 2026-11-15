#!/usr/bin/env python3
"""
Unit test verifying analysis/correlate.py against hand-calculated transitions.
Uses tools/testdata/ fixture (NOT evaluation data).
"""

import sys
import csv
from pathlib import Path

_site_pkg = Path(__file__).resolve().parent.parent / ".site-packages"
if _site_pkg.exists() and str(_site_pkg) not in sys.path:
    sys.path.insert(0, str(_site_pkg))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.correlate import correlate_single_run

def test_correlate_fixture():
    test_dir = REPO_ROOT / "tools" / "testdata"
    correlate_single_run(str(test_dir))

    feat_csv = test_dir / "features.csv"
    assert feat_csv.exists(), "features.csv not generated for unit test fixture"

    with open(feat_csv) as f:
        rows = {int(r["op_id"]): r for r in csv.DictReader(f)}

    # Hand-computed expectations:
    # Op 1: D-state sleep (prev_state=2, 6ms) + wakeup (2ms runnable) + 2ms running
    op1 = rows[1]
    assert int(op1["t_run"]) == 2000000, f"Op1 t_run expected 2000000, got {op1['t_run']}"
    assert int(op1["t_runnable"]) == 2000000, f"Op1 t_runnable expected 2000000, got {op1['t_runnable']}"
    assert int(op1["t_blocked"]) == 6000000, f"Op1 t_blocked expected 6000000, got {op1['t_blocked']}"
    assert int(op1["t_iowait"]) == 6000000, f"Op1 t_iowait expected 6000000, got {op1['t_iowait']}"
    assert int(op1["t_other_sleep"]) == 0, f"Op1 t_other_sleep expected 0, got {op1['t_other_sleep']}"
    assert int(op1["unknown_ns"]) == 0, f"Op1 unknown_ns expected 0, got {op1['unknown_ns']}"
    assert int(op1["residual_ns"]) == 0, f"Op1 residual_ns expected 0, got {op1['residual_ns']}"
    assert int(op1["cpu_gap_ns"]) == 0, f"Op1 cpu_gap_ns expected 0, got {op1['cpu_gap_ns']}"
    assert int(op1["wait_gap_ns"]) == 0, f"Op1 wait_gap_ns expected 0, got {op1['wait_gap_ns']}"
    print("Op 1 (D-state sleep): PASS (t_run=2ms, t_runnable=2ms, t_iowait=6ms, residual=0)")

    # Op 2: Interruptible sleep (prev_state=1, 3ms) + wakeup (1ms runnable) + 6ms running
    op2 = rows[2]
    assert int(op2["t_run"]) == 6000000, f"Op2 t_run expected 6000000, got {op2['t_run']}"
    assert int(op2["t_runnable"]) == 1000000, f"Op2 t_runnable expected 1000000, got {op2['t_runnable']}"
    assert int(op2["t_blocked"]) == 3000000, f"Op2 t_blocked expected 3000000, got {op2['t_blocked']}"
    assert int(op2["t_iowait"]) == 0, f"Op2 t_iowait expected 0, got {op2['t_iowait']}"
    assert int(op2["t_other_sleep"]) == 3000000, f"Op2 t_other_sleep expected 3000000, got {op2['t_other_sleep']}"
    assert int(op2["unknown_ns"]) == 0, f"Op2 unknown_ns expected 0, got {op2['unknown_ns']}"
    assert int(op2["residual_ns"]) == 0, f"Op2 residual_ns expected 0, got {op2['residual_ns']}"
    assert int(op2["cpu_gap_ns"]) == 0, f"Op2 cpu_gap_ns expected 0, got {op2['cpu_gap_ns']}"
    assert int(op2["wait_gap_ns"]) == 0, f"Op2 wait_gap_ns expected 0, got {op2['wait_gap_ns']}"
    print("Op 2 (Interruptible sleep): PASS (t_run=6ms, t_runnable=1ms, t_other_sleep=3ms, residual=0)")

    # Op 3: Preempted runnable wait (prev_state=0, 5ms) + switch_in + 5ms running
    op3 = rows[3]
    assert int(op3["t_run"]) == 5000000, f"Op3 t_run expected 5000000, got {op3['t_run']}"
    assert int(op3["t_runnable"]) == 5000000, f"Op3 t_runnable expected 5000000, got {op3['t_runnable']}"
    assert int(op3["t_blocked"]) == 0, f"Op3 t_blocked expected 0, got {op3['t_blocked']}"
    assert int(op3["t_iowait"]) == 0, f"Op3 t_iowait expected 0, got {op3['t_iowait']}"
    assert int(op3["t_other_sleep"]) == 0, f"Op3 t_other_sleep expected 0, got {op3['t_other_sleep']}"
    assert int(op3["unknown_ns"]) == 0, f"Op3 unknown_ns expected 0, got {op3['unknown_ns']}"
    assert int(op3["residual_ns"]) == 0, f"Op3 residual_ns expected 0, got {op3['residual_ns']}"
    assert int(op3["cpu_gap_ns"]) == 0, f"Op3 cpu_gap_ns expected 0, got {op3['cpu_gap_ns']}"
    assert int(op3["wait_gap_ns"]) == 0, f"Op3 wait_gap_ns expected 0, got {op3['wait_gap_ns']}"
    print("Op 3 (Preemption wait): PASS (t_run=5ms, t_runnable=5ms, t_blocked=0, residual=0)")

    print("\nAll unit test assertions passed. Correlate breakdown matches hand calculations.")

if __name__ == "__main__":
    test_correlate_fixture()

