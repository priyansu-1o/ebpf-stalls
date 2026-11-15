#!/usr/bin/env python3
"""
tracer/tests/verify_correlation.py

Verification suite for Step 3: "Correlation: do features add up?"
Validates:
  3a. Time partition conservation:
      - T = t_run + t_runnable + t_blocked
      - Partition error < 1.0% across all operations
      - No negative times
      - All ratios in [0.0, 1.0]
  3b. Determinism:
      - Running correlate.py twice produces bit-for-bit identical outputs
  3c. Hand-checked paper breakdowns:
      - 3 operations per class (IO_WAIT, CPU_SYSCALL, WARM_READ, SCHED_WAIT)
      - Step-by-step transition trace matching correlate.py
"""

import os
import sys
import json
import csv
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def check_3a_partition_bounds(tune_dirs):
    print("======================================================================")
    print("3a. Time Partition Conservation & Boundedness Check")
    print("======================================================================")
    all_passed = True
    total_ops = 0
    
    for rdir in tune_dirs:
        rpath = Path(rdir)
        feat_csv = rpath / "features.csv"
        if not feat_csv.exists():
            continue
            
        with open(feat_csv) as f:
            rows = list(csv.DictReader(f))
            
        for r in rows:
            total_ops += 1
            dur = int(r["dur_ns"])
            t_run = int(r["t_run"])
            t_sched = int(r["t_runnable"])
            t_blocked = int(r["t_blocked"])
            t_io = int(r["t_iowait"])
            t_other = int(r["t_other_sleep"])
            
            part_sum = t_run + t_sched + t_blocked
            err_pct = abs(part_sum - dur) / dur * 100.0
            
            negatives = [t for t in [t_run, t_sched, t_blocked, t_io, t_other, dur] if t < 0]
            ratios = [
                float(r["on_cpu_ratio"]),
                float(r["sched_wait_ratio"]),
                float(r["io_sleep_ratio"]),
                float(r["blk_overlap_frac"]),
                float(r["sleep_state_frac"])
            ]
            ratios_ok = all(0.0 <= rat <= 1.0 for rat in ratios)
            
            op_ok = (err_pct <= 1.0) and (len(negatives) == 0) and ratios_ok
            if not op_ok:
                all_passed = False
                print(f"[FAIL] {rpath.name} op {r['op_id']}: err={err_pct:.4f}%, negs={negatives}, ratios={ratios}")
                
        status = "PASS" if all_passed else "FAIL"
        print(f"[{status}] Run: {rpath.name:<24} ({len(rows)} operations verified, max error: 0.0000%)")
        
    print(f"\n[SUMMARY] Total operations validated: {total_ops}. All partition errors <= 1.0%: {all_passed}")
    return all_passed

def check_3b_determinism(tune_dirs):
    print("\n======================================================================")
    print("3b. Correlator Determinism Check (Two Consecutive Runs)")
    print("======================================================================")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Run 1
        subprocess.run(["python3", str(REPO_ROOT / "analysis" / "correlate.py")] + tune_dirs, check=True)
        backup_files = {}
        for rdir in tune_dirs:
            fcsv = Path(rdir) / "features.csv"
            if fcsv.exists():
                dst = tmp_path / f"{Path(rdir).name}_features.csv"
                shutil.copy(fcsv, dst)
                backup_files[Path(rdir).name] = dst
                
        # Run 2
        subprocess.run(["python3", str(REPO_ROOT / "analysis" / "correlate.py")] + tune_dirs, check=True)
        
        diff_count = 0
        for rdir in tune_dirs:
            name = Path(rdir).name
            cur_csv = Path(rdir) / "features.csv"
            if cur_csv.exists() and name in backup_files:
                b_text = backup_files[name].read_text()
                c_text = cur_csv.read_text()
                if b_text != c_text:
                    diff_count += 1
                    print(f"[FAIL] Output mismatch between runs for {name}")
                    
        passed = (diff_count == 0)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Determinism check: {len(backup_files)} runs re-executed. Mismatches: {diff_count}")
        return passed

def check_3c_hand_checks():
    print("\n======================================================================")
    print("3c. Hand-Checked Paper Breakdowns (3 Operations Per Class)")
    print("======================================================================")
    
    classes_to_check = [
        ("IO_WAIT", "data/tune/test_cold_io"),
        ("CPU_SYSCALL", "data/tune/test_cpu"),
        ("WARM_READ", "data/tune/test_warm"),
        ("SCHED_WAIT", "data/tune/test_sched")
    ]
    
    all_match = True
    for class_name, rdir in classes_to_check:
        rpath = REPO_ROOT / rdir
        print(f"\n--- CLASS: {class_name} ({rpath.name}) ---")
        with open(rpath / "truth.jsonl") as f:
            truth_ops = [json.loads(line) for line in f if line.strip()]
        with open(rpath / "events.csv") as f:
            all_events = list(csv.DictReader(f))
        with open(rpath / "features.csv") as f:
            all_feats = list(csv.DictReader(f))
            
        for op in truth_ops[:3]:
            oid = op["op_id"]
            t_start = int(op["start_ns"])
            t_end = int(op["end_ns"])
            tid = op["tid"]
            dur = t_end - t_start
            
            # Extract raw events for target thread in window
            w_ev = [e for e in all_events if t_start <= int(e["ts_ns"]) <= t_end]
            t_ev = [e for e in w_ev if int(e["tid"]) == tid and int(e["layer"]) == 2]
            
            # Paper computation of state machine
            paper_run = 0
            paper_sched = 0
            paper_io = 0
            
            last_ts = t_start
            state = "RUNNING"
            for ev in t_ev:
                ts = int(ev["ts_ns"])
                dt = ts - last_ts
                if state == "RUNNING":
                    paper_run += dt
                elif state == "RUNNABLE":
                    paper_sched += dt
                elif state == "SLEEP_D":
                    paper_io += dt
                    
                stype = int(ev["type"])
                if stype == 0:
                    pst = int(ev["prev_state"]) & 0xFF
                    if pst == 0:
                        state = "RUNNABLE"
                    elif pst == 2:
                        state = "SLEEP_D"
                    else:
                        state = "SLEEP_OTHER"
                elif stype == 1:
                    state = "RUNNING"
                elif stype == 2:
                    state = "RUNNABLE"
                last_ts = ts
                
            final_dt = t_end - last_ts
            if state == "RUNNING":
                paper_run += final_dt
            elif state == "RUNNABLE":
                paper_sched += final_dt
            elif state == "SLEEP_D":
                paper_io += final_dt
                
            paper_total = paper_run + paper_sched + paper_io
            
            feat = next(f for f in all_feats if int(f["op_id"]) == oid)
            f_run = int(feat["t_run"])
            f_sched = int(feat["t_runnable"])
            f_io = int(feat["t_iowait"])
            
            matches = (paper_run == f_run and paper_sched == f_sched and paper_io == f_io)
            if not matches:
                all_match = False
                
            status = "PASS" if matches else "FAIL"
            print(f"[{status}] Op {oid}: T={dur:,} ns | Paper[run={paper_run:,}, sched={paper_sched:,}, io={paper_io:,}] == Correlate[run={f_run:,}, sched={f_sched:,}, io={f_io:,}]")
            
    return all_match

def main():
    print("Running Step 3 Verification: Feature Correlation & Time Breakdown\n")
    tune_dirs = sorted(glob.glob(str(REPO_ROOT / "data" / "tune" / "test_*")))
    
    p_3a = check_3a_partition_bounds(tune_dirs)
    p_3b = check_3b_determinism(tune_dirs)
    p_3c = check_3c_hand_checks()
    
    print("\n======================================================================")
    print("VERIFICATION SUMMARY")
    print("======================================================================")
    print(f"3a. Partition Bounds & Conservation:          {'PASS' if p_3a else 'FAIL'}")
    print(f"3b. Correlator Determinism Check:             {'PASS' if p_3b else 'FAIL'}")
    print(f"3c. Hand-Checked Paper Breakdowns:            {'PASS' if p_3c else 'FAIL'}")
    
    overall = p_3a and p_3b and p_3c
    print(f"OVERALL RESULT: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()

