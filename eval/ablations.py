#!/usr/bin/env python3
"""
Scientific Ablation Studies: Isolating Layer Values and Window Margins.

Quantifies the marginal value of each telemetry layer:
1. Ablation 1 (Drop Block Layer): Removes Layer 3 storage telemetry.
2. Ablation 2 (Drop Scheduler Layer): Removes Layer 2 CPU runqueue & sleep telemetry.
3. Ablation 3 (Syscall Boundary Only): Removes Layers 2 and 3.
4. Full Fused Model: All layers active.

Generates:
- Rule-based & ML ablation comparisons
- Publication-quality tradeoff figure: eval/plots/ablation_study.png
"""

import sys
import json
import glob
from pathlib import Path

_site_pkg = Path(__file__).resolve().parent.parent / ".site-packages"
if _site_pkg.exists() and str(_site_pkg) not in sys.path:
    sys.path.insert(0, str(_site_pkg))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

try:
    from eval.metrics import CANONICAL_CLASSES, normalize_class
except ImportError:
    from metrics import CANONICAL_CLASSES, normalize_class

try:
    from analysis.rules import classify_row, THRESHOLDS
except ImportError:
    try:
        from rules import classify_row, THRESHOLDS
    except ImportError:
        classify_row = None
        THRESHOLDS = {
            "io_sleep_ratio_min": 0.18,
            "sched_wait_ratio_min": 0.45,
            "on_cpu_ratio_min": 0.80,
            "warm_read_dur_max_ms": 2.0,
        }

def classify_ablated_rules(row: dict, ablation_mode: str) -> str:
    """Classifies an operation under specific probe layer ablations."""
    dur_ms = float(row.get("duration_ms", 0.0))
    on_cpu = float(row.get("on_cpu_ratio", 0.0))
    sched_wait = float(row.get("sched_wait_ratio", 0.0))
    io_sleep = float(row.get("io_sleep_ratio", 0.0))
    block_issues = int(row.get("block_issues", 0))
    sys_enters = int(row.get("sys_enters", 0))

    if ablation_mode == "Full Fused Model":
        if classify_row is not None:
            return classify_row(row)[0]
        if io_sleep >= THRESHOLDS.get("abstain_phantom_io_sleep_min", 0.18) and block_issues == 0:
            return "INDETERMINATE"
        if block_issues > 0 and on_cpu >= THRESHOLDS.get("abstain_bg_noise_on_cpu_min", 0.80):
            return "INDETERMINATE"
        if io_sleep >= THRESHOLDS.get("io_sleep_ratio_min", 0.18) and block_issues > 0:
            return "IO_WAIT"
        if sched_wait >= THRESHOLDS.get("sched_wait_ratio_min", 0.45):
            return "SCHED_WAIT"
        if dur_ms <= THRESHOLDS.get("warm_read_dur_max_ms", 2.0) and block_issues == 0 and sys_enters > 0:
            return "WARM_READ"
        if on_cpu >= THRESHOLDS.get("on_cpu_ratio_min", 0.80) and block_issues == 0:
            return "CPU_SYSCALL"
        return "INDETERMINATE"

    elif ablation_mode == "No Block Layer":
        # Layer 3 stripped: no block events to confirm disk I/O.
        # Without block evidence, thread sleep cannot be verified as IO_WAIT.
        if sched_wait >= THRESHOLDS.get("sched_wait_ratio_min", 0.45):
            return "SCHED_WAIT"
        if dur_ms <= THRESHOLDS.get("warm_read_dur_max_ms", 2.0) and sys_enters > 0:
            return "WARM_READ"
        if on_cpu >= THRESHOLDS.get("on_cpu_ratio_min", 0.80):
            return "CPU_SYSCALL"
        return "INDETERMINATE"

    elif ablation_mode == "No Scheduler Layer":
        # Layer 2 stripped: blind to on_cpu, sched_wait, and sleep states.
        # Cannot detect runqueue latency -> SCHED_WAIT cannot be identified.
        if block_issues > 0 and dur_ms > THRESHOLDS.get("warm_read_dur_max_ms", 2.0):
            return "IO_WAIT"
        if dur_ms <= THRESHOLDS.get("warm_read_dur_max_ms", 2.0) and sys_enters > 0:
            return "WARM_READ"
        if dur_ms > 10.0 and sys_enters == 0:
            return "CPU_SYSCALL"
        return "INDETERMINATE"

    elif ablation_mode == "Syscall Boundary Only":
        # Layers 2 & 3 stripped: naive duration heuristics only.
        if dur_ms > 10.0 and sys_enters == 0:
            return "CPU_SYSCALL"
        if dur_ms <= THRESHOLDS.get("warm_read_dur_max_ms", 2.0) and sys_enters > 0:
            return "WARM_READ"
        if dur_ms > THRESHOLDS.get("warm_read_dur_max_ms", 2.0) and sys_enters > 0:
            return "IO_WAIT"
        return "INDETERMINATE"
        return "INDETERMINATE"

    return "INDETERMINATE"

def load_feature_records(runs_dir: str):
    target_path = Path(runs_dir)
    if target_path.is_dir():
        if (target_path / "truth.jsonl").exists():
            run_dirs = [target_path]
        else:
            run_dirs = [p for p in target_path.glob("**/") if (p / "truth.jsonl").exists()]
    else:
        run_dirs = [Path(m) for m in glob.glob(runs_dir) if (Path(m) / "truth.jsonl").exists()]

    records = []
    for rdir in run_dirs:
        feat_parquet = rdir / "features.parquet"
        feat_csv = rdir / "features.csv"

        df = None
        if feat_parquet.exists():
            try:
                df = pd.read_parquet(feat_parquet)
            except Exception:
                pass
        if df is None and feat_csv.exists():
            try:
                df = pd.read_csv(feat_csv)
            except Exception:
                pass

        if df is not None and not df.empty:
            for row in df.to_dict(orient="records"):
                row["_run_id"] = rdir.name
                records.append(row)

    return pd.DataFrame(records)

def run_layer_ablations(runs_dir: str):
    df = load_feature_records(runs_dir)
    if df.empty or len(df) < 2:
        print(f"Insufficient samples in {runs_dir} for ablation studies.")
        return

    df["target"] = df["true_label"].apply(normalize_class)
    y_true = df["target"].values

    modes = [
        "Full Fused Model",
        "No Block Layer",
        "No Scheduler Layer",
        "Syscall Boundary Only"
    ]

    rule_scores = {}
    abstention_rates = {}

    print("\n" + "=" * 95)
    print(" LAYER ABLATION BENCHMARKS: PROBE REMOVAL IMPACT")
    print("=" * 95)
    print(f"{'Ablation Configuration':<24} | {'Macro-F1':<10} | {'IO_WAIT Rec':<12} | {'SCHED_WAIT Rec':<14} | {'Abstain %':<10} | {'Decided':<8}")
    print("-" * 95)

    for mode in modes:
        preds = [classify_ablated_rules(row, mode) for row in df.to_dict(orient="records")]
        
        # Calculate decided metrics
        decided_indices = [i for i, p in enumerate(preds) if p != "INDETERMINATE"]
        total = len(preds)
        abstained = total - len(decided_indices)
        abst_pct = (abstained / total) * 100.0
        abstention_rates[mode] = abst_pct

        if decided_indices:
            y_dec_true = [y_true[i] for i in decided_indices]
            y_dec_pred = [preds[i] for i in decided_indices]
            f1 = f1_score(y_dec_true, y_dec_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
            from sklearn.metrics import recall_score
            recalls = recall_score(y_dec_true, y_dec_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
            io_rec = recalls[CANONICAL_CLASSES.index("IO_WAIT")]
            sched_rec = recalls[CANONICAL_CLASSES.index("SCHED_WAIT")]
        else:
            f1 = 0.0
            io_rec = 0.0
            sched_rec = 0.0

        rule_scores[mode] = f1
        print(f"{mode:<24} | {f1:<10.4f} | {io_rec:<12.4f} | {sched_rec:<14.4f} | {abst_pct:<9.1f}% | {len(decided_indices):<8}")
    print("=" * 95)

    # Plotting Ablation Impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))

    names = list(rule_scores.keys())
    scores = list(rule_scores.values())
    absts = [abstention_rates[m] for m in names]

    colors = ["#2ca02c" if "Full" in n else "#d62728" for n in names]
    bars1 = ax1.barh(names, scores, color=colors, height=0.5)
    ax1.set_xlabel("Macro-F1 Score (Decided Instances)", fontsize=10)
    ax1.set_xlim(0, 1.15)
    ax1.set_title("Classification Quality Degradation", fontsize=11, fontweight="bold")
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f"{w:.3f}",
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=9, fontweight="bold")

    bars2 = ax2.barh(names, absts, color="#ff7f0e", height=0.5)
    ax2.set_xlabel("Abstention Rate (%)", fontsize=10)
    ax2.set_xlim(0, 100)
    ax2.set_title("Uncertainty / Abstention Triggered", fontsize=11, fontweight="bold")
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.1f}%",
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=9)

    fig.suptitle("Layer Ablation Study: Impact of Stripping Observability Layers", fontsize=12, y=0.98, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_out = Path("eval/plots/ablation_study.png")
    plt.savefig(plot_out, dpi=200)
    plt.close(fig)
    print(f"\nSaved ablation figure to {plot_out}")

def extract_window_features(events: list[dict], t_start: int, t_end: int, tid: int) -> dict:
    dur_ns = max(t_end - t_start, 1)
    dur_ms = dur_ns / 1e6

    w_rows = [r for r in events if t_start <= int(r["ts_ns"]) <= t_end]
    t_rows = sorted([r for r in w_rows if int(r.get("tid", 0)) == tid], key=lambda x: int(x["ts_ns"]))

    b_rows = [r for r in w_rows if int(r.get("layer", 0)) == 3]
    block_issues = sum(1 for r in b_rows if int(r.get("type", -1)) == 0)

    sched_rows = [r for r in t_rows if int(r.get("layer", 0)) == 2]
    on_cpu_ns = 0
    sched_wait_ns = 0
    io_sleep_ns = 0

    last_ts = t_start
    curr_state = "RUNNING"

    for r in sched_rows:
        ts = int(r["ts_ns"])
        dt = max(ts - last_ts, 0)
        if curr_state == "RUNNING":
            on_cpu_ns += dt
        elif curr_state == "RUNNABLE":
            sched_wait_ns += dt
        elif curr_state == "SLEEP_D":
            io_sleep_ns += dt

        stype = int(r.get("type", -1))
        if stype == 0:
            pst = int(r.get("prev_state", 0)) & 0xFF
            if pst == 0:
                curr_state = "RUNNABLE"
            elif pst == 2:
                curr_state = "SLEEP_D"
            else:
                curr_state = "SLEEP_OTHER"
        elif stype == 1:
            curr_state = "RUNNING"
        elif stype == 2:
            curr_state = "RUNNABLE"

        last_ts = ts

    final_dt = max(t_end - last_ts, 0)
    if curr_state == "RUNNING":
        on_cpu_ns += final_dt
    elif curr_state == "RUNNABLE":
        sched_wait_ns += final_dt
    elif curr_state == "SLEEP_D":
        io_sleep_ns += final_dt

    sys_enters = sum(1 for r in t_rows if int(r.get("layer", 0)) == 1 and int(r.get("type", -1)) == 0)

    return {
        "duration_ms": dur_ms,
        "on_cpu_ratio": min(on_cpu_ns / dur_ns, 1.0),
        "sched_wait_ratio": min(sched_wait_ns / dur_ns, 1.0),
        "io_sleep_ratio": min(io_sleep_ns / dur_ns, 1.0),
        "block_issues": block_issues,
        "sys_enters": sys_enters,
    }

def run_window_margin_ablation(runs_dir: str):
    """
    Evaluates rules classifier robustness across window margins:
    Recomputes features per margin (0, 1, 10, 100 ms, 1 s, 5 s) and reports Macro-F1 and coverage.
    If insufficient real multi-class data exists, prints 'needs real runs' and returns.
    """
    import csv
    target_path = Path(runs_dir)
    if target_path.is_dir():
        if (target_path / "truth.jsonl").exists():
            run_dirs = [target_path]
        else:
            run_dirs = [p for p in target_path.glob("**/") if (p / "truth.jsonl").exists()]
    else:
        run_dirs = [Path(m) for m in glob.glob(runs_dir) if (Path(m) / "truth.jsonl").exists()]

    # Collect operations across runs
    all_runs_data = []
    for rdir in sorted(run_dirs):
        truth_file = rdir / "truth.jsonl"
        events_file = rdir / "events.csv"
        if not truth_file.exists() or not events_file.exists():
            continue
        try:
            truth = [json.loads(l) for l in truth_file.read_text().splitlines() if l.strip()]
            with events_file.open() as f:
                events = list(csv.DictReader(f))
        except (json.JSONDecodeError, csv.Error):
            continue

        all_runs_data.append((rdir, truth, events))

    total_ops = sum(len(truth) for _, truth, _ in all_runs_data)
    unique_labels = set(
        normalize_class(op.get("true_label", op.get("label", "")))
        for _, truth, _ in all_runs_data
        for op in truth
    )

    print("\n" + "=" * 80)
    print(" WINDOW MARGIN SWEEP: RULES CLASSIFIER MACRO-F1 & COVERAGE VS. MARGIN (L1)")
    print("=" * 80)

    # Check for sufficient real data
    if total_ops < 10 or len(unique_labels) < 2 or classify_row is None:
        print(f"needs real runs (found {total_ops} operations, {len(unique_labels)} classes in '{runs_dir}')")
        print("=" * 80)
        return

    deltas_ms = [0, 1, 10, 100, 1000, 5000]
    print(f"{'Window Margin':<20} | {'Coverage %':<14} | {'Macro-F1 (Decided)':<20} | {'Decided':<8}")
    print("-" * 80)

    for d in deltas_ms:
        d_ns = int(d * 1e6)
        y_true = []
        preds = []

        for rdir, truth, events in all_runs_data:
            for op in truth:
                t0 = max(0, int(op["start_ns"]) - d_ns)
                t1 = int(op["end_ns"]) + d_ns
                tid = int(op.get("tid", 0))

                feat = extract_window_features(events, t0, t1, tid)
                pred, _ = classify_row(feat)

                y_true.append(normalize_class(op.get("true_label", op.get("label", ""))))
                preds.append(pred)

        total = len(preds)
        decided_idx = [i for i, p in enumerate(preds) if p != "INDETERMINATE"]
        cov_pct = (len(decided_idx) / total * 100.0) if total > 0 else 0.0

        if decided_idx:
            yt_dec = [y_true[i] for i in decided_idx]
            yp_dec = [preds[i] for i in decided_idx]
            f1_dec = f1_score(yt_dec, yp_dec, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        else:
            f1_dec = 0.0

        label_d = f"±{d} ms" if d < 1000 else f"±{d // 1000} s"
        print(f"{label_d:<20} | {cov_pct:6.2f}%       | {f1_dec:10.4f}           | {len(decided_idx):<8}")
    print("=" * 80)

def run_attribution_ablation(runs_dir: str):
    """
    Evaluates block I/O submitter attribution accuracy against known ground-truth bytes (N)
    for cold read operations (with and without background fio noise).
    Compares three distinct attribution rules (no 'or' combinations):
      1. issuing_tid: matches ev['tid'] == op['tid']
      2. target_tid:  matches ev['target_tid'] == op['tid'] (bio_submitters)
      3. pid:         matches ev['pid'] == op_pid
    """
    import csv
    target_path = Path(runs_dir)
    if target_path.is_dir():
        if (target_path / "truth.jsonl").exists():
            run_dirs = [target_path]
        else:
            run_dirs = [p for p in target_path.glob("**/") if (p / "truth.jsonl").exists()]
    else:
        run_dirs = [Path(m) for m in glob.glob(runs_dir) if (Path(m) / "truth.jsonl").exists()]

    op_records = []

    for rdir in sorted(run_dirs):
        truth_file = rdir / "truth.jsonl"
        events_file = rdir / "events.csv"
        run_file = rdir / "run.json"
        if not truth_file.exists() or not events_file.exists():
            continue

        bg_noise = "none"
        if run_file.exists():
            try:
                run_meta = json.loads(run_file.read_text())
                bg_noise = run_meta.get("background_noise", "none")
            except json.JSONDecodeError:
                pass

        try:
            truth = [json.loads(l) for l in truth_file.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue

        with events_file.open() as f:
            events = list(csv.DictReader(f))

        for op in truth:
            known_bytes = int(op.get("bytes", 0))
            if known_bytes <= 0:
                continue

            syscall = str(op.get("syscall", "")).lower()
            is_write = "write" in syscall or "write" in bg_noise.lower() or "write" in rdir.name.lower()
            is_read = not is_write

            t0 = int(op["start_ns"])
            t1 = int(op["end_ns"])
            op_tid = int(op.get("tid", 0))
            op_pid = int(op.get("pid", op_tid))

            bytes_issuing_tid = 0
            bytes_target_tid = 0
            bytes_pid = 0

            for ev in events:
                ts = int(ev["ts_ns"])
                if t0 <= ts <= t1 and int(ev.get("layer", 0)) == 3 and int(ev.get("type", -1)) == 0:
                    rwbs = str(ev.get("rwbs", ""))
                    if is_read and "R" not in rwbs:
                        continue
                    if is_write and "W" not in rwbs:
                        continue

                    ev_target_tid = int(ev.get("target_tid", 0)) if ev.get("target_tid") else 0
                    ev_tid = int(ev.get("tid", 0)) if ev.get("tid") else 0
                    ev_pid = int(ev.get("pid", 0)) if ev.get("pid") else 0
                    nr_bytes = int(ev.get("nr_bytes", 0)) if ev.get("nr_bytes") else 0

                    if ev_tid == op_tid:
                        bytes_issuing_tid += nr_bytes
                    if ev_target_tid == op_tid:
                        bytes_target_tid += nr_bytes
                    if ev_pid == op_pid:
                        bytes_pid += nr_bytes

            op_records.append({
                "run_id": rdir.name,
                "op_id": op.get("op_id"),
                "bg_noise": bg_noise,
                "known_bytes": known_bytes,
                "bytes_issuing_tid": bytes_issuing_tid,
                "diff_issuing_tid": bytes_issuing_tid - known_bytes,
                "bytes_target_tid": bytes_target_tid,
                "diff_target_tid": bytes_target_tid - known_bytes,
                "bytes_pid": bytes_pid,
                "diff_pid": bytes_pid - known_bytes,
            })

    print("\n" + "=" * 90)
    print(" ATTRIBUTION ABLATION: ERROR AGAINST KNOWN BYTES (L4)")
    print("=" * 90)

    if not op_records:
        print(f"No operations with known byte counts (cold reads) found in '{runs_dir}' (needs real runs).")
        print("=" * 90)
        return

    df_ops = pd.DataFrame(op_records)
    print(f"{'Condition':<20} | {'Rule':<28} | {'Known Bytes':<12} | {'Attributed':<12} | {'Diff (Bytes)':<12}")
    print("-" * 90)

    for bg, grp in df_ops.groupby("bg_noise"):
        tot_known = grp["known_bytes"].sum()
        tot_issuing = grp["bytes_issuing_tid"].sum()
        tot_target = grp["bytes_target_tid"].sum()
        tot_pid = grp["bytes_pid"].sum()

        cond_label = f"bg={bg} (n={len(grp)})"
        print(f"{cond_label:<20} | {'1. Issuing TID':<28} | {tot_known:<12} | {tot_issuing:<12} | {tot_issuing - tot_known:+12d}")
        print(f"{'':<20} | {'2. Target TID (bio_submitters)':<28} | {tot_known:<12} | {tot_target:<12} | {tot_target - tot_known:+12d}")
        print(f"{'':<20} | {'3. Naive PID':<28} | {tot_known:<12} | {tot_pid:<12} | {tot_pid - tot_known:+12d}")
        print("-" * 90)
    print("=" * 90)

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tune"
    run_layer_ablations(target_dir)
    run_window_margin_ablation(target_dir)
    run_attribution_ablation(target_dir)

if __name__ == "__main__":
    main()
