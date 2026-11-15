#!/usr/bin/env python3
"""
Single-Layer Baselines vs. Fused Observability Evaluator.

Hypothesis H1 states that fusing multi-layer telemetry (syscall + block + sched)
significantly outperforms any isolated single-layer signal.

This script implements:
1. Syscall-Duration-Only Baseline (duration > 5.0 ms -> IO_WAIT, else WARM_READ)
2. Block-Metrics-Only Baseline (block_issues > 0 -> IO_WAIT, else CPU_STALL)
3. Scheduler-Only Baseline (sched latency only, blind to block layer)
4. Fused Heuristic Rules & Fused Decision Tree comparison
5. Generates comparison bar plot: eval/plots/baseline_comparison.png
"""

import sys
import json
import glob
from pathlib import Path

_site_pkg = Path(__file__).resolve().parent.parent / ".site-packages"
if _site_pkg.exists() and str(_site_pkg) not in sys.path:
    sys.path.insert(0, str(_site_pkg))

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import evaluate_predictions, CANONICAL_CLASSES

def predict_syscall_only(row: dict) -> tuple[str, float]:
    """Baseline 1: Uses duration and syscall presence only."""
    dur_ms = float(row.get("duration_ms", 0.0))
    sys_enters = int(row.get("sys_enters", 0))

    if sys_enters == 0 and dur_ms > 10.0:
        return "CPU_SYSCALL", 0.60
    if dur_ms > 5.0:
        return "IO_WAIT", 0.60
    return "WARM_READ", 0.70

def predict_block_only(row: dict) -> tuple[str, float]:
    block_issues = int(row.get("block_issues", 0))
    if block_issues > 0:
        return "IO_WAIT", 0.80
    return "CPU_SYSCALL", 0.60

def predict_sched_only(row: dict) -> tuple[str, float]:
    on_cpu = float(row.get("on_cpu_ratio", 0.0))
    sched_wait = float(row.get("sched_wait_ratio", 0.0))
    io_sleep = float(row.get("io_sleep_ratio", 0.0))

    if sched_wait >= 0.40:
        return "SCHED_WAIT", 0.70
    if io_sleep >= 0.40:
        return "IO_WAIT", 0.70
    if on_cpu >= 0.70:
        return "CPU_SYSCALL", 0.70
    return "WARM_READ", 0.50

def predict_paper(row: dict, dur_mean: float = 0.0, dur_std: float = 0.0, alpha: float = 0.5, tau_abstain: float = 0.0) -> tuple[str, float, int] | None:
    """
    Re-implementation of Darzi et al. (WoSC'25) correlation-based diagnosis:
    - Fixed/sliding window evaluation
    - Mean and 3-sigma spike score: (dur_ns - dur_mean) / (3.0 * dur_std)
    - If dur_std <= 0, returns None (skip window)
    - Cross-correlation / alignment to individual layer telemetry
    - Confidence: alpha * spike + (1 - alpha) * rho (alpha defaults to 0.5)
    - Optional abstention threshold tau_abstain tuned strictly on data/tune
    """
    if dur_std <= 0.0:
        return None

    dur_ns = int(row.get("dur_ns", int(float(row.get("duration_ms", 0.0)) * 1e6)))
    dur_ms = float(row.get("duration_ms", dur_ns / 1e6))
    on_cpu_r = float(row.get("on_cpu_ratio", 0.0))
    sched_r = float(row.get("sched_wait_ratio", 0.0))
    io_sleep_r = float(row.get("io_sleep_ratio", 0.0))
    blk_issues = int(row.get("block_issues", 0))
    sys_enters = int(row.get("sys_enters", 0))

    # Spike score strictly following Darzi et al.
    spike_score = min(max((dur_ns - dur_mean) / (3.0 * dur_std), 0.0), 1.0)

    # Cross-layer signal alignment proxies
    rho_io = io_sleep_r if blk_issues > 0 else (0.5 * io_sleep_r)
    rho_sched = sched_r
    rho_cpu = on_cpu_r

    s_io = alpha * spike_score + (1.0 - alpha) * rho_io
    s_sched = alpha * spike_score + (1.0 - alpha) * rho_sched
    s_cpu = alpha * spike_score + (1.0 - alpha) * rho_cpu

    # Optional abstention if tuned threshold tau_abstain > 0
    top_score = max(s_io, s_sched, s_cpu)
    if tau_abstain > 0.0 and top_score < tau_abstain:
        return "INDETERMINATE", round(top_score, 3), 0

    # Forced single-label diagnosis
    if s_io >= s_sched and s_io >= s_cpu and blk_issues > 0:
        stall_ns = int(row.get("t_iowait", int(io_sleep_r * dur_ns)))
        return "IO_WAIT", round(s_io, 3), stall_ns
    elif s_sched >= s_cpu:
        stall_ns = int(row.get("t_runnable", int(sched_r * dur_ns)))
        return "SCHED_WAIT", round(s_sched, 3), stall_ns
    else:
        if dur_ms <= 5.0 and sys_enters > 0 and blk_issues == 0:
            return "WARM_READ", round(s_cpu, 3), 0
        else:
            return "CPU_SYSCALL", round(s_cpu, 3), dur_ns

def run_single_layer_baselines(runs_dir: str, alpha: float = 0.5):
    target_path = Path(runs_dir)
    if target_path.is_dir():
        if (target_path / "truth.jsonl").exists():
            run_dirs = [target_path]
        else:
            run_dirs = [p for p in target_path.glob("**/") if (p / "truth.jsonl").exists()]
    else:
        run_dirs = [Path(m) for m in glob.glob(runs_dir) if (Path(m) / "truth.jsonl").exists()]

    total_ops = 0
    for rdir in run_dirs:
        feat_parquet = rdir / "features.parquet"
        feat_csv = rdir / "features.csv"

        rows = []
        if feat_parquet.exists():
            try:
                df = pd.read_parquet(feat_parquet)
                rows = df.to_dict(orient="records")
            except Exception:
                pass
        if not rows and feat_csv.exists():
            try:
                df = pd.read_csv(feat_csv)
                rows = df.to_dict(orient="records")
            except Exception:
                pass

        if not rows:
            continue

        total_ops += len(rows)

        # Baseline predictions
        syscall_preds = []
        block_preds = []
        sched_preds = []
        paper_preds = []
        paper_abstain_preds = []

        durs = [int(r.get("dur_ns", int(float(r.get("duration_ms", 0.0)) * 1e6))) for r in rows]
        dur_mean = float(np.mean(durs)) if (durs and np is not None) else 0.0
        dur_std = float(np.std(durs)) if (len(durs) > 1 and np is not None) else 0.0

        for row in rows:
            op_id = int(row.get("op_id", 0))
            tid = int(row.get("tid", 0))
            dur_ns = int(row.get("dur_ns", int(float(row.get("duration_ms", 0.0)) * 1e6)))
            true_lbl = str(row.get("true_label", row.get("label", "UNKNOWN")))

            p_sys, c_sys = predict_syscall_only(row)
            est_sys = dur_ns if p_sys in ("IO_WAIT", "CPU_SYSCALL") else 0
            syscall_preds.append({
                "run_id": rdir.name, "op_id": op_id, "tid": tid,
                "pred": p_sys, "pred_label": p_sys,
                "conf": c_sys, "confidence": c_sys,
                "method": "syscall_only", "est_stall_ns": est_sys,
                "true_label": true_lbl
            })

            p_blk, c_blk = predict_block_only(row)
            est_blk = int(row.get("t_iowait", int(float(row.get("io_sleep_ratio", 0)) * dur_ns))) if p_blk == "IO_WAIT" else dur_ns
            block_preds.append({
                "run_id": rdir.name, "op_id": op_id, "tid": tid,
                "pred": p_blk, "pred_label": p_blk,
                "conf": c_blk, "confidence": c_blk,
                "method": "block_only", "est_stall_ns": est_blk,
                "true_label": true_lbl
            })

            p_sch, c_sch = predict_sched_only(row)
            est_sch = int(row.get("t_runnable", int(float(row.get("sched_wait_ratio", 0)) * dur_ns))) if p_sch == "SCHED_WAIT" else dur_ns
            sched_preds.append({
                "run_id": rdir.name, "op_id": op_id, "tid": tid,
                "pred": p_sch, "pred_label": p_sch,
                "conf": c_sch, "confidence": c_sch,
                "method": "sched_only", "est_stall_ns": est_sch,
                "true_label": true_lbl
            })

            res_pap = predict_paper(row, dur_mean, dur_std, alpha=alpha, tau_abstain=0.0)
            if res_pap is None:
                print(f"Skipping paper baseline for op {op_id} in {rdir.name}: dur_std is zero.")
            else:
                p_pap, c_pap, est_pap = res_pap
                paper_preds.append({
                    "run_id": rdir.name, "op_id": op_id, "tid": tid,
                    "pred": p_pap, "pred_label": p_pap,
                    "conf": c_pap, "confidence": c_pap,
                    "method": "paper", "est_stall_ns": est_pap,
                    "true_label": true_lbl
                })

            res_pap_abs = predict_paper(row, dur_mean, dur_std, alpha=alpha, tau_abstain=0.30)
            if res_pap_abs is not None:
                p_pabs, c_pabs, est_pabs = res_pap_abs
                paper_abstain_preds.append({
                    "run_id": rdir.name, "op_id": op_id, "tid": tid,
                    "pred": p_pabs, "pred_label": p_pabs,
                    "conf": c_pabs, "confidence": c_pabs,
                    "method": "paper_abstain", "est_stall_ns": est_pabs,
                    "true_label": true_lbl
                })

        with open(rdir / "pred_syscall_only.jsonl", "w") as f:
            for p in syscall_preds:
                f.write(json.dumps(p) + "\n")

        with open(rdir / "pred_block_only.jsonl", "w") as f:
            for p in block_preds:
                f.write(json.dumps(p) + "\n")

        with open(rdir / "pred_sched_only.jsonl", "w") as f:
            for p in sched_preds:
                f.write(json.dumps(p) + "\n")

        with open(rdir / "pred_paper.jsonl", "w") as f:
            for p in paper_preds:
                f.write(json.dumps(p) + "\n")

        with open(rdir / "pred_paper_abstain.jsonl", "w") as f:
            for p in paper_abstain_preds:
                f.write(json.dumps(p) + "\n")

    print(f"Generated baseline predictions across {len(run_dirs)} runs ({total_ops} operations).")

    # Evaluate each method
    methods = [
        ("syscall_only",   "Syscall Duration Only"),
        ("block_only",     "Block Metrics Only"),
        ("sched_only",     "Scheduler Only"),
        ("paper",          "Correlation Baseline (Darzi et al.)"),
        ("paper_abstain",  "Correlation Baseline w/ Abstain (Darzi et al.)"),
        ("rules",          "Fused Heuristics (Ours)"),
        ("tree",           "Fused Decision Tree (Ours)")
    ]

    results = []
    for suffix, name in methods:
        print(f"\n---> Evaluating: {name} ({suffix})")
        res = evaluate_predictions(runs_dir, pred_suffix=suffix, save_plot=True)
        if res:
            res["display_name"] = name
            results.append(res)

    # Print Comparative Table
    print("\n" + "=" * 95)
    print(" BASELINE COMPARISON SUMMARY (Testing Hypothesis H1)")
    print("=" * 95)
    print(f"{'Method':<32} | {'All-Ops F1':<11} | {'95% Cluster CI':<18} | {'Coverage %':<12} | {'Ans Acc %':<11}")
    print("-" * 95)
    for r in results:
        ci_str = f"[{r['ci_95_all'][0]:.4f}, {r['ci_95_all'][1]:.4f}]"
        print(f"{r['display_name']:<32} | {r['macro_f1_all']:<11.4f} | {ci_str:<18} | {r['coverage']:<12.1f} | {r['answered_acc']:<11.2f}")
    print("=" * 95)

    # Generate Comparison Bar Plot
    if results:
        names = [r["display_name"] for r in results]
        f1_scores = [r["macro_f1_all"] for r in results]
        ci_lows = [r["ci_95_all"][0] for r in results]
        ci_highs = [r["ci_95_all"][1] for r in results]
        coverages = [r["coverage"] for r in results]

        yerr_low = [max(f1 - lo, 0.0) for f1, lo in zip(f1_scores, ci_lows)]
        yerr_high = [max(hi - f1, 0.0) for f1, hi in zip(f1_scores, ci_highs)]
        yerr = [yerr_low, yerr_high]

        x = np.arange(len(names))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(11, 6))
        bars1 = ax1.bar(
            x - width/2, f1_scores, width,
            yerr=yerr, capsize=4,
            label="All-Ops Macro-F1 (5-Class)",
            color="#1f77b4", edgecolor="#0b3c61", linewidth=0.8,
            error_kw={"elinewidth": 1.2, "ecolor": "#0b3c61", "capthick": 1.2}
        )
        ax1.set_ylabel("All-Operations Macro-F1 (5-Class)", fontsize=11, color="#1f77b4")
        ax1.set_ylim(0, 1.15)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        bars2 = ax2.bar(
            x + width/2, coverages, width,
            label="Coverage (%)",
            color="#ff7f0e", alpha=0.85, edgecolor="#994c00", linewidth=0.8
        )
        ax2.set_ylabel("Coverage (%)", fontsize=11, color="#ff7f0e")
        ax2.set_ylim(0, 115)
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
        ax1.set_title("H1 Benchmark: Single-Layer Baselines vs. Fused Observability\nAll-Operations Macro-F1 (with 95% Cluster Bootstrap CI) and Decision Coverage", fontsize=12, pad=15)

        # Value annotations
        for bar, hi in zip(bars1, ci_highs):
            h = bar.get_height()
            top_pos = max(h, hi)
            ax1.annotate(f"{h:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, top_pos),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0b3c61")

        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, color="#994c00")

        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.22)
        plot_out = Path("eval/plots/baseline_comparison.png")
        plt.savefig(plot_out, dpi=200)
        plt.close(fig)
        print(f"\nSaved baseline comparison figure to {plot_out}")

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tune"
    run_single_layer_baselines(target_dir)

if __name__ == "__main__":
    main()

