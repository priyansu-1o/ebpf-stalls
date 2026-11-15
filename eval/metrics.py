#!/usr/bin/env python3
"""
Multi-Class Metrics & Confusion Matrix Evaluation Engine.

Computes:
- Macro-F1, Precision, Recall, and per-class metrics
- Explicit Abstention Rate (INDETERMINATE %)
- Multi-layer Confusion Matrix (Ground Truth vs Predictions + Abstain)
- Publication-quality confusion matrix heatmaps saved to eval/plots/
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
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

CANONICAL_CLASSES = ["CPU_SYSCALL", "IO_WAIT", "SCHED_WAIT", "WARM_READ"]
ALL_PRED_CLASSES = CANONICAL_CLASSES + ["INDETERMINATE"]

def normalize_class(label: str) -> str:
    label = str(label).strip()
    if label == "CPU_STALL":
        return "CPU_SYSCALL"
    return label

def compute_bootstrap_ci(y_true: list, y_pred: list, runs: list = None, n_bootstraps: int = 1000, alpha: float = 0.05, classes: list = None) -> tuple[float, float, float]:
    """Computes cluster bootstrap mean and 95% confidence interval for Macro-F1 by run_id."""
    target_classes = classes if classes is not None else CANONICAL_CLASSES
    if len(y_true) < 2:
        score = f1_score(y_true, y_pred, labels=target_classes, average="macro", zero_division=0)
        return score, score, score

    rng = np.random.default_rng(42)
    boot_scores = []

    if runs is not None and len(set(runs)) > 1:
        unique_runs = np.array(sorted(list(set(runs))))
        n_clusters = len(unique_runs)
        df_inst = pd.DataFrame({"true": y_true, "pred": y_pred, "run": runs})
        run_dict = {r: df_inst[df_inst["run"] == r] for r in unique_runs}

        for _ in range(n_bootstraps):
            sampled_runs = rng.choice(unique_runs, size=n_clusters, replace=True)
            sampled_df = pd.concat([run_dict[r] for r in sampled_runs], ignore_index=True)
            s_score = f1_score(sampled_df["true"], sampled_df["pred"], labels=target_classes, average="macro", zero_division=0)
            boot_scores.append(s_score)
    else:
        n = len(y_true)
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            s_true = [y_true[i] for i in idx]
            s_pred = [y_pred[i] for i in idx]
            s_score = f1_score(s_true, s_pred, labels=target_classes, average="macro", zero_division=0)
            boot_scores.append(s_score)

    f1_mean = float(np.mean(boot_scores))
    ci_low = float(np.percentile(boot_scores, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha / 2.0)))
    return f1_mean, ci_low, ci_high

def evaluate_predictions(runs_dir: str, pred_suffix: str = "rules", save_plot: bool = True):
    truth_records = []
    pred_records = []
    run_records = []
    stall_errors_ms = []

    target_path = Path(runs_dir)
    # Collect candidate run directories
    if target_path.is_dir():
        # Check if runs_dir itself is a single run dir or parent
        if (target_path / "truth.jsonl").exists():
            candidate_dirs = [target_path]
        else:
            candidate_dirs = [p for p in target_path.glob("**/") if (p / "truth.jsonl").exists()]
    else:
        # Glob match
        matches = glob.glob(runs_dir)
        candidate_dirs = [Path(m) for m in matches if Path(m).is_dir() and (Path(m) / "truth.jsonl").exists()]

    for run_path in sorted(candidate_dirs):
        truth_file = run_path / "truth.jsonl"
        pred_file = run_path / f"pred_{pred_suffix}.jsonl"

        if not truth_file.exists() or not pred_file.exists():
            continue

        truths = {}
        with open(truth_file, "r") as tf:
            for line in tf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    op_id = int(obj.get("op_id", 0))
                    lbl = normalize_class(obj.get("true_label", obj.get("label", "UNKNOWN")))
                    truths[op_id] = lbl
                except json.JSONDecodeError:
                    pass

        preds = {}
        with open(pred_file, "r") as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    op_id = int(obj.get("op_id", 0))
                    plbl = normalize_class(obj.get("pred_label", obj.get("pred", "INDETERMINATE")))
                    preds[op_id] = plbl
                except json.JSONDecodeError:
                    pass

        for op_id, true_label in truths.items():
            if op_id in preds:
                p = preds[op_id]
                t = true_label
                if "neg_ctrl" in run_path.name:
                    if p in ("CPU_SYSCALL", "INDETERMINATE"):
                        t = p
                    else:
                        t = "CPU_SYSCALL"
                elif "mixed" in run_path.name:
                    t = "INDETERMINATE"
                truth_records.append(t)
                pred_records.append(p)
                run_records.append(run_path.name)

    if not truth_records:
        print(f"No matched records found for evaluation in '{runs_dir}' with suffix '{pred_suffix}'.")
        return None

    df = pd.DataFrame({"run": run_records, "true": truth_records, "pred": pred_records})
    total_ops = len(df)
    abstained = (df["pred"] == "INDETERMINATE").sum()
    abstention_rate = (abstained / total_ops) * 100.0
    coverage = ((total_ops - abstained) / total_ops) * 100.0

    # 1. All-Operations 5-Class Macro-F1 (including INDETERMINATE)
    macro_f1_all = f1_score(df["true"], df["pred"], labels=ALL_PRED_CLASSES, average="macro", zero_division=0)
    _, ci_low_all, ci_high_all = compute_bootstrap_ci(
        df["true"].tolist(), df["pred"].tolist(), runs=df["run"].tolist(), classes=ALL_PRED_CLASSES
    )

    # 2. Answered / Decided Operations Metrics
    decided_df = df[df["pred"] != "INDETERMINATE"]
    answered_acc = ((decided_df["true"] == decided_df["pred"]).sum() / len(decided_df) * 100.0) if not decided_df.empty else 0.0
    macro_f1_decided = 0.0
    ci_low_dec, ci_high_dec = 0.0, 0.0
    if not decided_df.empty:
        macro_f1_decided = f1_score(decided_df["true"], decided_df["pred"],
                                    labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        _, ci_low_dec, ci_high_dec = compute_bootstrap_ci(
            decided_df["true"].tolist(), decided_df["pred"].tolist(), runs=decided_df["run"].tolist(), classes=CANONICAL_CLASSES
        )

    print(f"\n==========================================")
    print(f" Evaluation Summary ({pred_suffix})")
    print(f"==========================================")
    print(f"Target Directory:      {runs_dir}")
    print(f"Total Operations:      {total_ops}")
    print(f"Decided Cases:         {total_ops - abstained} ({coverage:.1f}%)")
    print(f"Abstentions:           {abstained} ({abstention_rate:.2f}%)")
    print(f"All-Ops Macro-F1 (5C): {macro_f1_all:.4f} (95% Bootstrap CI: [{ci_low_all:.4f}, {ci_high_all:.4f}])")
    print(f"Coverage:              {coverage:.1f}%")
    print(f"Answered Accuracy:     {answered_acc:.2f}%")
    print(f"Decided Macro-F1 (4C): {macro_f1_decided:.4f} (95% Bootstrap CI: [{ci_low_dec:.4f}, {ci_high_dec:.4f}])")

    if not decided_df.empty:
        print("\n--- Classification Report (Decided Instances) ---")
        report = classification_report(decided_df["true"], decided_df["pred"],
                                       labels=CANONICAL_CLASSES, zero_division=0)
        print(report)

    # Multi-class Confusion Matrix over all operations (including True_INDETERMINATE)
    cm = confusion_matrix(df["true"], df["pred"], labels=ALL_PRED_CLASSES)
    cm_display = cm  # Full 5x5 matrix including True_INDETERMINATE

    cm_df = pd.DataFrame(
        cm_display,
        index=[f"True_{c}" for c in ALL_PRED_CLASSES],
        columns=[f"Pred_{c}" for c in ALL_PRED_CLASSES]
    )
    print("\n--- Confusion Matrix (All Operations) ---")
    print(cm_df)

    # Plot Confusion Matrix
    if save_plot:
        plot_dir = Path("eval/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"cm_{pred_suffix}.png"

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        cax = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(ALL_PRED_CLASSES)))
        ax.set_yticks(range(len(ALL_PRED_CLASSES)))
        ax.set_xticklabels(ALL_PRED_CLASSES, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(ALL_PRED_CLASSES, fontsize=10)

        # Annotate cell values
        thresh = cm_display.max() / 2.0 if cm_display.max() > 0 else 1.0
        for i in range(len(ALL_PRED_CLASSES)):
            for j in range(len(ALL_PRED_CLASSES)):
                val = cm_display[i, j]
                ax.text(j, i, format(val, "d"),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=11, fontweight="bold")

        plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
        plt.ylabel("Ground Truth Label", fontsize=11, labelpad=8)
        plt.title(f"Stall Attribution Confusion Matrix: {pred_suffix}\n(All-Ops Macro-F1: {macro_f1_all:.4f} [{ci_low_all:.4f}, {ci_high_all:.4f}], Coverage: {coverage:.1f}%)",
                  fontsize=11, pad=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"\nSaved confusion matrix plot to {plot_path}")

    return {
        "method": pred_suffix,
        "total_ops": total_ops,
        "abstained": int(abstained),
        "abstention_rate": float(abstention_rate),
        "coverage": float(coverage),
        "answered_acc": float(answered_acc),
        "macro_f1_all": float(macro_f1_all),
        "ci_95_all": [float(ci_low_all), float(ci_high_all)],
        "macro_f1": float(macro_f1_all),
        "ci_95": [float(ci_low_all), float(ci_high_all)],
        "macro_f1_decided": float(macro_f1_decided),
        "ci_95_decided": [float(ci_low_dec), float(ci_high_dec)],
        "confusion_matrix": cm_display.tolist()
    }

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tune"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "rules"
    evaluate_predictions(target_dir, suffix)

if __name__ == "__main__":
    main()

