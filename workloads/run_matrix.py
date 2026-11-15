#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_configs(split: str, scratch_dir: Path):
    test_bin_1mb = scratch_dir / "test_1mb.bin"
    test_bin_4mb = scratch_dir / "test_4mb.bin"

    configs = []
    # 3 repeat runs per configuration with suffix _r1, _r2, _r3
    for r in (1, 2, 3):
        # Base configs (40 iterations each)
        configs.append((f"{split}_warm_1mb_r{r}", "warm", str(test_bin_1mb), 1048576, 40, "none", "1MB", 3.0))
        configs.append((f"{split}_io_1mb_r{r}", "io", str(test_bin_1mb), 1048576, 40, "none", "1MB", 3.0))
        configs.append((f"{split}_cpu_50ms_r{r}", "cpu", "50", 1048576, 40, "none", "-", 3.0))
        configs.append((f"{split}_sched_4comp_r{r}", "sched", "4", 1048576, 40, "none", "-", 2.5))
        configs.append((f"{split}_mixed_1mb_r{r}", "mixed", str(test_bin_1mb), 1048576, 40, "none", "1MB", 3.0))
        configs.append((f"{split}_neg_ctrl_r{r}", "neg_ctrl", "50", 1048576, 40, "fio", "-", 4.0))

        # Transfer test configs for final evaluation only
        if split == "final":
            configs.append((f"final_io_4mb_256k_r{r}", "io", str(test_bin_4mb), 262144, 40, "none", "4MB", 3.5))
            configs.append((f"final_warm_4mb_256k_r{r}", "warm", str(test_bin_4mb), 262144, 40, "none", "4MB", 3.5))

    return configs

def run_matrix(split: str, dry_run: bool = False, max_runs: int = 0):
    user_home = Path(os.environ.get("SUDO_USER", "")).expanduser() if os.environ.get("SUDO_USER") else Path.home()
    scratch_dir = user_home / "ebpf_lab" / "scratch"

    if not dry_run:
        scratch_dir.mkdir(parents=True, exist_ok=True)

    out_base = REPO_ROOT / "data" / split
    if not dry_run:
        out_base.mkdir(parents=True, exist_ok=True)

    configs = build_configs(split, scratch_dir)
    if max_runs > 0:
        configs = configs[:max_runs]

    total_ops = sum(c[4] for c in configs)
    total_est_sec = sum(c[7] for c in configs)
    est_minutes = total_est_sec / 60.0

    print("=" * 86)
    print(f"Matrix Runner: {len(configs)} scenario runs configured for target '{split}'.")
    print(f"Total operations: {total_ops} | Estimated duration: {est_minutes:.1f} minutes ({total_est_sec:.0f}s)")
    print("=" * 86)

    run_script = REPO_ROOT / "workloads" / "run_scenario.sh"
    neg_ctrl_script = REPO_ROOT / "workloads" / "negative_controls" / "run_negative_control.sh"

    if dry_run:
        header = f"{'Run ID':<28} {'Mode':<10} {'Param / BS':<22} {'Ops':<5} {'Est Time':<8}"
        print(header)
        print("-" * len(header))
        for item in configs:
            run_name, mode, param, bs, iters, bg_noise, file_sz, est_sec = item
            desc = f"bs={bs}" if mode in ("io", "warm", "mixed") else f"param={param}"
            print(f"{run_name:<28} {mode:<10} {desc:<22} {iters:<5} {est_sec:>4.1f}s")
        print("-" * len(header))
        print(f"Total: {len(configs)} runs, {total_ops} operations, ~{est_minutes:.1f} minutes.\n")

    for i, item in enumerate(configs, 1):
        run_name, mode, param, bs, iters, bg_noise, file_sz, est_sec = item
        run_dir = out_base / run_name

        if mode == "neg_ctrl":
            cmd = [str(neg_ctrl_script), str(run_dir), str(iters)]
        else:
            cmd = [str(run_script), str(run_dir), mode, param, str(bs), str(iters)]

        if dry_run:
            cmd_prefix = "sudo " if os.geteuid() != 0 else ""
            print(f"[dry-run] {cmd_prefix}{' '.join(cmd)}")
            continue

        if os.geteuid() != 0:
            cmd = ["sudo"] + cmd

        print(f"\n---> [{i}/{len(configs)}] Executing scenario: {run_name} (mode: {mode}, iters: {iters}, bg: {bg_noise})")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error: command failed with code {res.returncode}: {' '.join(cmd)}")
            sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Factorial scenarios matrix runner")
    parser.add_argument("--split", choices=["tune", "final"], default="tune", help="Target data split directory")
    parser.add_argument("--dry-run", action="store_true", help="Print execution commands without running")
    parser.add_argument("--max-runs", type=int, default=0, help="Cap maximum number of runs executed")
    args = parser.parse_args()

    run_matrix(args.split, args.dry_run, args.max_runs)

if __name__ == "__main__":
    main()
