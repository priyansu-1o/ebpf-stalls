#!/usr/bin/env bash
set -e

# Orchestrates negative control: CPU spin under background fio storage noise
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT_DIR="${1:-$REPO_ROOT/data/tune/test_negative_control}"
ITERS="${2:-40}"
mkdir -p "$OUT_DIR"

export BG_NOISE=fio
echo "[run_negative_control] Executing negative control CPU run with background fio noise..."
"$REPO_ROOT/workloads/run_scenario.sh" "$OUT_DIR" cpu 50 1048576 "$ITERS"

echo "[run_negative_control] Finished negative control execution in $OUT_DIR"

