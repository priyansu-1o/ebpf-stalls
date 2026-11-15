#!/bin/bash
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Refreshing environment dumps in $ENV_DIR..."

uname -a > "$ENV_DIR/uname.txt" 2>&1 || true
lscpu > "$ENV_DIR/lscpu.txt" 2>&1 || true
lsblk > "$ENV_DIR/lsblk.txt" 2>&1 || true

{
    clang --version 2>&1 || true
    echo "---"
    gcc --version 2>&1 || true
} > "$ENV_DIR/compiler.txt"

bpftool version > "$ENV_DIR/bpftool.txt" 2>&1 || true

pip list > "$ENV_DIR/python_packages.txt" 2>&1 || python3 -m pip list > "$ENV_DIR/python_packages.txt" 2>&1 || true

echo "Environment dumps successfully refreshed:"
ls -la "$ENV_DIR"

