#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: 0054b requires macOS + Metal." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/native/PreprocessProbe"
BUILD="${ROOT}/.build-native/0054b"
mkdir -p "${BUILD}"

echo "[1/4] CPU/HF reference"
cd "${ROOT}"
uv run python scripts/54b_prepare_reference.py \
  --out-dir results/0054b_reference

echo "[2/4] Metal shader will compile at runtime via MTLDevice.makeLibrary(source:)"

echo "[3/4] Compile/run Swift Metal harness"
xcrun --sdk macosx swiftc \
  -swift-version 5 \
  -parse-as-library \
  -O \
  "${SRC}/PreprocessProbe.swift" \
  -o "${BUILD}/PreprocessProbe" \
  -framework Foundation \
  -framework Metal

"${BUILD}/PreprocessProbe" \
  --reference-dir results/0054b_reference \
  --metal-source "${SRC}/PreprocessKernels.metal" \
  --output results/metalground_metal_preprocess_0054b.json

echo "[4/4] Compare against CPU/HF reference"
uv run python scripts/54b_compare_preprocess.py \
  --reference-dir results/0054b_reference \
  --metal-json results/metalground_metal_preprocess_0054b.json \
  --output results/metalground_preprocess_compare_0054b.json


echo "[5/5] Diagnose interpolation semantics"
uv run python scripts/54b1_diagnose_interpolation.py
