#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SRC="${ROOT}/native/PreprocessProbe"; BUILD="${ROOT}/.build-native/0054b2"; mkdir -p "${BUILD}"; cd "${ROOT}"
echo '[1/5] CPU/HF reference'; uv run python scripts/54b_prepare_reference.py --out-dir results/0054b_reference
echo '[2/5] Pillow-compatible coefficient tables'; uv run python scripts/54b2_prepare_pillow_coeffs.py
echo '[3/5] Compile Swift harness'; xcrun --sdk macosx swiftc -swift-version 5 -parse-as-library -O "${SRC}/PillowCompatibleProbe.swift" -o "${BUILD}/PillowCompatibleProbe" -framework Foundation -framework Metal
echo '[4/5] Run candidate'; "${BUILD}/PillowCompatibleProbe" --reference-dir results/0054b_reference --coeff-dir results/0054b_pillow_coeffs --metal-source "${SRC}/PillowCompatibleResize.metal" --output results/metalground_metal_preprocess_0054b2.json
echo '[5/5] Reapply original 0054b gates'; uv run python scripts/54b2_compare.py
