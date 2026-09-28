#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: 0054c-3 requires macOS + Metal." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/native/ForeignMetalDLPack/ForeignMetalDLPack.mm"
BUILD="${ROOT}/.build-native/0054c3"
mkdir -p "${BUILD}"

PY_INC="$(uv run python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)"
EXT_SUFFIX="$(uv run python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
OUT="${BUILD}/metal_dlpack_native${EXT_SUFFIX}"

echo "[1/2] Build Objective-C++ DLPack extension"
xcrun --sdk macosx clang++ \
  -std=c++17 \
  -O2 \
  -fPIC \
  -bundle \
  -undefined dynamic_lookup \
  -I"${PY_INC}" \
  "${SRC}" \
  -o "${OUT}" \
  -framework Foundation \
  -framework Metal

echo "[2/2] Run foreign MTLBuffer -> MLX -> PyTorch probe"
cd "${ROOT}"
PYTHONPATH="${BUILD}:${PYTHONPATH:-}" \
caffeinate -i uv run python scripts/54c3_foreign_mtlbuffer_bridge.py \
  --output results/metalground_foreign_mtlbuffer_bridge_0054c3.json
