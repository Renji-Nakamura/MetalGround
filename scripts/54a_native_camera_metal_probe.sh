#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: Experiment 0054a requires macOS." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/native/CameraMetalProbe"
BUILD_ROOT="${REPO_ROOT}/.build-native/0054a"
APP="${BUILD_ROOT}/CameraMetalProbe.app"
CONTENTS="${APP}/Contents"
MACOS_DIR="${CONTENTS}/MacOS"
BINARY="${MACOS_DIR}/CameraMetalProbe"

mkdir -p "${MACOS_DIR}"
cp "${SRC_DIR}/Info.plist" "${CONTENTS}/Info.plist"

echo "Building native CameraMetalProbe..."
xcrun --sdk macosx swiftc \
  -swift-version 5 \
  -parse-as-library \
  -O \
  "${SRC_DIR}/CameraMetalProbe.swift" \
  -o "${BINARY}" \
  -framework Foundation \
  -framework AVFoundation \
  -framework CoreMedia \
  -framework CoreVideo \
  -framework Metal

codesign --force --sign - --timestamp=none "${APP}" >/dev/null

echo "Built: ${APP}"
echo "Launching probe from repository root..."
cd "${REPO_ROOT}"

"${BINARY}" "$@"
