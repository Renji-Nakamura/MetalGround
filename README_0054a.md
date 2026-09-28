# MetalGround Experiment 0054a

Native macOS ingress probe:

AVFoundation -> CMSampleBuffer -> CVPixelBuffer -> CVMetalTextureCache -> MTLTexture

This experiment intentionally does **not** run Grounding DINO, resize/normalize,
read camera pixels back to CPU, or render a preview.

## Install

Copy this package into the MetalGround repository root so that these paths exist:

- `scripts/54a_native_camera_metal_probe.sh`
- `native/CameraMetalProbe/CameraMetalProbe.swift`
- `native/CameraMetalProbe/Info.plist`

## Run

```bash
chmod +x scripts/54a_native_camera_metal_probe.sh

caffeinate -i bash scripts/54a_native_camera_metal_probe.sh \
  --output results/metalground_native_camera_probe_0054a.json
```

The first execution may trigger the macOS camera-permission dialog for
`MetalGround Camera Metal Probe`. Allow access and rerun if macOS requires it.

Default protocol:

- 1280x720
- 30 FPS
- 30 warmup callbacks
- 300 measured callbacks

Do not change those defaults for the authoritative 0054a run.

The harness builds an ad-hoc-signed `.app` bundle under `.build-native/0054a`
so the executable has an `NSCameraUsageDescription`.

## macOS compile fix

This revision adds `-parse-as-library` for the single-file `@main` Swift build,
removes the macOS-unavailable `.inputPriority` preset assignment, and applies
`activeFormat` / frame duration after the camera input has been added while the
session configuration transaction is open. The 0054a measurement protocol and
pre-registered gates are unchanged.


## 0054a protocol-validity rerun

The first completed run delivered 1920x1080 callback buffers despite the
camera device reporting a 1280x720 active format. This revision disables
automatic output-buffer dimension negotiation and explicitly requests
1280x720 in the video-output settings.

The pre-registered 0054a protocol and performance gates are unchanged.


## macOS availability correction

`automaticallyConfiguresOutputBufferDimensions` and
`deliversPreviewSizedOutputBuffers` are unavailable on macOS in the SDK used
for this experiment. This revision removes those two assignments while
retaining explicit `kCVPixelBufferWidthKey` / `kCVPixelBufferHeightKey`
requests in `videoSettings`.

The pre-registered 0054a workload and gates remain unchanged.
