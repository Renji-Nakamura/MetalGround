import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import Metal

private struct Config {
    var outputPath = "results/metalground_native_camera_probe_0054a.json"
    var width = 1280
    var height = 720
    var fps = 30.0
    var warmupFrames = 30
    var measuredFrames = 300
}

private enum ProbeError: Error, CustomStringConvertible {
    case argument(String)
    case cameraPermissionDenied
    case noCamera
    case noFormat(Int, Int, Double)
    case cannotAddInput
    case cannotAddOutput
    case noMetalDevice
    case textureCache(CVReturn)
    case timeout

    var description: String {
        switch self {
        case .argument(let s): return "Argument error: \(s)"
        case .cameraPermissionDenied: return "Camera permission denied."
        case .noCamera: return "No video capture device found."
        case .noFormat(let w, let h, let fps):
            return "No camera format supports \(w)x\(h) @ \(fps) FPS."
        case .cannotAddInput: return "AVCaptureSession cannot add camera input."
        case .cannotAddOutput: return "AVCaptureSession cannot add video data output."
        case .noMetalDevice: return "MTLCreateSystemDefaultDevice() returned nil."
        case .textureCache(let status):
            return "CVMetalTextureCacheCreate failed: \(status)"
        case .timeout: return "Timed out before collecting requested frames."
        }
    }
}

private func parseArgs() throws -> Config {
    var c = Config()
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0

    func value(_ flag: String) throws -> String {
        guard i + 1 < args.count else {
            throw ProbeError.argument("Missing value for \(flag)")
        }
        i += 1
        return args[i]
    }

    while i < args.count {
        switch args[i] {
        case "--output":
            c.outputPath = try value("--output")
        case "--width":
            guard let x = Int(try value("--width")) else {
                throw ProbeError.argument("Invalid --width")
            }
            c.width = x
        case "--height":
            guard let x = Int(try value("--height")) else {
                throw ProbeError.argument("Invalid --height")
            }
            c.height = x
        case "--fps":
            guard let x = Double(try value("--fps")), x > 0 else {
                throw ProbeError.argument("Invalid --fps")
            }
            c.fps = x
        case "--warmup":
            guard let x = Int(try value("--warmup")), x >= 0 else {
                throw ProbeError.argument("Invalid --warmup")
            }
            c.warmupFrames = x
        case "--samples":
            guard let x = Int(try value("--samples")), x > 1 else {
                throw ProbeError.argument("Invalid --samples")
            }
            c.measuredFrames = x
        case "--help", "-h":
            print("""
            MetalGround Experiment 0054a — Native AVFoundation → CVPixelBuffer → Metal probe

            Options:
              --output PATH     JSON output path
              --width N         requested capture width (default 1280)
              --height N        requested capture height (default 720)
              --fps X           requested capture FPS (default 30)
              --warmup N        warmup callbacks (default 30)
              --samples N       measured callbacks (default 300)
            """)
            exit(0)
        default:
            throw ProbeError.argument("Unknown argument: \(args[i])")
        }
        i += 1
    }
    return c
}

private func percentile(_ xs: [Double], _ p: Double) -> Double? {
    guard !xs.isEmpty else { return nil }
    let a = xs.sorted()
    if a.count == 1 { return a[0] }
    let pos = max(0.0, min(1.0, p)) * Double(a.count - 1)
    let lo = Int(floor(pos))
    let hi = Int(ceil(pos))
    if lo == hi { return a[lo] }
    let t = pos - Double(lo)
    return a[lo] * (1.0 - t) + a[hi] * t
}

private func stats(_ xs: [Double]) -> [String: Any] {
    guard !xs.isEmpty else {
        return ["n": 0]
    }
    let sum = xs.reduce(0.0, +)
    return [
        "n": xs.count,
        "mean": sum / Double(xs.count),
        "median": percentile(xs, 0.50)!,
        "p90": percentile(xs, 0.90)!,
        "p95": percentile(xs, 0.95)!,
        "min": xs.min()!,
        "max": xs.max()!,
    ]
}

private func fourCC(_ value: OSType) -> String {
    let bytes: [UInt8] = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
    return String(bytes: bytes, encoding: .ascii)
        ?? String(format: "0x%08X", value)
}

private func requestCameraPermission() throws {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
        return
    case .notDetermined:
        let sem = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .video) { ok in
            granted = ok
            sem.signal()
        }
        sem.wait()
        if !granted { throw ProbeError.cameraPermissionDenied }
    default:
        throw ProbeError.cameraPermissionDenied
    }
}

private func chooseFormat(
    device: AVCaptureDevice,
    width: Int,
    height: Int,
    fps: Double
) -> AVCaptureDevice.Format? {
    for format in device.formats {
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        guard Int(dims.width) == width, Int(dims.height) == height else {
            continue
        }
        for range in format.videoSupportedFrameRateRanges {
            if range.minFrameRate <= fps && fps <= range.maxFrameRate {
                return format
            }
        }
    }
    return nil
}

private final class CaptureProbe: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let config: Config
    let metalDevice: MTLDevice
    let textureCache: CVMetalTextureCache
    let completion: DispatchSemaphore

    private(set) var seenCallbacks = 0
    private(set) var measuredCallbacks = 0
    private(set) var droppedCallbacks = 0
    private(set) var textureSuccesses = 0
    private(set) var textureFailures = 0
    private(set) var signaled = false

    private var previousCallbackNs: UInt64?
    private var previousPTSSeconds: Double?
    private var firstMeasuredCallbackNs: UInt64?
    private var lastMeasuredCallbackNs: UInt64?

    private(set) var callbackIntervalsMs: [Double] = []
    private(set) var ptsIntervalsMs: [Double] = []
    private(set) var callbackToTextureReadyMs: [Double] = []
    private(set) var textureWrapMs: [Double] = []
    private(set) var widths: [Int] = []
    private(set) var heights: [Int] = []
    private(set) var bytesPerRow: [Int] = []
    private(set) var pixelFormats: [OSType] = []
    private(set) var samples: [[String: Any]] = []

    init(
        config: Config,
        metalDevice: MTLDevice,
        textureCache: CVMetalTextureCache,
        completion: DispatchSemaphore
    ) {
        self.config = config
        self.metalDevice = metalDevice
        self.textureCache = textureCache
        self.completion = completion
        super.init()
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        droppedCallbacks += 1
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        if signaled { return }

        let callbackEntryNs = DispatchTime.now().uptimeNanoseconds
        seenCallbacks += 1

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            if seenCallbacks > config.warmupFrames {
                measuredCallbacks += 1
                textureFailures += 1
                maybeFinish()
            }
            return
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let ptsSeconds = pts.isValid ? CMTimeGetSeconds(pts) : Double.nan

        let previousCallback = previousCallbackNs
        let previousPTS = previousPTSSeconds
        previousCallbackNs = callbackEntryNs
        if ptsSeconds.isFinite {
            previousPTSSeconds = ptsSeconds
        }

        let wrapStartNs = DispatchTime.now().uptimeNanoseconds
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvTexture
        )

        var metalTexture: MTLTexture?
        if status == kCVReturnSuccess, let cvTexture {
            metalTexture = CVMetalTextureGetTexture(cvTexture)
        }
        let readyNs = DispatchTime.now().uptimeNanoseconds

        let isMeasured = seenCallbacks > config.warmupFrames
        guard isMeasured else { return }

        measuredCallbacks += 1
        if firstMeasuredCallbackNs == nil {
            firstMeasuredCallbackNs = callbackEntryNs
        }
        lastMeasuredCallbackNs = callbackEntryNs

        widths.append(width)
        heights.append(height)
        bytesPerRow.append(rowBytes)
        pixelFormats.append(pixelFormat)

        let callbackToReady =
            Double(readyNs - callbackEntryNs) / 1_000_000.0
        let wrapMs =
            Double(readyNs - wrapStartNs) / 1_000_000.0

        callbackToTextureReadyMs.append(callbackToReady)
        textureWrapMs.append(wrapMs)

        var callbackInterval: Double? = nil
        if let previousCallback {
            callbackInterval =
                Double(callbackEntryNs - previousCallback) / 1_000_000.0
            callbackIntervalsMs.append(callbackInterval!)
        }

        var ptsInterval: Double? = nil
        if let previousPTS, ptsSeconds.isFinite {
            let dt = (ptsSeconds - previousPTS) * 1000.0
            if dt > 0 {
                ptsInterval = dt
                ptsIntervalsMs.append(dt)
            }
        }

        let ok = (status == kCVReturnSuccess && metalTexture != nil)
        if ok {
            textureSuccesses += 1
        } else {
            textureFailures += 1
        }

        var record: [String: Any] = [
            "sample_index": measuredCallbacks - 1,
            "width": width,
            "height": height,
            "bytes_per_row": rowBytes,
            "pixel_format": fourCC(pixelFormat),
            "cvmetal_status": Int(status),
            "metal_texture_created": ok,
            "callback_to_texture_ready_ms": callbackToReady,
            "texture_wrap_ms": wrapMs,
        ]
        if let callbackInterval {
            record["callback_interval_ms"] = callbackInterval
        }
        if let ptsInterval {
            record["capture_pts_interval_ms"] = ptsInterval
        }
        if let metalTexture {
            record["metal_texture_width"] = metalTexture.width
            record["metal_texture_height"] = metalTexture.height
            record["metal_texture_pixel_format_raw"] =
                metalTexture.pixelFormat.rawValue
        }
        samples.append(record)

        maybeFinish()
    }

    private func maybeFinish() {
        if measuredCallbacks >= config.measuredFrames && !signaled {
            signaled = true
            completion.signal()
        }
    }

    func effectiveCallbackFPS() -> Double? {
        guard
            measuredCallbacks >= 2,
            let first = firstMeasuredCallbackNs,
            let last = lastMeasuredCallbackNs,
            last > first
        else {
            return nil
        }
        let seconds = Double(last - first) / 1_000_000_000.0
        return Double(measuredCallbacks - 1) / seconds
    }
}

private func writeJSON(_ object: [String: Any], to path: String) throws {
    let url = URL(fileURLWithPath: path)
    let dir = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(
        at: dir,
        withIntermediateDirectories: true
    )
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.prettyPrinted, .sortedKeys]
    )
    try data.write(to: url)
}

@main
private struct MetalGroundCameraProbeMain {
    static func main() {
        do {
            let config = try parseArgs()
            try requestCameraPermission()

            guard let metalDevice = MTLCreateSystemDefaultDevice() else {
                throw ProbeError.noMetalDevice
            }

            var textureCacheOptional: CVMetalTextureCache?
            let cacheStatus = CVMetalTextureCacheCreate(
                kCFAllocatorDefault,
                nil,
                metalDevice,
                nil,
                &textureCacheOptional
            )
            guard
                cacheStatus == kCVReturnSuccess,
                let textureCache = textureCacheOptional
            else {
                throw ProbeError.textureCache(cacheStatus)
            }

            guard let camera = AVCaptureDevice.default(for: .video) else {
                throw ProbeError.noCamera
            }
            guard let format = chooseFormat(
                device: camera,
                width: config.width,
                height: config.height,
                fps: config.fps
            ) else {
                throw ProbeError.noFormat(
                    config.width,
                    config.height,
                    config.fps
                )
            }

            let session = AVCaptureSession()
            session.beginConfiguration()

            let input = try AVCaptureDeviceInput(device: camera)
            guard session.canAddInput(input) else {
                session.commitConfiguration()
                throw ProbeError.cannotAddInput
            }
            session.addInput(input)

            // AVCaptureSession.Preset.inputPriority is unavailable on macOS.
            // Apply the selected device format and frame duration directly
            // while the session configuration transaction is open.
            try camera.lockForConfiguration()
            camera.activeFormat = format
            let frameDuration = CMTime(
                seconds: 1.0 / config.fps,
                preferredTimescale: 60_000
            )
            camera.activeVideoMinFrameDuration = frameDuration
            camera.activeVideoMaxFrameDuration = frameDuration
            camera.unlockForConfiguration()

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true

            // Disable automatic preview/screen-sized output negotiation and
            // explicitly request the authoritative experiment dimensions.
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String:
                    Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String:
                    config.width,
                kCVPixelBufferHeightKey as String:
                    config.height
            ]

            guard session.canAddOutput(output) else {
                session.commitConfiguration()
                throw ProbeError.cannotAddOutput
            }
            session.addOutput(output)
            session.commitConfiguration()

            let completion = DispatchSemaphore(value: 0)
            let captureQueue = DispatchQueue(
                label: "org.metalground.camera.0054a.capture",
                qos: .userInteractive
            )
            let probe = CaptureProbe(
                config: config,
                metalDevice: metalDevice,
                textureCache: textureCache,
                completion: completion
            )
            output.setSampleBufferDelegate(probe, queue: captureQueue)

            let dims = CMVideoFormatDescriptionGetDimensions(
                camera.activeFormat.formatDescription
            )
            print(
                "Experiment 0054a — Native AVFoundation → CVPixelBuffer → Metal"
            )
            print("Camera: \(camera.localizedName)")
            print(
                "Active format: \(dims.width)x\(dims.height) "
                + "@ \(String(format: "%.3f", config.fps)) FPS"
            )
            print("Metal device: \(metalDevice.name)")
            print(
                "Warmup: \(config.warmupFrames), "
                + "measured: \(config.measuredFrames)"
            )
            print("Starting capture...")

            session.startRunning()

            let nominalSeconds =
                Double(config.warmupFrames + config.measuredFrames)
                / config.fps
            let timeoutSeconds = max(30.0, nominalSeconds * 4.0)
            let waitResult = completion.wait(
                timeout: .now() + timeoutSeconds
            )

            session.stopRunning()
            output.setSampleBufferDelegate(nil, queue: nil)

            if waitResult == .timedOut {
                throw ProbeError.timeout
            }

            captureQueue.sync {}

            let effectiveFPS = probe.effectiveCallbackFPS() ?? 0.0
            let callbackStats = stats(probe.callbackIntervalsMs)
            let ptsStats = stats(probe.ptsIntervalsMs)
            let callbackToReadyStats =
                stats(probe.callbackToTextureReadyMs)
            let wrapStats = stats(probe.textureWrapMs)

            let uniqueWidths = Array(Set(probe.widths)).sorted()
            let uniqueHeights = Array(Set(probe.heights)).sorted()
            let uniqueRowBytes = Array(Set(probe.bytesPerRow)).sorted()
            let uniqueFormats = Array(
                Set(probe.pixelFormats.map(fourCC))
            ).sorted()

            let completedPass =
                probe.measuredCallbacks == config.measuredFrames
            let fpsPass = effectiveFPS >= 29.0
            let intervalP95 =
                percentile(probe.callbackIntervalsMs, 0.95)
                ?? Double.infinity
            let intervalPass = intervalP95 <= 45.0
            let textureSuccessPass =
                probe.textureSuccesses == config.measuredFrames
                && probe.textureFailures == 0
            let callbackToReadyP95 =
                percentile(probe.callbackToTextureReadyMs, 0.95)
                ?? Double.infinity
            let callbackToReadyPass = callbackToReadyP95 <= 2.0
            let wrapP95 =
                percentile(probe.textureWrapMs, 0.95)
                ?? Double.infinity
            let wrapPass = wrapP95 <= 1.0
            let droppedPass = probe.droppedCallbacks <= 1

            let resolutionValid =
                uniqueWidths == [config.width]
                && uniqueHeights == [config.height]
            let formatValid = uniqueFormats == ["BGRA"]

            let gatePasses = [
                completedPass,
                fpsPass,
                intervalPass,
                textureSuccessPass,
                callbackToReadyPass,
                wrapPass,
                droppedPass,
            ]

            let result: [String: Any] = [
                "timestamp_utc":
                    ISO8601DateFormatter().string(from: Date()),
                "experiment": "0054a",
                "purpose":
                    "Native AVFoundation -> CVPixelBuffer -> "
                    + "CVMetalTextureCache -> MTLTexture ingress probe.",
                "scope": "native_camera_metal_ingress_only",
                "environment": [
                    "macos":
                        ProcessInfo.processInfo.operatingSystemVersionString,
                    "metal_device": metalDevice.name,
                    "process_arch":
                        ProcessInfo.processInfo.environment["ARCHS"]
                        ?? "runtime-native",
                ],
                "configuration": [
                    "requested_width": config.width,
                    "requested_height": config.height,
                    "requested_fps": config.fps,
                    "warmup_frames": config.warmupFrames,
                    "requested_samples": config.measuredFrames,
                    "pixel_format_requested": "BGRA",
                    "always_discards_late_video_frames": true,
                    "explicit_output_buffer_width": config.width,
                    "explicit_output_buffer_height": config.height,
                    "cpu_pixel_readback_performed": false,
                    "cpu_pixel_copy_performed": false,
                ],
                "camera": [
                    "device_name": camera.localizedName,
                    "active_format_width": Int(dims.width),
                    "active_format_height": Int(dims.height),
                    "measured_unique_widths": uniqueWidths,
                    "measured_unique_heights": uniqueHeights,
                    "measured_unique_bytes_per_row": uniqueRowBytes,
                    "measured_unique_pixel_formats": uniqueFormats,
                    "resolution_protocol_valid": resolutionValid,
                    "pixel_format_protocol_valid": formatValid,
                ],
                "summary": [
                    "seen_callbacks_including_warmup":
                        probe.seenCallbacks,
                    "completed_samples": probe.measuredCallbacks,
                    "dropped_callbacks_reported":
                        probe.droppedCallbacks,
                    "texture_successes": probe.textureSuccesses,
                    "texture_failures": probe.textureFailures,
                    "effective_callback_fps": effectiveFPS,
                    "callback_interval_ms": callbackStats,
                    "capture_pts_interval_ms": ptsStats,
                    "callback_to_texture_ready_ms":
                        callbackToReadyStats,
                    "texture_wrap_ms": wrapStats,
                ],
                "pre_registered_gates": [
                    "definition": [
                        "required_samples": config.measuredFrames,
                        "effective_callback_fps_min": 29.0,
                        "callback_interval_p95_ms_max": 45.0,
                        "texture_success_ratio_required": 1.0,
                        "callback_to_texture_ready_p95_ms_max": 2.0,
                        "texture_wrap_p95_ms_max": 1.0,
                        "dropped_callbacks_max": 1,
                    ],
                    "results": [
                        "required_samples": [
                            "observed": probe.measuredCallbacks,
                            "pass": completedPass,
                        ],
                        "effective_callback_fps": [
                            "observed": effectiveFPS,
                            "threshold_min": 29.0,
                            "pass": fpsPass,
                        ],
                        "callback_interval_p95_ms": [
                            "observed": intervalP95,
                            "threshold_max": 45.0,
                            "pass": intervalPass,
                        ],
                        "texture_success": [
                            "observed_successes":
                                probe.textureSuccesses,
                            "observed_failures":
                                probe.textureFailures,
                            "required_ratio": 1.0,
                            "pass": textureSuccessPass,
                        ],
                        "callback_to_texture_ready_p95_ms": [
                            "observed": callbackToReadyP95,
                            "threshold_max": 2.0,
                            "pass": callbackToReadyPass,
                        ],
                        "texture_wrap_p95_ms": [
                            "observed": wrapP95,
                            "threshold_max": 1.0,
                            "pass": wrapPass,
                        ],
                        "dropped_callbacks_reported": [
                            "observed": probe.droppedCallbacks,
                            "threshold_max": 1,
                            "pass": droppedPass,
                        ],
                    ],
                    "all_performance_gates_pass":
                        gatePasses.allSatisfy { $0 },
                    "protocol_valid":
                        resolutionValid && formatValid,
                ],
                "timing_semantics": [
                    "callback_entry":
                        "DispatchTime.uptimeNanoseconds at entry to "
                        + "captureOutput(didOutput:).",
                    "callback_to_texture_ready":
                        "Callback entry to successful retrieval of an "
                        + "MTLTexture object backed through "
                        + "CVMetalTextureCache. This is not GPU command "
                        + "completion latency.",
                    "texture_wrap":
                        "Immediately before "
                        + "CVMetalTextureCacheCreateTextureFromImage to "
                        + "retrieval of MTLTexture.",
                    "effective_callback_fps":
                        "(N-1) divided by elapsed monotonic time between "
                        + "first and last measured callback.",
                ],
                "notes": [
                    "This harness performs no CVPixelBuffer base-address "
                    + "lock, CPU pixel readback, memcpy, resize, "
                    + "normalization, inference, or rendering.",
                    "The absence of an explicit CPU pixel copy is a property "
                    + "of this harness implementation; it is not a claim "
                    + "that every lower-level camera/driver operation is "
                    + "physically copy-free.",
                    "Experiment 0054a measures native camera-to-Metal "
                    + "ingress only. Grounding DINO is intentionally absent.",
                ],
                "samples": probe.samples,
            ]

            try writeJSON(result, to: config.outputPath)

            print("")
            print("=== Experiment 0054a summary ===")
            print(
                String(
                    format:
                        "effective callback FPS: %.3f",
                    effectiveFPS
                )
            )
            print(
                String(
                    format:
                        "callback interval p95: %.4f ms",
                    intervalP95
                )
            )
            print(
                String(
                    format:
                        "callback -> texture ready p95: %.4f ms",
                    callbackToReadyP95
                )
            )
            print(
                String(
                    format:
                        "texture wrap p95: %.4f ms",
                    wrapP95
                )
            )
            print(
                "texture successes: "
                + "\(probe.textureSuccesses)/\(config.measuredFrames)"
            )
            print(
                "reported dropped callbacks: \(probe.droppedCallbacks)"
            )
            print(
                "protocol valid: \(resolutionValid && formatValid)"
            )
            print(
                "all performance gates pass: "
                + "\(gatePasses.allSatisfy { $0 })"
            )
            print("Saved: \(config.outputPath)")

        } catch {
            fputs("ERROR: \(error)\n", stderr)
            exit(1)
        }
    }
}
