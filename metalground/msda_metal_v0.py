from __future__ import annotations

import mlx.core as mx


# Metal MSDA v0
#
# Correctness-first implementation:
#   - FP32 only
#   - one Metal thread computes one output scalar
#   - fuses bilinear sampling + attention weighting + level/point reduction
#   - does NOT materialize sampled / stacked / weighted intermediate tensors
#
# Semantics target:
#   torch.nn.functional.grid_sample(
#       mode="bilinear",
#       padding_mode="zeros",
#       align_corners=False,
#   )
#
# Reference relation:
#   reference converts sampling locations [0, 1] to [-1, 1] by
#       grid = 2 * sampling_locations - 1
#   and grid_sample with align_corners=False maps normalized x to
#       x_src = ((grid_x + 1) * W - 1) / 2
#             = sampling_location_x * W - 0.5
#   similarly for y.


_METAL_SOURCE = r"""
    uint elem = thread_position_in_grid.x;

    constexpr uint TOTAL = B * Q * NH * HD;
    if (elem >= TOTAL) {
        return;
    }

    // Flat output layout is [B, Q, NH, HD], which is memory-equivalent
    // to the reference output [B, Q, NH * HD].
    uint tmp = elem;
    uint c = tmp % HD;
    tmp /= HD;
    uint h = tmp % NH;
    tmp /= NH;
    uint q = tmp % Q;
    uint b = tmp / Q;

    const uint qh = (b * Q + q) * NH + h;

    float acc = 0.0f;

    for (uint l = 0; l < NL; ++l) {
        const int height = spatial_shapes[l * 2 + 0];
        const int width  = spatial_shapes[l * 2 + 1];
        const int level_start = level_start_index[l];

        for (uint p = 0; p < NP; ++p) {
            const uint loc_base = ((qh * NL + l) * NP + p) * 2;

            // sampling_locations stores [x, y] in [0, 1]-style coordinates.
            // This is algebraically equivalent to:
            //   sampling_grid = 2 * sampling_locations - 1
            // followed by PyTorch grid_sample(..., align_corners=False).
            const float x = sampling_locations[loc_base + 0] * float(width)  - 0.5f;
            const float y = sampling_locations[loc_base + 1] * float(height) - 0.5f;

            const int x0 = int(metal::floor(x));
            const int y0 = int(metal::floor(y));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float lx = x - float(x0);
            const float ly = y - float(y0);
            const float hx = 1.0f - lx;
            const float hy = 1.0f - ly;

            float sampled = 0.0f;

            // padding_mode="zeros": out-of-range neighbors contribute zero.
            if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                const uint s = uint(level_start + y0 * width + x0);
                const uint vidx = ((b * S + s) * NH + h) * HD + c;
                sampled += value[vidx] * (hx * hy);
            }

            if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
                const uint s = uint(level_start + y0 * width + x1);
                const uint vidx = ((b * S + s) * NH + h) * HD + c;
                sampled += value[vidx] * (lx * hy);
            }

            if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
                const uint s = uint(level_start + y1 * width + x0);
                const uint vidx = ((b * S + s) * NH + h) * HD + c;
                sampled += value[vidx] * (hx * ly);
            }

            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                const uint s = uint(level_start + y1 * width + x1);
                const uint vidx = ((b * S + s) * NH + h) * HD + c;
                sampled += value[vidx] * (lx * ly);
            }

            const uint widx = (qh * NL + l) * NP + p;
            acc += sampled * attention_weights[widx];
        }
    }

    out[elem] = acc;
"""


_KERNEL = mx.fast.metal_kernel(
    name="metalground_msda_v0_fp32",
    input_names=[
        "value",
        "spatial_shapes",
        "level_start_index",
        "sampling_locations",
        "attention_weights",
    ],
    output_names=["out"],
    source=_METAL_SOURCE,
    ensure_row_contiguous=True,
    compile_options={"math_mode": "safe"},
)


def msda_metal_v0(
    value: mx.array,
    spatial_shapes: mx.array,
    level_start_index: mx.array,
    sampling_locations: mx.array,
    attention_weights: mx.array,
    *,
    threadgroup_size: int = 256,
    verbose: bool = False,
) -> mx.array:
    """Correctness-first fused FP32 Metal implementation of MSDA sampling core.

    Expected shapes:
      value:              [B, S, num_heads, head_dim]
      spatial_shapes:     [num_levels, 2] int32, rows are [height, width]
      level_start_index:  [num_levels] int32
      sampling_locations: [B, Q, num_heads, num_levels, num_points, 2]
      attention_weights:  [B, Q, num_heads, num_levels, num_points]

    Returns:
      [B, Q, num_heads * head_dim] float32
    """

    if value.dtype != mx.float32:
        raise TypeError(f"v0 is FP32-only; value dtype is {value.dtype}")
    if sampling_locations.dtype != mx.float32:
        raise TypeError(
            f"v0 is FP32-only; sampling_locations dtype is {sampling_locations.dtype}"
        )
    if attention_weights.dtype != mx.float32:
        raise TypeError(
            f"v0 is FP32-only; attention_weights dtype is {attention_weights.dtype}"
        )
    if spatial_shapes.dtype != mx.int32:
        raise TypeError(f"spatial_shapes must be int32; got {spatial_shapes.dtype}")
    if level_start_index.dtype != mx.int32:
        raise TypeError(
            f"level_start_index must be int32; got {level_start_index.dtype}"
        )

    if value.ndim != 4:
        raise ValueError(f"value must be rank 4; got {value.shape}")
    if sampling_locations.ndim != 6:
        raise ValueError(
            f"sampling_locations must be rank 6; got {sampling_locations.shape}"
        )
    if attention_weights.ndim != 5:
        raise ValueError(
            f"attention_weights must be rank 5; got {attention_weights.shape}"
        )

    B, S, NH, HD = map(int, value.shape)
    b2, Q, nh2, NL, NP, xy = map(int, sampling_locations.shape)

    if b2 != B or nh2 != NH or xy != 2:
        raise ValueError(
            "sampling_locations shape is incompatible with value: "
            f"value={value.shape}, locations={sampling_locations.shape}"
        )
    if tuple(map(int, attention_weights.shape)) != (B, Q, NH, NL, NP):
        raise ValueError(
            "attention_weights shape mismatch: "
            f"expected {(B, Q, NH, NL, NP)}, got {attention_weights.shape}"
        )
    if tuple(map(int, spatial_shapes.shape)) != (NL, 2):
        raise ValueError(
            f"spatial_shapes must be {(NL, 2)}, got {spatial_shapes.shape}"
        )
    if tuple(map(int, level_start_index.shape)) != (NL,):
        raise ValueError(
            f"level_start_index must be {(NL,)}, got {level_start_index.shape}"
        )

    total = B * Q * NH * HD

    outputs = _KERNEL(
        inputs=[
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        ],
        output_shapes=[(B, Q, NH * HD)],
        output_dtypes=[mx.float32],
        grid=(total, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        template=[
            ("B", B),
            ("S", S),
            ("Q", Q),
            ("NH", NH),
            ("HD", HD),
            ("NL", NL),
            ("NP", NP),
        ],
        stream=mx.gpu,
        verbose=verbose,
    )
    return outputs[0]
