# Experiment 0054b-1 — interpolation semantics diagnostic

0054b passed the performance gate but failed all three local numerical gates.

Observed 0054b:
- Metal median: 0.506667 ms
- Metal p95: 0.870834 ms
- mean abs error: 0.013901
- p99 abs error: 0.224401
- max abs error: 1.133579

The diagnostic does not alter any 0054b gate.

It adds:
1. pass-1 Metal letterbox readback and u8 comparison against the saved PIL letterbox PNG,
2. explicit PIL second resize + normalization comparison against the HF reference tensor,
3. spatial error localization by letterbox boundary/interior/padding,
4. error-vs-gradient analysis.

The diagnostic readback happens after all measured Metal timing.

Run the same command:

```bash
chmod +x scripts/54b_metal_preprocess_probe.sh
caffeinate -i bash scripts/54b_metal_preprocess_probe.sh
```

Upload:
`results/metalground_preprocess_diagnostic_0054b1.json`
