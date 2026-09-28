# MetalGround research charter (v0)

## Target
- Hardware: MacBook Pro 14-inch, Apple M3, 16 GB unified memory
- Workload: Grounding DINO open-vocabulary object detection
- First reference checkpoint: `IDEA-Research/grounding-dino-tiny`

## Initial research questions
1. Where does latency go on Apple M3 for faithful Grounding DINO inference?
2. Which operators/stages dominate MPS execution?
3. Does multi-scale deformable attention remain a dominant bottleneck after current framework improvements?
4. Can an Apple-GPU-native implementation improve operator and end-to-end latency without changing model semantics?

## Rule
No optimization is a contribution until it has:
- a correctness test,
- a baseline,
- a measurement protocol,
- an ablation showing what caused the improvement.
