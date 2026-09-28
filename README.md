# MetalGround: Grounding DINO Benchmark & Experiment Harness

Grounding DINO Tiny on Apple Silicon (M3) の推論最適化検証スクリプトおよび実測結果（JSON）のリポジトリです。  
PyTorch MPSベースライン（~1,002ms）からMetalGround最終統合ランタイム（~483ms）への約2.08倍の高速化を実測・再現できます。

## 1. 測定環境
- **マシン**: MacBook Pro 14-inch (Apple M3 / 16 GB Unified Memory)
- **OS**: macOS Sequoia 15.7.9 (Darwin arm64)
- **Python**: 3.12 (native arm64)

## 2. セットアップ
```bash
# 仮想環境の作成とインストール
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

# 検証用画像の取得
mkdir -p assets
curl -L --fail --retry 3 https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/cat_dog.jpeg -o assets/input.jpg
```

## 3. ベンチマーク実行方法
```bash
# ① ベースライン測定（PyTorch MPS: 約1,000ms）
uv run python scripts/01_baseline_hf.py --device mps --image assets/input.jpg --prompt "a cat" "a dog" --iters 10

# ② 主Headline測定（オリジナル vs 最終統合ランタイムの交互測定: 2.08倍高速化の実証）
uv run python scripts/50_original_vs_final_multiprocess.py
```

## 4. 計測データと詳細レポート
- **計測JSON生データ**: `results/`（全86ファイル）
- **実験判断ログ・詳細考察**: ポートフォリオサイト（MetalGroundドキュメント）をご覧ください。
