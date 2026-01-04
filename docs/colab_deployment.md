# Google Colabでの実行手順

このガイドでは、Google Colab上でIsaac Labを使って4WDロボットを学習させる詳細な手順を説明します。

## 事前準備

- Googleアカウント（Colab用）
- （オプション）Google Drive（チェックポイント保存用）
- （オプション）GitHubアカウント（プロジェクト管理用）

## 方法1: ノートブックを直接アップロード

### Step 1: プロジェクトファイルの準備

```bash
# ローカルマシンで実行
cd ~/workspace/isaac_4wd_rl

# 必要なファイルをzipに圧縮
zip -r isaac_4wd_rl.zip \
    assets/ \
    envs/ \
    scripts/ \
    requirements.txt
```

### Step 2: Google Colabでノートブックを開く

1. ブラウザで https://colab.research.google.com を開く
2. `ファイル` → `ノートブックをアップロード`
3. `colab/setup_isaac_lab.ipynb`をアップロード

### Step 3: ランタイム設定

```
ランタイム → ランタイムのタイプを変更

設定:
  - ランタイムのタイプ: Python 3
  - ハードウェア アクセラレータ: GPU
  - GPU タイプ: L4 (推奨) または A100
```

### Step 4: ノートブックを実行

1. **セル1-4**: システム設定とIsaac Labインストール
   - 約15-20分かかります
   - エラーが出なければ成功

2. **セル5**: プロジェクトファイルのアップロード
   - `isaac_4wd_rl.zip`をアップロード
   - または個別にファイルをアップロード

3. **セル6-9**: 環境のテスト
   - 1台の車両で動作確認

4. **セル11**: 本格的な学習開始
   ```python
   !python scripts/train.py \
       --num_envs 10 \
       --headless \
       --max_iterations 5000
   ```

## 方法2: GitHubから直接Clone（推奨）

### Step 1: GitHubにプロジェクトをプッシュ

```bash
# ローカルマシンで実行
cd ~/workspace/isaac_4wd_rl

# Gitリポジトリを初期化
git init
git add .
git commit -m "Initial commit: 4WD robot learning project"

# GitHubリポジトリを作成後
git remote add origin https://github.com/YOUR_USERNAME/isaac_4wd_rl.git
git branch -M main
git push -u origin main
```

### Step 2: Colabで専用ノートブックを作成

Colab上で新しいノートブックを作成し、以下のセルを実行：

#### セル1: GPU確認

```python
!nvidia-smi
```

#### セル2: システム依存関係のインストール

```python
!apt-get update && apt-get install -y \
    git git-lfs wget \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libglu1-mesa
```

#### セル3: Isaac Labのインストール

```python
import os

# Clone Isaac Lab
if not os.path.exists('/content/IsaacLab'):
    !git clone https://github.com/isaac-sim/IsaacLab.git /content/IsaacLab

%cd /content/IsaacLab
!./isaaclab.sh --install
```

#### セル4: プロジェクトをClone

```python
# あなたのGitHubリポジトリに置き換えてください
!git clone https://github.com/YOUR_USERNAME/isaac_4wd_rl.git /content/isaac_4wd_rl

%cd /content/isaac_4wd_rl
!pip install -r requirements.txt
```

#### セル5: 環境変数の設定

```python
import os
import sys

os.environ['PYTHONPATH'] = f"/content/IsaacLab:{os.environ.get('PYTHONPATH', '')}"
os.environ['PROJECT_PATH'] = '/content/isaac_4wd_rl'
os.environ['HEADLESS'] = '1'

sys.path.insert(0, '/content/isaac_4wd_rl')
print("Environment configured!")
```

#### セル6: テスト実行（1台のみ）

```python
%cd /content/isaac_4wd_rl

!python scripts/train.py \
    --num_envs 1 \
    --headless \
    --max_iterations 100
```

#### セル7: 本格的な学習

```python
!python scripts/train.py \
    --num_envs 10 \
    --headless \
    --max_iterations 5000
```

#### セル8: TensorBoard起動

```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

#### セル9: モデルのエクスポート

```python
# 学習完了後、最新のチェックポイントをONNXに変換
import glob
latest_checkpoint = sorted(glob.glob('logs/*/model_final.pt'))[-1]

!python scripts/export_onnx.py \
    --checkpoint {latest_checkpoint} \
    --output models/4wd_policy.onnx

print(f"Model exported from: {latest_checkpoint}")
```

#### セル10: モデルのダウンロード

```python
from google.colab import files
files.download('models/4wd_policy.onnx')
```

#### セル11: Google Driveへのバックアップ

```python
from google.colab import drive
drive.mount('/content/drive')

# ログとモデルをDriveにコピー
!mkdir -p /content/drive/MyDrive/isaac_4wd_training
!cp -r logs /content/drive/MyDrive/isaac_4wd_training/
!cp -r models /content/drive/MyDrive/isaac_4wd_training/

print("Backup completed!")
```

## トラブルシューティング

### エラー: "CUDA out of memory"

**解決策1**: 並列環境数を減らす
```python
!python scripts/train.py --num_envs 5 --headless --max_iterations 5000
```

**解決策2**: バッチサイズを減らす
`scripts/train.py`の`agent_cfg`を編集：
```python
num_steps_per_env=12,  # 24から12に減らす
```

### エラー: "Isaac Lab installation failed"

**解決策**:
```python
# ランタイムをリセット
# ランタイム → セッションを管理 → すべて終了

# 新しいノートブックで再試行
```

### エラー: "URDF file not found"

**解決策**: URDFファイルのパスを確認
```python
# envs/isaac_4wd_env.py の FourWDSceneCfg を編集
asset_path="/content/isaac_4wd_rl/assets/robots/4wd_vehicle.urdf"
```

### 警告: "Colab notebook timeout"

Colabの無料版は実行時間に制限があります。

**解決策**:
1. 定期的にチェックポイントを保存
2. Google Driveにバックアップ
3. 中断した場合は`--checkpoint`オプションで再開

```python
!python scripts/train.py \
    --num_envs 10 \
    --headless \
    --max_iterations 5000 \
    --checkpoint logs/4wd_ppo_XXXXXXXX_XXXXXX/model_XXXX.pt
```

## 学習パラメータの調整

### 並列環境数

| GPU | 推奨環境数 | メモリ使用量 |
|-----|-----------|-------------|
| T4  | 5-8       | ~12GB       |
| L4  | 10-15     | ~16GB       |
| A100| 20-30     | ~30GB       |

### イテレーション数と学習時間

| イテレーション | 環境数 | 所要時間 | 品質 |
|--------------|--------|----------|------|
| 100          | 1      | 5分      | テスト用 |
| 1000         | 10     | 2時間    | 低品質 |
| 5000         | 10     | 8時間    | 中品質（推奨） |
| 10000        | 20     | 15時間   | 高品質 |

## ベストプラクティス

### 1. 段階的な学習

```python
# Stage 1: 動作確認（5分）
!python scripts/train.py --num_envs 1 --headless --max_iterations 100

# Stage 2: 短期学習（2時間）
!python scripts/train.py --num_envs 10 --headless --max_iterations 1000

# Stage 3: 本格学習（8時間）
!python scripts/train.py --num_envs 10 --headless --max_iterations 5000
```

### 2. TensorBoardでの監視

学習中は別セルでTensorBoardを起動し、進捗を確認：

```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

重要な指標：
- `reward/mean`: 平均報酬（上昇が理想）
- `reward/max`: 最大報酬
- `policy/learning_rate`: 学習率

### 3. 定期的なバックアップ

```python
# 1時間ごとに実行
from google.colab import drive
drive.mount('/content/drive')

!cp -r logs /content/drive/MyDrive/isaac_4wd_backup/
!cp -r models /content/drive/MyDrive/isaac_4wd_backup/
```

### 4. 複数設定の並行実験

異なる報酬関数やパラメータで実験する場合：

```python
# 実験1: 標準設定
!python scripts/train.py --num_envs 10 --headless --max_iterations 3000 &

# 別のノートブックで実験2を実行
```

## 次のステップ

学習が完了したら：

1. **モデルの評価**
   - TensorBoardで学習曲線を確認
   - 報酬が収束しているか確認

2. **ONNX変換**
   - `scripts/export_onnx.py`で変換
   - ダウンロード

3. **実機テスト**
   - Raspberry Piにデプロイ
   - [hardware_setup_guide.md](hardware_setup_guide.md)を参照

---

**ヒント**: Colab Pro/Pro+を使用すると、より長い実行時間と高性能GPUが利用可能です。
