# 4WD Robot Learning with Isaac Lab (Sim-to-Real)

LiDARを用いた障害物回避を学習する4輪駆動ロボットの強化学習プロジェクト。Google Colab上のIsaac Labでシミュレーション学習を行い、学習済みモデルをRaspberry Pi実機に移植します。

## プロジェクト概要

- **学習環境**: Google Colab + NVIDIA Isaac Lab
- **アルゴリズム**: PPO (Proximal Policy Optimization)
- **センサー**: 仮想LiDAR (360度スキャン)
- **実機**: Raspberry Pi 4/5 + RPLIDAR + 4WDシャーシ
- **Domain Randomization**: LiDARノイズ、摩擦係数、遅延など

## プロジェクト構成

```
isaac_4wd_rl/
├── README.md                      # このファイル
├── requirements.txt               # Python依存関係
├── docs/
│   └── project_spec.md           # プロジェクト仕様書
├── colab/
│   └── setup_isaac_lab.ipynb     # Google Colabセットアップノートブック
├── assets/
│   ├── robots/
│   │   └── 4wd_vehicle.urdf      # 4WD車両モデル (URDF)
│   └── config/
│       └── vehicle_config.yaml   # 車両パラメータ設定
├── envs/
│   └── isaac_4wd_env.py          # Isaac Lab強化学習環境
├── scripts/
│   ├── train.py                  # PPO学習スクリプト
│   └── export_onnx.py            # モデルONNX変換スクリプト
└── raspberry_pi/
    ├── requirements.txt           # Raspberry Pi用依存関係
    └── inference.py               # 実機推論スクリプト
```

## セットアップ手順

### 1. Google Colabで学習環境をセットアップ

1. Google Colabを開く
2. `colab/setup_isaac_lab.ipynb`をアップロード
3. ランタイムを**GPU (L4またはA100推奨)**に設定
4. ノートブックのセルを順番に実行

### 2. プロジェクトファイルをColabにアップロード

以下のファイルをColab環境にアップロードします：

```bash
# ローカルでプロジェクトをzipに圧縮
cd ~/workspace
zip -r isaac_4wd_rl.zip isaac_4wd_rl/

# Colabでアップロード後、解凍
# (Colabノートブック内で実行)
!unzip isaac_4wd_rl.zip -d /content/
```

または、GitHubにプッシュしてCloneする方法もあります。

### 3. 学習を開始

```python
# Colabノートブック内で実行
%cd /content/isaac_4wd_rl
!python scripts/train.py --num_envs 10 --headless --max_iterations 5000
```

**学習パラメータ:**
- `--num_envs`: 並列環境数 (デフォルト: 10)
- `--headless`: GUI無効化 (Colabでは必須)
- `--max_iterations`: 最大イテレーション数 (5000 = 約2-3時間)

### 4. 学習済みモデルをONNX形式に変換

```python
!python scripts/export_onnx.py \
    --checkpoint logs/4wd_ppo_YYYYMMDD_HHMMSS/model_final.pt \
    --output models/4wd_policy.onnx
```

### 5. モデルをダウンロード

```python
from google.colab import files
files.download('/content/isaac_4wd_rl/models/4wd_policy.onnx')
```

## Raspberry Piへのデプロイ

### ハードウェア要件

- **Raspberry Pi 4/5** (4GB RAM以上推奨)
- **RPLIDAR A1/A2** または互換LiDAR
- **L298N** モータードライバー (または同等品)
- **4WDシャーシ** (DCモーター × 4)
- **バッテリー** (7.4V Li-Po など)

### 配線例

```
[Raspberry Pi]
  GPIO 17, 22, 23, 24 → モータードライバー (ENA, ENB, ENC, END)
  GPIO 27, 18, 15, 14 → 前輪方向制御
  GPIO 25, 8, 7, 1    → 後輪方向制御

[RPLIDAR]
  USB → /dev/ttyUSB0

[モータードライバー]
  OUT1/2 → 前左モーター
  OUT3/4 → 前右モーター
  OUT5/6 → 後左モーター
  OUT7/8 → 後右モーター
```

### ソフトウェアセットアップ

```bash
# Raspberry Pi上で実行

# 1. プロジェクトをコピー
scp 4wd_policy.onnx pi@raspberrypi.local:~/
scp -r raspberry_pi/ pi@raspberrypi.local:~/isaac_4wd_rl/

# 2. SSH接続
ssh pi@raspberrypi.local

# 3. 依存関係をインストール
cd ~/isaac_4wd_rl/raspberry_pi
pip install -r requirements.txt

# 4. LiDARの権限設定
sudo chmod 666 /dev/ttyUSB0

# 5. 推論スクリプトを実行
sudo python3 inference.py --model ~/4wd_policy.onnx
```

## 学習の監視 (TensorBoard)

学習中の進捗は、TensorBoardで確認できます：

```python
# Colabノートブック内で実行
%load_ext tensorboard
%tensorboard --logdir logs/
```

**主要な指標:**
- `reward/mean`: 平均報酬 (上昇傾向が理想)
- `policy/learning_rate`: 学習率
- `policy/entropy`: ポリシーのエントロピー

## トラブルシューティング

### Colab: GPU not available

- ランタイムタイプを「GPU」に変更
- `nvidia-smi`でGPUを確認

### Isaac Lab: Installation failed

- Colabのランタイムをリセット
- 最新のIsaac Labリポジトリを確認

### Raspberry Pi: LiDAR not detected

```bash
# デバイスを確認
ls -l /dev/ttyUSB*

# 権限を付与
sudo chmod 666 /dev/ttyUSB0
```

### Raspberry Pi: Motors not responding

- GPIO配線を確認
- モータードライバーの電源供給を確認
- `sudo`権限で実行しているか確認

## カスタマイズ

### LiDAR解像度の変更

`assets/config/vehicle_config.yaml`:
```yaml
lidar:
  resolution: 36  # 10度刻み（軽量）
  # resolution: 360  # 1度刻み（高精度）
```

### 報酬関数の調整

`envs/isaac_4wd_env.py`:
```python
@configclass
class RewardsCfg:
    forward_progress = RewTerm(func=reward_forward_progress, weight=1.0)
    collision = RewTerm(func=reward_collision_penalty, weight=-100.0)
    # カスタム報酬を追加可能
```

### 並列環境数の変更

```bash
# より高速に学習（GPU性能が必要）
python scripts/train.py --num_envs 20

# デバッグ用（1台のみ）
python scripts/train.py --num_envs 1
```

## 参考資料

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Domain Randomization](https://arxiv.org/abs/1703.06907)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します！

---

**作成日**: 2026-01-04
**バージョン**: 1.0.0
