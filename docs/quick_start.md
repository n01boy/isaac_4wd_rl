# クイックスタートガイド

最速で4WDロボットの学習を開始するための簡易ガイドです。

## ⚡ 5分でスタート (Google Colab)

### Step 1: Colabノートブックを開く

1. [Google Colab](https://colab.research.google.com/)にアクセス
2. `ファイル` → `ノートブックをアップロード`
3. `colab/setup_isaac_lab.ipynb`をアップロード

### Step 2: GPUランタイムに変更

```
ランタイム → ランタイムのタイプを変更 → GPU (L4またはA100)
```

### Step 3: セルを順番に実行

- `Runtime` → `Run all` または `Shift + Enter`で各セルを実行
- 初回は約15-20分かかります（Isaac Labのインストール）

### Step 4: 学習開始

ノートブックの指示に従ってプロジェクトファイルをアップロード後：

```python
!python scripts/train.py --num_envs 10 --headless --max_iterations 1000
```

これで学習が開始されます！

## 🎯 最小限の学習（テスト用）

本格的な学習の前に、環境が正しく動作するか確認：

```python
# 1台のみ、100イテレーション（約5分）
!python scripts/train.py --num_envs 1 --headless --max_iterations 100
```

## 📊 学習の進捗を確認

別のセルで以下を実行：

```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

TensorBoardが起動し、リアルタイムで学習曲線を確認できます。

## 💾 モデルの保存とダウンロード

学習完了後：

```python
# 1. ONNX形式に変換
!python scripts/export_onnx.py \
    --checkpoint logs/4wd_ppo_XXXXXXXX_XXXXXX/model_final.pt \
    --output models/4wd_policy.onnx

# 2. ダウンロード
from google.colab import files
files.download('models/4wd_policy.onnx')
```

## 🤖 Raspberry Piでの実行（超速版）

### 前提条件

- ハードウェアが組み立て済み
- Raspberry Pi OSがインストール済み

### 3つのコマンドで起動

```bash
# 1. 依存関係のインストール
pip install onnxruntime numpy rplidar-roboticia RPi.GPIO

# 2. モデルとスクリプトを転送
# (ローカルマシンから実行)
scp 4wd_policy.onnx pi@raspberrypi.local:~/
scp raspberry_pi/inference.py pi@raspberrypi.local:~/

# 3. 実行
# (Raspberry Pi上で実行)
sudo python3 ~/inference.py --model ~/4wd_policy.onnx
```

## 🐛 よくある問題と解決策

### Colab: "GPU not available"

```python
# セルで確認
!nvidia-smi
```

- 表示されない場合: `ランタイム` → `ランタイムのタイプを変更` → `GPU`

### Colab: "Isaac Lab installation failed"

- `ランタイム` → `セッションを管理` → すべて削除
- 新しいノートブックで再試行

### Pi: "Permission denied: /dev/ttyUSB0"

```bash
sudo chmod 666 /dev/ttyUSB0
```

### Pi: Motors not responding

```bash
# sudoで実行しているか確認
sudo python3 inference.py --model 4wd_policy.onnx
```

## 📚 次のステップ

学習が成功したら：

1. **学習の最適化**
   - [README.md](../README.md) の「カスタマイズ」セクション
   - 報酬関数の調整
   - 並列環境数の増加

2. **実機での調整**
   - Domain Randomizationパラメータの微調整
   - 実機でのセンサーノイズの測定

3. **ハードウェア改良**
   - [hardware_setup_guide.md](hardware_setup_guide.md)
   - IMUセンサーの追加
   - カメラモジュールの統合

## 🎓 推奨する学習手順

### 第1段階: 環境確認（30分）

- [ ] Google ColabでIsaac Lab起動
- [ ] 1台1環境で100イテレーション実行
- [ ] エラーなく完了することを確認

### 第2段階: 短期学習（2時間）

- [ ] 10台並列で1000イテレーション
- [ ] TensorBoardで学習曲線を観察
- [ ] 報酬が上昇傾向にあるか確認

### 第3段階: 本格学習（5-8時間）

- [ ] 10-20台並列で5000イテレーション
- [ ] 定期的にチェックポイントを保存
- [ ] Google Driveにバックアップ

### 第4段階: 実機テスト

- [ ] ONNX変換とダウンロード
- [ ] Raspberry Piにデプロイ
- [ ] まずはハードウェアテスト実行
- [ ] 安全な環境で実機走行テスト

## 💡 ヒント

### 学習時間の目安

| 並列環境数 | イテレーション | 所要時間 | 用途 |
|-----------|--------------|----------|------|
| 1 | 100 | 5分 | 動作確認 |
| 10 | 1000 | 2時間 | 短期テスト |
| 10 | 5000 | 8時間 | 標準学習 |
| 20 | 10000 | 15時間 | 高品質モデル |

### Colabの無料枠を効率的に使う

- 学習は連続実行（途中で止めない）
- TensorBoardは必要な時だけ起動
- チェックポイントを定期的にGoogle Driveに保存

### 最初の実機テストのコツ

- ロボットを台の上で浮かせて実行
- 低速（50%速度）から開始
- 広い平坦な場所でテスト
- 緊急停止用のSSH接続を常に確保

---

**トラブルが発生した場合**: [README.md](../README.md)の「トラブルシューティング」セクションを参照してください。
