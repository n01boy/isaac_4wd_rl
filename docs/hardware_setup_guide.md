# ハードウェアセットアップガイド

Raspberry Pi 4WDロボットの物理的なセットアップガイドです。

## 必要な部品リスト

### 主要コンポーネント

| 部品 | 数量 | 参考価格 | 備考 |
|------|------|----------|------|
| Raspberry Pi 4/5 | 1 | ¥7,000-15,000 | 4GB RAM以上推奨 |
| microSDカード | 1 | ¥1,000-2,000 | 32GB以上、Class 10 |
| RPLIDAR A1/A2 | 1 | ¥10,000-30,000 | または互換品 |
| 4WDシャーシキット | 1 | ¥3,000-5,000 | DCモーター×4付き |
| L298N モータードライバー | 2 | ¥500-1,000 | または同等品 |
| Li-Poバッテリー (7.4V) | 1 | ¥2,000-3,000 | 2S 2200mAh以上 |
| 5V降圧コンバータ | 1 | ¥500 | Raspberry Pi用電源 |
| ジャンパーワイヤー | 適量 | ¥500 | オス-メス、オス-オス |
| モバイルバッテリー | 1 | ¥2,000 | またはDC-DCコンバータ |

### オプション部品

- IMU (慣性計測装置): MPU6050など
- カメラモジュール: Raspberry Pi Camera V2
- 超音波センサー: 補助センサーとして
- LEDインジケータ: デバッグ用

## 配線図

### 全体構成図

```
[Li-Po Battery 7.4V]
    |
    ├─→ [L298N Motor Driver #1] → [Front Left Motor]
    |                            → [Front Right Motor]
    |
    ├─→ [L298N Motor Driver #2] → [Rear Left Motor]
    |                            → [Rear Right Motor]
    |
    └─→ [5V Buck Converter] → [Raspberry Pi]
                               (GPIO pins) → [Motor Drivers]

[USB Power Bank] → [RPLIDAR]
    (USB) → [Raspberry Pi USB Port]
```

### GPIO配線表 (BCM番号)

| GPIO | 機能 | 接続先 |
|------|------|--------|
| 17 | PWM | L298N #1 ENA (Front Left) |
| 22 | PWM | L298N #1 ENB (Front Right) |
| 23 | PWM | L298N #2 ENA (Rear Left) |
| 24 | PWM | L298N #2 ENB (Rear Right) |
| 27 | OUT | L298N #1 IN1 (FL Forward) |
| 18 | OUT | L298N #1 IN2 (FL Backward) |
| 15 | OUT | L298N #1 IN3 (FR Forward) |
| 14 | OUT | L298N #1 IN4 (FR Backward) |
| 25 | OUT | L298N #2 IN1 (RL Forward) |
| 8  | OUT | L298N #2 IN2 (RL Backward) |
| 7  | OUT | L298N #2 IN3 (RR Forward) |
| 1  | OUT | L298N #2 IN4 (RR Backward) |
| GND | GND | L298N GND (両方) |

**注意**: L298Nの5V出力ピンはRaspberry Piに**接続しない**でください（逆流防止のため）

### RPLIDARの接続

- **USB接続**: RPLIDAR → Raspberry Pi USBポート (`/dev/ttyUSB0`)
- **電源**: モバイルバッテリーまたはRaspberry PiのUSB電源を使用

## 組み立て手順

### 1. シャーシの組み立て

1. 4WDシャーシキットの説明書に従って基本フレームを組み立て
2. モーターを取り付け
3. 車輪を装着

### 2. 電子部品の取り付け

1. **Raspberry Piの固定**
   - シャーシ上部にスペーサーで固定
   - 振動対策として、ゴムワッシャーを使用推奨

2. **L298Nモータードライバーの取り付け**
   - シャーシ側面または後部に固定
   - 放熱を考慮して配置

3. **RPLIDARの取り付け**
   - シャーシ最上部、中央に配置
   - 360度スキャン可能な位置に設置
   - 高さは床から約12cm（`vehicle_config.yaml`のz=0.12に対応）

4. **バッテリーの配置**
   - シャーシ下部または後部に固定
   - 重心バランスを考慮

### 3. 配線

1. **モーター配線**
   ```
   Front Left Motor  → L298N #1 OUT1, OUT2
   Front Right Motor → L298N #1 OUT3, OUT4
   Rear Left Motor   → L298N #2 OUT1, OUT2
   Rear Right Motor  → L298N #2 OUT3, OUT4
   ```

2. **電源配線**
   ```
   Battery + → L298N #1 VCC, L298N #2 VCC
   Battery - → L298N #1 GND, L298N #2 GND, 5V Converter GND

   5V Converter OUT+ → Raspberry Pi GPIO Pin 2 (5V)
   5V Converter OUT- → Raspberry Pi GPIO Pin 6 (GND)
   ```

3. **GPIO配線**
   - 上記のGPIO配線表に従って接続
   - ジャンパーワイヤーに色分けを使用すると便利：
     - 赤: 電源
     - 黒: GND
     - その他: 信号線

4. **LiDAR配線**
   - RPLIDARをUSBケーブルでRaspberry Piに接続

### 4. 動作確認

```bash
# SSHでRaspberry Piに接続
ssh pi@raspberrypi.local

# LiDARデバイスの確認
ls -l /dev/ttyUSB*

# ハードウェアテストの実行
cd ~/isaac_4wd_rl/raspberry_pi
sudo python3 test_hardware.py --test all
```

## トラブルシューティング

### モーターが動かない

**原因と対策:**

1. **電源電圧不足**
   - バッテリー電圧を確認（マルチメータ使用）
   - 7.0V以下の場合は充電

2. **配線間違い**
   - GPIO配線を再確認
   - L298Nのジャンパーキャップがついているか確認

3. **PWM周波数の問題**
   - `inference.py`のPWM周波数を変更してみる（1000Hz → 500Hz）

### RPLIDARが認識されない

**原因と対策:**

1. **デバイスが表示されない**
   ```bash
   # USBデバイスを確認
   lsusb
   # CP2102またはCP2105が表示されるはず

   # カーネルログを確認
   dmesg | grep tty
   ```

2. **権限エラー**
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   # または
   sudo usermod -a -G dialout pi
   # 再ログイン後に有効
   ```

3. **電源不足**
   - RPLIDARを別電源（モバイルバッテリー）から供給

### Raspberry Piの電源が不安定

**原因と対策:**

1. **降圧コンバータの容量不足**
   - 3A以上出力可能なコンバータを使用

2. **電圧降下**
   - バッテリーから直接5V変換（長いケーブルを避ける）

3. **ノイズ対策**
   - コンデンサ（1000μF）をRaspberry Piの電源ラインに追加

## 安全上の注意

1. **バッテリー取り扱い**
   - Li-Poバッテリーは過充電・過放電に注意
   - 使用後は必ず専用充電器で充電
   - 保管時は半分程度の充電状態を保つ

2. **電源投入順序**
   - ① バッテリー接続
   - ② Raspberry Pi起動を確認
   - ③ プログラム実行

3. **緊急停止**
   - プログラム実行中はSSH端末を開いたまま
   - `Ctrl+C`でいつでも停止可能
   - 物理的な電源スイッチも設置推奨

4. **テスト環境**
   - 初回テストは必ず浮かせた状態で実行
   - 床での走行テストは広い平坦な場所で実施

## メンテナンス

### 定期点検項目

- [ ] 車輪の固定ネジ（1週間ごと）
- [ ] バッテリー電圧チェック（使用前）
- [ ] 配線の断線・ゆるみ（1ヶ月ごと）
- [ ] RPLIDARのレンズクリーニング（必要に応じて）
- [ ] モータードライバーの発熱確認（使用後）

### 推奨する改良

1. **バッテリー電圧モニター**
   - ADコンバータでバッテリー残量を監視

2. **緊急停止ボタン**
   - 物理スイッチをGPIOに接続

3. **LED インジケータ**
   - 動作状態の視覚的な確認

4. **IMUセンサー追加**
   - より正確な速度推定

---

**次のステップ**: ハードウェアセットアップが完了したら、[README.md](../README.md)の「Raspberry Piへのデプロイ」セクションに進んでください。
