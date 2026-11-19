# camera-recognization

ロボット搭載カメラで **青い箱** を認識し、ロボットから見た箱の相対位置・向き `(x, y, z, yaw)` を推定する。

## 概要

- カメラ: ELP 170° 魚眼 USB カメラ (720p / 30fps)
- 検出: YOLO-pose で箱の bbox と上縁 4 隅のキーポイントを推定
- 座標計算: 魚眼補正 (`cv2.fisheye`) + PnP で箱の実寸と 4 隅を対応付け、カメラ座標系の姿勢を求める
- 学習: X-AnyLabeling でアノテーション → Kaggle で学習

```
カメラ画像 ──▶ YOLO-pose (bbox + 4 隅) ──▶ 魚眼補正 + PnP ──▶ (x, y, z, yaw)
```

座標はカメラ基準 (OpenCV 準拠: x 右, y 下, z 前 [mm])。yaw は箱上面 (床) 上でカメラ x 軸から長辺への角度 (±90°)。

## 構成

| パス | 役割 |
| --- | --- |
| `yolo/collect.py` | カメラ列挙・動画撮影・フレーム切り出し |
| `yolo/train.py` | データセットのパッケージ化 (`pack`) と Kaggle 上での学習 (`train`) |
| `yolo/calib.py` | 魚眼キャリブレーション → `data/calib.yaml` |
| `yolo/pose.py` | 推論 + PnP による `(x, y, z, yaw)` 推定・表示 |
| `yolo/box.yaml` | 箱の実寸 (現在は仮の値) |

手順の詳細は [`yolo/CLAUDE.md`](yolo/CLAUDE.md) を参照。

## マイルストーン

1. [ ] **YOLO による推論**: 学習データ作成・学習・実機での推論
2. [ ] **座標計算**: 魚眼キャリブレーション・箱の実寸測定・PnP による `(x, y, z, yaw)` の精度検証
3. [ ] **3 カメラへの拡張**: 3 台同時取得・各カメラの結果をロボット座標系で統合
