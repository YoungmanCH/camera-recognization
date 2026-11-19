# Yolo

青い箱の認識を行う Yolo を実装する

## 認識情報

- 箱の相対座標
  - (x, y, z, yaw)

## 手順

1. `collect.py record/extract` で撮影・切り出し → `data/raw/images/`
2. X-AnyLabeling で `box` の矩形 + 上縁 4 隅の点 (同じ group_id、ラベル名は任意、隠れた隅は difficult)
3. `train.py pack` → Kaggle で `train.py train` (YOLO-pose, kpt 4 点。順序は pack が画像上で時計回りに揃える)
4. `calib.py capture` で魚眼キャリブレーション → `data/calib.yaml` (推論と同じ解像度)
5. `box.yaml` に箱の実寸 → `pose.py --model ... --source 1` でカメラ座標系の (x, y, z, yaw) を表示

- 座標はカメラ基準 (x 右, y 下, z 前 [mm])。yaw は箱上面 (床) 上でカメラ x 軸から長辺への角度、±90°
- 上面を浅い角度で遠目に見ると長辺/短辺が判別できず yaw が 90° 跳ぶことがあり、その場合は `yaw?` と表示
