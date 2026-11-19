"""
YOLO-pose で箱の上縁 4 隅を検出し、魚眼補正 + PnP でカメラ座標系の (x, y, z, yaw) を求めて表示する。
カメラ座標系は OpenCV 準拠: x 右, y 下, z 前方 [mm]。yaw は箱の上面 (床) 上で測った、カメラ x 軸から長辺への角度 (奥向きが正, ±90°、箱は 180° 対称)。
  python pose.py --model models/box-v1/best.pt --source 1
  python pose.py --model models/box-v1/best.onnx --source data/raw/videos/xxx.mp4
"""
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import yaml

from calib import CALIB_YAML, load_calib
from collect import open_camera

HERE = Path(__file__).resolve().parent
BOX_YAML = HERE / "box.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BoxPose:
    xyz: np.ndarray
    yaw_deg: float
    rvec: np.ndarray
    tvec: np.ndarray
    err_px: float
    ambiguous: bool


def floor_yaw(R: np.ndarray) -> float:
    # カメラは傾いているので、カメラの x-z 平面ではなく箱の上面 (=床と平行) 上で角度を測る
    n = R[:, 2]
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2[2] < 0:
        e2 = -e2
    return math.atan2(R[:, 0] @ e2, R[:, 0] @ e1)


class PoseSolver:
    def __init__(self, calib: dict, width: float, depth: float, height: float):
        self.K, self.D = calib["K"], calib["D"]
        self.height = height
        w, d = width / 2, depth / 2
        # 箱座標系: 原点は上面中心、x 長辺方向、y 短辺方向、z 上向き。上から見て時計回り
        self.top = np.array([[w, d, 0], [w, -d, 0], [-w, -d, 0], [-w, d, 0]], np.float64)
        self.wire = np.vstack([self.top, self.top + [0, 0, -height]])

    def solve(self, kpts: np.ndarray) -> BoxPose | None:
        undist = cv2.fisheye.undistortPoints(kpts.reshape(-1, 1, 2).astype(np.float64), self.K, self.D, P=self.K)
        cands = []
        # 箱は 180° 対称で隅に固有の名前を付けられないため、対応の巡回・向きを総当たりして再投影誤差最小を採る
        for direction in (1, -1):
            for shift in range(4):
                obj = np.roll(self.top[::direction], shift, axis=0)
                n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, undist, self.K, None, flags=cv2.SOLVEPNP_IPPE)
                for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
                    R, _ = cv2.Rodrigues(rvec)
                    # 上面の法線がカメラを向いていない解は鏡像
                    if float(R[:, 2] @ tvec.ravel()) >= 0:
                        continue
                    yaw = (math.degrees(floor_yaw(R)) + 90) % 180 - 90
                    cands.append((float(err), yaw, rvec, tvec, R))
        if not cands:
            return None
        err, yaw, rvec, tvec, R = min(cands, key=lambda c: c[0])
        # 上面を浅い角度から遠目に見ると長辺/短辺の取り違えでも誤差がほぼ同じになり、yaw が 90° 跳ぶ
        ambiguous = any(abs((c[1] - yaw + 90) % 180 - 90) > 45 and c[0] < max(2 * err, err + 1.0) for c in cands)
        center = tvec.ravel() + R @ np.array([0.0, 0.0, -self.height / 2])
        return BoxPose(center, yaw, rvec, tvec, err, ambiguous)

    def draw(self, img: np.ndarray, pose: BoxPose) -> None:
        pts, _ = cv2.fisheye.projectPoints(self.wire.reshape(-1, 1, 3), pose.rvec, pose.tvec, self.K, self.D)
        pts = pts.reshape(-1, 2).astype(np.int32)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 255) if a < 4 and b < 4 else (0, 180, 255), 2)


def frames(source: str, args: argparse.Namespace, size: tuple[int, int]):
    if source.isdigit():
        cap = open_camera(int(source), *size, args.fps)
    elif Path(source).is_dir():
        for f in sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_EXTS):
            yield f.name, cv2.imread(str(f)), True
        return
    elif Path(source).suffix.lower() in IMAGE_EXTS:
        yield Path(source).name, cv2.imread(source), True
        return
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"開けません: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield source, frame, False
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="青い箱の位置・向き推定 (カメラ座標系)")
    parser.add_argument("--model", required=True, help="train.py で出力した best.pt / best.onnx")
    parser.add_argument("--source", default="1", help="カメラ番号 / 動画 / 画像 / 画像ディレクトリ")
    parser.add_argument("--calib", default=str(CALIB_YAML))
    parser.add_argument("--box", default=str(BOX_YAML))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="4 隅すべてがこれ以上のときだけ PnP する")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    calib = load_calib(Path(args.calib))
    box = yaml.safe_load(Path(args.box).read_text(encoding="utf-8"))
    solver = PoseSolver(calib, box["width"], box["depth"], box["height"])
    model = YOLO(args.model, task="pose")

    print("[q/ESC] 終了  (画像入力時は任意キーで次へ)")
    for name, frame, still in frames(args.source, args, calib["size"]):
        if frame.shape[1::-1] != calib["size"]:
            raise SystemExit(f"{name}: 解像度 {frame.shape[1::-1]} がキャリブレーション時 {calib['size']} と違います")
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        view = res.plot(labels=False)
        if res.keypoints is not None:
            for i, (xy, kc) in enumerate(zip(res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy())):
                x1, y1 = res.boxes.xyxy[i, :2].cpu().numpy().astype(int)
                pose = solver.solve(xy) if (kc >= args.kpt_conf).all() else None
                if pose is None:
                    cv2.putText(view, "no pose", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    continue
                solver.draw(view, pose)
                x, y, z = pose.xyz
                text = f"x{x:+.0f} y{y:+.0f} z{z:.0f}mm yaw{pose.yaw_deg:+.1f}{'?' if pose.ambiguous else ''}"
                color = (0, 165, 255) if pose.ambiguous else (0, 255, 0)
                cv2.putText(view, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"{name} #{i}: {text} (reproj {pose.err_px:.1f}px)")
        cv2.imshow("pose", view)
        if cv2.waitKey(0 if still else 1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
