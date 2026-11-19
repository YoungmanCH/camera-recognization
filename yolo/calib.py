"""
魚眼カメラの内部パラメータ推定 (cv2.fisheye)。推論と同じカメラ・解像度で行うこと。
  python calib.py capture -c 1 --cols 9 --rows 6 --square 25
  python calib.py capture -c 1 --cols 9 --rows 6 --square 25 --images data/calib   # 撮影済み画像から
  python calib.py check -c 1
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from collect import open_camera

HERE = Path(__file__).resolve().parent
CALIB_YAML = HERE / "data" / "calib.yaml"
CALIB_IMG_DIR = HERE / "data" / "calib"
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def load_calib(path: Path = CALIB_YAML) -> dict:
    if not path.exists():
        raise SystemExit(f"キャリブレーション結果がありません: {path} (python calib.py capture で作成)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {"K": np.array(data["K"], dtype=np.float64), "D": np.array(data["D"], dtype=np.float64).reshape(4, 1),
            "size": tuple(data["size"])}


def find_corners(gray, pattern: tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)


def calibrate(img_points: list, pattern: tuple[int, int], square: float, size: tuple[int, int]) -> dict:
    obj = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
    obj[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    pts = [p.reshape(1, -1, 2).astype(np.float64) for p in img_points]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # 縁に寄せすぎた画像や誤検出で CHECK_COND が落ちるので、外れ値を 1 枚ずつ除いて再推定する
    while True:
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate([obj] * len(pts), pts, size, np.eye(3), np.zeros((4, 1)),
                                                    flags=flags | cv2.fisheye.CALIB_CHECK_COND, criteria=crit)
            break
        except cv2.error as e:
            bad = re.search(r"input array (\d+)", str(e))
            if not bad or len(pts) <= 10:
                raise SystemExit(f"キャリブレーション失敗: {e}")
            print(f"条件の悪い画像 #{bad.group(1)} を除外して再試行")
            pts.pop(int(bad.group(1)))
    return {"K": K.tolist(), "D": D.ravel().tolist(), "size": list(size), "rms": float(rms), "n_images": len(pts),
            "pattern": list(pattern), "square_mm": square}


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(f"RMS {result['rms']:.3f}px ({result['n_images']} images) -> {path}")
    if result["rms"] > 1.0:
        print("WARN: RMS が 1px を超えています。ボードを画面の端や斜めにも置いて撮り直すと改善します")


def cmd_capture(args: argparse.Namespace) -> None:
    pattern = (args.cols, args.rows)
    if args.images:
        files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".png"})
        points, size = [], None
        for f in files:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            size = gray.shape[::-1]
            corners = find_corners(gray, pattern)
            print(f"{f.name}: {'ok' if corners is not None else 'not found'}")
            if corners is not None:
                points.append(corners)
        if len(points) < 10:
            raise SystemExit(f"検出できた画像が {len(points)} 枚です。10 枚以上必要です")
        save(calibrate(points, pattern, args.square, size), Path(args.out))
        return

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    CALIB_IMG_DIR.mkdir(parents=True, exist_ok=True)
    points = []
    print("[space] 検出中のフレームを採用  [c] キャリブレーション実行  [q/ESC] 中止")
    print("ボードを画面中央・四隅・端、近く/遠く、傾けた状態で 20~30 枚程度")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("フレーム取得に失敗しました")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern)
        view = frame.copy()
        for p in points:
            cv2.polylines(view, [p.astype(np.int32)], False, (80, 80, 80), 1)
        if corners is not None:
            cv2.drawChessboardCorners(view, pattern, corners, True)
        cv2.putText(view, f"{len(points)} frames", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calib", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and corners is not None:
            cv2.imwrite(str(CALIB_IMG_DIR / f"{len(points):03d}.png"), frame)
            points.append(corners)
        elif key == ord("c"):
            if len(points) < 10:
                print("10 枚以上採用してください")
                continue
            save(calibrate(points, pattern, args.square, gray.shape[::-1]), Path(args.out))
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_check(args: argparse.Namespace) -> None:
    calib = load_calib(Path(args.calib))
    cap = open_camera(args.camera, *calib["size"], args.fps)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        calib["K"], calib["D"], calib["size"], np.eye(3), balance=args.balance)
    maps = cv2.fisheye.initUndistortRectifyMap(calib["K"], calib["D"], np.eye(3), newK, calib["size"], cv2.CV_16SC2)
    print("直線が直線に見えれば OK。[q/ESC] 終了")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1::-1] != calib["size"]:
            raise SystemExit(f"解像度 {frame.shape[1::-1]} がキャリブレーション時 {calib['size']} と違います")
        cv2.imshow("raw | undistorted", np.hstack([frame, cv2.remap(frame, *maps, cv2.INTER_LINEAR)]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="魚眼カメラのキャリブレーション")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="チェッカーボードを撮影して内部パラメータを推定")
    p.add_argument("-c", "--camera", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cols", type=int, required=True, help="内側コーナーの横の数 (マス数 - 1)")
    p.add_argument("--rows", type=int, required=True, help="内側コーナーの縦の数 (マス数 - 1)")
    p.add_argument("--square", type=float, required=True, help="1 マスの一辺 [mm]")
    p.add_argument("--images", default=None, help="撮影済み画像のディレクトリ。指定時はカメラを使わない")
    p.add_argument("-o", "--out", default=str(CALIB_YAML))
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("check", help="歪み補正した映像を並べて確認")
    p.add_argument("-c", "--camera", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--calib", default=str(CALIB_YAML))
    p.add_argument("--balance", type=float, default=0.0, help="0: 有効画素のみ ~ 1: 全画角を残す")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
