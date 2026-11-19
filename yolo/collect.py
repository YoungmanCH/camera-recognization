import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 存在しない index を開いたときの WARN がカメラ列挙のたびに出てノイズになる
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

ROOT = Path(__file__).resolve().parent / "data" / "raw"
VIDEO_DIR = ROOT / "videos"
IMAGE_DIR = ROOT / "images"


def open_camera(index: int, width: int, height: int, fps: int, exposure: float | None = None) -> cv2.VideoCapture:
    # MSMF は起動が遅く解像度指定が効かないカメラが多いため DirectShow を優先
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} を開けませんでした")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # YUY2 は USB2.0 帯域不足で 720p≒9fps しか出ない。DSHOW は解像度設定時に FOURCC を戻すので最後に指定する
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if exposure is not None:
        # ローリングシャッターなので、オート露出で暗所の露光が伸びると動体がブレ・斜めに歪む
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cmd_list(args: argparse.Namespace) -> None:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for i in range(args.max):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            default = frame.shape[1::-1] if ok else None
            # 過大な値を要求するとドライバが対応する最大解像度に丸めるので、それを上限として読む
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
            ok, frame = cap.read()
            maximum = frame.shape[1::-1] if ok else None
            print(f"camera {i}: default={default} max={maximum}")
        cap.release()


def cmd_record(args: argparse.Namespace) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.exposure)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera {args.camera}: {w}x{h} {fourcc} @ {fps:.1f}fps (要求値)")
    print("[r] 録画開始/停止  [s] 静止画保存  [p] ドライバ設定画面  [q/ESC] 終了")

    writer = None
    video_path = None
    rec_start = 0.0
    measured = 0.0
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("フレーム取得に失敗しました")
            break
        now = time.time()
        measured = 0.9 * measured + 0.1 / max(now - prev, 1e-6)
        prev = now

        if writer is not None:
            writer.write(frame)

        view = frame.copy()
        cv2.putText(view, f"{measured:4.1f}fps", (w - 130, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if writer is not None:
            cv2.circle(view, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(view, f"REC {time.time() - rec_start:5.1f}s", (42, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            if writer is None:
                video_path = VIDEO_DIR / f"{timestamp()}.mp4"
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                rec_start = time.time()
                print(f"録画開始: {video_path}")
            else:
                writer.release()
                writer = None
                print(f"録画停止: {video_path}")
        elif key == ord("p"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif key == ord("s"):
            path = IMAGE_DIR / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"保存: {path}")

    if writer is not None:
        writer.release()
        print(f"録画停止: {video_path}")
    cap.release()
    cv2.destroyAllWindows()


def sharpness(gray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_similar(a, b, threshold: float) -> bool:
    small_a = cv2.resize(a, (64, 64))
    small_b = cv2.resize(b, (64, 64))
    return cv2.absdiff(small_a, small_b).mean() < threshold


def extract(video: Path, out_dir: Path, every: float, every_frames: int | None, min_sharp: float, min_diff: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"開けません: {video}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, every_frames or round(fps * every))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    last_gray = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 手ブレ画像と静止中の重複フレームはアノテーションの手間に対して学習効果が薄い
            if sharpness(gray) >= min_sharp and (last_gray is None or not is_similar(gray, last_gray, min_diff)):
                cv2.imwrite(str(out_dir / f"{video.stem}_{idx:06d}.jpg"), frame)
                last_gray = gray
                saved += 1
        idx += 1
    cap.release()
    print(f"{video.name}: {idx} frames -> {saved} images")
    return saved


def cmd_extract(args: argparse.Namespace) -> None:
    targets = [Path(p) for p in args.videos] or sorted(VIDEO_DIR.glob("*.mp4"))
    if not targets:
        raise SystemExit(f"動画がありません: {VIDEO_DIR}")
    total = sum(extract(v, Path(args.out), args.every, args.every_frames, args.min_sharp, args.min_diff) for v in targets)
    print(f"合計 {total} images -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 学習用データ収集")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="接続中のカメラを列挙")
    p.add_argument("--max", type=int, default=5)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("record", help="プレビューしながら録画/静止画保存")
    p.add_argument("-c", "--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float, default=None,
                   help="手動露出 (DirectShow は 2^値 秒。例: -7 ≒ 1/128s)。省略時はオート")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("extract", help="動画からフレームを切り出し")
    p.add_argument("videos", nargs="*", help="省略時は data/raw/videos/*.mp4 全て")
    p.add_argument("-o", "--out", default=str(IMAGE_DIR))
    p.add_argument("--every", type=float, default=0.5, help="切り出し間隔 [秒]")
    p.add_argument("-n", "--every-frames", type=int, default=None, help="切り出し間隔 [フレーム]。指定時は --every より優先")
    p.add_argument("--min-sharp", type=float, default=50.0, help="Laplacian 分散がこれ未満はブレとして除外")
    p.add_argument("--min-diff", type=float, default=4.0, help="直前の保存画像との平均差分がこれ未満なら除外")
    p.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
