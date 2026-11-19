"""
ローカル:  python train.py pack --name box-v1       -> data/box-v1.zip を Kaggle Dataset としてアップロード
Kaggle :  !pip install -U ultralytics
          !python /kaggle/input/<dataset>/train.py train --name box-v1
"""
import argparse
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LABELED_DIR = HERE / "data" / "labeled"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
WORK_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else HERE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def group_key(stem: str) -> str:
    # 同じ動画から切り出した連続フレームは酷似しているため、分割を跨ると val/test が過大評価される
    return stem.rsplit("_", 1)[0]


def load_names(src: Path) -> list[str]:
    for name in ("classes.txt", "obj.names"):
        path = src / name
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise SystemExit(f"クラス定義がありません: {src / 'classes.txt'}")


def collect_pairs(src: Path, nc: int) -> list[tuple[Path, Path]]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"{images_dir} と {labels_dir} が必要です")

    pairs = []
    unlabeled = 0
    for img in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        label = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
        # 空の txt は「物体なし」の負例として扱い、txt 自体が無い画像は未アノテーションとして除外
        if not label.exists():
            unlabeled += 1
            continue
        for n, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if parts and not (parts[0].isdigit() and int(parts[0]) < nc and len(parts) >= 5):
                raise SystemExit(f"不正なラベル行: {label}:{n}: {line!r}")
        pairs.append((img, label))

    if unlabeled:
        print(f"ラベル無しのため除外: {unlabeled} images")
    if not pairs:
        raise SystemExit(f"ラベル付き画像がありません: {images_dir}")
    return pairs


def split(pairs: list, ratios: dict[str, float], seed: int) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    for pair in pairs:
        groups[group_key(pair[0].stem)].append(pair)

    rng = random.Random(seed)
    if len(groups) >= len([r for r in ratios.values() if r > 0]) + 1:
        units = [groups[k] for k in sorted(groups)]
    else:
        print("WARN: 撮影グループが少ないため画像単位で分割します (val/test が楽観的になります)")
        units = [[p] for p in pairs]
    rng.shuffle(units)

    out: dict[str, list] = {s: [] for s in SPLITS}
    for unit in units:
        for s in ("val", "test"):
            if len(out[s]) < len(pairs) * ratios[s]:
                out[s].extend(unit)
                break
        else:
            out["train"].extend(unit)
    if not out["train"] or not out["val"]:
        raise SystemExit(f"train/val が空になりました: { {s: len(v) for s, v in out.items()} }")
    return out


def cmd_pack(args: argparse.Namespace) -> None:
    src = Path(args.src)
    names = load_names(src)
    parts = split(collect_pairs(src, len(names)), {"val": args.val, "test": args.test}, args.seed)

    data = {s: f"{s}/images" for s in SPLITS if parts[s]}
    data["names"] = dict(enumerate(names))
    out = HERE / "data" / f"{args.name}.zip"
    # JPEG/PNG は既に圧縮済みで deflate しても縮まず時間だけかかる
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        zf.write(Path(__file__), "train.py")
        for s, items in parts.items():
            for img, label in items:
                zf.write(img, f"{s}/images/{img.name}")
                zf.write(label, f"{s}/labels/{img.stem}.txt")

    print(f"classes: {names}")
    print("  ".join(f"{s}: {len(parts[s])}" for s in SPLITS))
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def find_data_yaml(hint: str | None) -> Path:
    if hint:
        path = Path(hint)
        return path / "data.yaml" if path.is_dir() else path
    candidates = sorted(KAGGLE_INPUT.rglob("data.yaml")) if KAGGLE_INPUT.exists() else []
    if len(candidates) != 1:
        raise SystemExit(f"--data で data.yaml を指定してください (候補: {[str(c) for c in candidates]})")
    return candidates[0]


def resolve_data_yaml(data_yaml: Path, name: str) -> Path:
    # Dataset の展開先はアップロード方法で階層が変わるため、path を data.yaml の位置から決める
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = str(data_yaml.parent.resolve())
    out = WORK_DIR / f"{name}_data.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def cmd_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        data_yaml = resolve_data_yaml(find_data_yaml(args.data), args.name)
        batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            project=str(WORK_DIR / "runs"),
            name=args.name,
            exist_ok=True,
        )

    best = Path(model.trainer.best)
    onnx = YOLO(best).export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                             half=args.half, dynamic=args.dynamic, simplify=True)
    out_dir = WORK_DIR / "models" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in (best, Path(onnx)):
        shutil.copy2(f, out_dir / f.name)
    print(f"-> {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 学習 (Kaggle 想定)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="アノテーション済みデータを train/val/test に分割し Kaggle 用 zip を作る")
    p.add_argument("--name", required=True, help="データセット名 (zip 名)")
    p.add_argument("--src", default=str(LABELED_DIR), help="images/ labels/ classes.txt を含む YOLO 形式データ")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("train", help="学習して .pt と .onnx を出力")
    p.add_argument("--name", default="train", help="実行名。runs/<name>, models/<name> に出力")
    p.add_argument("--data", default=None, help="data.yaml かそのディレクトリ。省略時は /kaggle/input から探索")
    p.add_argument("--model", default="yolo26n.pt", help="事前学習済み重み or モデル yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=float, default=-1, help="-1 で GPU メモリから自動決定、0~1 でメモリ使用率指定")
    p.add_argument("--device", default=None, help="0 / 0,1 (Kaggle T4x2) / cpu。省略時は自動")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=30, help="val が改善しないまま経過したら早期終了するエポック数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="中断した学習の last.pt を指定して再開")
    p.add_argument("--opset", type=int, default=None, help="ONNX opset。省略時は ultralytics の既定")
    p.add_argument("--half", action="store_true", help="ONNX を FP16 で出力 (GPU 必須)")
    p.add_argument("--dynamic", action="store_true", help="ONNX の入力サイズ/バッチを可変にする")
    p.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
