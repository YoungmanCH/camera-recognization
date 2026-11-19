"""
ローカル:  python train.py pack --name box-v1       -> data/box-v1.zip を Kaggle Dataset としてアップロード
Kaggle :  !pip install -U ultralytics
          !python /kaggle/input/<dataset>/train.py train --name box-v1
"""
import argparse
import json
import math
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "data" / "raw" / "images"
CLASSES_TXT = HERE / "data" / "classes.txt"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
WORK_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else HERE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
# 箱上縁の 4 隅。順序は order_corners が決める (画像上で左上寄りから時計回り)
KPT_NAMES = ["c0", "c1", "c2", "c3"]
# 左右反転で時計回りが反時計回りになり、c0<->c1, c2<->c3 が入れ替わる
FLIP_IDX = [1, 0, 3, 2]


def group_key(stem: str) -> str:
    # 同じ動画から切り出した連続フレームは酷似しているため、分割を跨ると val/test が過大評価される
    return stem.rsplit("_", 1)[0]


def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"クラス定義がありません: {path} (1 行 1 クラス、X-AnyLabeling と共用)")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def order_corners(pts: list[tuple[float, float]], box: tuple[float, float, float, float]) -> list[int]:
    # 箱は上から見て対称なので点の名前に意味を持たせられない。画像上で bbox 左上に最も近い隅から時計回りに並べ、
    # 推論側の PnP で長辺/短辺の対応を総当たりする
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    cw = sorted(range(len(pts)), key=lambda i: math.atan2(pts[i][1] - cy, pts[i][0] - cx))
    start = min(range(len(cw)), key=lambda k: abs(pts[cw[k]][0] - box[0]) + abs(pts[cw[k]][1] - box[1]))
    return cw[start:] + cw[:start]


def json_to_yolo(path: Path, names: list[str]) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    w, h = data["imageWidth"], data["imageHeight"]
    boxes, points = [], []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "point":
            points.append(shape)
            continue
        if shape["label"] not in names:
            raise SystemExit(f"classes.txt に無いラベル {shape['label']!r}: {path}")
        # rectangle 以外 (polygon 等) も外接矩形にすれば検出の教師として使える
        xs = [min(max(p[0], 0), w) for p in shape["points"]]
        ys = [min(max(p[1], 0), h) for p in shape["points"]]
        if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
            continue
        boxes.append({"cls": names.index(shape["label"]), "gid": shape.get("group_id"),
                      "xyxy": (min(xs), min(ys), max(xs), max(ys)), "kpts": []})

    for pt in points:
        x, y = pt["points"][0]
        owner = [b for b in boxes if pt.get("group_id") is not None and b["gid"] == pt.get("group_id")]
        if not owner:
            owner = [b for b in boxes if b["xyxy"][0] <= x <= b["xyxy"][2] and b["xyxy"][1] <= y <= b["xyxy"][3]]
        if len(owner) != 1:
            raise SystemExit(f"隅の点 {pt['label']!r} ({x:.0f},{y:.0f}) の箱が決まりません。矩形と同じ group_id を付けてください: {path}")
        owner[0]["kpts"].append((x, y, 1 if pt.get("difficult") else 2, pt["label"]))

    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        kpts = [(0.0, 0.0, 0)] * len(KPT_NAMES)
        if len(b["kpts"]) == len(KPT_NAMES):
            order = order_corners([k[:2] for k in b["kpts"]], b["xyxy"])
            kpts = [b["kpts"][i][:3] for i in order]
        elif b["kpts"] and all(k[3] in KPT_NAMES for k in b["kpts"]):
            # 自動ラベリングで低信頼の隅が落ちた場合。名前は order_corners の順で付いている
            for x, y, v, label in b["kpts"]:
                kpts[KPT_NAMES.index(label)] = (x, y, v)
        elif b["kpts"]:
            print(f"WARN: 隅が {len(b['kpts'])} 点で順序を決められないため矩形のみ使用: {path.name}")
        kpt_str = " ".join(f"{x / w:.6f} {y / h:.6f} {v}" if v else "0 0 0" for x, y, v in kpts)
        lines.append(f"{b['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} {kpt_str}")
    return "\n".join(lines) + ("\n" if lines else "")


def collect(images_dir: Path, names: list[str], unlabeled_as_negative: bool) -> list[tuple[Path, str]]:
    if not images_dir.is_dir():
        raise SystemExit(f"画像ディレクトリがありません: {images_dir}")

    items = []
    skipped = 0
    for img in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        js = img.with_suffix(".json")
        if js.exists():
            items.append((img, json_to_yolo(js, names)))
        elif unlabeled_as_negative:
            items.append((img, ""))
        else:
            skipped += 1

    if skipped:
        print(f"JSON が無いため除外: {skipped} images (対象なし画像として使うなら --unlabeled-as-negative)")
    if not items:
        raise SystemExit(f"アノテーション済み画像がありません: {images_dir}")
    return items


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
    names = load_names(Path(args.classes))
    items = collect(Path(args.src), names, args.unlabeled_as_negative)
    parts = split(items, {"val": args.val, "test": args.test}, args.seed)

    data = {s: f"{s}/images" for s in SPLITS if parts[s]}
    data["names"] = dict(enumerate(names))
    data["kpt_shape"] = [len(KPT_NAMES), 3]
    data["flip_idx"] = FLIP_IDX
    out = HERE / "data" / f"{args.name}.zip"
    # JPEG/PNG は既に圧縮済みで deflate しても縮まず時間だけかかる
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        zf.write(Path(__file__), "train.py")
        for s, part in parts.items():
            for img, label in part:
                zf.write(img, f"{s}/images/{img.name}")
                zf.writestr(f"{s}/labels/{img.stem}.txt", label)

    rows = [line.split() for _, label in items for line in label.splitlines()]
    n_kpt = sum(1 for r in rows if any(v != "0" for v in r[7::3]))
    n_neg = sum(1 for _, label in items if not label)
    print(f"classes: {names}")
    print(f"images: {len(items)} (対象なし {n_neg})  boxes: {len(rows)} (隅あり {n_kpt})")
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
    write_xanylabeling_config(YOLO(best), out_dir / "xanylabeling.yaml", Path(onnx).name, args.name)
    print(f"-> {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")


def write_xanylabeling_config(model, path: Path, onnx_name: str, name: str) -> None:
    # YOLO26 の既定 export は NMS 込み (end2end) で、X-AnyLabeling 側は yolo26 型でのみその出力を解釈できる
    end2end = bool(getattr(model.model.model[-1], "end2end", False))
    config = {
        "type": "yolo26_pose" if end2end else "yolo11_pose",
        "name": f"{name}-custom",
        "provider": "custom",
        "display_name": name,
        "model_path": onnx_name,
        "conf_threshold": 0.25,
        "kpt_threshold": 0.25,
        "has_visible": True,
        "classes": {c: KPT_NAMES for c in model.names.values()},
    }
    if end2end:
        config.update(end2end=True, max_det=300)
    else:
        config["iou_threshold"] = 0.45
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 学習 (Kaggle 想定)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="X-AnyLabeling の JSON (矩形+上縁 4 隅の点) を YOLO-pose 形式に変換・分割し Kaggle 用 zip を作る")
    p.add_argument("--name", required=True, help="データセット名 (zip 名)")
    p.add_argument("--src", default=str(IMAGES_DIR), help="画像と同名 JSON が並ぶディレクトリ")
    p.add_argument("--classes", default=str(CLASSES_TXT))
    p.add_argument("--unlabeled-as-negative", action="store_true",
                   help="JSON の無い画像を「対象なし」として含める (全画像を確認し終えた後に使う)")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("train", help="学習して .pt と .onnx を出力")
    p.add_argument("--name", default="train", help="実行名。runs/<name>, models/<name> に出力")
    p.add_argument("--data", default=None, help="data.yaml かそのディレクトリ。省略時は /kaggle/input から探索")
    p.add_argument("--model", default="yolo26n-pose.pt", help="事前学習済み重み or モデル yaml")
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
