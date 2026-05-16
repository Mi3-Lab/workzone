import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _build_image_index(images: List[dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    file_to_id = {}
    id_to_file = {}
    for img in images:
        fname = img["file_name"]
        img_id = img["id"]
        file_to_id[fname] = img_id
        id_to_file[img_id] = fname
    return file_to_id, id_to_file


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _build_category_map(categories: List[dict]) -> Dict[str, int]:
    mapping = {}
    for cat in categories:
        mapping[_normalize_name(cat["name"])] = cat["id"]
    return mapping


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_image_paths(images_root: Path, file_names: List[str]) -> Dict[str, Path]:
    missing = []
    out = {}
    for name in file_names:
        path = images_root / name
        if path.exists():
            out[name] = path
        else:
            missing.append(name)
    if missing:
        print(f"Warning: {len(missing)} images missing under {images_root}")
    return out


def _precision_at_recall(coco_eval, iou_index: int, recall_target: float) -> float:
    # precision dims: [T, R, K, A, M]
    precision = coco_eval.eval["precision"]
    if precision is None:
        return float("nan")
    # Use area=all (0) and maxDets=100 (last index)
    p = precision[iou_index, :, :, 0, -1]
    # p shape: [R, K], values in [0,1] or -1
    r = coco_eval.params.recThrs
    valid = r >= recall_target
    if not valid.any():
        return float("nan")
    subset = p[valid, :]
    # ignore invalid entries (-1)
    vals = subset[subset > -1]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on COCO images and evaluate.")
    parser.add_argument("--gt", required=True, help="Path to COCO ground truth JSON.")
    parser.add_argument("--images", required=True, help="Images root directory.")
    parser.add_argument("--weights", required=True, help="YOLO weights path.")
    parser.add_argument("--out", required=True, help="Output predictions JSON path.")
    parser.add_argument("--summary", required=True, help="Output summary JSON path.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Device string for inference.")
    parser.add_argument("--recall-target", type=float, default=0.9, help="Recall target for precision@high-recall.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("ultralytics not installed; install torch + ultralytics first") from exc
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise SystemExit("pycocotools not installed; install it before running") from exc

    gt_path = Path(args.gt)
    images_root = Path(args.images)
    weights_path = Path(args.weights)

    gt = _load_json(gt_path)
    file_to_id, _ = _build_image_index(gt["images"])
    cat_map = _build_category_map(gt["categories"])

    image_paths = _collect_image_paths(images_root, list(file_to_id.keys()))
    if not image_paths:
        raise SystemExit("No images found; check --images path.")

    model = YOLO(str(weights_path))

    predictions = []
    paths = [str(p) for p in image_paths.values()]
    results = model.predict(
        source=paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for result in results:
        img_path = Path(result.path)
        fname = img_path.name
        img_id = file_to_id.get(fname)
        if img_id is None:
            continue
        if result.boxes is None or result.boxes.shape[0] == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        for box, score, cls_id in zip(boxes, scores, classes):
            class_name = names[int(cls_id)]
            cat_id = cat_map.get(_normalize_name(class_name))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(out_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision_at_recall = _precision_at_recall(
        coco_eval,
        iou_index=0,
        recall_target=args.recall_target,
    )

    summary = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR1": float(coco_eval.stats[6]),
        "AR10": float(coco_eval.stats[7]),
        "AR100": float(coco_eval.stats[8]),
        "precision_at_recall": precision_at_recall,
        "recall_target": args.recall_target,
        "images_evaluated": len(image_paths),
        "predictions": len(predictions),
        "weights": str(weights_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
