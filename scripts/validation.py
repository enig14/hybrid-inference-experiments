#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Optional, Set

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_filename_to_id(coco_gt: COCO) -> Dict[str, int]:
    fn2id = {}
    for img in coco_gt.dataset.get("images", []):
        fn = img.get("file_name")
        iid = img.get("id")
        if fn is not None and iid is not None:
            fn2id[fn] = int(iid)
    return fn2id


def filter_img_ids_by_folder(coco_gt: COCO, images_dir: str) -> List[int]:
    files = set(os.listdir(images_dir))
    fn2id = build_filename_to_id(coco_gt)

    img_ids = []
    missing = 0
    for fn in files:
        if fn in fn2id:
            img_ids.append(fn2id[fn])
        else:
            missing += 1

    img_ids = sorted(set(img_ids))
    if len(img_ids) == 0:
        raise RuntimeError(
            f"No matching COCO image_ids found for files in: {images_dir}\n"
            f"- Check that GT file_name matches your folder filenames.\n"
            f"- Example expected: 000000397133.jpg"
        )

    if missing > 0:
        print(f"[WARN] {missing} files in {images_dir} not found in GT file_name list.")

    return img_ids


def sanity_check_pred_json(pred_path: str) -> None:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Pred JSON must be a list of detections.")

    # quick structural check (first few)
    for i, det in enumerate(data[:5]):
        if not isinstance(det, dict):
            raise ValueError(f"Pred JSON item {i} is not an object.")
        for k in ("image_id", "category_id", "bbox", "score"):
            if k not in det:
                raise ValueError(f"Pred JSON item {i} missing key: {k}")
        bbox = det["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError(f"Pred JSON item {i} bbox must be [x,y,w,h].")

    print(f"[OK] Loaded {len(data)} detections from {pred_path}")


def pred_image_ids(pred_path: str) -> Set[int]:
    """Return set of image_id present in predictions.json."""
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Set[int] = set()
    for det in data:
        try:
            out.add(int(det["image_id"]))
        except Exception:
            pass
    return out


def parse_exclude_ids(exclude_ids: Optional[str], exclude_file: Optional[str]) -> Set[int]:
    """Parse excluded image ids from either a comma-separated string or a file."""
    out: Set[int] = set()

    if exclude_ids:
        # allow commas and spaces
        tokens = exclude_ids.replace(",", " ").split()
        for t in tokens:
            out.add(int(t))

    if exclude_file:
        if not os.path.exists(exclude_file):
            raise FileNotFoundError(f"exclude_file not found: {exclude_file}")

        if exclude_file.lower().endswith(".json"):
            with open(exclude_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                out.update(int(x) for x in obj)
            elif isinstance(obj, dict) and "exclude" in obj and isinstance(obj["exclude"], list):
                out.update(int(x) for x in obj["exclude"])
            else:
                raise ValueError("exclude_file .json must be a list [..] or {'exclude':[...]} format")
        else:
            # txt/others: accept whitespace or comma-separated
            with open(exclude_file, "r", encoding="utf-8") as f:
                txt = f.read()
            tokens = txt.replace(",", " ").split()
            for t in tokens:
                out.add(int(t))

    return out


def eval_coco_bbox(
    gt_json: str,
    pred_json: str,
    images_dir: Optional[str] = None,
    use_cats: bool = True,
    exclude_img_ids: Optional[Set[int]] = None,
    exclude_missing_preds: bool = False,
) -> Dict[str, float]:
    coco_gt = COCO(gt_json)

    # base evaluation set
    img_ids = coco_gt.getImgIds()
    if images_dir is not None:
        img_ids = filter_img_ids_by_folder(coco_gt, images_dir)

    sanity_check_pred_json(pred_json)

    # exclusions
    exclude_img_ids = exclude_img_ids or set()

    if exclude_missing_preds:
        pred_ids = pred_image_ids(pred_json)
        missing_pred = set(img_ids) - pred_ids
        if missing_pred:
            print(f"[INFO] exclude_missing_preds: excluding {len(missing_pred)} GT images missing in pred_json")
            exclude_img_ids = set(exclude_img_ids) | missing_pred

    if exclude_img_ids:
        before = len(img_ids)
        img_ids = [iid for iid in img_ids if iid not in exclude_img_ids]
        after = len(img_ids)
        print(f"[INFO] Excluded {before - after} images by exclude list. Evaluating {after} images.")

        if after == 0:
            raise RuntimeError("All images were excluded; nothing left to evaluate.")

    coco_dt = coco_gt.loadRes(pred_json)

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = img_ids
    ev.params.useCats = 1 if use_cats else 0
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    keys = [
        "AP50-95", "AP50", "AP75", "APS", "APM", "APL",
        "AR1", "AR10", "AR100", "ARS", "ARM", "ARL"
    ]
    out = {k: float(ev.stats[i]) for i, k in enumerate(keys)}
    out["num_images"] = float(len(img_ids))
    return out


def print_summary(tag: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {tag} ===")
    print(f"images: {int(metrics['num_images'])}")
    print(f"AP50-95: {metrics['AP50-95']:.6f}")
    print(f"AP50   : {metrics['AP50']:.6f}")
    print(f"AP75   : {metrics['AP75']:.6f}")
    print(f"APS/APM/APL: {metrics['APS']:.6f} / {metrics['APM']:.6f} / {metrics['APL']:.6f}")
    print(f"AR1/AR10/AR100: {metrics['AR1']:.6f} / {metrics['AR10']:.6f} / {metrics['AR100']:.6f}")


def print_delta(a_tag: str, a: Dict[str, float], b_tag: str, b: Dict[str, float]) -> None:
    print(f"\n=== Delta ({b_tag} - {a_tag}) ===")
    for k in ["AP50-95", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100"]:
        print(f"{k:8s}: {b[k] - a[k]:+.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="COCO ground-truth json, e.g. instances_val2017.json")
    ap.add_argument("--pred", required=True, help="COCO detection result json (your C++ output)")
    ap.add_argument("--pred2", default=None, help="Optional second predictions.json to compare")
    ap.add_argument("--images_dir", default=None, help="Optional folder to evaluate subset by file names")
    ap.add_argument("--no_cats", action="store_true", help="If set, disable category matching (rarely desired)")

    # B-plan options
    ap.add_argument("--exclude_ids", default=None,
                    help="Comma/space separated COCO image_ids to exclude, e.g. '1,5,10'")
    ap.add_argument("--exclude_file", default=None,
                    help="File with image_ids to exclude (.txt one per line or .json list or {'exclude':[...]}).")
    ap.add_argument("--exclude_missing_preds", action="store_true",
                    help="Exclude GT images that do not appear in pred_json image_id set (use with caution).")

    args = ap.parse_args()

    exclude_set = parse_exclude_ids(args.exclude_ids, args.exclude_file)

    m1 = eval_coco_bbox(
        args.gt, args.pred,
        images_dir=args.images_dir,
        use_cats=(not args.no_cats),
        exclude_img_ids=exclude_set,
        exclude_missing_preds=args.exclude_missing_preds,
    )
    print_summary("PRED1", m1)

    if args.pred2:
        m2 = eval_coco_bbox(
            args.gt, args.pred2,
            images_dir=args.images_dir,
            use_cats=(not args.no_cats),
            exclude_img_ids=exclude_set,
            exclude_missing_preds=args.exclude_missing_preds,
        )
        print_summary("PRED2", m2)
        print_delta("PRED1", m1, "PRED2", m2)


if __name__ == "__main__":
    main()
