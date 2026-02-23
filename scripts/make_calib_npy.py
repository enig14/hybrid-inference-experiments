#!/usr/bin/env python3
import os
import glob
import argparse
import random
import hashlib
import numpy as np
import cv2
from numpy.lib.format import open_memmap
import json


def _norm_abs(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def stable_hash_u64(seed: int, s: str) -> int:
    """
    Deterministic 64-bit key from (seed, string).
    Uses BLAKE2b (stable across Python versions, unlike built-in hash()).
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def stable_rng_from(seed: int, s: str) -> np.random.Generator:
    key = stable_hash_u64(seed, s)
    ss = np.random.SeedSequence([seed, int(key & 0xFFFFFFFF), int((key >> 32) & 0xFFFFFFFF)])
    return np.random.default_rng(ss)


def load_pad_stats(pad_stats_json: str | None):
    """
    Expected JSON:
      {
        "mean":  [m0,m1,m2],
        "std":   [s0,s1,s2],
        "scale": "0_255" | "0_1",
        "order": "BGR" | "RGB"
      }
    """
    if not pad_stats_json:
        return None
    with open(pad_stats_json, "r", encoding="utf-8") as f:
        st = json.load(f)

    mean = np.array(st["mean"], dtype=np.float32)
    std = np.array(st["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise RuntimeError("pad_stats_json must contain mean/std arrays of length 3")

    scale = st.get("scale", "0_255")
    order = st.get("order", "BGR").upper()
    if order not in ("BGR", "RGB"):
        raise RuntimeError("pad_stats_json.order must be 'BGR' or 'RGB'")

    if scale == "0_1":
        mean *= 255.0
        std *= 255.0
    elif scale != "0_255":
        raise RuntimeError("pad_stats_json.scale must be '0_255' or '0_1'")

    return {"mean255": mean, "std255": std, "order": order, "scale": scale}


def letterbox(
    im,
    new_shape=(640, 640),
    auto=False,
    scaleup=True,
    pad_mode="const",
    pad_const=114,
    pad_stats=None,
    sigma_clip=3.0,
    rng: np.random.Generator | None = None,
):
    shape = im.shape[:2]  # (h, w)
    h0, w0 = shape
    h, w = new_shape

    r = min(h / h0, w / w0)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    pad_w = w - new_unpad[0]
    pad_h = h - new_unpad[1]
    if auto:
        pad_w %= 32
        pad_h %= 32

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if pad_mode == "mean":
        m = im.reshape(-1, 3).mean(axis=0)
        pad_color = (int(round(m[0])), int(round(m[1])), int(round(m[2])))
    elif pad_mode == "rand":
        if pad_stats is None:
            raise RuntimeError("pad_mode=rand requires --pad_stats_json")

        mu = pad_stats["mean255"]
        sd = pad_stats["std255"]
        lo = mu - float(sigma_clip) * sd
        hi = mu + float(sigma_clip) * sd

        c = rng.normal(mu, sd).astype(np.float32)
        c = np.clip(c, lo, hi)
        c = np.clip(c, 0.0, 255.0)
        pad_color = (int(round(c[0])), int(round(c[1])), int(round(c[2])))
    else:
        v = int(pad_const)
        pad_color = (v, v, v)

    im = cv2.copyMakeBorder(
        im, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color
    )

    meta = {
        "orig_w": int(w0), "orig_h": int(h0),
        "dst_w": int(w),  "dst_h": int(h),
        "new_w": int(new_unpad[0]), "new_h": int(new_unpad[1]),
        "pad_left": int(pad_left), "pad_right": int(pad_right),
        "pad_top": int(pad_top), "pad_bottom": int(pad_bottom),
        "scale": float(r),
        "pad_mode": pad_mode,
        "pad_color_bgr": [int(pad_color[0]), int(pad_color[1]), int(pad_color[2])],
        "bbox_xyxy": [int(pad_left), int(pad_top),
                      int(pad_left + new_unpad[0]), int(pad_top + new_unpad[1])],
        "auto": bool(auto),
        "scaleup": bool(scaleup),
    }
    if pad_mode == "rand" and pad_stats is not None:
        meta["sigma_clip"] = float(sigma_clip)
        meta["pad_stats_order"] = pad_stats["order"]
        meta["pad_stats_scale"] = pad_stats["scale"]
        meta["pad_stats_mean"] = [float(x) for x in pad_stats["mean255"]]
        meta["pad_stats_std"] = [float(x) for x in pad_stats["std255"]]
    return im, meta


def load_images(image_dir, exts=("jpg", "jpeg", "png")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(image_dir, f"*.{e}"))
        paths += glob.glob(os.path.join(image_dir, f"*.{e.upper()}"))
    return sorted(paths)


def load_list(list_path: str, image_dir: str | None):
    out = []
    with open(list_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if not os.path.isabs(p) and image_dir is not None:
                p = os.path.join(image_dir, p)
            out.append(p)
    return out


def load_exclude_idx(exclude_idx_txt: str | None) -> set[int]:
    if not exclude_idx_txt:
        return set()
    s = set()
    with open(exclude_idx_txt, "r") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            t = t.split(",")[0].strip()
            try:
                s.add(int(t))
            except ValueError:
                raise RuntimeError(f"Invalid idx in exclude file: '{line.rstrip()}'")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default=None,
                    help="Folder with images (optional if --list is absolute paths)")
    ap.add_argument("--list", default=None,
                    help="Text file with one image path per line. If set, overrides sampling from --image_dir.")
    ap.add_argument("--shuffle_list", action="store_true",
                    help="If set with --list, shuffle before selection (not needed for select_mode=hash).")
    ap.add_argument("--out_list", default=None,
                    help="If set, write selected image paths to this file (after exclusion filtering).")

    ap.add_argument("--exclude_idx_txt", default=None,
                    help="Text file with idx (one per line) to EXCLUDE. idx refers to selection order before exclusion.")
    ap.add_argument("--num", type=int, default=300, help="Number of images to use (before exclusion)")
    ap.add_argument("--out", default="calib.npy", help="Output npy path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=640, help="Letterbox target size (square)")

    ap.add_argument("--to_rgb", action="store_true",
                    help="If set, convert BGR->RGB (recommended if your model expects RGB)")
    ap.add_argument("--no_norm", action="store_true",
                    help="If set, DO NOT divide by 255 (default: divide by 255)")

    ap.add_argument("--pad_mode", choices=["const", "mean", "rand"], default="const",
                    help="Letterbox padding mode. const=pad_const, mean=per-image mean color, rand=global random N(mean,std) clipped.")
    ap.add_argument("--pad_const", type=int, default=114,
                    help="Padding value when pad_mode=const (default: 114)")
    ap.add_argument("--pad_stats_json", default="",
                    help="Required if pad_mode=rand. JSON with mean/std/order/scale.")
    ap.add_argument("--sigma_clip", type=float, default=3.0,
                    help="For pad_mode=rand: clip color to mean±sigma_clip*std. (default: 3.0)")

    ap.add_argument("--select_mode", choices=["hash", "random"], default="hash",
                    help="Selection policy. hash=deterministic sort by hash(seed,path), random=Python random sampling.")
    ap.add_argument("--out_meta", default="",
                    help="Metadata output path (.jsonl). Default: <out>.meta.jsonl")
    args = ap.parse_args()

    # Make OpenCV more deterministic / less noisy
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    random.seed(args.seed)
    exclude_idx = load_exclude_idx(args.exclude_idx_txt)

    pad_stats = load_pad_stats(args.pad_stats_json) if args.pad_mode == "rand" else None

    # 1) 후보 경로 수집 + 정규화
    if args.list:
        paths = load_list(args.list, args.image_dir)
    else:
        if not args.image_dir:
            raise RuntimeError("Need --image_dir if --list is not provided.")
        paths = load_images(args.image_dir)

    # 파일 존재하는 것만 + 절대경로 정규화
    paths2 = []
    for p in paths:
        p2 = _norm_abs(p) if not os.path.isabs(p) else _norm_abs(p)
        if os.path.isfile(p2):
            paths2.append(p2)
    paths = sorted(set(paths2))  # 중복 제거 + 안정적 정렬

    if len(paths) == 0:
        raise RuntimeError("No valid image paths found.")

    # 2) 선택
    if args.select_mode == "hash":
        # deterministic: sort by hash(seed, abs_path), then take top-K
        keyed = [(stable_hash_u64(args.seed, p), p) for p in paths]
        keyed.sort(key=lambda x: x[0])
        sel = [p for _, p in keyed[:min(args.num, len(keyed))]]
    else:
        # random: preserve old behavior
        if args.list and args.shuffle_list:
            random.shuffle(paths)
            sel = paths[:min(args.num, len(paths))]
        else:
            if args.num >= len(paths):
                sel = paths
            else:
                sel = random.sample(paths, args.num)

    # 3) idx 제외 (idx는 'sel'의 원래 인덱스 기준)
    if exclude_idx:
        kept = []
        dropped = 0
        for i, p in enumerate(sel):
            if i in exclude_idx:
                dropped += 1
                continue
            kept.append(p)
        print(f"[INFO] exclude_idx: {len(exclude_idx)} requested, dropped={dropped}, kept={len(kept)}")
        sel = kept

    if len(sel) == 0:
        raise RuntimeError("Selection is empty after exclusion. Check --exclude_idx_txt and --num.")

    # 선택 리스트 저장(재현성)
    if args.out_list:
        os.makedirs(os.path.dirname(args.out_list) or ".", exist_ok=True)
        with open(args.out_list, "w") as f:
            for p in sel:
                f.write(p + "\n")
        print("[OK] wrote out_list:", args.out_list, "n=", len(sel))

    N = len(sel)
    H = W = args.size
    arr = open_memmap(args.out, mode="w+", dtype=np.float32, shape=(N, 3, H, W))
    out_meta = args.out_meta or (args.out + ".meta.jsonl")
    os.makedirs(os.path.dirname(out_meta) or ".", exist_ok=True)
    mf = open(out_meta, "w", encoding="utf-8")

    for i, p in enumerate(sel):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read: {p}")

        if args.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # pad_stats order를 현재 im order와 맞춤 (필요 시 채널 swap)
        rng = None
        pad_stats_use = None
        if args.pad_mode == "rand":
            rng = stable_rng_from(args.seed, p)  # per-image deterministic
            im_order = "RGB" if args.to_rgb else "BGR"
            if pad_stats is None:
                raise RuntimeError("pad_mode=rand requires --pad_stats_json")
            if pad_stats["order"] != im_order:
                pad_stats_use = {
                    **pad_stats,
                    "mean255": pad_stats["mean255"][::-1].copy(),
                    "std255": pad_stats["std255"][::-1].copy(),
                    "order": im_order,
                }
            else:
                pad_stats_use = pad_stats

        im, meta = letterbox(
            im, new_shape=(H, W),
            auto=False, scaleup=True,
            pad_mode=args.pad_mode,
            pad_const=args.pad_const,
            pad_stats=pad_stats_use,
            sigma_clip=args.sigma_clip,
            rng=rng,
        )

        im = im.astype(np.float32)
        if not args.no_norm:
            im *= (1.0 / 255.0)
        im = np.transpose(im, (2, 0, 1))  # CHW
        arr[i] = np.ascontiguousarray(im)

        meta["idx"] = i
        meta["src"] = p
        meta["to_rgb"] = bool(args.to_rgb)
        meta["no_norm"] = bool(args.no_norm)
        meta["dtype"] = "float32"
        meta["layout"] = "CHW"
        meta["value_range"] = "0_255" if args.no_norm else "0_1"
        mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

        if (i + 1) % 100 == 0:
            print(f"[write] {i+1}/{N}")

    mf.flush()
    mf.close()
    print("[OK] saved meta:", out_meta)
    arr.flush()
    print("[OK] saved:", args.out, arr.shape, arr.dtype)
    print("stats min/max/mean/std =", float(arr.min()), float(arr.max()),
          float(arr.mean()), float(arr.std()))


if __name__ == "__main__":
    main()