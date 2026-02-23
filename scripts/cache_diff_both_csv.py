#!/usr/bin/env python3
"""
cache_diff_both_v2.py

Compare two TensorRT INT8 calibration caches ("tensor_name: value") and report BOTH:
- precision (step size Δ / quant resolution)
- clipping tendency (dynamic range threshold α)

Adds directional sorting/filtering:
- --sort bits_desc : sort by signed bits (log2(stepA/stepB)) descending (A coarser first)
- --sort bits_asc  : sort by signed bits ascending (B coarser first)
- --keep bits_pos  : keep only rows where bits > 0  (stepA > stepB)
- --keep bits_neg  : keep only rows where bits < 0  (stepA < stepB)

NEW:
- --out_csv <path> : write ALL filtered rows to CSV (not just topk)
  Columns include: name, A_raw, B_raw, stepA, stepB, alphaA, alphaB, ratio, bits, ...

Interpretation (assume step, typical for your cache value magnitude):
  bits = log2(stepA/stepB)
    bits > 0  => stepA > stepB  => A is coarser (precision worse than B), alphaA > alphaB (less clipping than B)
    bits < 0  => stepA < stepB  => A is finer   (precision better than B), alphaA < alphaB (more clipping risk than B)

If you want "precision got better from A->B", that's bits > 0 (because step decreased).
If you want "precision got worse from A->B", that's bits < 0 (because step increased).
"""
import argparse
import csv
import math
import os
import re
import struct

HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

def hex_to_float(hexstr: str, endian: str) -> float:
    hs = hexstr.strip()
    if not HEX_RE.match(hs):
        raise ValueError("not hex")
    if len(hs) == 8:
        b = bytes.fromhex(hs)
        return struct.unpack((">f" if endian == "big" else "<f"), b)[0]
    if len(hs) == 16:
        b = bytes.fromhex(hs)
        return struct.unpack((">d" if endian == "big" else "<d"), b)[0]
    raise ValueError("hex length must be 8 or 16")

def parse_val(s: str, endian: str) -> float:
    s = s.strip()
    if s.lower().startswith("0x"):
        return hex_to_float(s[2:], endian)
    if len(s) in (8, 16) and HEX_RE.match(s):
        return hex_to_float(s, endian)
    return float(s)

def load_cache(path: str, endian: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "TRT-" in line:
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                x = parse_val(v, endian)
                if math.isfinite(x):
                    out[k] = x
            except Exception:
                continue
    return out

def plausibility_score(values: list[float]) -> float:
    if not values:
        return -1e9
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return -1e9
    n = len(values); nf = len(finite)
    pos = sum(1 for x in finite if x > 0.0)
    huge = sum(1 for x in finite if abs(x) > 1e10)
    tiny = sum(1 for x in finite if 0.0 < abs(x) < 1e-20)
    return (nf/n) + 0.2*(pos/nf) - 0.2*(huge/nf) - 0.1*(tiny/nf)

def load_cache_auto(path: str) -> tuple[str, dict[str, float]]:
    big = load_cache(path, "big")
    lit = load_cache(path, "little")
    return ("big", big) if plausibility_score(list(big.values())) >= plausibility_score(list(lit.values())) else ("little", lit)

def step_alpha_from_value(v: float, assume: str) -> tuple[float, float]:
    v = abs(v)
    if assume == "step":
        step = v
        alpha = 127.0 * step
        return step, alpha
    if assume == "inv":
        step = 1.0 / max(v, 1e-12)
        alpha = 127.0 * step
        return step, alpha
    raise ValueError("assume must be step or inv")

def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b if abs(b) > eps else (eps if b >= 0 else -eps))

def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        # still write header for consistency
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("name,A_raw,B_raw,stepA,stepB,alphaA,alphaB,step_ratio,ratio,bits,abs_bits,alpha_delta,abs_alpha_delta,step_delta,abs_step_delta\n")
        return
    # stable column order
    cols = [
        "name",
        "A_raw","B_raw",
        "stepA","stepB",
        "alphaA","alphaB",
        "step_ratio","ratio",
        "bits","abs_bits",
        "alpha_delta","abs_alpha_delta",
        "step_delta","abs_step_delta",
    ]
    # include any extra columns at end
    extra = [c for c in rows[0].keys() if c not in cols]
    cols = cols + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_a", required=True)
    ap.add_argument("--cache_b", required=True)
    ap.add_argument("--endian", choices=["big", "little", "auto"], default="big")
    ap.add_argument("--assume", choices=["step", "inv"], default="step",
                    help="step: cache==Δ ; inv: cache==1/Δ")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--min_ratio", type=float, default=1.0)
    ap.add_argument("--keep", choices=["all", "bits_pos", "bits_neg"], default="all",
                    help="Filter by signed bits: bits_pos keeps bits>0 (stepA>stepB), bits_neg keeps bits<0 (stepA<stepB)")
    ap.add_argument("--sort", choices=[
        "abs_bits",
        "bits_desc",
        "bits_asc",
        "abs_alpha_delta",
        "abs_step_delta",
        "ratio",
        "name",
    ], default="abs_bits")
    ap.add_argument("--out_csv", default="", help="If set, write ALL filtered rows to CSV.")
    args = ap.parse_args()

    if args.endian == "auto":
        ea, A = load_cache_auto(args.cache_a)
        eb, B = load_cache_auto(args.cache_b)
        print(f"[auto] cache_a endian={ea}, cache_b endian={eb}")
    else:
        A = load_cache(args.cache_a, args.endian)
        B = load_cache(args.cache_b, args.endian)
        print(f"[fixed] endian={args.endian}")

    keys = sorted(set(A.keys()) & set(B.keys()))
    rows = []
    for k in keys:
        va, vb = A[k], B[k]
        stepA, alphaA = step_alpha_from_value(va, args.assume)
        stepB, alphaB = step_alpha_from_value(vb, args.assume)

        step_ratio = safe_div(stepA, stepB)
        ratio = max(abs(step_ratio), 1.0 / max(abs(step_ratio), 1e-12))
        if ratio < args.min_ratio or not math.isfinite(ratio):
            continue

        bits = math.log2(max(abs(step_ratio), 1e-12))  # signed
        if args.keep == "bits_pos" and not (bits > 0):
            continue
        if args.keep == "bits_neg" and not (bits < 0):
            continue

        abs_bits = abs(bits)
        alpha_delta = alphaA - alphaB
        step_delta = stepA - stepB

        rows.append({
            "name": k,
            "A_raw": va, "B_raw": vb,
            "stepA": stepA, "stepB": stepB,
            "alphaA": alphaA, "alphaB": alphaB,
            "step_ratio": step_ratio,
            "ratio": ratio,
            "bits": bits,
            "abs_bits": abs_bits,
            "alpha_delta": alpha_delta,
            "abs_alpha_delta": abs(alpha_delta),
            "step_delta": step_delta,
            "abs_step_delta": abs(step_delta),
        })

    # sorting
    if args.sort == "name":
        rows.sort(key=lambda r: r["name"])
    elif args.sort == "bits_desc":
        rows.sort(key=lambda r: r["bits"], reverse=True)
    elif args.sort == "bits_asc":
        rows.sort(key=lambda r: r["bits"])
    else:
        rows.sort(key=lambda r: r[args.sort], reverse=True)

    print(f"common={len(keys)}  shown={min(args.topk, len(rows))} (filtered by min_ratio>={args.min_ratio}, keep={args.keep})")
    print("abs_bits  bits(log2 stepA/stepB)  ratio  name  A_raw  B_raw  stepA  stepB  alphaA  alphaB  alpha_delta")
    for r in rows[:args.topk]:
        print(
            f"{r['abs_bits']:7.3f}  {r['bits']:9.3f}  {r['ratio']:6.3f}  {r['name']}  "
            f"{r['A_raw']:.7g}  {r['B_raw']:.7g}  "
            f"{r['stepA']:.7g}  {r['stepB']:.7g}  "
            f"{r['alphaA']:.6g}  {r['alphaB']:.6g}  {r['alpha_delta']:.6g}"
        )

    if args.out_csv:
        write_csv(args.out_csv, rows)
        print("[OK] wrote CSV:", args.out_csv, f"(rows={len(rows)})")

if __name__ == "__main__":
    main()
