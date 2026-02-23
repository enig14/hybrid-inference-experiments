#!/usr/bin/env python3
"""
pc_kl_sweep_from_onnx_curves.py

Extension of pc_kl_sweep_from_onnx.py:
- Runs ONNXRuntime inference for specified output tensors over an input NCHW .npy
- Accumulates histogram per tensor (optionally |x|)
- Computes TensorRT-style entropy(KL) sweep: KL(P||Q) for thresholds across histogram bins
- Saves:
  (A) summary CSV (best_t / best_thr / best_kl / sat_rate / last_bin_frac)
  (B) per-tensor KL-curve CSV: (t, k, T, kl)
  (C) per-tensor plots: KL vs T (linear + log-y), plus optional k_view and zoom_best views

This aims to match the sweep definition used in plot_kl_curve_from_dump_patched_trtlike_v3.py.

Example:
python3 pc_kl_sweep_from_onnx_curves.py \
  --onnx model.onnx \
  --input_npy calib.npy \
  --outputs /model.2/...Conv_output_0 \
  --out_dir kl_curves_onnx \
  --out_csv_summary summary.csv \
  --hist_abs \
  --bins 2048 \
  --num_quant_bins 127 \
  --warmup 8 \
  --k_view 8:512 \
  --zoom_best 64
"""
import argparse, csv, math, os
import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def entropy_calib_kl_sweep(hist, num_quant_bins=127, start_bin=None, eps=1e-12, prefer_large_t_on_tie=True, tie_eps=0.0):
    """
    TensorRT-style entropy(KL) sweep:
      For each candidate t (inclusive last kept bin index),
        P = hist[:t+1] with tail merged into last bin,
        Q: quantize P into num_quant_bins bins with step=(t+1)/num_quant_bins,
           then dequantize uniformly within each quant span.
        KL(P||Q)

    Returns:
      best_t (int), best_kl (float), kls (np.ndarray shape (n,), indexed by t)
    """
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.size
    if start_bin is None:
        start_bin = max(num_quant_bins, 16)

    kls = np.full(n, np.nan, dtype=np.float64)

    for t in range(start_bin, n):
        P = hist[:t+1].copy()
        if t + 1 < n:
            P[-1] += hist[t+1:].sum()
        if P.sum() <= 0:
            continue

        step = (t + 1) / float(num_quant_bins)
        Qq = np.zeros(num_quant_bins, dtype=np.float64)
        for i in range(t + 1):
            qb = int(i / step)
            if qb >= num_quant_bins:
                qb = num_quant_bins - 1
            Qq[qb] += P[i]

        Q = np.zeros(t + 1, dtype=np.float64)
        for qb in range(num_quant_bins):
            lo = int(math.floor(qb * step))
            hi = int(math.floor((qb + 1) * step)) - 1
            if qb == num_quant_bins - 1:
                hi = t
            if hi < lo:
                continue
            mass = Qq[qb]
            width = hi - lo + 1
            if mass > 0:
                Q[lo:hi+1] = mass / float(width)

        kls[t] = kl_divergence(P, Q, eps=eps)

    finite = np.isfinite(kls)
    if not finite.any():
        return None, None, kls

    min_kl = float(np.nanmin(kls))
    if tie_eps and tie_eps > 0:
        cand = np.where(finite & (np.abs(kls - min_kl) <= tie_eps))[0]
        best_t = int(cand.max() if prefer_large_t_on_tie else cand.min())
    else:
        if prefer_large_t_on_tie:
            cand = np.where(finite & (kls == min_kl))[0]
            best_t = int(cand.max())
        else:
            best_t = int(np.nanargmin(kls))

    return best_t, float(kls[best_t]), kls


def parse_k_view(s: str):
    if not s:
        return None
    if ":" not in s:
        raise ValueError("--k_view must be like 'a:b'")
    a, b = s.split(":")
    a = int(a); b = int(b)
    if b < a:
        a, b = b, a
    return a, b


def save_curve_plot(df, best_k, out_path, title, k_lo=None, k_hi=None, logy=False):
    d = df.copy()
    if k_lo is not None and k_hi is not None:
        d = d[(d["k"] >= k_lo) & (d["k"] <= k_hi)].copy()

    plt.figure()
    plt.plot(d["T"], d["kl"], label="KL")
    r = d[d["k"] == int(best_k)]
    if len(r):
        y = float(r.iloc[0]["kl"])
        if logy:
            y = max(y, 1e-30)
        plt.scatter([float(r.iloc[0]["T"])], [y], s=25)

    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("T")
    plt.ylabel("KL(p||q)" + (" [log]" if logy else ""))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input_npy", required=True)          # NCHW float32
    ap.add_argument("--input_name", default="images")
    ap.add_argument("--outputs", required=True)            # comma-separated output names
    ap.add_argument("--providers", default="auto", choices=["auto","cpu","cuda"])

    ap.add_argument("--bins", type=int, default=2048)
    ap.add_argument("--hist_abs", action="store_true", help="use |x| histogram")
    ap.add_argument("--hist_min", type=float, default=None)
    ap.add_argument("--hist_max", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--sample_per_image", type=int, default=0,
                    help="0=full (PC ok). If >0, random sample per image.")
    ap.add_argument("--num_quant_bins", type=int, default=127)
    ap.add_argument("--seed", type=int, default=123)

    # outputs
    ap.add_argument("--out_dir", required=True, help="directory for per-tensor curve CSVs and plots")
    ap.add_argument("--out_csv_summary", default="", help="optional summary CSV path (inside out_dir if relative)")
    ap.add_argument("--tie_eps", type=float, default=1e-12)
    ap.add_argument("--prefer_large_t_on_tie", action="store_true", default=False)

    # plot views (same semantics as dump plotter)
    ap.add_argument("--k_view", default="", help="k range view like '8:512' (inclusive)")
    ap.add_argument("--zoom_best", type=int, default=0, help="plot around best_k +/- N (0 disables)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # providers
    if args.providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.providers == "cuda":
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    else:
        providers = ort.get_available_providers()

    sess = ort.InferenceSession(args.onnx, providers=providers)

    outs = [s.strip() for s in args.outputs.split(",") if s.strip()]
    x = np.load(args.input_npy).astype(np.float32, copy=False)
    N = x.shape[0]

    rng = np.random.default_rng(args.seed)

    # auto range by warmup
    hist_min = args.hist_min
    hist_max = args.hist_max
    if hist_min is None or hist_max is None:
        mx = 0.0
        for i in range(min(args.warmup, N)):
            y = sess.run(outs, {args.input_name: x[i:i+1]})
            for arr in y:
                v = arr.ravel()
                if args.hist_abs:
                    v = np.abs(v)
                if args.sample_per_image and v.size > args.sample_per_image:
                    idx = rng.integers(0, v.size, size=args.sample_per_image, dtype=np.int64)
                    v = v[idx]
                mx = max(mx, float(np.max(v)))
        if hist_min is None:
            hist_min = 0.0 if args.hist_abs else -mx
        if hist_max is None:
            hist_max = mx

    edges = np.linspace(hist_min, hist_max, args.bins+1, dtype=np.float32)
    acc = {t: np.zeros(args.bins, dtype=np.int64) for t in outs}
    absmax = {t: 0.0 for t in outs}
    sat_cnt = {t: 0 for t in outs}
    tot_cnt = {t: 0 for t in outs}

    for i in range(N):
        y = sess.run(outs, {args.input_name: x[i:i+1]})
        for name, arr in zip(outs, y):
            v = arr.ravel()
            if args.hist_abs:
                v = np.abs(v)
            absmax[name] = max(absmax[name], float(np.max(v)))

            if args.sample_per_image and v.size > args.sample_per_image:
                idx = rng.integers(0, v.size, size=args.sample_per_image, dtype=np.int64)
                vs = v[idx]
            else:
                vs = v

            tot_cnt[name] += vs.size
            # saturation defined as values that would fall outside histogram max
            sat_cnt[name] += int((vs >= hist_max).sum()) if args.hist_abs and hist_min == 0.0 else int(((vs < hist_min) | (vs >= hist_max)).sum())

            vs = np.clip(vs, hist_min, np.nextafter(hist_max, -np.inf))
            h, _ = np.histogram(vs, bins=edges)
            acc[name] += h.astype(np.int64)

        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"infer+hist: {i+1}/{N}")

    # summary CSV path
    out_csv_summary = args.out_csv_summary.strip()
    if out_csv_summary:
        if not os.path.isabs(out_csv_summary):
            out_csv_summary = os.path.join(args.out_dir, out_csv_summary)
    else:
        out_csv_summary = os.path.join(args.out_dir, "summary.csv")

    k_view = parse_k_view(args.k_view)

    # write summary + per-tensor curves/plots
    with open(out_csv_summary, "w", newline="") as f:
        cols = ["tensor","bins","hist_min","hist_max","hist_abs","absmax_fp32",
                "best_t","best_thr_value","best_kl","sat_rate","last_bin_frac"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for tname in outs:
            best_t, best_kl, kls = entropy_calib_kl_sweep(
                acc[tname],
                num_quant_bins=args.num_quant_bins,
                start_bin=None,
                eps=1e-12,
                prefer_large_t_on_tie=args.prefer_large_t_on_tie,
                tie_eps=args.tie_eps
            )
            thr = "" if best_t is None else float(edges[best_t+1])

            sat_rate = sat_cnt[tname] / tot_cnt[tname] if tot_cnt[tname] else ""
            last_bin_frac = acc[tname][-1] / acc[tname].sum() if acc[tname].sum() else ""

            w.writerow({
                "tensor": tname, "bins": args.bins,
                "hist_min": hist_min, "hist_max": hist_max, "hist_abs": int(args.hist_abs),
                "absmax_fp32": absmax[tname],
                "best_t": "" if best_t is None else int(best_t),
                "best_thr_value": thr,
                "best_kl": "" if best_kl is None else float(best_kl),
                "sat_rate": sat_rate,
                "last_bin_frac": last_bin_frac,
            })

            # curve CSV: t, k, T, kl (T uses upper edge edges[t+1])
            t = np.arange(args.bins, dtype=np.int32)
            k = t + 1
            T = edges[t+1].astype(np.float64)  # upper-edge threshold for each t
            df = pd.DataFrame({"t": t, "k": k, "T": T, "kl": kls})
            safe_name = tname.replace("/", "_").replace(":", "_")
            base = f"{safe_name}.b{args.bins}.nq{args.num_quant_bins}"
            curve_csv = os.path.join(args.out_dir, f"curve.{base}.csv")
            df.to_csv(curve_csv, index=False)

            # plots
            if best_t is None:
                print(f"[{tname}] best_t=None (no finite KL). saved curve: {curve_csv}")
                continue
            best_k = int(best_t + 1)

            # full plots
            save_curve_plot(
                df, best_k,
                out_path=os.path.join(args.out_dir, f"plot.{base}.png"),
                title=f"KL-curve | {tname} | best_k={best_k} (T={thr:.6g})",
                logy=False
            )
            save_curve_plot(
                df, best_k,
                out_path=os.path.join(args.out_dir, f"plot.{base}.log.png"),
                title=f"KL-curve (log-y) | {tname} | best_k={best_k} (T={thr:.6g})",
                logy=True
            )

            # k_view
            if k_view is not None:
                a, b = k_view
                save_curve_plot(
                    df, best_k,
                    out_path=os.path.join(args.out_dir, f"plot.{base}.kview_{a}_{b}.png"),
                    title=f"KL-curve | {tname} | k_view={a}:{b} | best_k={best_k} (T={thr:.6g})",
                    k_lo=a, k_hi=b, logy=False
                )
                save_curve_plot(
                    df, best_k,
                    out_path=os.path.join(args.out_dir, f"plot.{base}.kview_{a}_{b}.log.png"),
                    title=f"KL-curve (log-y) | {tname} | k_view={a}:{b} | best_k={best_k} (T={thr:.6g})",
                    k_lo=a, k_hi=b, logy=True
                )

            # zoom_best
            if args.zoom_best and args.zoom_best > 0:
                z = int(args.zoom_best)
                lo = max(1, best_k - z)
                hi = min(args.bins, best_k + z)
                save_curve_plot(
                    df, best_k,
                    out_path=os.path.join(args.out_dir, f"plot.{base}.zoom_best_{z}.png"),
                    title=f"KL-curve | {tname} | zoom_best=±{z} (k={lo}..{hi}) | best_k={best_k} (T={thr:.6g})",
                    k_lo=lo, k_hi=hi, logy=False
                )
                save_curve_plot(
                    df, best_k,
                    out_path=os.path.join(args.out_dir, f"plot.{base}.zoom_best_{z}.log.png"),
                    title=f"KL-curve (log-y) | {tname} | zoom_best=±{z} (k={lo}..{hi}) | best_k={best_k} (T={thr:.6g})",
                    k_lo=lo, k_hi=hi, logy=True
                )

            print(f"[{tname}] best_t={best_t} best_k={best_k} T={thr:.6g} KL={best_kl:.3e}")
            print(f"  curve_csv: {curve_csv}")

    print(f"Saved summary: {out_csv_summary}")
    print(f"providers: {sess.get_providers()}")
    print(f"[range] hist_min={hist_min} hist_max={hist_max} hist_abs={args.hist_abs}")


if __name__ == "__main__":
    main()
