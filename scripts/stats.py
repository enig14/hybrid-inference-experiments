#!/usr/bin/env python3
"""stats_v4.py

Extends stats_v3.py with jtop util logging support.

Inputs (4-run C-2):
  - raw power csvs (t_ns, power_key)
  - power_frames csvs (frame,t0_ns,t1_ns,dt_s,p_mean_w,...) for LOAD runs
  - optional timing csvs
  - optional jtop util csvs for LOAD runs (ts_ns, gpu_util_pct, ..., dla0_util_pct, dla1_util_pct)

Outputs:
  - prints a single summary table including:
      raw_power.*, raw_power.dpdt, frame.power_frames metrics,
      excess_energy (idle-subtracted),
      raw_util.* and frame.util_* (if util files provided)

Design:
  - frame.* metrics are forced to exactly frame_count (default 4982) post-skip.
  - util is joined to frames by timestamp window [t0_ns,t1_ns] from power_frames.
  - missing-frame util values are filled using the same 3-pt strategy.
"""

import argparse
import os
import numpy as np
import pandas as pd


# --------------------------
# IO / cleaning
# --------------------------
def try_read_csv(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def must_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def clean_series(s: pd.Series, *, drop_zero: bool = False) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce")
    if drop_zero:
        v = v.mask(v == 0)
    return v.dropna().to_numpy(dtype=float)


def robust_stats_arr(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"n": 0}
    vs = np.sort(v)
    med = float(np.median(vs))
    mad = float(np.median(np.abs(vs - med)))
    mean = float(np.mean(vs))
    p99 = float(np.percentile(vs, 99))
    return {
        "n": int(vs.size),
        "mean": mean,
        "std": float(np.std(vs, ddof=1)) if vs.size > 1 else 0.0,
        "p50": float(np.percentile(vs, 50)),
        "p95": float(np.percentile(vs, 95)),
        "p99": p99,
        "max": float(vs[-1]),
        "mad": mad,
        "p99_over_mean": (p99 / mean) if mean > 0 else float("nan"),
    }


def row_stats(group: str, metric: str, v: np.ndarray, *, n_override: int | None = None) -> dict:
    d = robust_stats_arr(v)
    if n_override is not None:
        d["n"] = int(n_override)
    d.update({"group": group, "metric": metric})
    return d


def filter_valid_power_frames(df: pd.DataFrame) -> pd.DataFrame:
    need = ["frame", "t0_ns", "t1_ns", "dt_s", "n", "p_mean_w"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"power_frames missing '{c}'. columns={list(df.columns)}")

    out = df.copy()
    for c in ["frame", "t0_ns", "t1_ns", "dt_s", "n", "p_mean_w", "p_max_w", "e_j"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[
        (out["t0_ns"] > 0)
        & (out["t1_ns"] > 0)
        & (out["dt_s"] > 0)
        & (out["n"] > 0)
        & (out["p_mean_w"].notna())
    ].copy()

    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out = out[out["frame"].notna()].copy()
    out["frame"] = out["frame"].astype(int)

    out["tmid_ns"] = (out["t0_ns"] + out["t1_ns"]) / 2.0
    return out


def filter_timing(df: pd.DataFrame | None, skip_frames: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "frame" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out = out[out["frame"].notna()].copy()
    out["frame"] = out["frame"].astype(int)
    if skip_frames > 0:
        out = out[out["frame"] >= skip_frames].copy()
    return out


def cut_by_seconds(df: pd.DataFrame, seconds: float, *, tcol: str = "t_ns") -> pd.DataFrame:
    if df is None or len(df) == 0 or not seconds or seconds <= 0:
        return df
    out = df.copy()
    out[tcol] = pd.to_numeric(out[tcol], errors="coerce")
    out = out[out[tcol].notna()].sort_values(tcol)
    if len(out) == 0:
        return out
    t0 = float(out[tcol].iloc[0])
    cut = t0 + float(seconds) * 1e9
    return out[out[tcol] >= cut].copy()


def dpdt_series(power_df: pd.DataFrame, pcol: str = "p_w") -> np.ndarray:
    t = pd.to_numeric(power_df["t_ns"], errors="coerce")
    p = pd.to_numeric(power_df[pcol], errors="coerce")
    m = t.notna() & p.notna()
    t = t[m].to_numpy(dtype=float)
    p = p[m].to_numpy(dtype=float)
    if t.size < 2:
        return np.array([], dtype=float)
    dt = np.abs(t[1:] - t[:-1]) * 1e-9
    dp = np.abs(p[1:] - p[:-1])
    m2 = dt > 0
    return (dp[m2] / dt[m2]).astype(float)


# --------------------------
# Missing-frame handling
# --------------------------
def fill_missing_series_3pt(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x.interpolate(method="linear", limit_direction="both")
    x = x.ffill().bfill()

    if x.isna().any():
        arr = x.to_numpy(dtype=float)
        idx = np.arange(arr.size)
        for i in idx[np.isnan(arr)]:
            vals = []
            for j in (i - 1, i + 1, i - 2, i + 2):
                if 0 <= j < arr.size and np.isfinite(arr[j]):
                    vals.append(arr[j])
            arr[i] = float(np.mean(vals)) if vals else float("nan")
        x = pd.Series(arr, index=x.index).ffill().bfill()
    return x


def reindex_frames_full(df: pd.DataFrame, *, skip_frames: int, frame_count: int, cols_to_fill: list[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    start = int(skip_frames)
    frames = pd.Index(range(start, start + int(frame_count)), name="frame")
    d = df.copy()
    d["frame"] = pd.to_numeric(d["frame"], errors="coerce")
    d = d[d["frame"].notna()].copy()
    d["frame"] = d["frame"].astype(int)
    d = d.sort_values("frame").drop_duplicates(subset=["frame"], keep="last")
    d = d.set_index("frame").reindex(frames)
    for col in cols_to_fill:
        if col in d.columns:
            d[col] = fill_missing_series_3pt(d[col])
    return d.reset_index()


def reindex_power_frames_full(pf: pd.DataFrame, *, skip_frames: int, frame_count: int) -> pd.DataFrame:
    cols = ["p_mean_w", "p_max_w", "e_j", "dt_s", "n", "t0_ns", "t1_ns", "tmid_ns"]
    return reindex_frames_full(pf, skip_frames=skip_frames, frame_count=frame_count, cols_to_fill=cols)


# --------------------------
# Util join (jtop)
# --------------------------
DEFAULT_UTIL_COLS = [
    "gpu_util_pct",
    "emc_util_pct",
    "cpu_avg_pct",
    "cpu_max_pct",
]


def filter_util(df: pd.DataFrame | None, *, tcol: str = "ts_ns") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if tcol not in df.columns:
        raise KeyError(f"util csv missing '{tcol}'. columns={list(df.columns)}")
    out = df.copy()
    out[tcol] = pd.to_numeric(out[tcol], errors="coerce")
    out = out[out[tcol].notna()].sort_values(tcol)
    return out


def join_timeseries_to_frames(
    ts: pd.DataFrame,
    pf_full: pd.DataFrame,
    *,
    tcol: str,
    cols: list[str],
    how: str = "mean",
) -> pd.DataFrame:
    """Join a sampled timeseries to per-frame windows [t0_ns,t1_ns].

    Returns df with columns: frame + joined cols.
    Missing windows get NaN (filled later by reindex_frames_full).
    """
    if ts is None or len(ts) == 0:
        return pd.DataFrame({"frame": pf_full["frame"].astype(int)})

    t = ts.copy()
    t[tcol] = pd.to_numeric(t[tcol], errors="coerce")
    t = t[t[tcol].notna()].sort_values(tcol)

    # Ensure util cols numeric
    cols2 = [c for c in cols if c in t.columns]
    for c in cols2:
        t[c] = pd.to_numeric(t[c], errors="coerce")

    # Vectorized-ish join using searchsorted per frame (fast enough for ~5k frames)
    tt = t[tcol].to_numpy(dtype=np.int64)
    out_rows = []
    for _, r in pf_full.iterrows():
        f = int(r["frame"])
        t0 = int(r["t0_ns"])
        t1 = int(r["t1_ns"])
        i0 = int(np.searchsorted(tt, t0, side="left"))
        i1 = int(np.searchsorted(tt, t1, side="right"))
        row = {"frame": f}
        if i1 <= i0:
            for c in cols2:
                row[c] = np.nan
        else:
            seg = t.iloc[i0:i1]
            for c in cols2:
                vv = seg[c].to_numpy(dtype=float)
                vv = vv[np.isfinite(vv)]
                if vv.size == 0:
                    row[c] = np.nan
                else:
                    if how == "max":
                        row[c] = float(np.max(vv))
                    else:
                        row[c] = float(np.mean(vv))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# --------------------------
# Energy (idle-subtracted)
# --------------------------
def idle_baseline_w(power_idle: pd.DataFrame, pcol: str, skip_seconds: float) -> float:
    d = power_idle.copy()
    d["t_ns"] = pd.to_numeric(d["t_ns"], errors="coerce")
    d[pcol] = pd.to_numeric(d[pcol], errors="coerce")
    d = d[d["t_ns"].notna() & d[pcol].notna()].sort_values("t_ns")
    if len(d) == 0:
        return float("nan")
    if skip_seconds > 0:
        t0 = float(d["t_ns"].iloc[0])
        cut = t0 + skip_seconds * 1e9
        d = d[d["t_ns"] >= cut]
        if len(d) == 0:
            return float("nan")
    return float(np.median(d[pcol].to_numpy(dtype=float)))


def infer_n_frames(pf: pd.DataFrame, timing: pd.DataFrame, skip_frames: int, *, frame_count: int | None = None) -> int:
    if frame_count is not None and frame_count > 0:
        return int(frame_count)
    if timing is not None and len(timing) and "frame" in timing.columns:
        fr = pd.to_numeric(timing["frame"], errors="coerce").dropna().astype(int)
        fr = fr[fr >= skip_frames]
        if len(fr):
            return int(fr.nunique())
    if pf is not None and len(pf) and "frame" in pf.columns:
        fr = pd.to_numeric(pf["frame"], errors="coerce").dropna().astype(int)
        fr = fr[fr >= skip_frames]
        if len(fr):
            return int(fr.max() - fr.min() + 1)
    return 0


def energy_trapz_excess(power_load: pd.DataFrame, t0_ns: int, t1_ns: int, p_idle_w: float, pcol: str) -> float:
    d = power_load.copy()
    d["t_ns"] = pd.to_numeric(d["t_ns"], errors="coerce")
    d[pcol] = pd.to_numeric(d[pcol], errors="coerce")
    d = d[d["t_ns"].notna() & d[pcol].notna()].sort_values("t_ns")
    d = d[(d["t_ns"] >= t0_ns) & (d["t_ns"] <= t1_ns)]
    if len(d) < 2 or not np.isfinite(p_idle_w):
        return float("nan")
    t = d["t_ns"].to_numpy(dtype=np.float64) * 1e-9
    p = d[pcol].to_numpy(dtype=np.float64) - float(p_idle_w)
    return float(np.trapz(p, t))


def excess_energy_per_frame(power_df, pf, p_idle_w, *, skip_frames=10, load_skip_seconds=0.0, pcol="p_w"):
    if pf is None or len(pf) == 0:
        return None
    pf2 = pf.copy()
    if load_skip_seconds and load_skip_seconds > 0:
        t_start_ns = float(pd.to_numeric(pf2["t0_ns"], errors="coerce").min())
        cut_ns = t_start_ns + float(load_skip_seconds) * 1e9
        if "tmid_ns" in pf2.columns:
            pf2 = pf2[pf2["tmid_ns"] >= cut_ns].copy()
        else:
            tmid = (pd.to_numeric(pf2["t0_ns"], errors="coerce") + pd.to_numeric(pf2["t1_ns"], errors="coerce")) / 2.0
            pf2 = pf2[tmid >= cut_ns].copy()
        if len(pf2) == 0:
            return None
    t0_ns = int(pd.to_numeric(pf2["t0_ns"], errors="coerce").min())
    t1_ns = int(pd.to_numeric(pf2["t1_ns"], errors="coerce").max())
    dt_s = (t1_ns - t0_ns) * 1e-9
    if not (dt_s > 0):
        return None
    n_frames = infer_n_frames(pf2, pd.DataFrame(), skip_frames, frame_count=None)
    if n_frames <= 0:
        return None
    E_excess = energy_trapz_excess(power_df, t0_ns, t1_ns, p_idle_w, pcol=pcol)
    if not np.isfinite(E_excess):
        return None
    Epf = E_excess / n_frames
    fps = n_frames / dt_s
    Pavg_excess = E_excess / dt_s
    coverage = (len(pf2) / n_frames) if n_frames > 0 else np.nan
    return {
        "n_frames": n_frames,
        "dt_s": dt_s,
        "fps": fps,
        "P_idle_W": float(p_idle_w),
        "Pavg_W_excess": float(Pavg_excess),
        "E_total_J": float(E_excess),
        "E_per_frame_J": float(Epf),
        "coverage_pf_rows_over_frames": float(coverage),
    }


# --------------------------
# Reporting
# --------------------------
def print_table(title: str, rows: list[dict]):
    print("\n" + title)
    if not rows:
        print("(no rows)")
        return
    df = pd.DataFrame(rows)
    preferred = [
        "group",
        "metric",
        "n",
        "mean",
        "std",
        "p50",
        "p95",
        "p99",
        "max",
        "mad",
        "p99_over_mean",
        "fps",
        "P_idle_W",
        "Pavg_W_excess",
        "E_per_frame_J",
        "E_total_J",
        "dt_s",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    with pd.option_context("display.max_rows", 400, "display.max_columns", 200, "display.width", 240):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()

    # --- 4 runs: gpu_idle/gpu_load/hybrid_idle/hybrid_load ---
    ap.add_argument("--gpu_idle_power", required=True)
    ap.add_argument("--gpu_load_power", required=True)
    ap.add_argument("--gpu_load_pframes", required=True)
    ap.add_argument("--gpu_load_timing", default="", help="optional")
    ap.add_argument("--gpu_load_util", default="", help="optional jtop util csv (LOAD)")

    ap.add_argument("--hybrid_idle_power", required=True)
    ap.add_argument("--hybrid_load_power", required=True)
    ap.add_argument("--hybrid_load_pframes", required=True)
    ap.add_argument("--hybrid_load_timing", default="", help="optional")
    ap.add_argument("--hybrid_load_util", default="", help="optional jtop util csv (LOAD)")

    # common options
    ap.add_argument("--power_key", default="p_w", help="column in raw power csv: p_w or p_avg_w")
    ap.add_argument("--skip_frames", type=int, default=10)
    ap.add_argument("--frame_count", type=int, default=4982, help="post-skip frame count policy for frame.* rows")
    ap.add_argument("--idle_skip_seconds", type=float, default=1.0)
    ap.add_argument("--load_skip_seconds", type=float, default=0.0)
    ap.add_argument("--timing_cols", default="", help="comma-separated timing cols")
    ap.add_argument("--timing_drop_zero", action="store_true")
    ap.add_argument("--util_cols", default=",".join(DEFAULT_UTIL_COLS), help="comma-separated util columns to report")
    ap.add_argument("--util_join", default="mean", choices=["mean", "max"], help="util aggregation within frame window")

    # Add output file argument
    ap.add_argument("--output_csv", required=True, help="Path to save the output CSV file.")
    args = ap.parse_args()

    pcol = args.power_key
    util_cols = [c.strip() for c in args.util_cols.split(",") if c.strip()]

    # Load CSVs
    gpu_idle_pwr = must_read_csv(args.gpu_idle_power)
    gpu_load_pwr = must_read_csv(args.gpu_load_power)
    gpu_load_pf = must_read_csv(args.gpu_load_pframes)
    gpu_load_t = try_read_csv(args.gpu_load_timing)
    gpu_load_u = try_read_csv(args.gpu_load_util)

    hyb_idle_pwr = must_read_csv(args.hybrid_idle_power)
    hyb_load_pwr = must_read_csv(args.hybrid_load_power)
    hyb_load_pf = must_read_csv(args.hybrid_load_pframes)
    hyb_load_t = try_read_csv(args.hybrid_load_timing)
    hyb_load_u = try_read_csv(args.hybrid_load_util)

    # Validate power col
    for name, df in [
        ("gpu_idle_power", gpu_idle_pwr),
        ("gpu_load_power", gpu_load_pwr),
        ("hybrid_idle_power", hyb_idle_pwr),
        ("hybrid_load_power", hyb_load_pwr),
    ]:
        if pcol not in df.columns:
            raise KeyError(f"{name}: power_key '{pcol}' not found. columns={list(df.columns)}")

    # Apply LOAD time-cut for raw_power.* stats and dpdt
    gpu_load_pwr_cut = cut_by_seconds(gpu_load_pwr, args.load_skip_seconds, tcol="t_ns")
    hyb_load_pwr_cut = cut_by_seconds(hyb_load_pwr, args.load_skip_seconds, tcol="t_ns")

    # Power frames filtering + skip_frames
    gpu_load_pf = filter_valid_power_frames(gpu_load_pf)
    hyb_load_pf = filter_valid_power_frames(hyb_load_pf)
    if args.skip_frames > 0:
        gpu_load_pf = gpu_load_pf[gpu_load_pf["frame"] >= args.skip_frames].copy()
        hyb_load_pf = hyb_load_pf[hyb_load_pf["frame"] >= args.skip_frames].copy()

    # Force frame policy: exactly N frames for frame.* stats, fill gaps
    gpu_pf_full = reindex_power_frames_full(gpu_load_pf, skip_frames=args.skip_frames, frame_count=args.frame_count)
    hyb_pf_full = reindex_power_frames_full(hyb_load_pf, skip_frames=args.skip_frames, frame_count=args.frame_count)

    # Timing (optional)
    gpu_load_t = filter_timing(gpu_load_t, args.skip_frames)
    hyb_load_t = filter_timing(hyb_load_t, args.skip_frames)
    timing_cols = [c.strip() for c in args.timing_cols.split(",") if c.strip()]

    rows: list[dict] = []

    # (1) Idle baselines (median)
    gpu_idle_w = idle_baseline_w(gpu_idle_pwr, pcol=pcol, skip_seconds=args.idle_skip_seconds)
    hyb_idle_w = idle_baseline_w(hyb_idle_pwr, pcol=pcol, skip_seconds=args.idle_skip_seconds)

    rows.append(row_stats("gpu_only.idle", f"raw_power.{pcol}", clean_series(gpu_idle_pwr[pcol])))
    rows.append(row_stats("hybrid.idle", f"raw_power.{pcol}", clean_series(hyb_idle_pwr[pcol])))

    # (2) Load raw power stats (after load_skip_seconds cut)
    rows.append(row_stats("gpu_only.load", f"raw_power.{pcol}", clean_series(gpu_load_pwr_cut[pcol])))
    rows.append(row_stats("hybrid.load", f"raw_power.{pcol}", clean_series(hyb_load_pwr_cut[pcol])))
    rows.append(row_stats("gpu_only.load", "raw_power.dpdt(W/s)", dpdt_series(gpu_load_pwr_cut, pcol=pcol)))
    rows.append(row_stats("hybrid.load", "raw_power.dpdt(W/s)", dpdt_series(hyb_load_pwr_cut, pcol=pcol)))

    # (3) Load per-frame stats (force n=frame_count, fill missing frames)
    for col in ["p_mean_w", "p_max_w", "e_j", "dt_s"]:
        if col in gpu_pf_full.columns:
            rows.append(row_stats("gpu_only.load", f"frame.{col}", clean_series(gpu_pf_full[col]), n_override=args.frame_count))
        if col in hyb_pf_full.columns:
            rows.append(row_stats("hybrid.load", f"frame.{col}", clean_series(hyb_pf_full[col]), n_override=args.frame_count))

    # (4) Excess energy (load - idle) using raw power integration
    gpu_ex = excess_energy_per_frame(gpu_load_pwr, gpu_load_pf, gpu_idle_w, skip_frames=args.skip_frames, load_skip_seconds=args.load_skip_seconds, pcol=pcol)
    hyb_ex = excess_energy_per_frame(hyb_load_pwr, hyb_load_pf, hyb_idle_w, skip_frames=args.skip_frames, load_skip_seconds=args.load_skip_seconds, pcol=pcol)

    if gpu_ex is not None:
        rows.append({
            "group": "gpu_only.delta",
            "metric": "excess_energy.raw_int",
            "n": gpu_ex["n_frames"],
            "mean": gpu_ex["E_per_frame_J"],
            "fps": gpu_ex["fps"],
            "P_idle_W": gpu_ex["P_idle_W"],
            "Pavg_W_excess": gpu_ex["Pavg_W_excess"],
            "E_per_frame_J": gpu_ex["E_per_frame_J"],
            "E_total_J": gpu_ex["E_total_J"],
            "dt_s": gpu_ex["dt_s"],
            "coverage_pf_rows_over_frames": gpu_ex["coverage_pf_rows_over_frames"],
        })
    else:
        rows.append({"group": "gpu_only.delta", "metric": "excess_energy.raw_int", "n": 0})

    if hyb_ex is not None:
        rows.append({
            "group": "hybrid.delta",
            "metric": "excess_energy.raw_int",
            "n": hyb_ex["n_frames"],
            "mean": hyb_ex["E_per_frame_J"],
            "fps": hyb_ex["fps"],
            "P_idle_W": hyb_ex["P_idle_W"],
            "Pavg_W_excess": hyb_ex["Pavg_W_excess"],
            "E_per_frame_J": hyb_ex["E_per_frame_J"],
            "E_total_J": hyb_ex["E_total_J"],
            "dt_s": hyb_ex["dt_s"],
            "coverage_pf_rows_over_frames": hyb_ex["coverage_pf_rows_over_frames"],
        })
    else:
        rows.append({"group": "hybrid.delta", "metric": "excess_energy.raw_int", "n": 0})

    # (5) Optional util stats (raw + per-frame) for LOAD runs
    if args.gpu_load_util:
        gpu_u = filter_util(gpu_load_u, tcol="ts_ns")
        gpu_u_cut = cut_by_seconds(gpu_u, args.load_skip_seconds, tcol="ts_ns")
        for c in util_cols:
            if c in gpu_u_cut.columns:
                rows.append(row_stats("gpu_only.load", f"raw_util.{c}", clean_series(gpu_u_cut[c])))
        # join to frames
        gpu_uf = join_timeseries_to_frames(gpu_u, gpu_pf_full, tcol="ts_ns", cols=util_cols, how=args.util_join)
        gpu_uf_full = reindex_frames_full(gpu_uf, skip_frames=args.skip_frames, frame_count=args.frame_count, cols_to_fill=[c for c in util_cols if c in gpu_uf.columns])
        for c in util_cols:
            if c in gpu_uf_full.columns:
                rows.append(row_stats("gpu_only.load", f"frame.util_{args.util_join}.{c}", clean_series(gpu_uf_full[c]), n_override=args.frame_count))

    if args.hybrid_load_util:
        hyb_u = filter_util(hyb_load_u, tcol="ts_ns")
        hyb_u_cut = cut_by_seconds(hyb_u, args.load_skip_seconds, tcol="ts_ns")
        for c in util_cols:
            if c in hyb_u_cut.columns:
                rows.append(row_stats("hybrid.load", f"raw_util.{c}", clean_series(hyb_u_cut[c])))
        hyb_uf = join_timeseries_to_frames(hyb_u, hyb_pf_full, tcol="ts_ns", cols=util_cols, how=args.util_join)
        hyb_uf_full = reindex_frames_full(hyb_uf, skip_frames=args.skip_frames, frame_count=args.frame_count, cols_to_fill=[c for c in util_cols if c in hyb_uf.columns])
        for c in util_cols:
            if c in hyb_uf_full.columns:
                rows.append(row_stats("hybrid.load", f"frame.util_{args.util_join}.{c}", clean_series(hyb_uf_full[c]), n_override=args.frame_count))

    # (6) Optional timing stats
    if timing_cols:
        for col in timing_cols:
            if col in gpu_load_t.columns:
                rows.append(row_stats("gpu_only.load", f"timing.{col}(ms)", clean_series(gpu_load_t[col], drop_zero=args.timing_drop_zero)))
            else:
                rows.append({"group": "gpu_only.load", "metric": f"timing.{col}(ms)", "n": 0})
            if col in hyb_load_t.columns:
                rows.append(row_stats("hybrid.load", f"timing.{col}(ms)", clean_series(hyb_load_t[col], drop_zero=args.timing_drop_zero)))
            else:
                rows.append({"group": "hybrid.load", "metric": f"timing.{col}(ms)", "n": 0})
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    print_table(
        f"Summary (skip_frames={args.skip_frames}, frame_count={args.frame_count}, idle_skip_seconds={args.idle_skip_seconds}, load_skip_seconds={args.load_skip_seconds}, power_key={pcol}, util_join={args.util_join})",
        rows,
    )


if __name__ == "__main__":
    main()
