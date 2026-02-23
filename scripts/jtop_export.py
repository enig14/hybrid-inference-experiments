#!/usr/bin/env python3
import argparse, csv, time, signal, sys
import re
from typing import Any, Optional, Iterable, Tuple

def now_ns() -> int:
    # Use monotonic time so we can join with steady_clock / CLOCK_MONOTONIC timestamps.
    return time.monotonic_ns()

def to_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        # common decorations
        s = s.replace("%", "")
        s = s.replace("MHz", "").replace("mhz", "")
        s = s.replace("kHz", "").replace("khz", "")
        s = s.replace("@", " ")
        s = s.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None

def iter_kv(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield str(i), v

def walk(obj: Any, path: str = ""):
    # DFS yielding (path, key, value)
    stack = [(path, obj)]
    while stack:
        p, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                np = f"{p}.{k}" if p else str(k)
                yield np, k, v
                if isinstance(v, (dict, list, tuple)):
                    stack.append((np, v))
        elif isinstance(cur, (list, tuple)):
            for i, v in enumerate(cur):
                np = f"{p}[{i}]"
                yield np, str(i), v
                if isinstance(v, (dict, list, tuple)):
                    stack.append((np, v))

def find_first_number(stats: Any, key_substr: Iterable[str]) -> Optional[float]:
    subs = [s.lower() for s in key_substr]
    for p, k, v in walk(stats):
        kl = str(k).lower()
        if any(s in kl for s in subs):
            # value itself numeric?
            n = to_num(v)
            if n is not None:
                return n
            # common nested patterns: {"load":..}, {"util":..}, {"val":..}
            if isinstance(v, dict):
                for kk in ("load", "util", "val", "use", "percent", "perc"):
                    if kk in v:
                        n2 = to_num(v.get(kk))
                        if n2 is not None:
                            return n2
    return None

def extract_gpu_util(st: dict) -> Optional[float]:
    # Prefer explicit GPU containers if present
    for keys in (("GPU","load"), ("GPU","util"), ("GPU","val"), ("gpu","load"), ("gr3d","load")):
        cur = st
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            n = to_num(cur)
            if n is not None:
                return n
    # generic search
    return find_first_number(st, ["gr3d", "gpu", "gpu_util", "gpu load"])

def extract_emc_util(st: dict) -> Optional[float]:
    # generic search is usually enough
    return find_first_number(st, ["emc", "emc_freq", "memory controller", "mc"])

def extract_cpu_utils(st: dict) -> Tuple[Optional[float], Optional[float]]:
    # Many schemas: a list of per-core loads somewhere under CPU
    vals = []
    cpu = st.get("CPU")
    if isinstance(cpu, dict):
        # CPU may contain per-core entries or a list under "load"
        if isinstance(cpu.get("load"), (list, tuple)):
            for x in cpu["load"]:
                n = to_num(x)
                if n is not None:
                    vals.append(n)
        else:
            for _, v in cpu.items():
                if isinstance(v, dict):
                    for kk in ("load","util","val","use","percent"):
                        n = to_num(v.get(kk))
                        if n is not None:
                            vals.append(n)
                            break
                else:
                    n = to_num(v)
                    if n is not None:
                        vals.append(n)

    # Fallback: look for keys like cpu0, cpu1, etc with load/util
    if not vals:
        for p, k, v in walk(st):
            kl = str(k).lower()
            if re.fullmatch(r"cpu\d+", kl) or kl.startswith("cpu"):
                if isinstance(v, dict):
                    for kk in ("load","util","val","use","percent"):
                        n = to_num(v.get(kk))
                        if n is not None:
                            vals.append(n); break
                else:
                    n = to_num(v)
                    if n is not None:
                        vals.append(n)

    if not vals:
        return None, None
    avg = sum(vals) / len(vals)
    mx = max(vals)
    return avg, mx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval_ms", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug_schema", action="store_true",
                    help="print top-level keys once (stderr) to help schema debugging")
    args = ap.parse_args()

    try:
        from jtop import jtop
    except Exception as e:
        print(f"[jtop_export] Failed to import jtop: {e}", file=sys.stderr)
        sys.exit(2)

    running = True
    def _sig(_s, _f):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    interval_s = max(1, args.interval_ms) / 1000.0

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_ns","gpu_util_pct","emc_util_pct","cpu_avg_pct","cpu_max_pct"])
        f.flush()

        with jtop() as jetson:
            time.sleep(interval_s)
            printed = False

            while running:
                st = jetson.stats
                if args.debug_schema and not printed:
                    try:
                        print("[jtop_export] top-level keys:", sorted(list(st.keys()))[:80], file=sys.stderr)
                    except Exception:
                        pass
                    printed = True

                ts = now_ns()
                gpu = extract_gpu_util(st)
                emc = extract_emc_util(st)
                cpu_avg, cpu_max = extract_cpu_utils(st)

                w.writerow([ts, gpu, emc, cpu_avg, cpu_max])
                f.flush()
                time.sleep(interval_s)

if __name__ == "__main__":
    main()
