"""Run placer on all 17 IBM benchmarks, in parallel, and produce a summary.

Each benchmark runs in a separate process so they don't fight over GPU/CPU.
Results are accumulated and printed at the end.

Usage:
    python submissions/hrt_mit/run_all.py [--max-parallel 2]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

IBM = [f"ibm{i:02d}" for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]

SA_BASELINES = {
    "ibm01": 1.3166, "ibm02": 1.9072, "ibm03": 1.7401, "ibm04": 1.5037,
    "ibm06": 2.5057, "ibm07": 2.0229, "ibm08": 1.9239, "ibm09": 1.3875,
    "ibm10": 2.1108, "ibm11": 1.7111, "ibm12": 2.8261, "ibm13": 1.9141,
    "ibm14": 2.2750, "ibm15": 2.3000, "ibm16": 2.2337, "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINES = {
    "ibm01": 0.9976, "ibm02": 1.8370, "ibm03": 1.3222, "ibm04": 1.3024,
    "ibm06": 1.6187, "ibm07": 1.4633, "ibm08": 1.4285, "ibm09": 1.1194,
    "ibm10": 1.5009, "ibm11": 1.1774, "ibm12": 1.7261, "ibm13": 1.3355,
    "ibm14": 1.5436, "ibm15": 1.5159, "ibm16": 1.4780, "ibm17": 1.6446,
    "ibm18": 1.7722,
}


def run_one(name: str) -> dict:
    """Run placer on a single benchmark and parse result."""
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "macro_place.evaluate",
        "submissions/hrt_mit/placer.py", "-b", name,
    ]
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr

    # Parse line like: "proxy=1.0542  (wl=0.068 den=0.759 cong=1.213)  VALID  [32.35s]"
    import re
    m = re.search(r"proxy=([0-9.]+)\s+\(wl=([0-9.]+)\s+den=([0-9.]+)\s+cong=([0-9.]+)\)\s+(\S+)\s+\[([0-9.]+)s\]", out)
    if not m:
        return {"name": name, "error": "parse_failed", "stderr_tail": out[-500:], "elapsed": elapsed}

    valid_str = m.group(5)
    overlaps = 0
    if "INVALID" in valid_str:
        m2 = re.search(r"INVALID \((\d+) overlaps\)", out)
        if m2:
            overlaps = int(m2.group(1))

    return {
        "name": name,
        "proxy": float(m.group(1)),
        "wl": float(m.group(2)),
        "den": float(m.group(3)),
        "cong": float(m.group(4)),
        "valid": valid_str == "VALID",
        "overlaps": overlaps,
        "runtime": float(m.group(6)),
        "wall_time": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="Max concurrent benchmarks. TILOS eval is CPU-bound so 4 is fine on 16-core box.")
    ap.add_argument("--out", type=str, default="output/results_all.json")
    ap.add_argument("--benchmarks", type=str, nargs="+", default=None,
                    help="Subset to run; default is all 17")
    args = ap.parse_args()

    benchmarks = args.benchmarks or IBM
    Path("output").mkdir(exist_ok=True)

    print(f"Running {len(benchmarks)} benchmarks with max_parallel={args.max_parallel}")
    print(f"Saving to: {args.out}")
    print()

    results = {}
    t_total = time.time()

    with ProcessPoolExecutor(max_workers=args.max_parallel) as exe:
        futures = {exe.submit(run_one, name): name for name in benchmarks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                r = future.result()
            except Exception as e:
                r = {"name": name, "error": str(e)}
            results[name] = r
            if "error" in r:
                print(f"  {name:8s}  ERROR: {r['error']}")
            else:
                vs_sa = (SA_BASELINES.get(name, 0) - r["proxy"]) / SA_BASELINES.get(name, 1) * 100 if SA_BASELINES.get(name) else 0
                vs_rep = (REPLACE_BASELINES.get(name, 0) - r["proxy"]) / REPLACE_BASELINES.get(name, 1) * 100 if REPLACE_BASELINES.get(name) else 0
                valid_mark = "✓" if r["valid"] else "✗"
                print(f"  {name:8s}  proxy={r['proxy']:.4f}  vs_SA={vs_sa:+.1f}%  vs_RP={vs_rep:+.1f}%  overlaps={r['overlaps']}  {valid_mark}  [{r['wall_time']:.0f}s wall]")

            # Save incremental results
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print()
    print(f"=== Summary ({time.time() - t_total:.0f}s total wall time) ===")
    print("-" * 90)
    print(f"{'Benchmark':>10}  {'Proxy':>8}  {'SA':>8}  {'RP':>8}  {'vs SA':>7}  {'vs RP':>7}  {'overlaps':>9}  runtime")
    print("-" * 90)

    proxies = []
    sa_total = 0
    rep_total = 0
    count = 0
    bad = 0
    for name in benchmarks:
        r = results.get(name, {})
        if "error" in r:
            print(f"{name:>10}  {'ERROR':>8}")
            bad += 1
            continue
        proxy = r["proxy"]
        sa = SA_BASELINES.get(name, 0)
        rp = REPLACE_BASELINES.get(name, 0)
        vs_sa = (sa - proxy) / sa * 100 if sa else 0
        vs_rep = (rp - proxy) / rp * 100 if rp else 0
        proxies.append(proxy)
        sa_total += sa
        rep_total += rp
        count += 1
        valid_mark = "" if r["valid"] else f" [{r['overlaps']} overlaps]"
        print(f"{name:>10}  {proxy:>8.4f}  {sa:>8.4f}  {rp:>8.4f}  {vs_sa:>+6.1f}%  {vs_rep:>+6.1f}%  {r['overlaps']:>9}{valid_mark}  {r['runtime']:.1f}s")

    if proxies:
        avg = sum(proxies) / len(proxies)
        avg_sa = sa_total / count
        avg_rep = rep_total / count
        print("-" * 90)
        print(f"{'AVG':>10}  {avg:>8.4f}  {avg_sa:>8.4f}  {avg_rep:>8.4f}  "
              f"{(avg_sa-avg)/avg_sa*100:>+6.1f}%  {(avg_rep-avg)/avg_rep*100:>+6.1f}%")
        print()
        print(f"AVG PROXY: {avg:.4f}  ({len(proxies)}/{len(benchmarks)} valid, {bad} errored)")


if __name__ == "__main__":
    main()
