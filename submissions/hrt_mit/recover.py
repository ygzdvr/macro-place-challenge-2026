"""Recover proxy scores for benchmarks that errored in run_all.

Loads saved placements from /tmp/hrt_placements/{benchmark}.pt and computes
their TILOS proxy. Useful when the outer evaluator crashes mid-run.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, ".")

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


IBM = [f"ibm{i:02d}" for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]


def main():
    save_dir = os.environ.get("HRT_SAVE_DIR", "/tmp/hrt_placements")
    results_path = "output/results_all.json"

    try:
        with open(results_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    n_recovered = 0
    for name in IBM:
        v = results.get(name, {})
        if "proxy" in v:
            continue  # already has valid result

        save_path = f"{save_dir}/{name}.pt"
        if not Path(save_path).exists():
            print(f"  {name}: no saved placement at {save_path}")
            continue

        try:
            data = torch.load(save_path, weights_only=False)
            placement = data["placement"]
            saved_proxy = data.get("proxy", -1)

            # Re-evaluate with fresh TILOS
            t0 = time.time()
            benchmark, plc = load_benchmark_from_dir(
                f"external/MacroPlacement/Testcases/ICCAD04/{name}"
            )
            costs = compute_proxy_cost(placement, benchmark, plc)
            elapsed = time.time() - t0

            results[name] = {
                "name": name,
                "proxy": float(costs["proxy_cost"]),
                "wl": float(costs["wirelength_cost"]),
                "den": float(costs["density_cost"]),
                "cong": float(costs["congestion_cost"]),
                "valid": True,
                "overlaps": int(costs["overlap_count"]),
                "runtime": elapsed,
                "wall_time": elapsed,
                "recovered_from": save_path,
                "saved_proxy": saved_proxy,
            }
            print(f"  {name}: recovered proxy={costs['proxy_cost']:.4f} "
                  f"(saved={saved_proxy:.4f})  overlaps={costs['overlap_count']}  [{elapsed:.1f}s]")
            n_recovered += 1
        except Exception as e:
            print(f"  {name}: recover failed: {e}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRecovered {n_recovered} benchmarks.")


if __name__ == "__main__":
    main()
