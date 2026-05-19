"""
Surrogate-vs-TILOS correlation check.

Generates K random legal placements, computes both our surrogate cost
(WL_LSE + density + RUDY congestion) and TILOS's `compute_proxy_cost`,
then reports rank correlation.  If rank correlation < 0.8, our surrogate
is misaligned and gradient descent will lead to wrong solutions.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add the placer's directory so we can import its costs
sys.path.insert(0, str(Path(__file__).parent))
from placer import (
    NetTensors,
    build_net_tensors,
    compute_pin_positions,
    hpwl_lse,
    density_loss,
    rudy_congestion_loss,
    legalize_min_displacement,
)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def diverse_placement(benchmark, rng, kind: str):
    """Generate a placement with chosen amount of variation.

    kinds:
      "initial"          → just the initial placement (has overlaps)
      "perturb_small"    → small perturbation around initial, legalized
      "perturb_large"    → large perturbation around initial, legalized
      "random_legal"     → fully random hard macros, legalized
      "random_overlap"   → fully random hard macros (no legalize)
      "tight_left"       → all hard macros clustered to left half (no legalize)
      "tight_right"      → clustered to right half
      "spread"           → grid-uniform spread (legalized)
    """
    pos = benchmark.macro_positions.clone().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.numpy()
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros

    if kind == "initial":
        return torch.from_numpy(pos).float()

    perturb_scale = {"perturb_small": 0.05, "perturb_large": 0.20}.get(kind, None)
    if perturb_scale is not None:
        for i in range(n_hard):
            if fixed[i]:
                continue
            pos[i, 0] += rng.normal(0, cw * perturb_scale)
            pos[i, 1] += rng.normal(0, ch * perturb_scale)
            pos[i, 0] = np.clip(pos[i, 0], sizes[i, 0]/2, cw - sizes[i, 0]/2)
            pos[i, 1] = np.clip(pos[i, 1], sizes[i, 1]/2, ch - sizes[i, 1]/2)
        legal = legalize_min_displacement(pos[:n_hard], sizes[:n_hard], fixed[:n_hard], cw, ch)
        pos[:n_hard] = legal
        return torch.from_numpy(pos).float()

    if kind == "random_legal":
        for i in range(n_hard):
            if fixed[i]:
                continue
            w, h = sizes[i]
            pos[i, 0] = rng.uniform(w/2, cw - w/2)
            pos[i, 1] = rng.uniform(h/2, ch - h/2)
        legal = legalize_min_displacement(pos[:n_hard], sizes[:n_hard], fixed[:n_hard], cw, ch)
        pos[:n_hard] = legal
        return torch.from_numpy(pos).float()

    if kind == "random_overlap":
        for i in range(n_hard):
            if fixed[i]:
                continue
            w, h = sizes[i]
            pos[i, 0] = rng.uniform(w/2, cw - w/2)
            pos[i, 1] = rng.uniform(h/2, ch - h/2)
        return torch.from_numpy(pos).float()

    if kind in ("tight_left", "tight_right"):
        x_offset = 0.0 if kind == "tight_left" else cw * 0.5
        for i in range(n_hard):
            if fixed[i]:
                continue
            w, h = sizes[i]
            pos[i, 0] = x_offset + rng.uniform(w/2, cw * 0.5 - w/2)
            pos[i, 1] = rng.uniform(h/2, ch - h/2)
        return torch.from_numpy(pos).float()

    if kind == "spread":
        # Uniform grid spread
        cells = int(np.ceil(np.sqrt(n_hard)))
        dx, dy = cw / cells, ch / cells
        movable_idx = [i for i in range(n_hard) if not fixed[i]]
        for k, i in enumerate(movable_idx):
            row, col = divmod(k, cells)
            pos[i, 0] = col * dx + dx / 2
            pos[i, 1] = row * dy + dy / 2
        legal = legalize_min_displacement(pos[:n_hard], sizes[:n_hard], fixed[:n_hard], cw, ch)
        pos[:n_hard] = legal
        return torch.from_numpy(pos).float()

    raise ValueError(f"unknown kind {kind}")


def evaluate_surrogate(placement, benchmark, net, gamma_frac=0.005):
    """Compute surrogate cost given a placement."""
    pos = placement.cuda() if torch.cuda.is_available() else placement
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_size = max(canvas_w, canvas_h)
    sizes = benchmark.macro_sizes.to(pos.device)

    gamma = gamma_frac * canvas_size
    pin_xy = compute_pin_positions(pos, net)
    wl = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma)
    # All macros (hard + soft) for density — matches TILOS
    den = density_loss(pos, sizes, canvas_w, canvas_h,
                       benchmark.grid_rows, benchmark.grid_cols,
                       top_k_frac=0.10, use_smooth_topk=False)
    cong = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                benchmark.grid_rows, benchmark.grid_cols,
                                top_k_frac=0.05, smooth_range=2,
                                use_smooth_topk=False)
    return {
        "wl_surr": wl.item() / (canvas_size * net.num_nets),
        "den_surr": den.item(),
        "cong_surr": cong.item(),
    }


def main():
    rng = np.random.default_rng(42)
    benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net_tensors(benchmark).to(device)

    kinds = ["initial", "perturb_small", "perturb_small", "perturb_large",
             "perturb_large", "random_legal", "random_legal",
             "random_overlap", "random_overlap",
             "tight_left", "tight_right", "spread"]

    print(f"Generating {len(kinds)} diverse placements for {benchmark.name}...")
    placements = []
    for k, kind in enumerate(kinds):
        t0 = time.time()
        p = diverse_placement(benchmark, rng, kind)
        placements.append((kind, p))
        print(f"  [{k+1}/{len(kinds)}] {kind:18s}: {time.time()-t0:.2f}s")

    print("\nComputing surrogate + TILOS costs...")
    surr = []
    tilos = []
    for k, (kind, p) in enumerate(placements):
        s = evaluate_surrogate(p, benchmark, net)
        t0 = time.time()
        c = compute_proxy_cost(p.cpu(), benchmark, plc)
        t = time.time() - t0
        surr.append(s)
        tilos.append({
            "proxy": c["proxy_cost"],
            "wl_t": c["wirelength_cost"],
            "den_t": c["density_cost"],
            "cong_t": c["congestion_cost"],
        })
        print(f"  [{k+1:2d}] {kind:18s} TILOS[{t:.1f}s] proxy={c['proxy_cost']:.4f} wl={c['wirelength_cost']:.4f} den={c['density_cost']:.4f} cong={c['congestion_cost']:.4f} | surr wl={s['wl_surr']:.4e} den={s['den_surr']:.4e} cong={s['cong_surr']:.4e}")

    def rank_corr(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        n = len(a)
        return 1 - 6 * np.sum((ra - rb) ** 2) / (n * (n*n - 1)) if n > 1 else 0

    def lin_corr(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.corrcoef(a, b)[0, 1]

    print("\n=== Rank correlations (Spearman) ===")
    print(f"  WL    surr vs TILOS: {rank_corr([s['wl_surr'] for s in surr], [t['wl_t'] for t in tilos]):.3f}")
    print(f"  Den   surr vs TILOS: {rank_corr([s['den_surr'] for s in surr], [t['den_t'] for t in tilos]):.3f}")
    print(f"  Cong  surr vs TILOS: {rank_corr([s['cong_surr'] for s in surr], [t['cong_t'] for t in tilos]):.3f}")

    surr_proxy = [s['wl_surr'] + 0.5*s['den_surr'] + 0.5*s['cong_surr'] for s in surr]
    tilos_proxy = [t['proxy'] for t in tilos]
    print(f"  Proxy surr vs TILOS: {rank_corr(surr_proxy, tilos_proxy):.3f}")

    print("\n=== Pearson correlations (linear) ===")
    print(f"  WL    surr vs TILOS: {lin_corr([s['wl_surr'] for s in surr], [t['wl_t'] for t in tilos]):.3f}")
    print(f"  Den   surr vs TILOS: {lin_corr([s['den_surr'] for s in surr], [t['den_t'] for t in tilos]):.3f}")
    print(f"  Cong  surr vs TILOS: {lin_corr([s['cong_surr'] for s in surr], [t['cong_t'] for t in tilos]):.3f}")
    print(f"  Proxy surr vs TILOS: {lin_corr(surr_proxy, tilos_proxy):.3f}")


if __name__ == "__main__":
    main()
