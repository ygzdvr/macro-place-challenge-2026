"""Diagnostic: run gradient descent, evaluate TILOS proxy at several checkpoints.

Confirms whether surrogate descent improves or degrades TILOS proxy along
the optimization trajectory.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from placer import (
    build_net_tensors,
    compute_pin_positions,
    hpwl_lse,
    density_loss,
    rudy_congestion_loss,
    overlap_repulsion_loss,
)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def main():
    benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net_tensors(benchmark).to(device)

    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_size = max(canvas_w, canvas_h)
    sizes = benchmark.macro_sizes.to(device)
    fixed = benchmark.macro_fixed.to(device)
    movable_mask = (~fixed).float().unsqueeze(1).to(device)

    pos_init = benchmark.macro_positions.clone().to(device)
    pos = pos_init.clone().detach().requires_grad_(True)

    # Auto-balance from init magnitudes
    with torch.no_grad():
        pin_xy = compute_pin_positions(pos, net)
        gamma_init = 0.04 * canvas_size
        wl0 = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma_init).item()
        den0 = density_loss(pos, sizes, canvas_w, canvas_h,
                            benchmark.grid_rows, benchmark.grid_cols,
                            top_k_frac=0.10, use_smooth_topk=True, smooth_tau=1.0).item()
        cong0 = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                     benchmark.grid_rows, benchmark.grid_cols,
                                     top_k_frac=0.05, smooth_range=2,
                                     use_smooth_topk=True, smooth_tau=1.0).item()

    w_wl = 1.0 / (abs(wl0) + 1e-9)
    w_den = 0.5 / (abs(den0) + 1e-9)
    w_cong = 0.5 / (abs(cong0) + 1e-9)

    # Smaller LR per advisor recommendation
    lr = 0.0005 * canvas_size
    optimizer = torch.optim.Adam([pos], lr=lr)

    global_steps = 300
    gamma_final = 0.002 * canvas_size

    def eval_tilos(step):
        cur = pos.detach().cpu()
        c = compute_proxy_cost(cur, benchmark, plc)
        print(f"  step {step:4d}  TILOS: proxy={c['proxy_cost']:.4f}  wl={c['wirelength_cost']:.4f}  den={c['density_cost']:.4f}  cong={c['congestion_cost']:.4f}  overlaps={c['overlap_count']}")

    print(f"=== Diagnostic trajectory on ibm01 (start: initial) ===")
    eval_tilos(0)

    for step in range(global_steps):
        frac = step / max(1, global_steps - 1)
        gamma = gamma_init * (gamma_final / gamma_init) ** frac

        optimizer.zero_grad()
        pin_xy = compute_pin_positions(pos, net)
        wl = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma)
        den = density_loss(pos, sizes, canvas_w, canvas_h,
                           benchmark.grid_rows, benchmark.grid_cols,
                           top_k_frac=0.10, use_smooth_topk=True, smooth_tau=1.0)
        cong = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                    benchmark.grid_rows, benchmark.grid_cols,
                                    top_k_frac=0.05, smooth_range=2,
                                    use_smooth_topk=True, smooth_tau=1.0)
        loss = w_wl * wl + w_den * den + w_cong * cong
        loss.backward()

        with torch.no_grad():
            pos.grad.mul_(movable_mask)
            pos.grad.clamp_(-canvas_size * 0.05, canvas_size * 0.05)
        optimizer.step()
        with torch.no_grad():
            pos[:, 0].clamp_(sizes[:, 0] / 2, canvas_w - sizes[:, 0] / 2)
            pos[:, 1].clamp_(sizes[:, 1] / 2, canvas_h - sizes[:, 1] / 2)

        if step in (10, 25, 50, 100, 200, 299):
            eval_tilos(step)


if __name__ == "__main__":
    main()
