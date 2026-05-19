"""Diagnostic: what's happening on ibm10? Test multiple hyperparameter settings.
Each evaluates TILOS proxy at: initial, post-gradient, post-legalization.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from placer import (
    build_net_tensors, compute_pin_positions, hpwl_lse, density_loss,
    rudy_congestion_loss, legalize_min_displacement,
)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def run_setting(benchmark, plc, name, global_steps, lr_frac, smooth_tau):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net_tensors(benchmark).to(device)
    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_size = max(canvas_w, canvas_h)
    sizes = benchmark.macro_sizes.to(device)
    fixed = benchmark.macro_fixed.to(device)
    movable_mask = (~fixed).float().unsqueeze(1).to(device)

    pos = benchmark.macro_positions.clone().to(device).detach().requires_grad_(True)

    if global_steps > 0:
        with torch.no_grad():
            pin_xy = compute_pin_positions(pos, net)
            gamma_init = 0.04 * canvas_size
            wl0 = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma_init).item()
            den0 = density_loss(pos, sizes, canvas_w, canvas_h,
                                benchmark.grid_rows, benchmark.grid_cols, smooth_tau=smooth_tau).item()
            cong0 = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                         benchmark.grid_rows, benchmark.grid_cols, smooth_tau=smooth_tau).item()
        w_wl = 1.0 / wl0
        w_den = 0.5 / den0
        w_cong = 0.5 / cong0

        lr = lr_frac * canvas_size
        optimizer = torch.optim.Adam([pos], lr=lr)

        for step in range(global_steps):
            frac = step / max(1, global_steps - 1)
            gamma = gamma_init * (0.05) ** frac
            optimizer.zero_grad()
            pin_xy = compute_pin_positions(pos, net)
            wl = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma)
            den = density_loss(pos, sizes, canvas_w, canvas_h,
                               benchmark.grid_rows, benchmark.grid_cols, smooth_tau=smooth_tau)
            cong = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                        benchmark.grid_rows, benchmark.grid_cols, smooth_tau=smooth_tau)
            loss = w_wl * wl + w_den * den + w_cong * cong
            loss.backward()
            with torch.no_grad():
                pos.grad.mul_(movable_mask)
                pos.grad.clamp_(-canvas_size * 0.05, canvas_size * 0.05)
            optimizer.step()
            with torch.no_grad():
                pos[:, 0].clamp_(sizes[:, 0] / 2, canvas_w - sizes[:, 0] / 2)
                pos[:, 1].clamp_(sizes[:, 1] / 2, canvas_h - sizes[:, 1] / 2)

    # Post-gradient TILOS eval
    c_grad = compute_proxy_cost(pos.detach().cpu(), benchmark, plc)

    # Legalize
    hard_pos = pos[:n_hard].detach().cpu().numpy().astype(np.float64)
    hard_sizes = sizes[:n_hard].cpu().numpy().astype(np.float64)
    hard_fixed = fixed[:n_hard].cpu().numpy()
    legal = legalize_min_displacement(hard_pos, hard_sizes, hard_fixed, canvas_w, canvas_h)
    full = pos.detach().cpu().clone()
    full[:n_hard] = torch.from_numpy(legal).float()
    c_legal = compute_proxy_cost(full, benchmark, plc)

    print(f"  [{name:30s}] post-grad: {c_grad['proxy_cost']:.4f} (overlaps={c_grad['overlap_count']})  legal: {c_legal['proxy_cost']:.4f} (overlaps={c_legal['overlap_count']})")
    return c_legal['proxy_cost']


def main():
    bench_name = "ibm10"
    print(f"=== {bench_name} hyperparameter sweep ===")
    benchmark, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")

    # Baseline: initial
    c = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    print(f"  [initial (illegal)] proxy={c['proxy_cost']:.4f}  overlaps={c['overlap_count']}")

    settings = [
        ("no_grad", 0, 0.0, 1.0),
        ("grad_25_lr0001", 25, 0.0001, 1.0),
        ("grad_25_lr0002", 25, 0.0002, 1.0),
        ("grad_50_lr0001", 50, 0.0001, 1.0),
        ("grad_50_lr0005", 50, 0.0005, 1.0),
    ]

    for name, steps, lr, tau in settings:
        run_setting(benchmark, plc, name, steps, lr, tau)


if __name__ == "__main__":
    main()
