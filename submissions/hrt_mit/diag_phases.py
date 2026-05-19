"""Track TILOS proxy at each phase of the placer to find biggest loss."""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from placer import (
    build_net_tensors, compute_pin_positions, hpwl_lse, density_loss,
    rudy_congestion_loss, legalize_diffuse_push, legalize_spiral_fallback,
)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def eval_tilos(label, pos, benchmark, plc):
    c = compute_proxy_cost(pos, benchmark, plc)
    print(f"  [{label:30s}] proxy={c['proxy_cost']:.4f}  wl={c['wirelength_cost']:.4f}  den={c['density_cost']:.4f}  cong={c['congestion_cost']:.4f}  overlaps={c['overlap_count']}")
    return c


def main():
    for bench_name in ["ibm01"]:
        print(f"\n=== {bench_name} ===")
        benchmark, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")
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

        # Phase 0: initial
        eval_tilos("phase 0 — initial (illegal)", pos.detach().cpu(), benchmark, plc)

        # Auto-balance
        with torch.no_grad():
            pin_xy = compute_pin_positions(pos, net)
            gamma_init = 0.04 * canvas_size
            wl0 = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma_init).item()
            den0 = density_loss(pos, sizes, canvas_w, canvas_h,
                                benchmark.grid_rows, benchmark.grid_cols, smooth_tau=1.0).item()
            cong0 = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                         benchmark.grid_rows, benchmark.grid_cols, smooth_tau=1.0).item()

        w_wl = 1.0 / wl0
        w_den = 0.5 / den0
        w_cong = 0.5 / cong0

        global_steps = 50
        lr = 0.0005 * canvas_size
        optimizer = torch.optim.Adam([pos], lr=lr)

        for step in range(global_steps):
            frac = step / max(1, global_steps - 1)
            gamma = gamma_init * (0.05) ** frac

            optimizer.zero_grad()
            pin_xy = compute_pin_positions(pos, net)
            wl = hpwl_lse(pin_xy, net.pin_net, net.num_nets, gamma)
            den = density_loss(pos, sizes, canvas_w, canvas_h,
                               benchmark.grid_rows, benchmark.grid_cols, smooth_tau=1.0)
            cong = rudy_congestion_loss(pin_xy, net.pin_net, net.num_nets, canvas_w, canvas_h,
                                        benchmark.grid_rows, benchmark.grid_cols, smooth_tau=1.0)
            loss = w_wl * wl + w_den * den + w_cong * cong
            loss.backward()
            with torch.no_grad():
                pos.grad.mul_(movable_mask)
                pos.grad.clamp_(-canvas_size * 0.05, canvas_size * 0.05)
            optimizer.step()
            with torch.no_grad():
                pos[:, 0].clamp_(sizes[:, 0] / 2, canvas_w - sizes[:, 0] / 2)
                pos[:, 1].clamp_(sizes[:, 1] / 2, canvas_h - sizes[:, 1] / 2)

        eval_tilos("phase 1 — after gradient", pos.detach().cpu(), benchmark, plc)

        # Legalize: phase 2a diffuse push
        pos_np = pos[:n_hard].detach().cpu().numpy().astype(np.float64)
        sizes_np = sizes[:n_hard].cpu().numpy().astype(np.float64)
        fixed_np = fixed[:n_hard].cpu().numpy()
        diffuse_legal = legalize_diffuse_push(pos_np, sizes_np, fixed_np, canvas_w, canvas_h)
        legal_full = pos.detach().cpu().clone()
        legal_full[:n_hard] = torch.from_numpy(diffuse_legal).float()
        eval_tilos("phase 2a — diffuse push only", legal_full, benchmark, plc)

        # Phase 2b: spiral fallback
        final_legal = legalize_spiral_fallback(diffuse_legal, sizes_np, fixed_np, canvas_w, canvas_h)
        legal_full[:n_hard] = torch.from_numpy(final_legal).float()
        eval_tilos("phase 2b — spiral fallback", legal_full, benchmark, plc)

        # Measure displacement caused by legalization
        gradient_pos = pos[:n_hard].detach().cpu().numpy().astype(np.float64)
        disp = np.sqrt(((final_legal - gradient_pos) ** 2).sum(axis=1))
        print(f"  legalization displacement: mean={disp.mean():.3f}μm max={disp.max():.3f}μm  >2x_avg_size={np.sum(disp > 2*sizes_np.mean()):d}")


if __name__ == "__main__":
    main()
