"""Klein-4 orientation flip optimization.

For each hard macro, try the 4 allowed orientations (N, FN, FS, S) and pick
the one minimizing total wirelength of nets touching that macro. Orientation
flips don't change macro footprint (only pin offsets), so density / congestion
are roughly unchanged — main benefit is WL.

Per README, orientations propagate to Tier 2 via optional `orientations.pt`
sidecar (Klein-4 only — R0/R180/MX/MY).
"""
import numpy as np
import torch
from typing import List


# Klein-4 orientation transformations of pin offsets
# (assumes original orientation is N / R0)
def transform_offset(offset_xy, orient: str):
    """Apply orientation transform to a pin offset.

    Orientations:
      N  (R0):   (x, y) → ( x,  y)
      FN (MY):   (x, y) → (-x,  y)   flip about y-axis
      FS (MX):   (x, y) → ( x, -y)   flip about x-axis
      S  (R180): (x, y) → (-x, -y)   180° rotation
    """
    x, y = offset_xy[..., 0], offset_xy[..., 1]
    if orient == 'N':
        return torch.stack([x, y], dim=-1)
    elif orient == 'FN':
        return torch.stack([-x, y], dim=-1)
    elif orient == 'FS':
        return torch.stack([x, -y], dim=-1)
    elif orient == 'S':
        return torch.stack([-x, -y], dim=-1)
    else:
        raise ValueError(f"unsupported orientation: {orient}")


def optimize_orientations_gpu(
    placement,                    # [N_macros, 2] tensor, hard macros at [:n_hard]
    benchmark,
    net,                          # NetTensors
    device,
    n_passes: int = 2,
) -> List[str]:
    """Greedy coordinate-descent over orientations.

    For each hard macro, evaluate WL for each of 4 orientations and pick the best.
    Repeat for n_passes. Returns list of chosen orientations for each hard macro.

    Returns:
        orientations: list[str] of length n_hard, e.g. ['N', 'FN', 'S', ...]
    """
    n_hard = benchmark.num_hard_macros
    if n_hard == 0:
        return []

    orient_map = ['N', 'FN', 'FS', 'S']
    current_orient = ['N'] * n_hard  # start with all-N (no flip)

    # Get current pin positions baseline
    canvas_size = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
    gamma = 0.002 * canvas_size

    # Pre-compute: which macros are in which nets (for fast per-macro WL eval)
    # We'll just compute total WL after each move
    pin_owner = net.pin_owner
    pin_offset = net.pin_offset.clone()  # [total_pins, 2]
    pin_net = net.pin_net
    num_nets = net.num_nets

    # The offset transforms below need to be re-applied on demand
    # base_offset[i] = offset for pin i in N orientation (from netlist)
    base_offset = net.pin_offset.clone()  # this is the original

    def compute_total_wl(current_pin_offset):
        """Compute total LSE WL given pin offsets."""
        owner = pin_owner
        is_port = owner >= net.num_macros
        # Macro positions for non-port pins
        macro_idx = torch.where(is_port, torch.zeros_like(owner), owner)
        macro_xy = placement[macro_idx]
        pin_xy = macro_xy + current_pin_offset
        if net.num_ports > 0 and is_port.any():
            port_idx = (owner - net.num_macros).clamp_min(0)
            port_xy = net.port_pos[port_idx]
            pin_xy = torch.where(is_port.unsqueeze(1), port_xy, pin_xy)

        # LSE HPWL
        from placer import hpwl_lse
        return hpwl_lse(pin_xy, pin_net, num_nets, gamma).item()

    # For each macro, the pins owned by it: indices where pin_owner == i
    # We'll iterate over macros and for each, try all 4 orientations
    pin_to_macro = pin_owner.cpu().numpy()
    macro_pins = {}  # macro_idx -> list of pin indices
    for pi in range(len(pin_to_macro)):
        m = int(pin_to_macro[pi])
        if m < n_hard:  # only hard macros
            macro_pins.setdefault(m, []).append(pi)

    # Cache base offsets per macro
    current_offset = base_offset.clone()  # mutable

    best_wl = compute_total_wl(current_offset)

    for pass_idx in range(n_passes):
        n_changed = 0
        for m in range(n_hard):
            pins = macro_pins.get(m, [])
            if not pins:
                continue
            pin_idx = torch.tensor(pins, device=device, dtype=torch.long)
            base_pin_offsets = base_offset[pin_idx]  # [num_pins, 2]

            best_local_wl = best_wl
            best_orient = current_orient[m]
            for orient in orient_map:
                new_offsets = transform_offset(base_pin_offsets, orient)
                # Temp update
                tmp_offset = current_offset.clone()
                tmp_offset[pin_idx] = new_offsets
                wl = compute_total_wl(tmp_offset)
                if wl < best_local_wl:
                    best_local_wl = wl
                    best_orient = orient

            if best_orient != current_orient[m]:
                # Apply the move
                current_orient[m] = best_orient
                new_offsets = transform_offset(base_pin_offsets, best_orient)
                current_offset[pin_idx] = new_offsets
                best_wl = best_local_wl
                n_changed += 1
        print(f"  [orient pass {pass_idx+1}] changed {n_changed} macros, total WL={best_wl:.3e}", flush=True)
        if n_changed == 0:
            break

    return current_orient


def save_orientations_pt(orientations: List[str], path: str):
    """Save orientations as torch.LongTensor sidecar.

    Encoding: N=0, FN=1, FS=2, S=3
    """
    enc = {'N': 0, 'FN': 1, 'FS': 2, 'S': 3}
    t = torch.tensor([enc[o] for o in orientations], dtype=torch.long)
    torch.save(t, path)
    print(f"Saved {len(orientations)} orientations to {path}", flush=True)
