"""FFT-based electrostatic density loss (DREAMPlace / ePlace style).

Treats macros as charges. Density violations create an "electric potential field"
solved via Poisson equation in frequency domain. Pure PyTorch — no CUDA extensions.

Mathematical formulation (ePlace, Cheng & Kahng 2018):
  Poisson:  ∇²φ(x,y) = -(ρ(x,y) - ρ_avg)
  Loss:     U = 1/2 ∫∫ |∇φ|² dx dy = 1/2 Σ_k (|k|² · |φ_k|²) = 1/2 Σ_k |ρ_k|²/|k|²
"""
import math
import torch
import torch.nn.functional as F


def fft_density_loss(
    pos: torch.Tensor,           # [N_all, 2] macro centers (hard + soft)
    size: torch.Tensor,          # [N_all, 2] (w, h)
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    target_density: float = 1.0,
) -> torch.Tensor:
    """FFT-based density penalty matching ePlace energy formulation.

    Returns the electrostatic potential energy of the density field.
    Lower = density is more uniformly distributed.
    Differentiable through both the density grid construction and the FFT.
    """
    device = pos.device
    cw = canvas_w / grid_cols
    ch = canvas_h / grid_rows

    # Build density grid via differentiable rectangle-cell overlap.
    x_lo = pos[:, 0] - size[:, 0] / 2
    x_hi = pos[:, 0] + size[:, 0] / 2
    y_lo = pos[:, 1] - size[:, 1] / 2
    y_hi = pos[:, 1] + size[:, 1] / 2

    with torch.no_grad():
        col_lo = torch.floor(x_lo / cw).long().clamp(0, grid_cols - 1)
        col_hi = (torch.ceil(x_hi / cw).long() - 1).clamp(0, grid_cols - 1)
        row_lo = torch.floor(y_lo / ch).long().clamp(0, grid_rows - 1)
        row_hi = (torch.ceil(y_hi / ch).long() - 1).clamp(0, grid_rows - 1)

    n = pos.shape[0]
    if n == 0:
        return torch.zeros((), device=device, dtype=pos.dtype)

    max_dcol = int((col_hi - col_lo).max().item()) + 1
    max_drow = int((row_hi - row_lo).max().item()) + 1

    drow = torch.arange(max_drow, device=device).unsqueeze(0)
    dcol = torch.arange(max_dcol, device=device).unsqueeze(0)
    rows = row_lo.unsqueeze(1) + drow
    cols = col_lo.unsqueeze(1) + dcol
    rows_valid = (rows <= row_hi.unsqueeze(1)).float()
    cols_valid = (cols <= col_hi.unsqueeze(1)).float()
    rows = rows.clamp(0, grid_rows - 1)
    cols = cols.clamp(0, grid_cols - 1)

    cell_L = cols.float() * cw
    cell_R = cell_L + cw
    cell_B = rows.float() * ch
    cell_T = cell_B + ch

    ox = torch.clamp(torch.minimum(x_hi.unsqueeze(1), cell_R) - torch.maximum(x_lo.unsqueeze(1), cell_L), min=0.0) * cols_valid
    oy = torch.clamp(torch.minimum(y_hi.unsqueeze(1), cell_T) - torch.maximum(y_lo.unsqueeze(1), cell_B), min=0.0) * rows_valid

    overlap = oy.unsqueeze(2) * ox.unsqueeze(1)
    cell_idx = rows.unsqueeze(2) * grid_cols + cols.unsqueeze(1)
    grid_flat = torch.zeros(grid_rows * grid_cols, device=device, dtype=pos.dtype)
    grid_flat = grid_flat.scatter_add(0, cell_idx.view(-1), overlap.view(-1))

    # Density grid (filled fraction per cell)
    density = grid_flat.view(grid_rows, grid_cols) / (cw * ch)

    # Subtract mean to make it zero-mean (charge neutrality)
    rho = density - density.mean()

    # 2D FFT
    rho_freq = torch.fft.rfft2(rho)  # [grid_rows, grid_cols//2+1] complex

    # Wavenumbers (cycles per unit length × 2π)
    ky = torch.fft.fftfreq(grid_rows, d=ch, device=device) * 2 * math.pi  # [grid_rows]
    kx = torch.fft.rfftfreq(grid_cols, d=cw, device=device) * 2 * math.pi  # [grid_cols//2+1]

    # k² grid
    ky2, kx2 = torch.meshgrid(ky ** 2, kx ** 2, indexing='ij')
    k2 = ky2 + kx2
    # Avoid div by 0 at DC (we already subtracted mean so DC is 0)
    k2 = k2.clamp_min(1e-12)
    k2[0, 0] = 1.0  # safe value, will zero out below

    # Potential energy in frequency domain:
    #   U = 1/2 ∫|∇φ|² = 1/2 Σ_k |ρ_k|² / |k|²
    # (Parseval / Plancherel)
    spectrum = (rho_freq.abs() ** 2) / k2
    spectrum[0, 0] = 0.0

    # Sum over frequency (with the right normalization for rfft2)
    # Multiply by 2 except for k=0 and Nyquist columns (rfft only stores half)
    energy = spectrum.sum() * 2 - spectrum[:, 0].sum() - spectrum[:, -1].sum()
    return energy
