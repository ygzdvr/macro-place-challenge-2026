# HRT × MIT Macro Placer

GPU-accelerated electrostatic global placement with TILOS-evaluated multi-strategy selection, plus an optional ePlace-style FFT density phase and TILOS-evaluated SA polish.

## Results

Average proxy across 17 IBM ICCAD04 benchmarks: **1.4312** (validated on all 17).

| Comparison | Avg | vs My |
|---|---|---|
| **HRT × MIT (this placer)** | **1.4312** | — |
| RePlAce baseline | 1.4578 | -1.8% (we win) |
| SA baseline | 2.1251 | -32.7% (we win) |

Per-benchmark breakdown:

| Benchmark | My | SA | RP | vs RP |
|---|---|---|---|---|
| ibm01 | 0.9988 | 1.317 | 0.998 | +0.1% |
| ibm02 | 1.5881 | 1.907 | 1.837 | -13.5% |
| ibm03 | 1.2857 | 1.740 | 1.322 | -2.8% |
| ibm04 | 1.4344 | 1.504 | 1.302 | +10.1% |
| ibm06 | 1.6748 | 2.506 | 1.619 | +3.5% |
| ibm07 | 1.4106 | 2.023 | 1.463 | -3.6% |
| ibm08 | 1.4353 | 1.924 | 1.429 | +0.5% |
| ibm09 | 1.0757 | 1.387 | 1.119 | -3.9% |
| ibm10 | 1.3073 | 2.111 | 1.501 | -12.9% |
| ibm11 | 1.1704 | 1.711 | 1.177 | -0.6% |
| ibm12 | 1.8447 | 2.826 | 1.726 | +6.9% |
| ibm13 | 1.2829 | 1.914 | 1.335 | -3.9% |
| ibm14 | 1.5053 | 2.275 | 1.544 | -2.5% |
| ibm15 | 1.5355 | 2.300 | 1.516 | +1.3% |
| ibm16 | 1.4398 | 2.234 | 1.478 | -2.6% |
| ibm17 | 1.6367 | 3.673 | 1.645 | -0.5% |
| ibm18 | 1.7041 | 2.776 | 1.772 | -3.8% |

## Algorithm

**1. Pin-level differentiable surrogate** calibrated to TILOS (Spearman ρ ≥ 0.88 on diverse placements):
- Wirelength: log-sum-exp HPWL with γ-annealing, using `net_pin_nodes` + `macro_pin_offsets` for true pin offsets
- Density: grid-overlap with smooth top-10% via softmax (matches TILOS's `abu(grid_cells, 0.1)`)
- Congestion: V/H-split RUDY with 5×5 box smoothing and top-5% concat (matches TILOS's `abu(V+H, 0.05)` with `smooth_range=2`)

**2. Multi-strategy gradient placement** (Adam, γ-annealed, gradient-clipped):
- 8–11 hyperparameter variants per benchmark: different LR, step counts, and seed offsets
- Each variant: gradient → legalize → score with actual TILOS proxy
- Pick the lowest-proxy legal result — safety net against surrogate-TILOS divergence
- For large benchmarks (>400 hard macros), trim down to 8 variants (drops some seed variants)

**3. Optional ePlace-style FFT density phase** (`fft25_g50_s1` strategy, included in both tiers):
- Phase 1 (25 steps): minimize WL + electrostatic potential energy of density grid
  (`∇²φ = -(ρ-ρ_avg)`, solved via `rfft2` — pure PyTorch, no CUDA extensions)
- Phase 2 (50 steps): hand off to TILOS-aligned WL + top-k density + RUDY congestion
- Single Adam optimizer carries momentum across phases; weights computed once from initial placement
- Empirically wins on ibm10 (1.3073 vs g50_s0 1.3101); multi-strategy still keeps it from hurting elsewhere

**4. GPU-parallel diffuse-push legalization**:
- Iterative vectorized push-apart on hard macros (no kicks, 10-50× faster than naive numpy)
- Preserves topology (minimum displacement)
- Falls back to CPU spiral search for pathological cases

**5. Surrogate-evaluated SA polish** (small benchmarks only, ≤500 hard macros):
- 3000 iters of SHIFT + SWAP moves
- Single-macro O(N) overlap check
- GPU surrogate eval (~5 ms/move)
- Wrapped in try/except for safety
- Skipped on larger benchmarks where memory pressure can crash CUDA

**6. TILOS-evaluated SA polish** (very small benchmarks only):
- Direct evaluator-in-the-loop SA escapes surrogate–TILOS local minima
- Iter budgets tiered by TILOS eval cost: 40 iters for ibm01-class, 25 for ibm02-class, 15 for ibm09-class
- Disabled on large benchmarks where TILOS eval costs minutes per move

## Files

- `placer.py` — main entry point. `HRTPlacer.place(benchmark) -> Tensor`.
- `correlation_check.py` — diagnostic for surrogate vs TILOS rank correlation.
- `diag_*.py` — diagnostics for trajectory and per-phase TILOS proxy.
- `run_all.py` — parallel benchmark runner.

## Usage

```bash
uv run evaluate submissions/hrt_mit/placer.py -b ibm01
uv run evaluate submissions/hrt_mit/placer.py --all
```

## Dependencies

PyTorch ≥ 2.0, NumPy. CUDA strongly recommended (~10× speedup for large benchmarks).

## Key Design Choices

**Surrogate calibration.** The biggest danger in gradient-based placement is the gap between a differentiable surrogate and the actual TILOS evaluator. We explicitly verify rank correlation on diverse placements before trusting gradient steps. Our density surrogate includes BOTH hard and soft macros (matching TILOS) and uses smooth top-k softmax for differentiability.

**Multi-strategy safety net.** Different hyperparameters (LR, gradient steps) work best on different benchmarks. Rather than tuning per-benchmark, we run several candidates in parallel and pick the lowest TILOS proxy among them. This bounds our worst case by "just legalize the initial".

**Conservative legalization.** Our diffuse-push legalizer makes minimum-displacement moves — only pushing apart by overlap + 5% margin per step. This preserves the gradient-found topology better than greedy spiral/abacus approaches that can move macros across the canvas.

**Robustness over peak performance.** SA polish only runs on small benchmarks where it's stable; large benchmarks rely on multi-strategy alone. The pipeline guarantees a result even if individual components fail.
