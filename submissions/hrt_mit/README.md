# HRT × MIT Macro Placer

GPU-accelerated electrostatic global placement with TILOS-evaluated multi-strategy selection.

## Results

Average proxy across 17 IBM ICCAD04 benchmarks: **1.4350** (validated on all 17).

| Comparison | Avg | vs My |
|---|---|---|
| **HRT × MIT (this placer)** | **1.4350** | — |
| RePlAce baseline | 1.4578 | -1.6% (we win) |
| SA baseline | 2.1251 | -32.5% (we win) |

Per-benchmark breakdown:

| Benchmark | My | SA | RP | vs RP |
|---|---|---|---|---|
| ibm01 | 1.0005 | 1.317 | 0.998 | +0.3% |
| ibm02 | 1.5914 | 1.907 | 1.837 | -13.4% |
| ibm03 | 1.3438 | 1.740 | 1.322 | +1.6% |
| ibm04 | 1.2729 | 1.504 | 1.302 | -2.3% |
| ibm06 | 1.6875 | 2.506 | 1.619 | +4.3% |
| ibm07 | 1.4268 | 2.023 | 1.463 | -2.5% |
| ibm08 | 1.5409 | 1.924 | 1.429 | +7.9% |
| ibm09 | 1.0926 | 1.387 | 1.119 | -2.4% |
| ibm10 | 1.3101 | 2.111 | 1.501 | -12.7% |
| ibm11 | 1.3466 | 1.711 | 1.177 | +14.4% |
| ibm12 | 1.6807 | 2.826 | 1.726 | -2.6% |
| ibm13 | 1.2754 | 1.914 | 1.335 | -4.5% |
| ibm14 | 1.5080 | 2.275 | 1.544 | -2.3% |
| ibm15 | 1.5350 | 2.300 | 1.516 | +1.3% |
| ibm16 | 1.4400 | 2.234 | 1.478 | -2.6% |
| ibm17 | 1.6380 | 3.673 | 1.645 | -0.4% |
| ibm18 | 1.7042 | 2.776 | 1.772 | -3.8% |

## Algorithm

**1. Pin-level differentiable surrogate** calibrated to TILOS (Spearman ρ ≥ 0.88 on diverse placements):
- Wirelength: log-sum-exp HPWL with γ-annealing, using `net_pin_nodes` + `macro_pin_offsets` for true pin offsets
- Density: grid-overlap with smooth top-10% via softmax (matches TILOS's `abu(grid_cells, 0.1)`)
- Congestion: V/H-split RUDY with 5×5 box smoothing and top-5% concat (matches TILOS's `abu(V+H, 0.05)` with `smooth_range=2`)

**2. Multi-strategy gradient placement** (Adam, γ-annealed, gradient-clipped):
- 4–5 hyperparameter variants per benchmark (different LR + step counts)
- Each variant: gradient → legalize → score with actual TILOS proxy
- Pick the lowest-proxy legal result — safety net against surrogate-TILOS divergence
- For large benchmarks (>400 hard macros), trim down to top-3 variants

**3. GPU-parallel diffuse-push legalization**:
- Iterative vectorized push-apart on hard macros (no kicks, 10-50× faster than naive numpy)
- Preserves topology (minimum displacement)
- Falls back to CPU spiral search for pathological cases

**4. Surrogate-evaluated SA polish** (small benchmarks only, ≤500 hard macros):
- 3000 iters of SHIFT + SWAP moves
- Single-macro O(N) overlap check
- GPU surrogate eval (~5 ms/move)
- Wrapped in try/except for safety
- Skipped on larger benchmarks where memory pressure can crash CUDA

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
