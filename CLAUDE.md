# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is the **Partcl × HRT Macro Placement Challenge 2026** — a $20K open competition. Submissions place hard macros (SRAMs, IPs) on a chip canvas to minimize proxy cost; top 7 are then run through the full OpenROAD PnR flow on NG45 designs for the Grand Prize. Submission deadline: **2026-05-21**.

The `macro_place/` Python package is the evaluation infrastructure participants depend on. Treat it as a stable API — don't refactor signatures or break loader/objective behavior without strong reason. Participants' placers import `macro_place.benchmark.Benchmark` and `macro_place.objective.compute_proxy_cost`.

## Setup & Common Commands

```bash
# One-time setup (submodule provides TILOS PlacementCost + ICCAD04 benchmarks)
git submodule update --init external/MacroPlacement
uv sync

# Run a placer on a single benchmark (default: ibm01)
uv run evaluate submissions/examples/greedy_row_placer.py
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03

# Full IBM suite (17 benchmarks — Tier 1 ranking metric)
uv run evaluate submissions/examples/greedy_row_placer.py --all

# NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize placements to vis/<benchmark>.png
uv run evaluate <placer.py> --vis

# Tests (skip gracefully if submodule isn't initialized)
uv run pytest                              # all
uv run pytest test/test_smoke.py -v        # one file
uv run pytest test/test_smoke.py::test_compute_proxy_cost  # one test

# Tier 2 — full ORFS flow (requires ../OpenROAD-flow-scripts checked out)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker \
    --placement <placement.pt>

# Air-gapped judging harness (mirrors the eval environment)
./eval_docker/run_eval.sh <team_name> <path/to/placer.py> [extra_mount...]
```

The `evaluate` CLI is wired via `pyproject.toml`'s `[project.scripts]` entry — `evaluate = "macro_place.evaluate:main"`. It auto-imports the first class in the placer file that has a `place` attribute and instantiates it with no arguments.

## Architecture: Two-Tier Evaluation

**Tier 1 (proxy cost, ranks everyone):** 17 IBM ICCAD04 benchmarks (`ibm01–04, 06–18` — no `ibm05`). Score = `1.0·HPWL + 0.5·density + 0.5·congestion`, computed by TILOS `PlacementCost` (not reimplemented).

**Tier 2 (Grand Prize, top 7 only):** Full OpenROAD-flow-scripts on NG45 designs + 1–2 hidden. Scored by weighted geometric mean of improvement ratios over (SA+RePlAce)/2 baselines, with weights `(WNS:3, TNS:2, Area:1)`. See `SCORING.md` for the exact math and feasibility gate.

## Architecture: The Benchmark / PlacementCost Split

There are **two parallel representations** of every design, and they must stay in sync:

1. `Benchmark` (`macro_place/benchmark.py`): a dataclass of PyTorch tensors. What participants see.
2. `PlacementCost` (TILOS `plc_client_os.py`, loaded via `macro_place/_plc.py`): the C++/Python evaluator. Holds canonical state and computes costs.

`loader.py` builds both from one parse of `netlist.pb.txt` + `initial.plc` and returns `(benchmark, plc)`. `objective.compute_proxy_cost` accepts a placement tensor, then `_set_placement` writes those coords back into `plc.modules_w_pins[*]` **and also updates every pin's absolute position** because the evaluator caches stale pin coords. Forgetting the pin update silently gives wrong wirelength.

Tensor index → PlacementCost module index mapping:
- Hard macros: tensor `[0, num_hard_macros)` ↔ `benchmark.hard_macro_indices[i]`
- Soft macros: tensor `[num_hard_macros, num_macros)` ↔ `benchmark.soft_macro_indices[i]`
- I/O ports: tensor `[num_macros, num_macros + num_ports)` (only in `net_pin_nodes`)

After `_set_placement`, three `FLAG_UPDATE_*` flags are set so the next `get_cost()` / `get_density_cost()` / `get_congestion_cost()` recomputes. The congestion arrays are also re-sized via `_ensure_congestion_arrays` because the grid dimensions can shift on first call.

`objective.py` also monkey-patches `PlacementCost.__get_grid_cell_location` to clamp `row`/`col` into the valid range — fixes a boundary bug in the upstream evaluator that produces out-of-bounds index errors at the canvas edge. This patch runs at import time.

## Architecture: Hard vs Soft Macros

`Benchmark` packs both into one tensor: hard first, soft second. Differences that matter:

- **Hard macros are the optimization target.** Zero hard-hard overlaps is a hard constraint; both `validate_placement` and `compute_overlap_metrics` only check hard pairs.
- **Soft macros may overlap.** They are pre-clustered stdcell groups; their sizes are locked (resizing is explicitly disallowed by rules) and their proxy contribution is density/congestion only.
- **Both are movable.** The SA baseline re-runs `plc.optimize_stdcells()` between hard-macro batches to chase the moved hard macros. Most submissions ignore soft macros and accept the wirelength/density penalty.

`benchmark.get_hard_macro_mask()` / `get_soft_macro_mask()` / `get_movable_mask()` slice the right indices. `macro_fixed` blocks individual macros from moving regardless of hard/soft.

## Architecture: Net Connectivity

Stored in two parallel forms inside `Benchmark`:

- `net_nodes[i]`: int64 tensor of unique owner indices in net `i` — macro-level, dedup'd. Use for analytical placement / GNNs over macros.
- `net_pin_nodes[i]`: int64 tensor `[num_pins, 2]` of `(owner_idx, pin_slot)`. **Preserves multi-pin endpoints on the same macro.** Hard macros use `pin_slot` to index into `macro_pin_offsets[owner]`; soft macros and ports always use slot `0`. Needed for pin-accurate differentiable HPWL.

Net data also lives inside `plc.nets` (dict of driver → sinks), used by `compute_proxy_cost`. The two views are kept consistent by the loader; don't write through one and expect the other to update.

## Architecture: Tier-2 ORFS Integration

`scripts/evaluate_with_orfs.py` is the Tier-2 driver — it's intricate because it patches around real ORFS quirks. Key things future-you needs to know:

**Proxy canvas vs ORFS core area:** They are different sizes. `generate_macro_placement_tcl.py:write_orfs_macro_placement` **center-offsets** the proxy placement into the ORFS core (no scaling — preserves macro-to-macro distances exactly), then clamps with a 2 μm margin.

**Tier-2-only spacing enforcement (`generate_macro_placement_tcl.py:262-355`):** PDN metal5 needs ~10 μm channels between macros, so the evaluator runs up to 50 iterations pushing any pair closer than `MIN_GAP = 12 μm` apart along the larger axis. A sidecar `macros.tcl.spacing_diff.txt` logs every displacement. **This does not run at Tier 1** — proxy cost uses submitted coordinates unchanged. Submissions wanting full Tier-2 control must leave ≥12 μm gaps themselves.

**Name mapping `.plc` ↔ ODB:** The `.plc` protobuf uses flat indexing `…/macro_mem[K].i_ram`, but Yosys-flattened ODB uses `…/genblk1_G__i_ram.macro_mem_M` and the `K → (G, M)` mapping depends on the SRAM wrapper. Rather than hardcoding it, the generated TCL does **runtime two-pass matching**: pass 1 groups ODB instances by SRAM-block prefix; pass 2 sorts by `(genblk_idx, mem_idx)` and assigns linear `K` within each group. For Genus gate netlists (used to fix ariane133's missing-SRAM bug), `use_genus_names=True` uses the simpler `_plc_to_odb_name` rule (`'/'` preserved, `'.'`→`'_'`, `[N]`→`_N_`).

**Design-specific patches** (see `evaluate_benchmark` in `scripts/evaluate_with_orfs.py`):
- `ariane133`: uses pre-mapped Genus gate netlist (Yosys + `PRESERVE_CELLS` drops 89/133 SRAMs); patches missing `lzc_*` modules from `scripts/ariane133_lzc_patches.v`; reclassifies constant nets mistyped as POWER → SIGNAL via `PRE_GLOBAL_ROUTE_TCL`.
- `mempool_tile`: disables hierarchical flow, increases die to 2000×2000 (1272 IO pins need the room), opens all 4 die sides for pin placement.
- `ariane{133,136}`: reduces `MACRO_PLACE_HALO` to 5.0×5.0.
- `nvdla`: generates the ORFS config from scratch (no upstream collateral).
- All ASAP7: copies SRAM LEF/LIB from `Enablements/`.

The driver patches ORFS's `flow/scripts/macro_place_util.tcl` once to honor a `SKIP_RTLMP` env var (rtl_macro_placer crashes on already-placed macros in some OpenROAD versions). It also cleans stale `results/`, `logs/`, `objects/` subdirs so changed `DIE_AREA`/`CORE_AREA` take effect.

## Hard Submission Constraints (from rules)

These are enforced by validation or judging — don't suggest workarounds that violate them:

- **Zero hard-macro overlaps** — no tolerance. `validate_placement` checks bounding boxes; participants are told to add small gaps (~0.001 μm) to avoid float32 touching-edge false positives.
- **Klein-4 orientations only** (`N`, `FN`, `FS`, `S`) — no 90° rotations. The fakeram45 SRAMs aren't designed for rotation (pin access + internal metal direction assume fixed orientation). Orientations propagate to Tier 2 via optional `orientations.pt` sidecar.
- **Soft macro sizes are locked.** Sizing is a proxy-only abstraction; sizes get reset to initial `.plc` values on every `compute_proxy_cost` call. Don't resize.
- **1-hour wall-clock per benchmark.** Hard timeout.
- **No modifying the evaluation functions.** Don't suggest patches to `objective.py` for participants.
- **`--network none` at eval time.** Any `pip install` / `apt-get install` must happen at build time in a custom `Dockerfile`.

## Submission Interface

A submission is a `.py` file with a class exposing `place(benchmark) -> Tensor`:

```python
import torch
from macro_place.benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return placement  # [num_macros, 2] CENTER positions, hard then soft
```

The CLI loader (`macro_place/evaluate.py:_load_placer`) picks the first class defined in the file with a `place` attribute and instantiates with no args. Don't break this convention — many existing submissions rely on it.

Optional Tier-2 sidecar files next to `placer.py`:
- `orientations.pt`: a `[num_hard_macros]` int tensor of Klein-4 orientation codes.
- `Dockerfile`: if present, judges build it and run the placer inside, with the same `--network none` constraint.

## When Updating the Leaderboard

The README leaderboard is hand-edited via PRs (see `git log --grep="leaderboard"`). Each row: rank, team name in quotes, avg proxy in bold, best/worst (if known), overlaps, runtime, verified checkmark, notes. The `RePlAce` and `SA` baseline rows are pinned at proxy 1.4578 and 2.1251 respectively. Ordering is by avg proxy; verified scores supersede self-reported. When a team resubmits, update the existing row and note "Resubmitted M/D" — don't add a duplicate. Disqualified rows go at the bottom with `DQ` prefix.

## Useful Files When Reasoning About a Bug

- Cost mismatch / pin-position bug → `macro_place/objective.py:_set_placement` (pin update is easy to miss).
- Boundary index error → already patched by `_patched_get_grid_cell_location`; if a participant reports one, check whether they're calling `PlacementCost` directly (bypassing the patch).
- Tier-1 vs Tier-2 placement diff → the spacing enforcement in `scripts/generate_macro_placement_tcl.py:262-355` and the center-offset transform.
- ORFS macro not found → name mapping rules in `_plc_to_odb_name` and `_plc_extract_group_and_index`.
- Overlap reported on a placement that "looks" non-overlapping → float32 touching-edge; bump the legalization gap.
