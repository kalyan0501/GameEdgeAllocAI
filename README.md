# GameEdgeAllocAI: Complete Research Implementation

A Stackelberg Game-Based Resource Allocation Framework for Fair and Energy-Efficient Multi-User Mobile Edge Computing

Runnable Python MEC simulator; analytical best-response optimisation; original dynamic-pricing mode; capacity-clearing pricing extension; personalised fairness pricing; EWRE online demand learning; robust feedback filtering; fairness-target adaptation; DDPG and parameter-shared multi-agent DRL training/evaluation; six classical comparators; controlled ablations and sensitivities; statistics; 600-dpi figures; NS-3 Wi-Fi packet-level replay source; an automated test suite.

## 1. Installation

Python 3.10+ is required. Run these commands from the extracted `GameEdgeAllocAI/` directory:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[rl,test]'
python -m pytest -q
```

If you only need the analytical/classical algorithms and not the DRL baselines, install `pip install -e '.[test]'`. CUDA is optional. DDPG/MADRL training automatically selects a CUDA device when available; all other modules run on CPU. All Python imports and CLI commands also work with `PYTHONPATH=src` before installation.

### Immediate demonstration

```bash
python -m gameedgealloc.cli demo --users 10 --slots 15 --output results/demo
```

This runs the seven *non-DRL* methods on a common trace and exports `slot_metrics.csv`, per-method `allocations_*.csv` and equilibrium-iteration traces. The `gameedgealloc` terminal command works after installation (`gameedgealloc demo ...`).

## 2. Full research workflow

```bash
# Complete quick pipeline, including DDPG and MADRL training and verification:
python -m gameedgealloc.cli experiment \
  --config configs/smoke.yaml --output results/smoke

# Publication-scale configuration (resource-intensive; edit to suit hardware):
python -m gameedgealloc.cli experiment \
  --config configs/study.yaml --output results/study

# Train a comparator independently:
python -m gameedgealloc.cli train-rl --algo ddpg --users 25 \
  --load 1.5 --seed 42 --steps 5000 --device auto --output results/ddpg_25.pt
python -m gameedgealloc.cli train-rl --algo madrl --users 25 \
  --load 1.5 --seed 42 --steps 5000 --device auto --output results/madrl_25.pt

# Run controlled ablations:
python -m gameedgealloc.ablations --mode ablation \
  --config configs/ablation.yaml --output results/ablations

# Run parameter sensitivity analysis:
python -m gameedgealloc.ablations --mode sensitivity \
  --config configs/ablation.yaml --output results/sensitivity

# Generate high-resolution figures from an existing run:
python -m gameedgealloc.cli plot \
  --summary results/smoke/seed_summary.csv --output results/smoke/figures
```

The comprehensive `study.yaml` has 5 user counts, 4 loads, 10 independent seeds, 1,000 slots/seed, two independently trained DRL methods and nine algorithms. **It is a configurable experimental plan, not a claim that this entire grid has been executed.** Reduce the matrix for a quick run. Ten independent seeds are already specified in the study configuration.


## 3. What experiment files mean

* `slot_metrics.csv`: raw per-slot outcomes; includes method, seed, users, load and convergence/feasibility flags.
* `seed_summary.csv`: within-seed means after warm-up. **Paired comparisons must be across seeds**, not by treating dependent time slots as independent replications.
* `descriptive_statistics.csv`: across-seed mean, SD and two-sided 95% Student-t CI (CI unavailable for only one seed).
* `paired_tests_holm.csv`: paired t and Wilcoxon tests, effect size `dz`, Holm corrections **within each scenario and metric**. No fabricated small p-values.
* `manifest.json`: executed config, dependencies, OS, versions, seeds and figure paths.
* `rl_training_log.csv` and `checkpoints/*.pt`: per-scenario DRL train metadata and trained actors.
* `figures/*.png`: 600-dpi figures generated from observed CSV results, not manually entered bars.
* If `save_trace: true`, the exact common task/channel trace is saved in JSON Lines.
* If `save_detailed: true`, per-user allocations, personalised prices and iteration histories are saved. An NS-3-compatible CSV is one of these allocation files.

**Caveat on graphs:** no figure contains presumed 25% latency, 15–20% energy or p<0.001 results. They reflect only executed experiments, which may confirm or contradict the draft.

## 4. Reproducibility requirements

* Validate your installation: `python -m pytest -q`.
* Compare all algorithms on **the same** generated trace per workload/seed. Each allocator starts with independent queue and learned-price state.
* Report ten seed-level replications where the paper calls for ten, plus 95% CI, paired t/Wilcoxon, paired effect sizes and Holm correction.
* Use `deadline_mode: soft` for direct tests of the contraction theorem; hard deadline response can be discontinuous and needs separate treatment.
* Do not claim that centralised welfare optimality implies best latency or lowest energy separately.
* Change throughput and workload units only in one place (`models.py` and `workload.py`), document all physical calibration data, and rerun tests.
* Do not treat modelled signalling latency as a measured NS-3 control-plane round trip.

## 6. Code structure

```
src/gameedgealloc/
  models.py         # units, physical metrics, common config/data structures
  workload.py       # heterogeneous Poisson slot/task/channel generator
  follower.py       # exact follower soft/hard best responses
  pricing.py        # congestion and scarcity price functions
  fairness.py       # Jain index, leader-side fairness, adaptation
  equilibrium.py    # price/fairness/learning coupled iterative control
  online_learning.py# online per-UE inverse-demand coefficient learner
  baselines.py      # six classical baselines, allocation factory
  rl.py             # DDPG/MADRL PyTorch train and inference
  simulator.py      # common-trace comparative and queue simulation
  metrics.py        # end-to-end analytical metrics and signalling
  statistics.py     # confidence intervals, paired tests, Holm
  plotting.py       # experimental plot generation (600 dpi)
  experiments.py    # experiment orchestration and manifests
  ablations.py      # ablations and parameter sensitivity
  cli.py            # command line
simulation/ns3/     # NS-3 source, shell driver and explanatory README
configs/            # smoke, full study, ablation presets
tests/              # mathematical, integration and RL verification
```

