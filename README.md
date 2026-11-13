# GameEdgeAllocAI: Complete Research Implementation

**Title:** GameEdgeAllocAI: A Stackelberg Game-Based Resource Allocation Framework for Fair and Energy-Efficient Multi-User Mobile Edge Computing

**Contents:** runnable Python MEC simulator; analytical best-response optimisation; original dynamic-pricing mode; capacity-clearing pricing extension; personalised fairness pricing; EWRE online demand learning; robust feedback filtering; fairness-target adaptation; DDPG and parameter-shared multi-agent DRL training/evaluation; six classical comparators; controlled ablations and sensitivities; statistics; 600-dpi figures; NS-3 Wi-Fi packet-level replay source; an automated test suite. All simulated results are generated from the supplied code and **no previously claimed journal result is hardcoded**.

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

## 3. Mathematical model and exact module mapping

| Manuscript specification | Source file | Implementation |
|---|---|---|
| Eqs. (1)–(3d) | `models.py`, `simulator.py`, `workload.py` | Tasks, aggregate CPU/bandwidth bounds, latency, energy and optional queue state |
| Eqs. (4), (18) | `models.py` | Latency/energy/cost user utility |
| Eqs. (23a), (24a) | `follower.py` | CPU first-order root; closed-form bandwidth demand; optional hard deadline KKT dual |
| Eqs. (21), (22), (34), (35) | `pricing.py`, `equilibrium.py` | Congestion-adjusted leader fixed point, damping and optional scarcity price |
| Eqs. (6), (30), (30a) | `fairness.py` | Deadline-satisfaction Jain index, normalised deviation and personalised price |
| Eqs. (38)–(41) | `online_learning.py`, `equilibrium.py` | EWRE, robust filtering, bisection warm start, adaptive fairness/step |
| Eqs. (42)–(44) | `metrics.py` | Quantised-message overhead and estimated signalling latency |
| Section 5.2 | `baselines.py`, `rl.py` | Six classical and two trained neural comparators |
| Sections 5.3–5.12 (referenced but not fully supplied) | `experiments.py`, `statistics.py`, `plotting.py`, `ablations.py` | Configurable new comparisons, paired multi-seed evaluation, tests, sensitivity |

### Units and non-arbitrary dimensional choices

* `D`: **Mbit**, `b`: **MHz**. Therefore `D/(b*s)` is seconds.
* `C`: **Gcycle** (10^9 cycles), `f`: **GHz**, so `C/f` is seconds.
* `P`: W. `P*D/(b*s)` gives joules.
* `zeta`: **J/(Gcycle·GHz²)**. The `dvfs_zeta=0.07` default is **an assumed simulation coefficient**, not a measured hardware switched-capacitance value.
* Base prices are currency-like **utility units per GHz and MHz**, not externally calibrated market prices. The user utility weights are dimensionless simulation preferences with compatible weighting units.

The manuscript leaves capacities, slot length, per-device distributions, communication rates and many RL hyperparameters unspecified in the supplied pages. Defaults in `models.py` and `workload.py` are **explicit, independently chosen simulation assumptions**. Edit them for empirical replication before manuscript claims.

### Model fidelity and limitations that must not be concealed

1. **Follower optimality:** `follower.py` solves the unique optimum of the continuous, individual objective at announced prices. The `soft` setting uses continuous best responses without a hard deadline restriction, aligning with the manuscript's contraction assumptions. `hard` performs KKT dual optimisation for each user's deadline; collective deadline feasibility is not guaranteed under congestion.
2. **Leader-side equilibrium:** the paper's revenue objective (Eq. 5), fixed-point pricing (Eq. 34), and social-welfare KKT theorem (Theorem 1) are not identical objectives. We compute and record the **price fixed-point residual**, not a proven global leader-optimality certificate. Do not describe it as a formally verified Stackelberg equilibrium without reconciling the leader objective.
3. **Capacity correctness:** `capacity_clearing: true` adds **shadow/scarcity price floors** to ensure unconstrained (`soft`) follower allocations can collectively meet capacity limits while preserving their best-response character. This is a **labeled extension** to manuscript Eq. 34. To run the manuscript's pure fixed-point rule and observe any excess capacity directly, use `capacity_clearing: false`. Hard deadlines can make overall feasibility impossible, so the hard mode reports violations rather than hiding them.
4. **Fairness:** the leader uses normalised per-user deadline-satisfaction ratios `x_i=tau_i/L_i` to set clipped personalised CPU and bandwidth prices. Its multiplier is updated in a coupled fixed-point loop. **It is not an after-the-fact resource projection**. Convexity of the surrogate in the nonlinear resource variables is *not proven by the implementation*; the paper's claimed KKT equivalence requires mathematical scrutiny.
5. **Demand learning:** EWRE learns only from past observed demands and prices. The CPU inverse-price model is an approximation when DVFS coefficient `zeta>0`, as admitted in the manuscript. Corruption filtering affects feedback reports/pricing and demand-model updates. Physical allocations are calculated from actual follower decisions; safety clearing is applied on actual curves.
6. **Queue approximation:** when enabled, per-policy input-data and work-cycle backlogs are carried into the next slot and served at separate allocated rates. This is a *fluid-slot approximation*; individual per-task deadlines are not carried forward with age, and the simulator does not resolve pipeline timing in packet granularity. It is appropriate for controlled algorithm tests, **not a full packet-level end-to-end MEC trace**.
7. **NS-3:** see `simulation/ns3/README.md`. It is an independently executable Wi-Fi packet simulator for **offline** replay of the Python allocation trace, not a real-time shared-memory interface. It measures packet delay, packet deliveries and offered-rate shaping. It does not assert strict Wi-Fi radio bandwidth partitioning, or whole-task completion latency from individual packets.
8. **Baseline DRL:** DDPG uses a global deterministic actor + critic, replay buffer and target networks; MADRL uses shared per-device stochastic actors with central value and cooperative reward. These are functioning, documented implementations of those method families, not exact replicas of unprovided neural architectures or hyperparameters in [4] and [35]. Evaluation uses separate random sources for training and held-out load traces.
9. **Manuscript coverage:** the supplied 24-page file ends in Section 5.2. Claims about Tables 6, 8, 12, 17, Section 5.12 and specific reported latency/energy gains cannot be reproduced byte-for-byte without missing setup values and results.

## 4. What experiment files mean

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

## 5. Reproducibility requirements

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

## 7. Optional NS-3 network-level validation

Please see `simulation/ns3/README.md` for commands. ns-3 must be installed independently (it is not provided as a pip package in this ZIP). The C++ source uses ns-3 tutorial Wi-Fi, UDP OnOff, PacketSink and FlowMonitor primitives. It was **not compiled in the packaging environment**, which does not include ns-3 libraries. Do not label network-level results as completed until it is compiled and run on a working ns-3 installation.

Official NS-3 tutorial: https://www2.nsnam.org/docs/tutorial/singlehtml/index.html

---

This is a complete executable **research reference implementation** with an optional NS-3 source integration. It is not a claim of exact numerical reproduction of the manuscript, nor a proof of all the paper's disputed game-theoretic propositions. Any full SCI reproducibility claim must distinguish verified Python outputs, independently executed NS-3 outputs, and theoretical results that still require proof.

### Numerical theorem audit

To inspect the continuous best-response Hessian signs, monotonicity, maximum finite-difference elasticity, and the conservative contraction bound in Section 4.9, run:

```bash
python -m gameedgealloc.verification --users 20 --load 1.0 \
  --output results/equilibrium_numerical_check.json
```
