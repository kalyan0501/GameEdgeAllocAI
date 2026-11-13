"""Task model, physical equations, allocators' common IO and unit system.

Internal units: D in Mbit, b in MHz, C in billion CPU cycles (Gcycle),
f in GHz, deadlines and latency in seconds, transmission power in watts,
energy in joules. Since Mbit/MHz = second and Gcycle/GHz = second, the
manuscript's latency equations retain their exact form in these units.
DVFS zeta is the numerical coefficient in J/(Gcycle*GHz**2).
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Task:
    user_id: int
    data_mbit: float
    cycles_g: float
    deadline_s: float
    spectral_eff: float
    power_w: float
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    arrivals: int = 1
    profile: str = "mixed"

    def __post_init__(self):
        for name in ("data_mbit", "cycles_g", "deadline_s", "spectral_eff", "power_w", "beta", "gamma", "delta"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive")
        if self.user_id < 0:
            raise ValueError("user_id must be non-negative")

    def with_workload(self, data_mbit: float, cycles_g: float) -> 'Task':
        return replace(self, data_mbit=max(1e-9, data_mbit), cycles_g=max(1e-9, cycles_g))


@dataclass
class SystemConfig:
    cpu_capacity_ghz: float = 24.0
    bandwidth_capacity_mhz: float = 35.0
    slot_seconds: float = 1.0
    dvfs_zeta: float = 0.07
    base_cpu_price: float = 0.11
    base_bandwidth_price: float = 0.11
    kappa_cpu: float = 0.5
    kappa_bw: float = 0.5
    eta_cpu: float = 0.3
    eta_bw: float = 0.3
    initial_mu: float = 0.3
    mu_max: float = 1.5
    target_jain: float = 0.90
    mu_step: float = 2.0
    forgetting: float = 0.20
    feedback_clip: float = 2.5
    feedback_corruption: float = 0.0
    feedback_noise: float = 0.0
    min_cpu_ghz: float = 0.015
    min_bw_mhz: float = 0.015
    max_iter: int = 200
    tolerance: float = 1e-5
    step_size: float = 0.65
    adaptive_step: bool = True
    adaptive_fairness: bool = True
    enable_learning: bool = True
    enable_filtering: bool = True
    enable_warm_start: bool = True
    capacity_clearing: bool = True
    deadline_mode: str = "soft"  # soft (theorem) or hard (coupled constrained BR)
    control_bits: int = 32
    downlink_bps: float = 2e6
    uplink_bps: float = 2e6
    queue_enabled: bool = True
    queue_cap_gcycles: float = 30.0
    queue_cap_mbit: float = 25.0

    def __post_init__(self):
        for key in ("cpu_capacity_ghz", "bandwidth_capacity_mhz", "slot_seconds", "base_cpu_price", "base_bandwidth_price"):
            if getattr(self, key) <= 0: raise ValueError(f"{key} must be positive")
        if self.deadline_mode not in ("soft", "hard"):
            raise ValueError("deadline_mode must be soft or hard")
        if self.max_iter < 1 or not 0 < self.step_size <= 1:
            raise ValueError("max_iter >= 1 and step_size in (0,1] are required")
        if self.mu_max < 0 or not (0 < self.target_jain <= 1):
            raise ValueError("Invalid fairness limits")
        if not (0 <= self.feedback_corruption <= 1):
            raise ValueError("feedback_corruption must be within [0,1]")

    @classmethod
    def from_dict(cls, config: dict | None = None) -> 'SystemConfig':
        return cls(**(config or {}))


def latency(tasks: Sequence[Task], f: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not tasks: return np.empty(0)
    data = np.array([t.data_mbit for t in tasks])
    cyc = np.array([t.cycles_g for t in tasks])
    s = np.array([t.spectral_eff for t in tasks])
    return data / np.maximum(b * s, 1e-12) + cyc / np.maximum(f, 1e-12)


def energy(tasks: Sequence[Task], f: np.ndarray, b: np.ndarray, zeta: float) -> np.ndarray:
    if not tasks: return np.empty(0)
    data = np.array([t.data_mbit for t in tasks]); cyc = np.array([t.cycles_g for t in tasks])
    s = np.array([t.spectral_eff for t in tasks]); p = np.array([t.power_w for t in tasks])
    return p * data / np.maximum(b * s, 1e-12) + zeta * cyc * f**2


def utility(tasks: Sequence[Task], f: np.ndarray, b: np.ndarray, zeta: float,
            pf: np.ndarray | float = 0.0, pb: np.ndarray | float = 0.0) -> np.ndarray:
    ell = latency(tasks, f, b); en = energy(tasks, f, b, zeta)
    return np.array([t.alpha for t in tasks]) - np.array([t.beta for t in tasks])*ell - np.array([t.gamma for t in tasks])*en - np.array([t.delta for t in tasks])*(pf*f + pb*b)


def jain_index(x: np.ndarray) -> float:
    """Jain's index of the nonnegative deadline-satisfaction ratios x_i=tau_i/L_i."""
    if not len(x): return 1.0
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    denom = len(x)*float(np.sum(x*x))
    return float(np.sum(x)**2 / denom) if denom > 1e-18 else 1.0


def satisfaction(tasks: Sequence[Task], f: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([t.deadline_s for t in tasks]) / np.maximum(latency(tasks, f, b), 1e-12)


@dataclass
class Allocation:
    cpu: np.ndarray
    bandwidth: np.ndarray
    price_cpu: np.ndarray = field(default_factory=lambda: np.empty(0))
    price_bw: np.ndarray = field(default_factory=lambda: np.empty(0))
    iterations: int = 1
    converged: bool = True
    method: str = "unknown"
    price_residual: float = 0.0
    objective_note: str = ""
    history: list[dict] = field(default_factory=list)
    solve_s: float = 0.0
    filtered_report_count: int = 0

    def check(self, tasks: Sequence[Task], cfg: SystemConfig) -> dict:
        assert len(self.cpu)==len(self.bandwidth)==len(tasks)
        assert np.all(np.isfinite(self.cpu)) and np.all(np.isfinite(self.bandwidth))
        assert np.all(self.cpu >= -1e-12) and np.all(self.bandwidth >= -1e-12)
        return {
            "cpu_excess": float(max(0, np.sum(self.cpu)-cfg.cpu_capacity_ghz)),
            "bw_excess": float(max(0, np.sum(self.bandwidth)-cfg.bandwidth_capacity_mhz)),
            "capacity_feasible": bool(np.sum(self.cpu)<=cfg.cpu_capacity_ghz+1e-6 and np.sum(self.bandwidth)<=cfg.bandwidth_capacity_mhz+1e-6),
        }
