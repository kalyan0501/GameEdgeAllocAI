"""Strictly concave follower optimisation; explicit soft/hard deadline modes."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .models import Task, SystemConfig


def cpu_best_response(t: Task, price: float, zeta: float, minimum: float=0.0,
                      maximum: float=np.inf, latency_multiplier: float=0.0) -> float:
    """Unique root: (beta+lambda)*C/f^2-2gamma*zeta*C*f=delta*price.

    lambda is a nonnegative multiplier on L<=deadline in hard deadline mode.
    """
    if price < 0 or latency_multiplier < 0 or zeta < 0: raise ValueError("Negative price/multiplier/zeta")
    a = (t.beta+latency_multiplier)*t.cycles_g
    c = 2.0*t.gamma*zeta*t.cycles_g
    q = t.delta*price
    if c == 0:
        raw = np.sqrt(a/max(q, 1e-300))
    else:
        fn = lambda x: a/(x*x) - c*x - q
        high = max(1.0, np.cbrt(a/c) + 1.0)
        raw = brentq(fn, 1e-12, high, xtol=1e-13)
    return float(np.clip(raw, minimum, maximum))


def bandwidth_best_response(t: Task, price: float, minimum: float=0.0,
                            maximum: float=np.inf, latency_multiplier: float=0.0) -> float:
    if price <= 0: return float(maximum) if np.isfinite(maximum) else np.inf
    v = (t.beta+latency_multiplier+t.gamma*t.power_w)*t.data_mbit/(t.delta*t.spectral_eff*price)
    return float(np.clip(np.sqrt(v), minimum, maximum))


def follower_response(t: Task, price_cpu: float, price_bw: float,
                      cfg: SystemConfig) -> tuple[float, float, bool]:
    """Returns exact utility-maximizing individual demand at announced prices.

    soft: continuous interior best response (the contraction-theory regime).
    hard: convex deadline-constrained optimum by KKT dual bisection, with
    individual resource maxima used to detect even per-user infeasibility.
    Shared capacity feasibility is *not* guaranteed by individual responses.
    """
    if min(price_cpu, price_bw) <= 0: raise ValueError("Prices must be positive")
    fcap,bcap=cfg.cpu_capacity_ghz,cfg.bandwidth_capacity_mhz
    def resp(lam):
        f=cpu_best_response(t,price_cpu,cfg.dvfs_zeta,cfg.min_cpu_ghz,fcap,lam)
        b=bandwidth_best_response(t,price_bw,cfg.min_bw_mhz,bcap,lam)
        return f,b
    f,b=resp(0.0)
    def laten(f,b): return t.data_mbit/(b*t.spectral_eff)+t.cycles_g/f
    if cfg.deadline_mode=="soft" or laten(f,b)<=t.deadline_s:
        return f,b,bool(laten(f,b)<=t.deadline_s)
    if laten(fcap,bcap)>t.deadline_s:
        return fcap,bcap,False
    hi=1.0
    while laten(*resp(hi))>t.deadline_s and hi<1e12: hi*=2
    lam=brentq(lambda v:laten(*resp(v))-t.deadline_s,0.0,hi,xtol=1e-9)
    f,b=resp(lam)
    return f,b,bool(laten(f,b)<=t.deadline_s+1e-6)


def batch_best_responses(tasks, pf, pb, cfg):
    pfs=np.broadcast_to(np.asarray(pf,dtype=float),(len(tasks),))
    pbs=np.broadcast_to(np.asarray(pb,dtype=float),(len(tasks),))
    ans=[follower_response(t,float(fp),float(bp),cfg) for t,fp,bp in zip(tasks,pfs,pbs)]
    if not ans:return np.empty(0),np.empty(0),np.empty(0,dtype=bool)
    return np.array([a[0] for a in ans]),np.array([a[1] for a in ans]),np.array([a[2] for a in ans])
