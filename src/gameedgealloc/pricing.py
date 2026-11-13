"""Damped utilization and congestion prices, with optional scarcity clearing."""
import numpy as np
from scipy.optimize import brentq
from .follower import cpu_best_response, bandwidth_best_response
from .models import SystemConfig


def price_target(price0: float, utilization: float, kappa: float, eta: float) -> float:
    """Eq. (34) consistent normalisation: p0[1+k*rho+eta*(rho-1)_+]."""
    return price0*(1+kappa*max(0.,utilization)+eta*max(utilization-1.,0.))


def contraction_bound(mu: float, kappa: float, eta: float, rho0: float) -> float:
    return (1+mu)*(kappa+eta)*rho0/2.0


def adaptive_step_size(mu,kappa,eta,utilization,base_price,price):
    local_bound=(1+mu)*(kappa+eta)*max(utilization,0)*base_price/(2*max(price,1e-12))
    return float(np.clip(min(1.,1.6/(1+local_bound)),0.05,1.0))


def clearing_price(tasks, multipliers, capacity, price_floor, cfg:SystemConfig,
                   dimension='cpu'):
    """Find minimal common price above floor for feasible uncoupled demands.

    Used only for deadline_mode=soft. This is a documented capacity-clearing
    modification, *not* an identity implied by the manuscript's Eq. (34).
    """
    if len(tasks)==0:return float(price_floor)
    if capacity <= len(tasks)*(cfg.min_cpu_ghz if dimension=='cpu' else cfg.min_bw_mhz):
        raise ValueError('Capacity insufficient for minimum per-user reservations')
    def demand(p):
        if dimension=='cpu':
            return sum(cpu_best_response(t,p*m,cfg.dvfs_zeta,cfg.min_cpu_ghz,cfg.cpu_capacity_ghz)
                       for t,m in zip(tasks,multipliers))
        return sum(bandwidth_best_response(t,p*m,cfg.min_bw_mhz,cfg.bandwidth_capacity_mhz)
                   for t,m in zip(tasks,multipliers))
    lo=float(price_floor)
    if demand(lo)<=capacity:return lo
    hi=lo*2
    for _ in range(100):
        if demand(hi)<=capacity: break
        hi*=2
    else:
        raise RuntimeError('Could not find capacity-clearing price: check minima')
    return float(brentq(lambda p:demand(p)-capacity,lo,hi,xtol=1e-11))
