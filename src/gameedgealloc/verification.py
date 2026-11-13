"""Independent numerical checks for analytical conditions in Sec. 4.9.

The checks report observed finite-difference slopes/monotonicity and Hessian
signs. They are numerical checks, not deductive proofs of equilibrium existence
or of the paper's leader optimality assertions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .models import Task,SystemConfig
from .follower import cpu_best_response,bandwidth_best_response
from .pricing import price_target,contraction_bound


def verify_followers_and_prices(tasks:list[Task],cfg:SystemConfig)->dict:
    if not tasks:raise ValueError('Verification requires active users')
    p0=cfg.base_cpu_price
    kappa=cfg.kappa_cpu;eta=cfg.eta_cpu
    def response(p):
        return np.array([cpu_best_response(t,p,cfg.dvfs_zeta) for t in tasks])
    def target(p):
        return price_target(p0,float(response(p).sum())/cfg.cpu_capacity_ghz,kappa,eta)
    grid=np.geomspace(p0,p0*40,60)
    demand=np.array([response(p).sum() for p in grid])
    T=np.array([target(p) for p in grid])
    elasticities=[];hessians=[]
    for t in tasks:
        f=cpu_best_response(t,p0,cfg.dvfs_zeta)
        b=bandwidth_best_response(t,cfg.base_bandwidth_price)
        hessian_cpu=-2*t.beta*t.cycles_g/f**3-2*t.gamma*cfg.dvfs_zeta*t.cycles_g
        hessian_bw=-2*(t.beta+t.gamma*t.power_w)*t.data_mbit/(t.spectral_eff*b**3)
        hessians.append([hessian_cpu,hessian_bw])
        hp=p0*1e-4
        dp=(cpu_best_response(t,p0+hp,cfg.dvfs_zeta)-
            cpu_best_response(t,p0-hp,cfg.dvfs_zeta))/(2*hp)
        elasticities.append(-p0*dp/f)
    slopes=np.diff(T)/np.diff(grid)
    rho0=float(demand[0]/cfg.cpu_capacity_ghz)
    Lbound=contraction_bound(cfg.initial_mu,kappa,eta,rho0)
    return {
        'n_users':len(tasks),
        'negative_hessian_diagonals_all':bool((np.asarray(hessians)<0).all()),
        'demand_monotone_decreasing':bool((np.diff(demand)<0).all()),
        'price_map_monotone_decreasing':bool((np.diff(T)<=1e-9).all()),
        'max_cpu_price_elasticity':float(np.max(elasticities)),
        'elasticity_half_bound_respected':bool(np.max(elasticities)<=.5001),
        'empirical_max_abs_price_slope':float(np.max(np.abs(slopes))),
        'conservative_L_bound':float(Lbound),
        'slope_within_conservative_bound':bool(np.max(np.abs(slopes))<=Lbound+1e-6),
        'global_sufficient_condition_L_below_one':bool(Lbound<1.),
        'warning':'Finite-difference audit for uniform-price/soft-response only; fairness coupling and hard deadlines are excluded.'}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--users',type=int,default=20)
    p.add_argument('--load',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='results/equilibrium_numerical_check.json')
    a=p.parse_args(argv)
    from .workload import WorkloadConfig,generate_trace
    tasks=generate_trace(WorkloadConfig(n_users=a.users,slots=1,offered_load=a.load,seed=a.seed)).slots[0]
    result=verify_followers_and_prices(tasks,SystemConfig())
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
