"""Evaluate actual allocated resources; never substitute requested amounts."""
import numpy as np
from .models import SystemConfig,Allocation,latency,energy,utility,jain_index,satisfaction


def slot_metrics(tasks,allocation:Allocation,cfg:SystemConfig,slot:int=0)->dict:
    n=len(tasks)
    record={'slot':slot,'method':allocation.method,'n_active':n,
            'iterations':allocation.iterations,'converged':int(allocation.converged),
            'price_residual':allocation.price_residual,'solver_time_s':allocation.solve_s,
            'filtered_reports':allocation.filtered_report_count}
    q=cfg.control_bits
    record['signalling_bits']=int(allocation.iterations*q*(2+3*n)) if n and allocation.method in ('gameedgealloc','uniform_stackelberg') else 0
    rate_delay=q*(2+n)/cfg.downlink_bps+2*q*n/cfg.uplink_bps
    record['signalling_latency_s']=(allocation.iterations*rate_delay if n and allocation.method in ('gameedgealloc','uniform_stackelberg') else 0.)+allocation.solve_s
    if n==0:
        record.update({k:0. for k in ('cpu_utilization','bw_utilization','mean_cost','capacity_violation')})
        record.update({k:np.nan for k in ('latency_s','latency_p95_s','energy_j','deadline_satisfaction','mean_utility','avg_satisfaction_ratio')})
        record['jain']=np.nan
        return record
    ell=latency(tasks,allocation.cpu,allocation.bandwidth)
    en=energy(tasks,allocation.cpu,allocation.bandwidth,cfg.dvfs_zeta)
    x=satisfaction(tasks,allocation.cpu,allocation.bandwidth)
    pf=allocation.price_cpu if len(allocation.price_cpu)==n else np.zeros(n)
    pb=allocation.price_bw if len(allocation.price_bw)==n else np.zeros(n)
    record.update({'latency_s':float(np.mean(ell)),
                   'latency_p95_s':float(np.percentile(ell,95)),
                   'energy_j':float(np.mean(en)),
                   'deadline_satisfaction':float(np.mean(ell<=np.array([t.deadline_s for t in tasks]))),
                   'cpu_utilization':float(np.sum(allocation.cpu)/cfg.cpu_capacity_ghz),
                   'bw_utilization':float(np.sum(allocation.bandwidth)/cfg.bandwidth_capacity_mhz),
                   'jain':jain_index(x),'avg_satisfaction_ratio':float(np.mean(x)),
                   'mean_utility':float(np.mean(utility(tasks,allocation.cpu,allocation.bandwidth,cfg.dvfs_zeta,pf,pb))),
                   'mean_cost':float(np.mean(pf*allocation.cpu+pb*allocation.bandwidth)),
                   'capacity_violation':float(max(0,np.sum(allocation.cpu)/cfg.cpu_capacity_ghz-1)+max(0,np.sum(allocation.bandwidth)/cfg.bandwidth_capacity_mhz-1))})
    return record
