"""Common-trace simulator with policy-specific per-user workload queues."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from .models import SystemConfig,Task
from .workload import Trace
from .metrics import slot_metrics


def evaluate_trace(allocator,trace:Trace,cfg:SystemConfig,save_allocations:str|Path|None=None,
                   save_history:str|Path|None=None)->pd.DataFrame:
    back_cpu:dict[int,float]={};back_data:dict[int,float]={};last_known:dict[int,Task]={}
    rows=[];alloc_rows=[];history_rows=[]
    for k,arrivals in enumerate(trace.slots):
        curr={t.user_id:t for t in arrivals}
        last_known.update(curr)
        # Queues are per-policy state, so a slow method does not quietly share
        # the fast method's backlog or its prior allocation decisions.
        if cfg.queue_enabled:
            ids=sorted(set(curr)|{i for i in back_cpu if back_cpu.get(i,0)>1e-8 or back_data.get(i,0)>1e-8})
            tasks=[]
            for i in ids:
                src=curr.get(i,last_known[i])
                d=(curr[i].data_mbit if i in curr else 0.)+back_data.get(i,0.)
                c=(curr[i].cycles_g if i in curr else 0.)+back_cpu.get(i,0.)
                if d>1e-8 or c>1e-8:
                    # Different task stages may be exhausted separately, so
                    # 1e-9 is a numerical nonzero placeholder, not new demand.
                    tasks.append(src.with_workload(d,c))
        else:tasks=list(arrivals)
        alloc=allocator.allocate(tasks)
        check=alloc.check(tasks,cfg)
        row=slot_metrics(tasks,alloc,cfg,k)
        row['active_arrivals']=len(arrivals)
        row['capacity_feasible']=int(check['capacity_feasible'])
        row['cpu_capacity_excess_ghz']=check['cpu_excess']
        row['bw_capacity_excess_mhz']=check['bw_excess']
        if cfg.queue_enabled:
            back_cpu={};back_data={}
            dropped_gcycles=0.;dropped_mbit=0.
            for i,t in enumerate(tasks):
                leftc=max(0,t.cycles_g-alloc.cpu[i]*cfg.slot_seconds)
                leftd=max(0,t.data_mbit-alloc.bandwidth[i]*t.spectral_eff*cfg.slot_seconds)
                dropped_gcycles+=max(0,leftc-cfg.queue_cap_gcycles)
                dropped_mbit+=max(0,leftd-cfg.queue_cap_mbit)
                back_cpu[t.user_id]=min(leftc,cfg.queue_cap_gcycles)
                back_data[t.user_id]=min(leftd,cfg.queue_cap_mbit)
            row['queue_gcycles']=float(sum(back_cpu.values()))
            row['queue_mbit']=float(sum(back_data.values()))
            row['queue_dropped_gcycles']=dropped_gcycles
            row['queue_dropped_mbit']=dropped_mbit
        else:
            row['queue_gcycles']=0.;row['queue_mbit']=0.
            row['queue_dropped_gcycles']=0.;row['queue_dropped_mbit']=0.
        rows.append(row)
        for i,t in enumerate(tasks):
            alloc_rows.append({'slot':k,'method':allocator.name,'user_id':t.user_id,
                               'data_mbit':t.data_mbit,'cycles_g':t.cycles_g,
                               'deadline_s':t.deadline_s,'spectral_eff':t.spectral_eff,
                               'power_w':t.power_w,'cpu_ghz':alloc.cpu[i],
                               'bandwidth_mhz':alloc.bandwidth[i],
                               'price_cpu':alloc.price_cpu[i] if len(alloc.price_cpu) else 0.,
                               'price_bw':alloc.price_bw[i] if len(alloc.price_bw) else 0.})
        history_rows.extend([{'slot':k,'method':allocator.name,**h} for h in alloc.history])
    if save_allocations:
        path=Path(save_allocations);path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(alloc_rows).to_csv(path,index=False)
    if save_history:
        path=Path(save_history);path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(history_rows).to_csv(path,index=False)
    return pd.DataFrame(rows)


def aggregate_results(df:pd.DataFrame,warmup:int=0)->pd.DataFrame:
    df=df[df.slot>=warmup].copy()
    if df.empty:return pd.DataFrame()
    metrics=['latency_s','latency_p95_s','energy_j','deadline_satisfaction',
             'jain','cpu_utilization','bw_utilization','mean_utility',
             'iterations','converged','capacity_feasible','capacity_violation',
             'signalling_bits','signalling_latency_s','solver_time_s','queue_gcycles',
             'queue_dropped_gcycles','queue_dropped_mbit']
    cols=[k for k in ('method','n_users','load','seed') if k in df]
    return df.groupby(cols,dropna=False)[metrics].mean().reset_index() if cols else df[metrics].mean().to_frame().T
