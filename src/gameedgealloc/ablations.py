"""Controlled ablation and parameter sensitivity sweeps."""
from __future__ import annotations
from dataclasses import replace
import argparse
from pathlib import Path
import yaml
import pandas as pd
from .models import SystemConfig
from .workload import WorkloadConfig,generate_trace
from .equilibrium import GameEdgeAllocator
from .simulator import evaluate_trace,aggregate_results
from .statistics import descriptive_summary,paired_comparisons


def ablation_configs(cfg:SystemConfig):
    return {
        'Full':cfg,
        'No fairness':replace(cfg,initial_mu=0,adaptive_fairness=False),
        'No congestion':replace(cfg,eta_cpu=0,eta_bw=0),
        'No online learning':replace(cfg,enable_learning=False,enable_warm_start=False,enable_filtering=False),
        'No warm start':replace(cfg,enable_warm_start=False),
        'No filtering':replace(cfg,enable_filtering=False),
        'Fixed fairness weight':replace(cfg,adaptive_fairness=False),
        'Fixed step size':replace(cfg,adaptive_step=False),
        'Paper fixed point (no clearing)':replace(cfg,capacity_clearing=False),
    }


def run_ablations(config:dict,output_dir:str|Path):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    settings=config.get('experiment',{})
    n=int(settings.get('users',25));load=float(settings.get('load',1.5))
    seeds=settings.get('seeds',[42,123,2024,2025,3407])
    slots=int(settings.get('slots',100));warmup=int(settings.get('warmup',10))
    cfg=SystemConfig.from_dict(config.get('system',{}))
    frames=[];summary=[]
    for seed in seeds:
        trace=generate_trace(WorkloadConfig(n_users=n,slots=slots,offered_load=load,seed=seed))
        for name,variant in ablation_configs(cfg).items():
            model=GameEdgeAllocator(variant,seed=seed)
            df=evaluate_trace(model,trace,variant)
            df['method']=name;df['seed']=seed;df['n_users']=n;df['load']=load
            frames.append(df);summary.append(aggregate_results(df,warmup))
    joined=pd.concat(frames,ignore_index=True)
    seeds_df=pd.concat(summary,ignore_index=True)
    joined.to_csv(out/'ablation_slots.csv',index=False)
    seeds_df.to_csv(out/'ablation_seed_summary.csv',index=False)
    descriptive_summary(seeds_df).to_csv(out/'ablation_descriptive.csv',index=False)
    paired_comparisons(seeds_df,reference='Full').to_csv(out/'ablation_holm.csv',index=False)
    return seeds_df


def run_sensitivity(config:dict,output_dir:str|Path):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=SystemConfig.from_dict(config.get('system',{}))
    settings=config.get('experiment',{})
    n=int(settings.get('users',25));load=float(settings.get('load',1.5))
    slots=int(settings.get('slots',100));warmup=int(settings.get('warmup',10))
    seeds=settings.get('seeds',[42,123,2024,2025,3407])
    params={'kappa_cpu':[.2,.5,.8,1.0], 'eta_cpu':[0,.3,.6,.9],
            'initial_mu':[0,.3,.7,1.2], 'target_jain':[.8,.9,.95,.98],
            'feedback_corruption':[0,.10,.25,.40]}
    records=[]
    for seed in seeds:
        trace=generate_trace(WorkloadConfig(n_users=n,slots=slots,offered_load=load,seed=seed))
        for field,values in params.items():
            for v in values:
                overrides={field:v}
                if field=='kappa_cpu':overrides['kappa_bw']=v
                if field=='eta_cpu':overrides['eta_bw']=v
                if field=='initial_mu':overrides['adaptive_fairness']=False
                variant=replace(cfg,**overrides)
                df=evaluate_trace(GameEdgeAllocator(variant,seed),trace,variant)
                df=df[df.slot>=warmup]
                row={key:df[key].mean() for key in ['latency_s','energy_j','jain',
                       'deadline_satisfaction','iterations','converged','capacity_feasible']}
                records.append({'parameter':field,'value':v,'seed':seed,**row})
    table=pd.DataFrame(records)
    table.to_csv(out/'sensitivity_seed_results.csv',index=False)
    plot_sensitivity(table,out/'sensitivity.png')
    return table


def plot_sensitivity(table,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=list(table.parameter.unique())
    fig,axes=plt.subplots(len(groups),3,figsize=(13,3*len(groups)),constrained_layout=True)
    for i,param in enumerate(groups):
        group=table[table.parameter==param].groupby('value').mean(numeric_only=True)
        for j,key in enumerate(['latency_s','jain','iterations']):
            axes[i,j].plot(group.index,group[key],marker='o')
            axes[i,j].set(xlabel=param,ylabel=key);axes[i,j].grid(alpha=.2)
    fig.savefig(path,dpi=600);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ablation.yaml')
    p.add_argument('--output',default='results/ablations')
    p.add_argument('--mode',choices=['ablation','sensitivity'],default='ablation')
    args=p.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text())
    if args.mode=='ablation':print(run_ablations(config,args.output).to_string(index=False))
    else:print(run_sensitivity(config,args.output).head().to_string(index=False))

if __name__=='__main__':main()
