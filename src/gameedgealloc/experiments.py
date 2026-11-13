"""Manifested, paired-seed research experiments and optional baseline training."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import platform,json,sys,datetime
import numpy as np
import pandas as pd
from .models import SystemConfig
from .workload import WorkloadConfig,generate_trace
from .baselines import make_allocator,ALLOCATORS
from .simulator import evaluate_trace,aggregate_results
from .statistics import paired_comparisons,descriptive_summary
from .plotting import plot_all,plot_iteration_history


def _json_dump(path,data):
    with Path(path).open('w',encoding='utf8') as h:json.dump(data,h,indent=2,default=str)


def run_experiments(config:dict,output_dir:str|Path,generate_plots=True)->dict:
    dst=Path(output_dir);dst.mkdir(parents=True,exist_ok=True)
    cfg=SystemConfig.from_dict(config.get('system',{}))
    settings=config.get('experiment',{})
    nusers=settings.get('user_counts',[10])
    loads=settings.get('loads',[1.])
    seeds=settings.get('seeds',[42])
    slots=int(settings.get('slots',25))
    warmup=int(settings.get('warmup',min(5,slots//5)))
    methods=settings.get('methods',list(ALLOCATORS))
    rl_steps=int(settings.get('rl_train_steps',500))
    rl_device=settings.get('rl_device','auto')
    save_trace=bool(settings.get('save_trace',False))
    save_detailed=bool(settings.get('save_detailed',False))
    results=[];summaries=[];train_log=[]
    for n in nusers:
        for load in loads:
            for seed in seeds:
                wcfg=WorkloadConfig(n_users=int(n),slots=slots,offered_load=float(load),
                     seed=int(seed),**config.get('workload',{}))
                trace=generate_trace(wcfg)
                tag=f'n{n}_load{float(load):.2f}_seed{seed}'
                if save_trace:trace.save(dst/f'trace_{tag}.jsonl')
                for method in methods:
                    if method in ('ddpg','madrl'):
                        from .rl import RLAllocator,train_algorithm
                        checkpoint=dst/'checkpoints'/f'{method}_{tag}.pt'
                        train_log.append({'method':method,'n_users':n,'load':load,'seed':seed,
                              **train_algorithm(method,cfg,int(n),float(load),int(seed),rl_steps,checkpoint,device=rl_device)})
                        allocator=RLAllocator(cfg,checkpoint)
                    else:allocator=make_allocator(method,cfg,seed=int(seed))
                    alloc_path=dst/'details'/f'allocations_{method}_{tag}.csv' if save_detailed else None
                    hist_path=dst/'details'/f'history_{method}_{tag}.csv' if save_detailed and method in ('gameedgealloc','uniform_stackelberg') else None
                    frame=evaluate_trace(allocator,trace,cfg,alloc_path,hist_path)
                    frame['seed']=seed;frame['n_users']=n;frame['load']=float(load)
                    results.append(frame)
                    summaries.append(aggregate_results(frame,warmup))
                    print(f'{tag:>24} {method:<21} latency={frame.latency_s.mean():.4f} fairness={frame.jain.mean():.4f} capacity={frame.capacity_feasible.mean():.3f}',flush=True)
                    if hist_path and Path(hist_path).exists() and method=='gameedgealloc':
                        plot_iteration_history(hist_path,dst/'details'/f'price_iteration_{tag}.png')
    slots_frame=pd.concat(results,ignore_index=True)
    summary=pd.concat(summaries,ignore_index=True)
    slots_frame.to_csv(dst/'slot_metrics.csv',index=False)
    summary.to_csv(dst/'seed_summary.csv',index=False)
    descript=descriptive_summary(summary)
    descript.to_csv(dst/'descriptive_statistics.csv',index=False)
    paired=paired_comparisons(summary)
    paired.to_csv(dst/'paired_tests_holm.csv',index=False)
    if train_log:pd.DataFrame(train_log).to_csv(dst/'rl_training_log.csv',index=False)
    figures=plot_all(summary,dst/'figures') if generate_plots else []
    versions={'python':sys.version,'platform':platform.platform(),
              'numpy':np.__version__,'pandas':pd.__version__}
    try:
        import torch
        versions['torch']=torch.__version__
    except ImportError:pass
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'config':config,'software':versions,
              'methods':methods,'traces_are_paired_by_seed':True,
              'warmup_slots_excluded_from_summary':warmup,
              'figures':[str(p.relative_to(dst)) for p in figures],
              'interpretation':'Simulation outcomes; no manuscript performance claims are assumed.'}
    _json_dump(dst/'manifest.json',manifest)
    return {'results_dir':str(dst),'rows':len(slots_frame),'seed_summaries':len(summary),
            'figures':[str(p) for p in figures]}
