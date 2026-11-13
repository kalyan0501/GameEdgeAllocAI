"""Reproducible, heterogeneous Poisson MEC arrivals and slot-channel dynamics."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import numpy as np
from .models import Task


PROFILES={
 'latency':{'alpha':1.0,'beta':1.8,'gamma':0.4,'delta':0.9,'deadline_range':(.9,1.6)},
 'energy':{'alpha':1.0,'beta':0.8,'gamma':1.8,'delta':1.0,'deadline_range':(1.4,3.2)},
 'cost':{'alpha':1.0,'beta':0.8,'gamma':0.7,'delta':1.8,'deadline_range':(1.7,4.0)},
 'mixed':{'alpha':1.0,'beta':1.2,'gamma':1.0,'delta':1.1,'deadline_range':(1.1,2.6)},
}


@dataclass(frozen=True)
class WorkloadConfig:
    n_users:int=10
    slots:int=30
    offered_load:float=1.0
    arrival_rate:float=0.65
    channel_mode:str='varying' # fixed / varying / poor
    data_mean_mbit:float=0.60
    cycles_mean_g:float=0.60
    variability:float=0.35
    seed:int=42


@dataclass
class Trace:
    config:WorkloadConfig
    slots:list[list[Task]]

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8') as h:
            h.write(json.dumps({'type':'config',**asdict(self.config)})+'\n')
            for i,ts in enumerate(self.slots):
                h.write(json.dumps({'type':'slot','slot':i,'tasks':[asdict(t) for t in ts]})+'\n')

    @staticmethod
    def load(path):
        with open(path,encoding='utf-8') as h:
            cfg=json.loads(next(h));cfg.pop('type')
            slots=[]
            for row in h:
                d=json.loads(row)
                if d['type']=='slot':slots.append([Task(**t) for t in d['tasks']])
        return Trace(WorkloadConfig(**cfg),slots)


def generate_trace(cfg:WorkloadConfig)->Trace:
    if cfg.n_users<1 or cfg.slots<1 or cfg.offered_load<0:raise ValueError('Bad workload configuration')
    rng=np.random.default_rng(cfg.seed)
    profile_names=list(PROFILES)
    assigned=[profile_names[i%len(profile_names)] for i in rng.permutation(cfg.n_users)]
    out=[]
    # Fixed individual/channel parameters; channels fluctuate between slots.
    powers=rng.uniform(.09,.25,cfg.n_users)
    channel_base=rng.uniform(1.7,3.2,cfg.n_users)
    dline=[rng.uniform(*PROFILES[p]['deadline_range']) for p in assigned]
    for slot in range(cfg.slots):
        tasks=[]
        for i in range(cfg.n_users):
            arrivals=int(rng.poisson(cfg.arrival_rate*cfg.offered_load))
            if not arrivals:continue
            profile=assigned[i];spec=PROFILES[profile]
            # Multiplicative heterogeneity avoids negative task demands.
            d=float(rng.lognormal(np.log(cfg.data_mean_mbit),cfg.variability,size=arrivals).sum())
            cycles=float(rng.lognormal(np.log(cfg.cycles_mean_g),cfg.variability,size=arrivals).sum())
            s=channel_base[i]
            if cfg.channel_mode=='varying':s*=rng.lognormal(0.,.20)
            if cfg.channel_mode=='poor':s*=.50*rng.lognormal(0.,.25)
            s=float(np.clip(s,.35,6.0))
            tasks.append(Task(i,d,cycles,dline[i],s,powers[i],
                              alpha=spec['alpha'],beta=spec['beta'],
                              gamma=spec['gamma'],delta=spec['delta'],
                              profile=profile,arrivals=arrivals))
        out.append(tasks)
    return Trace(cfg,out)
