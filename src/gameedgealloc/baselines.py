"""Six executable non-RL comparator allocators under the same capacity model."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import brentq
from .models import Allocation,Task,SystemConfig
from .follower import batch_best_responses,cpu_best_response,bandwidth_best_response
from .pricing import clearing_price
from .equilibrium import GameEdgeAllocator


def _split(weights,capacity,minimum):
    weights=np.maximum(np.asarray(weights,dtype=float),1e-12)
    n=len(weights)
    if not n:return np.zeros(0)
    if n*minimum>capacity:raise ValueError('Minimum resource demand exceeds capacity')
    return minimum+(capacity-n*minimum)*weights/weights.sum()


class SimpleAllocator:
    name='simple'
    def __init__(self,cfg:SystemConfig,seed:int=42):self.cfg=cfg;self.rng=np.random.default_rng(seed)
    def allocate(self,tasks:list[Task])->Allocation:
        tic=time.perf_counter();n=len(tasks)
        if not n:return Allocation(np.zeros(0),np.zeros(0),method=self.name)
        f,b=self._allocate(tasks)
        return Allocation(np.asarray(f),np.asarray(b),method=self.name,
                          solve_s=time.perf_counter()-tic)


class EqualAllocator(SimpleAllocator):
    name='equal'
    def _allocate(self,tasks):
        return np.full(len(tasks),self.cfg.cpu_capacity_ghz/len(tasks)),np.full(len(tasks),self.cfg.bandwidth_capacity_mhz/len(tasks))


class ProportionalAllocator(SimpleAllocator):
    name='proportional'
    def _allocate(self,tasks):
        c=np.array([t.cycles_g/t.deadline_s for t in tasks])
        b=np.array([t.data_mbit/(t.spectral_eff*t.deadline_s) for t in tasks])
        return _split(c,self.cfg.cpu_capacity_ghz,self.cfg.min_cpu_ghz),_split(b,self.cfg.bandwidth_capacity_mhz,self.cfg.min_bw_mhz)


class GreedyAllocator(SimpleAllocator):
    name='greedy'
    def _allocate(self,tasks):
        # Deadline-priority greedy; a small reservation prevents starvation.
        n=len(tasks);f=np.full(n,self.cfg.min_cpu_ghz);b=np.full(n,self.cfg.min_bw_mhz)
        urgency=np.array([(t.cycles_g/self.cfg.cpu_capacity_ghz+
                           t.data_mbit/(t.spectral_eff*self.cfg.bandwidth_capacity_mhz))/t.deadline_s
                          for t in tasks])
        order=np.argsort(-urgency)
        # Greedy capacity goes to urgent users, but per-user allocations are
        # capped to avoid a single unbounded demand monopolising the server.
        rem_f=self.cfg.cpu_capacity_ghz-f.sum();rem_b=self.cfg.bandwidth_capacity_mhz-b.sum()
        wants_f=np.array([max(t.cycles_g/t.deadline_s,cfg_min) for t,cfg_min in zip(tasks,f)])*1.6
        wants_b=np.array([max(t.data_mbit/(t.deadline_s*t.spectral_eff),cfg_min) for t,cfg_min in zip(tasks,b)])*1.6
        for i in order:
            step=min(rem_f,wants_f[i]);f[i]+=step;rem_f-=step
            step=min(rem_b,wants_b[i]);b[i]+=step;rem_b-=step
        if rem_f>0:f+=rem_f/n
        if rem_b>0:b+=rem_b/n
        return f,b


class SelfishAllocator(SimpleAllocator):
    name='selfish'
    def _allocate(self,tasks):
        c=self.cfg;n=len(tasks)
        f,b,_=batch_best_responses(tasks,np.full(n,c.base_cpu_price*.4),np.full(n,c.base_bandwidth_price*.4),c)
        # Noncooperative requests are admitted in queue order; unserved
        # users retain minimum reserves. This is a congestion/rationing model.
        f0=np.full(n,c.min_cpu_ghz);b0=np.full(n,c.min_bw_mhz)
        ff=c.cpu_capacity_ghz-f0.sum();bb=c.bandwidth_capacity_mhz-b0.sum()
        for i in range(n):
            x=min(ff,max(0,f[i]-f0[i]));ff-=x;f0[i]+=x
            y=min(bb,max(0,b[i]-b0[i]));bb-=y;b0[i]+=y
        return f0,b0


class CentralizedAllocator(SimpleAllocator):
    name='centralized'
    def _allocate(self,tasks):
        """Solve strictly concave social welfare via separable KKT shadow prices.

        W = sum[alpha/delta - (beta/delta)*L - (gamma/delta)*E].
        Deadline mode soft is exact for this W and total capacities.
        This is not a universal upper bound for Jain fairness or deadline QoS.
        """
        c=self.cfg;n=len(tasks)
        def aggregate(p,dim):
            if dim=='cpu':
                return np.array([cpu_best_response(t,p,c.dvfs_zeta,c.min_cpu_ghz,c.cpu_capacity_ghz)
                                 for t in tasks])
            return np.array([bandwidth_best_response(t,p,c.min_bw_mhz,c.bandwidth_capacity_mhz)
                             for t in tasks])
        def price_for(dim,cap):
            lo=1e-11
            if aggregate(lo,dim).sum()<=cap:return lo
            hi=1.
            while aggregate(hi,dim).sum()>cap and hi<1e12:hi*=2
            return brentq(lambda p:aggregate(p,dim).sum()-cap,lo,hi)
        f=aggregate(price_for('cpu',c.cpu_capacity_ghz),'cpu')
        b=aggregate(price_for('bw',c.bandwidth_capacity_mhz),'bw')
        return f,b


class UniformStackelbergAllocator:
    name='uniform_stackelberg'
    def __init__(self,cfg:SystemConfig,seed:int=42):
        from dataclasses import replace
        cc=replace(cfg,initial_mu=0,mu_max=0,adaptive_fairness=False,
                   enable_learning=False,enable_filtering=False,enable_warm_start=False,
                   kappa_cpu=0.5,kappa_bw=0.5,eta_cpu=0,eta_bw=0)
        self.inner=GameEdgeAllocator(cc,seed)
    def allocate(self,tasks):
        ans=self.inner.allocate(tasks)
        ans.method=self.name
        return ans


ALLOCATORS={'equal':EqualAllocator,'proportional':ProportionalAllocator,
            'greedy':GreedyAllocator,'selfish':SelfishAllocator,
            'centralized':CentralizedAllocator,'uniform_stackelberg':UniformStackelbergAllocator,
            'gameedgealloc':GameEdgeAllocator}


def make_allocator(name:str,cfg:SystemConfig,seed:int=42):
    if name not in ALLOCATORS:raise ValueError(f'Unknown allocator {name}; available: {list(ALLOCATORS)}')
    return ALLOCATORS[name](cfg,seed)
