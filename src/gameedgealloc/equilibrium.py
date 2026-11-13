from __future__ import annotations
import time
import numpy as np
from .models import Task,Allocation,SystemConfig,latency,satisfaction,jain_index
from .follower import batch_best_responses
from .fairness import fairness_multipliers,adapt_fairness_weight
from .pricing import price_target,clearing_price,adaptive_step_size
from .online_learning import DemandLearner


class GameEdgeAllocator:
    name='gameedgealloc'
    def __init__(self,cfg:SystemConfig,seed:int=42):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed+4543)
        self.mu=cfg.initial_mu
        self.last_satisfaction:dict[int,float]={}
        self.last_prices=np.array([cfg.base_cpu_price,cfg.base_bandwidth_price],dtype=float)
        self.learner=DemandLearner(cfg)

    def allocate(self,tasks:list[Task]) -> Allocation:
        tic=time.perf_counter()
        n=len(tasks)
        if not n:return Allocation(np.zeros(0),np.zeros(0),method=self.name,solve_s=time.perf_counter()-tic)
        c=self.cfg
        if n*c.min_cpu_ghz>=c.cpu_capacity_ghz or n*c.min_bw_mhz>=c.bandwidth_capacity_mhz:
            raise ValueError('Infeasible minimum resource reservations for active users')
        price=self.last_prices.copy()
        mu=self.mu
        # Prior satisfaction is used only as a warm start; once the loop runs,
        # fairness multipliers are recomputed from contemporaneous responses.
        prior=np.array([self.last_satisfaction.get(t.user_id,1.) for t in tasks])
        multipliers=fairness_multipliers(prior,mu)
        if c.enable_learning and c.enable_warm_start:
            price=self.learner.warm_prices(tasks,multipliers,price)
        price=np.maximum(price,[c.base_cpu_price,c.base_bandwidth_price])
        corrupt=np.ones((2,n))
        if c.feedback_corruption>0:
            hit=self.rng.random(n)<c.feedback_corruption
            corrupt[:,hit]=np.exp(self.rng.choice([-1.,1.],size=(2,hit.sum()))*
                                      self.rng.uniform(1.,2.5,size=(2,hit.sum())))
        if c.feedback_noise>0:
            corrupt*=np.exp(self.rng.normal(0,c.feedback_noise,size=(2,n)))
        last_f=np.zeros(n);last_b=np.zeros(n)
        history=[]
        count_filtered=0
        converged=False
        base=np.array([c.base_cpu_price,c.base_bandwidth_price])
        caps=np.array([c.cpu_capacity_ghz,c.bandwidth_capacity_mhz])
        kp=np.array([c.kappa_cpu,c.kappa_bw]); eta=np.array([c.eta_cpu,c.eta_bw])
        for iteration in range(1,c.max_iter+1):
            pf=price[0]*multipliers;pb=price[1]*multipliers
            f,b,_=batch_best_responses(tasks,pf,pb,c)
            rf=f*corrupt[0];rb=b*corrupt[1]
            if c.enable_filtering and c.enable_learning:
                rf,rb,filtered=self.learner.filter_reports(tasks,rf,rb,pf,pb)
                count_filtered+=filtered
            rho=np.array([np.sum(rf),np.sum(rb)])/caps
            targets=np.array([price_target(base[j],rho[j],kp[j],eta[j]) for j in range(2)])
            if c.capacity_clearing and c.deadline_mode=='soft':
                floors=np.array([clearing_price(tasks,multipliers,caps[0],base[0],c,'cpu'),
                                 clearing_price(tasks,multipliers,caps[1],base[1],c,'bw')])
                targets=np.maximum(targets,floors)
            if c.adaptive_step:
                step=min(adaptive_step_size(mu,kp[j],eta[j],rho[j],base[j],price[j]) for j in range(2))
            else:
                step=c.step_size
            new_price=(1-step)*price+step*targets
            # Compute fairness at the *candidate* leader decision.
            nf,nb,_=batch_best_responses(tasks,new_price[0]*multipliers,new_price[1]*multipliers,c)
            sat=satisfaction(tasks,nf,nb)
            new_m=fairness_multipliers(sat,mu)
            # Damping the nonlinear fairness feedback improves robustness.
            new_m=np.clip(0.55*multipliers+0.45*new_m,0.5,2.0)
            relative=float(np.max(np.abs(new_price-price)/np.maximum(price,1e-12)))
            mult_gap=float(np.max(np.abs(new_m-multipliers)))
            request_gap=max(float(np.max(np.abs(nf-last_f)/np.maximum(np.abs(nf),.05))),
                            float(np.max(np.abs(nb-last_b)/np.maximum(np.abs(nb),.05)))) if iteration>1 else 1.
            history.append({'iteration':iteration,'price_cpu':float(new_price[0]),
                            'price_bw':float(new_price[1]),'requested_cpu':float(np.sum(nf)),
                            'requested_bw':float(np.sum(nb)), 'jain':jain_index(sat),
                            'price_change':relative,'fairness_change':mult_gap})
            price,multipliers=new_price,new_m
            last_f,last_b=nf,nb
            if max(relative,mult_gap,request_gap)<c.tolerance:
                converged=True
                break
        # A final scarcity calculation uses *actual* follower curves, never
        # distorted reports, and ensures feasible resources under soft mode.
        if c.capacity_clearing and c.deadline_mode=='soft':
            pf_floor=clearing_price(tasks,multipliers,caps[0],base[0],c,'cpu')
            pb_floor=clearing_price(tasks,multipliers,caps[1],base[1],c,'bw')
            newprice=np.maximum(price,[pf_floor,pb_floor])
            if np.any(newprice>price*(1+max(c.tolerance,1e-6))):
                converged=False # a final correction is not called equilibrium
            price=newprice
        f,b,_=batch_best_responses(tasks,price[0]*multipliers,price[1]*multipliers,c)
        sat=satisfaction(tasks,f,b)
        prices_f=price[0]*multipliers;prices_b=price[1]*multipliers
        rf,rb=f*corrupt[0],b*corrupt[1]
        if c.enable_filtering and c.enable_learning:
            rf,rb,changed=self.learner.filter_reports(tasks,rf,rb,prices_f,prices_b)
            count_filtered+=changed
        if c.enable_learning:
            self.learner.update(tasks,rf,rb,prices_f,prices_b)
        self.last_prices=price.copy()
        self.last_satisfaction.update({t.user_id:float(x) for t,x in zip(tasks,sat)})
        if c.adaptive_fairness:self.mu=adapt_fairness_weight(mu,jain_index(sat),c.target_jain,c.mu_step,c.mu_max)
        # This residual is the implemented (price + scarcity floor) map, not
        # the residual of a proved revenue-maximizing leader objective.
        rfho=np.array([np.sum(rf),np.sum(rb)])/caps
        targ=np.array([price_target(base[j],rfho[j],kp[j],eta[j]) for j in range(2)])
        if c.capacity_clearing and c.deadline_mode=='soft':
            targ=np.maximum(targ,[clearing_price(tasks,multipliers,caps[0],base[0],c,'cpu'),
                                  clearing_price(tasks,multipliers,caps[1],base[1],c,'bw')])
        residual=float(np.max(np.abs(price-targ)/np.maximum(targ,1e-12)))
        converged=bool(converged and residual<max(c.tolerance*3,1e-6))
        return Allocation(f,b,price_cpu=prices_f,price_bw=prices_b,iterations=iteration,
                          converged=converged,price_residual=residual,method=self.name,
                          history=history,solve_s=time.perf_counter()-tic,
                          filtered_report_count=count_filtered,
                          objective_note='Capacity-clearing price extension; no general revenue-optimality claim' if c.capacity_clearing else 'Manuscript fixed-point mapping; may be capacity-infeasible')
