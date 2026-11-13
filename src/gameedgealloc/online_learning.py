"""Online demand identification, corrupted-report filtering, and warm-starting."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .models import SystemConfig, Task
from .pricing import price_target


class DemandLearner:
    """EWRE from reported choices only: the leader does not read private beta/gamma/delta."""
    def __init__(self,cfg:SystemConfig):
        self.cfg=cfg
        self.theta_cpu:dict[int,float]={}
        self.theta_bw:dict[int,float]={}

    def predict(self,tasks:list[Task],prices_cpu:np.ndarray,prices_bw:np.ndarray):
        f=np.full(len(tasks),np.nan)
        b=np.full(len(tasks),np.nan)
        for i,t in enumerate(tasks):
            if t.user_id in self.theta_cpu:
                f[i]=np.sqrt(max(self.theta_cpu[t.user_id]*t.cycles_g/max(prices_cpu[i],1e-12),1e-20))
            if t.user_id in self.theta_bw:
                b[i]=np.sqrt(max(self.theta_bw[t.user_id]*t.data_mbit/
                                 (t.spectral_eff*max(prices_bw[i],1e-12)),1e-20))
        return f,b

    def filter_reports(self,tasks:list[Task],reported_f,reported_b,prices_cpu,prices_bw):
        """Per-user ±2.5 prediction band, then robust log-residual MAD winsorisation.

        Only estimates with prior observations are filtered. Raw report sums
        and reliable-filtered sums can be compared in the corruption ablation.
        """
        reported_f=np.asarray(reported_f,dtype=float).copy()
        reported_b=np.asarray(reported_b,dtype=float).copy()
        predicted_f,predicted_b=self.predict(tasks,prices_cpu,prices_bw)
        changed=0
        for reports,pred in ((reported_f,predicted_f),(reported_b,predicted_b)):
            good=np.isfinite(pred)
            if not np.any(good):continue
            prev=reports.copy()
            reports[good]=np.clip(reports[good],pred[good]/self.cfg.feedback_clip,
                                  pred[good]*self.cfg.feedback_clip)
            resid=np.log(np.maximum(reports[good],1e-12)/np.maximum(pred[good],1e-12))
            median=float(np.median(resid))
            mad=float(np.median(np.abs(resid-median)))
            width=max(3*1.4826*mad,0.20)
            reports[good]=pred[good]*np.exp(np.clip(resid,median-width,median+width))
            changed+=int(np.count_nonzero(np.abs(prev-reports)>1e-10))
        return reported_f,reported_b,changed

    def update(self,tasks:list[Task],reports_f,reports_b,prices_cpu,prices_bw):
        omega=self.cfg.forgetting
        for i,t in enumerate(tasks):
            theta_f=max(float(reports_f[i]**2*prices_cpu[i]/t.cycles_g),1e-12)
            theta_b=max(float(reports_b[i]**2*prices_bw[i]*t.spectral_eff/t.data_mbit),1e-12)
            if t.user_id in self.theta_cpu:
                self.theta_cpu[t.user_id]=(1-omega)*self.theta_cpu[t.user_id]+omega*theta_f
                self.theta_bw[t.user_id]=(1-omega)*self.theta_bw[t.user_id]+omega*theta_b
            else:
                self.theta_cpu[t.user_id]=theta_f
                self.theta_bw[t.user_id]=theta_b

    def warm_prices(self,tasks:list[Task],multipliers,previous_prices):
        """Solve predicted leader fixed points via scalar bisection, when possible."""
        if not tasks or not all(t.user_id in self.theta_cpu for t in tasks):
            return np.asarray(previous_prices,dtype=float).copy()
        out=[]
        for dimension in ('cpu','bw'):
            cpu=dimension=='cpu'
            p0=self.cfg.base_cpu_price if cpu else self.cfg.base_bandwidth_price
            cap=self.cfg.cpu_capacity_ghz if cpu else self.cfg.bandwidth_capacity_mhz
            k=self.cfg.kappa_cpu if cpu else self.cfg.kappa_bw
            eta=self.cfg.eta_cpu if cpu else self.cfg.eta_bw
            def rho(p):
                tot=0.
                for t,m in zip(tasks,multipliers):
                    theta=(self.theta_cpu if cpu else self.theta_bw)[t.user_id]
                    feature=t.cycles_g if cpu else t.data_mbit/t.spectral_eff
                    tot+=np.sqrt(theta*feature/max(p*m,1e-12))
                return tot/cap
            def g(p):return p-price_target(p0,rho(p),k,eta)
            lo=p0;hi=max(p0*2,previous_prices[0 if cpu else 1]*2)
            while g(hi)<0 and hi<1e9:hi*=2
            out.append(float(brentq(g,lo,hi)) if g(lo)<=0 else lo)
        return np.asarray(out,dtype=float)
