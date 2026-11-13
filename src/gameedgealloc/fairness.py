"""Leader-side fairness pricing; no post-hoc clipping of follower choices."""
import numpy as np
from .models import jain_index


def fairness_multipliers(satisfaction_ratios, mu: float) -> np.ndarray:
    x=np.maximum(np.asarray(satisfaction_ratios,dtype=float),1e-12)
    if x.size==0: return x
    mean=float(np.mean(x))
    deviations=(x-mean)/max(mean,1e-12)
    return np.clip(1+mu*deviations,0.5,2.0)


def normalized_fairness_penalty(satisfaction_ratios) -> float:
    x=np.asarray(satisfaction_ratios,dtype=float)
    if len(x)==0:return 0.0
    delta=(x-x.mean())/max(float(x.mean()),1e-12)
    return float(np.mean(delta**2)/2)


def adapt_fairness_weight(mu, achieved_jain, minimum_jain, gain=2.0, max_mu=1.5):
    return float(np.clip(mu+gain*(minimum_jain-achieved_jain),0,max_mu))


__all__=['fairness_multipliers','normalized_fairness_penalty','adapt_fairness_weight','jain_index']
