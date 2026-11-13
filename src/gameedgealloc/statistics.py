"""Paired seed-level inference; confidence intervals, paired tests and Holm."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


LOWER_BETTER={'latency_s','energy_j','latency_p95_s','iterations','signalling_bits','solver_time_s','signalling_latency_s'}


def confidence_interval(values,confidence=.95):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan
    mean=float(np.mean(values))
    if len(values)<2:return np.nan,np.nan
    if np.std(values,ddof=1)<1e-13:return mean,mean
    sem=stats.sem(values)
    return tuple(float(z) for z in stats.t.interval(confidence,len(values)-1,loc=mean,scale=sem))


def holm_adjust(pvalues):
    """Strong FWER control; nan values remain nan."""
    arr=np.asarray(pvalues,dtype=float);out=np.full(len(arr),np.nan)
    valid=np.flatnonzero(np.isfinite(arr))
    order=valid[np.argsort(arr[valid])]
    prev=0.0;m=len(order)
    for rank,idx in enumerate(order):
        prev=max(prev,min(1.,arr[idx]*(m-rank)))
        out[idx]=prev
    return out


def paired_comparisons(summary:pd.DataFrame,reference='gameedgealloc',metrics=None):
    if metrics is None:metrics=['latency_s','energy_j','jain','deadline_satisfaction','iterations']
    rows=[]
    grouping=[k for k in ('n_users','load') if k in summary]
    grouper=summary.groupby(grouping,dropna=False) if grouping else [((),summary)]
    for key,part in grouper:
        ref=part[part.method==reference].set_index('seed')
        for method in sorted(set(part.method)-{reference}):
            comp=part[part.method==method].set_index('seed')
            for metric in metrics:
                if metric not in ref or metric not in comp:continue
                aligned=ref[[metric]].join(comp[[metric]],lsuffix='_ref',rsuffix='_comparison',how='inner').dropna()
                dif=(aligned[metric+'_ref']-aligned[metric+'_comparison']).to_numpy(dtype=float)
                n=len(dif)
                ci_lo,ci_hi=confidence_interval(dif)
                if n>=2 and not np.allclose(dif,0) and np.std(dif,ddof=1)>1e-13:
                    tp=float(stats.ttest_1samp(dif,0).pvalue)
                    try: wp=float(stats.wilcoxon(dif).pvalue)
                    except ValueError: wp=np.nan
                    dz=float(np.mean(dif)/np.std(dif,ddof=1)) if np.std(dif,ddof=1)>0 else np.nan
                elif n>=2 and np.allclose(dif,0):
                    tp=1.;wp=1.;dz=0.
                else:tp=wp=dz=np.nan
                meta={}
                if grouping:
                    keys=key if isinstance(key,tuple) else (key,)
                    meta=dict(zip(grouping,keys))
                rows.append({**meta,'reference':reference,'comparator':method,'metric':metric,
                             'n_pairs':n,'difference_mean':float(np.mean(dif)) if n else np.nan,
                             'difference_ci_low':ci_lo,'difference_ci_high':ci_hi,
                             'paired_t_p':tp,'wilcoxon_p':wp,'paired_effect_dz':dz,
                             'reference_better':bool(np.mean(dif)<0) if metric in LOWER_BETTER and n else
                             (bool(np.mean(dif)>0) if n else False)})
    df=pd.DataFrame(rows)
    if len(df):
        # Family of comparisons is defined per scenario and outcome metric.
        cols=grouping+['metric']
        for _,idx in df.groupby(cols).groups.items():
            df.loc[idx,'paired_t_p_holm']=holm_adjust(df.loc[idx,'paired_t_p'])
            df.loc[idx,'wilcoxon_p_holm']=holm_adjust(df.loc[idx,'wilcoxon_p'])
    return df


def descriptive_summary(summary:pd.DataFrame,metrics=None):
    if metrics is None:
        metrics=['latency_s','energy_j','jain','deadline_satisfaction',
                 'iterations','converged','capacity_feasible','solver_time_s','signalling_bits']
    out=[]
    keys=[k for k in ('n_users','load','method') if k in summary]
    for label,group in summary.groupby(keys,dropna=False):
        key=label if isinstance(label,tuple) else (label,)
        for metric in metrics:
            values=group[metric].to_numpy(float)
            ci=confidence_interval(values)
            out.append({**dict(zip(keys,key)),'metric':metric,'n':len(values),
                        'mean':float(np.mean(values)),
                        'sd':float(np.std(values,ddof=1)) if len(values)>1 else np.nan,
                        'ci_low':ci[0],'ci_high':ci[1]})
    return pd.DataFrame(out)
