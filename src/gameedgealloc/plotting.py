"""Plot directly from recorded results; never hardcode method performance."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=[('latency_s','Mean latency (s)'),('energy_j','Mean energy (J)'),
         ('jain','Jain fairness index'),('deadline_satisfaction','Deadline satisfaction')]


def plot_all(seed_summary:pd.DataFrame,output_dir:str|Path,dpi:int=600):
    dst=Path(output_dir);dst.mkdir(parents=True,exist_ok=True)
    if seed_summary.empty:return []
    paths=[]
    for n_users in sorted(seed_summary.n_users.unique()):
        df=seed_summary[seed_summary.n_users==n_users]
        loads=sorted(df.load.unique())
        fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        methods=sorted(df.method.unique())
        for ax,(field,label) in zip(axes.ravel(),METRICS):
            for method in methods:
                subset=df[df.method==method]
                means=subset.groupby('load')[field].mean().reindex(loads)
                std=subset.groupby('load')[field].std().reindex(loads).fillna(0)
                ax.errorbar(loads,means,yerr=std,marker='o',linewidth=1.6,capsize=2,label=method)
            ax.set(xlabel='Offered load multiplier',ylabel=label)
            ax.grid(alpha=.25)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.02),ncol=min(4,len(methods)),fontsize=9)
        path=dst/f'main_metrics_users_{n_users}.png';fig.savefig(path,dpi=dpi,bbox_inches='tight');plt.close(fig);paths.append(path)
    if 'iterations' in seed_summary and 'signalling_bits' in seed_summary:
        for n_users in sorted(seed_summary.n_users.unique()):
            df=seed_summary[(seed_summary.n_users==n_users)&(seed_summary.method.isin(['gameedgealloc','uniform_stackelberg']))]
            if df.empty:continue
            fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
            for method,grp in df.groupby('method'):
                for ax,key,yl in zip(axes,['iterations','signalling_bits'],['Iterations per slot','Control signalling (bits/slot)']):
                    mean=grp.groupby('load')[key].mean()
                    ax.plot(mean.index,mean.values,marker='o',label=method)
                    ax.set(xlabel='Offered load',ylabel=yl);ax.grid(alpha=.25)
            axes[0].legend(fontsize=8)
            p=dst/f'convergence_signalling_users_{n_users}.png';fig.savefig(p,dpi=dpi);plt.close(fig);paths.append(p)
    return paths


def plot_iteration_history(csv_path,output_png,dpi=600):
    df=pd.read_csv(csv_path)
    if df.empty:return
    df=df[df.slot==df.slot.min()]
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot(df.iteration,df.price_cpu,label='CPU price')
    axes[0].plot(df.iteration,df.price_bw,label='Bandwidth price')
    axes[0].set(xlabel='Iteration',ylabel='Common resource price');axes[0].legend()
    axes[1].semilogy(df.iteration,np.maximum(df.price_change,1e-12),label='Price residual')
    axes[1].semilogy(df.iteration,np.maximum(df.fairness_change,1e-12),label='Fairness residual')
    axes[1].set(xlabel='Iteration',ylabel='Residual');axes[1].legend()
    for ax in axes:ax.grid(alpha=.3)
    fig.savefig(output_png,dpi=dpi);plt.close(fig)
