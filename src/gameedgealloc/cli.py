"""Command-line entry point: demo, experiment, train-rl, plot, ns3-report."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import yaml
from .models import SystemConfig
from .experiments import run_experiments
from .workload import WorkloadConfig,generate_trace
from .simulator import evaluate_trace
from .baselines import make_allocator
from .plotting import plot_all


def main(argv=None):
    parser=argparse.ArgumentParser(description='GameEdgeAllocAI reproducible MEC resource allocation')
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('demo',help='Quick deterministic all-classical-method demonstration')
    a.add_argument('--users',type=int,default=10);a.add_argument('--slots',type=int,default=12)
    a.add_argument('--seed',type=int,default=42);a.add_argument('--load',type=float,default=1.)
    a.add_argument('--output',default='results/demo');a.add_argument('--method',default='all')
    e=sub.add_parser('experiment',help='Run configured multi-seed comparisons and export figures')
    e.add_argument('--config',default='configs/smoke.yaml');e.add_argument('--output',default='results/experiment')
    e.add_argument('--no-plots',action='store_true')
    t=sub.add_parser('train-rl',help='Train DDPG or multi-agent DRL and save usable checkpoint')
    t.add_argument('--algo',choices=['ddpg','madrl'],required=True)
    t.add_argument('--users',type=int,default=10);t.add_argument('--load',type=float,default=1.)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--steps',type=int,default=1000)
    t.add_argument('--output',default='results/agent.pt')
    t.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p=sub.add_parser('plot',help='Create 600-DPI plots from a seed_summary.csv')
    p.add_argument('--summary',required=True);p.add_argument('--output',default='results/figures')
    s=sub.add_parser('ns3-report',help='Summarize NS-3 flow monitor CSV and preserve original data')
    s.add_argument('--input',required=True);s.add_argument('--output',default='results/ns3_flow_summary.csv')
    args=parser.parse_args(argv)
    if args.command=='demo':
        c=SystemConfig();tr=generate_trace(WorkloadConfig(n_users=args.users,slots=args.slots,seed=args.seed,offered_load=args.load))
        methods=list(__import__('gameedgealloc.baselines',fromlist=['ALLOCATORS']).ALLOCATORS) if args.method=='all' else [args.method]
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        results=[]
        for method in methods:
            path=out/f'allocations_{method}.csv'
            hist=out/f'iterations_{method}.csv' if method=='gameedgealloc' else None
            frame=evaluate_trace(make_allocator(method,c,args.seed),tr,c,path,hist)
            frame['method']=method;results.append(frame)
        df=pd.concat(results,ignore_index=True);df.to_csv(out/'slot_metrics.csv',index=False)
        print(df.groupby('method')[['latency_s','energy_j','jain','deadline_satisfaction','capacity_feasible','iterations']].mean().round(5).to_string())
        print('Demo output:',out.resolve())
    elif args.command=='experiment':
        config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        print(json.dumps(run_experiments(config,args.output,not args.no_plots),indent=2))
    elif args.command=='train-rl':
        from .rl import train_algorithm
        print(json.dumps(train_algorithm(args.algo,SystemConfig(),args.users,args.load,args.seed,args.steps,args.output,device=args.device),indent=2))
    elif args.command=='plot':
        print('\n'.join(map(str,plot_all(pd.read_csv(args.summary),args.output))))
    elif args.command=='ns3-report':
        df=pd.read_csv(args.input)
        if not {'user_id','tx_packets','rx_packets','rx_bytes','mean_delay_s'}.issubset(df.columns):
            raise ValueError('Input must contain ns3 flow columns from the supplied .cc example')
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        report=df.groupby('user_id',as_index=False).agg({'tx_packets':'sum','rx_packets':'sum',
                                                          'rx_bytes':'sum','mean_delay_s':'mean'})
        report['delivery_ratio']=report.rx_packets/report.tx_packets.clip(lower=1)
        report.to_csv(out,index=False)
        print(report.to_string(index=False))
        print('Saved',out)

if __name__=='__main__':main()
