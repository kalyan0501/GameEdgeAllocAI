import numpy as np
import pandas as pd
from gameedgealloc.models import SystemConfig
from gameedgealloc.workload import WorkloadConfig,generate_trace,Trace
from gameedgealloc.baselines import make_allocator,ALLOCATORS
from gameedgealloc.simulator import evaluate_trace
from gameedgealloc.online_learning import DemandLearner
from gameedgealloc.statistics import confidence_interval,holm_adjust,paired_comparisons
from gameedgealloc.experiments import run_experiments


def test_all_methods_capacity_and_finiteness():
    cfg=SystemConfig()
    tr=generate_trace(WorkloadConfig(n_users=15,slots=7,offered_load=2.,seed=99))
    for name in ALLOCATORS:
        rows=evaluate_trace(make_allocator(name,cfg,seed=99),tr,cfg)
        assert rows.n_active.ge(0).all()
        assert np.isfinite(rows.latency_s).all()
        assert np.isfinite(rows.energy_j).all()
        assert rows.capacity_feasible.eq(1).all(),name
        assert rows.jain.le(1+1e-8).all()


def test_same_seed_same_trace_and_jsonl(tmp_path):
    settings=WorkloadConfig(n_users=10,slots=6,seed=17)
    a=generate_trace(settings);b=generate_trace(settings)
    assert a.slots==b.slots
    path=tmp_path/'trace.jsonl';a.save(path)
    assert Trace.load(path).slots==a.slots


def test_online_learner_filters_anomalies():
    cfg=SystemConfig()
    learner=DemandLearner(cfg)
    tasks=generate_trace(WorkloadConfig(n_users=8,slots=4,seed=33)).slots[0]
    assert tasks
    n=len(tasks);p=np.full(n,.2)
    learner.update(tasks,np.full(n,1.2),np.full(n,1.),p,p)
    weird=np.full(n,100.)
    filt_f,filt_b,nchanged=learner.filter_reports(tasks,weird,weird,p,p)
    assert filt_f.max()<100 and filt_b.max()<100 and nchanged>0
    guess=learner.warm_prices(tasks,np.ones(n),[.11,.11])
    assert np.all(guess>0)


def test_statistical_pairing_and_holm():
    df=pd.DataFrame([{'method':m,'seed':s,'n_users':10,'load':1.,
                      'latency_s':base+.02*s,'energy_j':base,
                      'jain':1-base*.1,'deadline_satisfaction':1-base*.01,
                      'iterations':5+base} for m,base in [('gameedgealloc',.5),('equal',1.)]
                     for s in range(1,6)])
    p=paired_comparisons(df)
    assert not p.empty
    assert p.n_pairs.eq(5).all()
    np.testing.assert_allclose(holm_adjust([.01,.03,.05]),[.03,.06,.06])
    lo,hi=confidence_interval([1,2,3,4,5]);assert lo<3<hi


def test_experiment_end_to_end(tmp_path):
    config={'experiment':{'user_counts':[5],'loads':[.7],'seeds':[42,123],
                          'slots':5,'warmup':1,'methods':['gameedgealloc','equal'],
                          'save_trace':True,'save_detailed':True},
             'system':{'queue_enabled':False}}
    result=run_experiments(config,tmp_path,generate_plots=False)
    assert result['rows']==20
    assert (tmp_path/'manifest.json').exists()
    assert (tmp_path/'paired_tests_holm.csv').exists()
    assert (tmp_path/'seed_summary.csv').exists()
