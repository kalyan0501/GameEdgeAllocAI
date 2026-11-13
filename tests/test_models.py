import numpy as np
import pytest
from gameedgealloc.models import Task,SystemConfig,latency,energy,jain_index,utility
from gameedgealloc.follower import cpu_best_response,bandwidth_best_response,follower_response
from gameedgealloc.fairness import fairness_multipliers,adapt_fairness_weight,normalized_fairness_penalty
from gameedgealloc.pricing import price_target,contraction_bound,adaptive_step_size,clearing_price

T=Task(0,1.5,.8,1.5,2.0,.2,beta=1.5,gamma=.8,delta=1.)


def test_physical_units_and_math():
    np.testing.assert_allclose(latency([T],np.array([2.]),np.array([3.])),[1.5/(3*2)+.8/2])
    np.testing.assert_allclose(energy([T],np.array([2.]),np.array([3.]),.07),[.2*1.5/(3*2)+.07*.8*4])
    assert np.isfinite(utility([T],np.array([2.]),np.array([3.]),.07)).all()


def test_follower_foc_and_price_monotonicity():
    cfg=SystemConfig()
    for p in (.05,.1,.5,2):
        f=cpu_best_response(T,p,cfg.dvfs_zeta)
        assert abs(T.beta*T.cycles_g/f**2-2*T.gamma*cfg.dvfs_zeta*T.cycles_g*f-T.delta*p)<1e-8
        b=bandwidth_best_response(T,p)
        assert abs((T.beta+T.gamma*T.power_w)*T.data_mbit/(T.spectral_eff*b*b)-T.delta*p)<1e-8
    assert cpu_best_response(T,.4,.07)<cpu_best_response(T,.2,.07)
    assert bandwidth_best_response(T,.4)<bandwidth_best_response(T,.2)
    assert abs(cpu_best_response(T,.2,0)-np.sqrt(T.beta*T.cycles_g/(T.delta*.2)))<1e-9


def test_hard_deadline_dual_solver():
    cfg=SystemConfig(deadline_mode='hard')
    t=Task(1,1.,1.,.9,2.,.1,beta=1.,gamma=1.,delta=1.)
    f,b,ok=follower_response(t,1.5,1.5,cfg)
    assert ok
    assert latency([t],np.array([f]),np.array([b]))[0]<=.9+1e-6
    impossible=Task(1,10000.,10000.,.001,1.,.2)
    _,_,ok=follower_response(impossible,1.,1.,cfg)
    assert not ok


def test_fairness_and_pricing():
    assert np.isclose(jain_index(np.ones(5)),1)
    assert np.isclose(jain_index(np.array([1.,0.,0.])),1/3)
    m=fairness_multipliers([2.,1.,.1],.9)
    assert m[0]>1 and m[-1]<1
    assert np.all((m>=.5)&(m<=2.))
    assert normalized_fairness_penalty([1,1,1])==0
    assert adapt_fairness_weight(.3,.7,.9)>.3
    assert np.isclose(price_target(.1,2.,.5,.3),.1*(1+1+.3))
    assert contraction_bound(.3,.5,.3,2.0)>0
    assert 0<adaptive_step_size(.3,.5,.3,3.,.1,.2)<=1


def test_capacity_clearing_prices():
    cfg=SystemConfig(cpu_capacity_ghz=2.,bandwidth_capacity_mhz=3.)
    tasks=[T,Task(1,2.,1.,1.,1.7,.1)]
    m=np.ones(2)
    p=clearing_price(tasks,m,cfg.cpu_capacity_ghz,cfg.base_cpu_price,cfg,'cpu')
    f=sum(cpu_best_response(t,p,cfg.dvfs_zeta,cfg.min_cpu_ghz,cfg.cpu_capacity_ghz) for t in tasks)
    assert f<=cfg.cpu_capacity_ghz+1e-7
    q=clearing_price(tasks,m,cfg.bandwidth_capacity_mhz,cfg.base_bandwidth_price,cfg,'bw')
    b=sum(bandwidth_best_response(t,q,cfg.min_bw_mhz,cfg.bandwidth_capacity_mhz) for t in tasks)
    assert b<=cfg.bandwidth_capacity_mhz+1e-7


def test_validate_input():
    with pytest.raises(ValueError):Task(0,-1,1,1,1,1)
    with pytest.raises(ValueError):SystemConfig(deadline_mode='invalid')
