import numpy as np
import pytest
from gameedgealloc.models import SystemConfig
from gameedgealloc.workload import WorkloadConfig,generate_trace
from gameedgealloc.rl import train_algorithm,RLAllocator


@pytest.mark.parametrize('algo',['ddpg','madrl'])
def test_rl_training_checkpoint_and_allocation(tmp_path,algo):
    pytest.importorskip('torch')
    cfg=SystemConfig()
    path=tmp_path/f'{algo}.pt'
    info=train_algorithm(algo,cfg,6,1.,42,55,path)
    assert path.exists() and info['steps']==55
    policy=RLAllocator(cfg,path,method=algo)
    tasks=generate_trace(WorkloadConfig(n_users=6,slots=1,seed=71)).slots[0]
    alloc=policy.allocate(tasks)
    check=alloc.check(tasks,cfg)
    assert check['capacity_feasible']
    assert np.isfinite(alloc.cpu).all()
    assert alloc.method==algo
