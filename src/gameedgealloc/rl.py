from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from copy import deepcopy
from pathlib import Path
import time
import random
import numpy as np
from .models import Task,SystemConfig,Allocation,latency,energy,jain_index,satisfaction
from .workload import WorkloadConfig,generate_trace

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch=None;nn=None;F=None

N_FEATURES=8


def _require_torch():
    if torch is None:raise ImportError('PyTorch is required for DRL: pip install -e .[rl]')


def encode(tasks:list[Task],max_users:int,cfg:SystemConfig):
    arr=np.zeros((max_users,N_FEATURES),dtype=np.float32)
    mask=np.zeros(max_users,dtype=bool)
    for t in tasks:
        if t.user_id>=max_users:raise ValueError('Agent user_id exceeds trained n_users')
        i=t.user_id;mask[i]=True
        arr[i]=[np.log1p(t.data_mbit),np.log1p(t.cycles_g),t.deadline_s/3,
                t.spectral_eff/3,t.power_w/.25,
                t.data_mbit/(t.spectral_eff*t.deadline_s),t.cycles_g/t.deadline_s,1.]
    glob=np.array([mask.sum()/max_users,cfg.cpu_capacity_ghz/max_users/5],dtype=np.float32)
    return np.concatenate((arr.flatten(),glob)),mask


def _probability_alloc(weights,mask,capacity,minimum):
    if not mask.any():return np.zeros_like(weights,dtype=float)
    weights=np.asarray(weights,float)
    exp=np.exp(np.clip(weights[mask]-np.max(weights[mask]),-50,50))
    x=np.zeros(len(mask),float)
    x[mask]=minimum+(capacity-minimum*mask.sum())*exp/exp.sum()
    return x


def logits_to_resources(cpu_logits,bw_logits,mask,cfg):
    return (_probability_alloc(cpu_logits,mask,cfg.cpu_capacity_ghz,cfg.min_cpu_ghz),
            _probability_alloc(bw_logits,mask,cfg.bandwidth_capacity_mhz,cfg.min_bw_mhz))


def _torch_share(logits,mask,capacity,minimum):
    """Differentiable, floor-respecting masked softmax; shapes [batch,users]."""
    masked=logits.masked_fill(~mask,-1e9)
    weights=torch.softmax(masked,dim=-1)*mask.float()
    valid=mask.float().sum(dim=-1,keepdim=True)
    weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-12)
    return (minimum*mask.float() + (capacity-minimum*valid)*weights)/capacity


if nn is not None:
    class GlobalActor(nn.Module):
        def __init__(self,n):
            super().__init__();self.n=n
            self.net=nn.Sequential(nn.Linear(n*N_FEATURES+2,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2*n))
        def forward(self,states):return self.net(states).reshape(-1,2,self.n)

    class GlobalCritic(nn.Module):
        def __init__(self,n):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(n*N_FEATURES+2+2*n,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
        def forward(self,s,action):return self.net(torch.cat((s,action.reshape(len(s),-1)),dim=-1)).squeeze(-1)

    class SharedAgentActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(N_FEATURES,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
        def forward(self,user_features):return self.net(user_features)

    class CentralValue(nn.Module):
        def __init__(self,n):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(n*N_FEATURES+2,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,s):return self.net(s).squeeze(-1)


def _actions_from_global_torch(actor,states,masks,cfg):
    logits=actor(states)
    f=_torch_share(logits[:,0,:],masks,cfg.cpu_capacity_ghz,cfg.min_cpu_ghz)
    b=_torch_share(logits[:,1,:],masks,cfg.bandwidth_capacity_mhz,cfg.min_bw_mhz)
    return torch.stack((f,b),dim=1)


def cooperative_reward(tasks,f,b,cfg):
    if not tasks:return 0.
    ell=latency(tasks,f,b);en=energy(tasks,f,b,cfg.dvfs_zeta)
    d=np.array([t.deadline_s for t in tasks])
    sat=satisfaction(tasks,f,b)
    return float(-.45*np.mean(np.minimum(ell/d,10))-.20*np.mean(np.minimum(en/.5,10))
                 +.30*jain_index(sat)+.50*np.mean(ell<=d))


def _environment_steps(n_users,load,seed,steps):
    return generate_trace(WorkloadConfig(n_users=n_users,slots=steps,
                                         offered_load=load,seed=seed+100_001)).slots


def train_ddpg(cfg:SystemConfig,n_users:int,offered_load:float,seed:int,steps:int,
               save_path:str|Path,batch_size:int=48,learning_rate:float=3e-4,device:str="auto"):
    _require_torch()
    torch.set_num_threads(1)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested PyTorch device is unavailable')
    device=torch.device(device)
    actor=GlobalActor(n_users).to(device);critic=GlobalCritic(n_users).to(device)
    ta=deepcopy(actor);tc=deepcopy(critic)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=learning_rate)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=learning_rate)
    memory=deque(maxlen=20000)
    trace=_environment_steps(n_users,offered_load,seed,steps+1)
    rng=np.random.default_rng(seed+222)
    reward_log=[]
    gamma=.97;tau=.01
    for it in range(steps):
        tasks=trace[it];tasks_next=trace[it+1]
        state,mask=encode(tasks,n_users,cfg);nextstate,nextmask=encode(tasks_next,n_users,cfg)
        with torch.no_grad():
            logits=actor(torch.tensor(state,device=device).reshape(1,-1)).reshape(2,n_users).cpu().numpy()
        noise=rng.normal(0.,.35,size=logits.shape)
        f,b=logits_to_resources(logits[0]+noise[0],logits[1]+noise[1],mask,cfg)
        req_ids=np.array([t.user_id for t in tasks],dtype=int)
        r=cooperative_reward(tasks,f[req_ids],b[req_ids],cfg)
        action=np.stack((f/cfg.cpu_capacity_ghz,b/cfg.bandwidth_capacity_mhz))
        memory.append((state,mask,action,r,nextstate,nextmask,float(it%64==63)))
        reward_log.append(r)
        if len(memory)<batch_size:continue
        ix=rng.choice(len(memory),batch_size,replace=False)
        sample=[memory[i] for i in ix]
        ss=torch.tensor(np.stack([x[0] for x in sample]),dtype=torch.float32,device=device)
        mm=torch.tensor(np.stack([x[1] for x in sample]),dtype=torch.bool,device=device)
        aa=torch.tensor(np.stack([x[2] for x in sample]),dtype=torch.float32,device=device)
        rr=torch.tensor([x[3] for x in sample],dtype=torch.float32,device=device)
        ns=torch.tensor(np.stack([x[4] for x in sample]),dtype=torch.float32,device=device)
        nm=torch.tensor(np.stack([x[5] for x in sample]),dtype=torch.bool,device=device)
        done=torch.tensor([x[6] for x in sample],dtype=torch.float32,device=device)
        with torch.no_grad():y=rr+gamma*(1-done)*tc(ns,_actions_from_global_torch(ta,ns,nm,cfg))
        loss=F.smooth_l1_loss(critic(ss,aa),y)
        critic_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5);critic_opt.step()
        aloss=-critic(ss,_actions_from_global_torch(actor,ss,mm,cfg)).mean()
        actor_opt.zero_grad();aloss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),5);actor_opt.step()
        with torch.no_grad():
            for x,y in zip(ta.parameters(),actor.parameters()):x.mul_(1-tau).add_(tau*y)
            for x,y in zip(tc.parameters(),critic.parameters()):x.mul_(1-tau).add_(tau*y)
    path=Path(save_path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'algo':'ddpg','n_users':n_users,'actor':actor.state_dict(),
                'train_steps':steps,'seed':seed,'load':offered_load,
                'last100_reward':float(np.mean(reward_log[-100:]))},path)
    return {'checkpoint':str(path),'steps':steps,'last100_reward':float(np.mean(reward_log[-100:]))}


def train_madrl(cfg:SystemConfig,n_users:int,offered_load:float,seed:int,steps:int,
                save_path:str|Path,learning_rate:float=3e-4,device:str="auto"):
    _require_torch();torch.set_num_threads(1)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested PyTorch device is unavailable')
    device=torch.device(device)
    actor=SharedAgentActor().to(device);critic=CentralValue(n_users).to(device)
    opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=learning_rate)
    trace=_environment_steps(n_users,offered_load,seed+123,steps+1)
    rewards=[]
    for it in range(steps):
        tasks=trace[it];state,mask=encode(tasks,n_users,cfg)
        nextstate,_=encode(trace[it+1],n_users,cfg)
        st=torch.tensor(state,dtype=torch.float32,device=device).reshape(1,-1)
        ns=torch.tensor(nextstate,dtype=torch.float32,device=device).reshape(1,-1)
        local=st[:,:n_users*N_FEATURES].reshape(1,n_users,N_FEATURES)
        means=actor(local).permute(0,2,1)
        std=.45
        distribution=torch.distributions.Normal(means,torch.full_like(means,std))
        action_logits=distribution.sample()
        logp=distribution.log_prob(action_logits)
        mask_t=torch.tensor(mask,device=device).reshape(1,-1)
        ff=_torch_share(action_logits[:,0,:],mask_t,cfg.cpu_capacity_ghz,cfg.min_cpu_ghz)
        bb=_torch_share(action_logits[:,1,:],mask_t,cfg.bandwidth_capacity_mhz,cfg.min_bw_mhz)
        ids=np.array([t.user_id for t in tasks],dtype=int)
        f=ff.detach().cpu().numpy()[0,ids]*cfg.cpu_capacity_ghz
        b=bb.detach().cpu().numpy()[0,ids]*cfg.bandwidth_capacity_mhz
        reward=cooperative_reward(tasks,f,b,cfg)
        rewards.append(reward)
        rv=torch.tensor(reward,dtype=torch.float32,device=device)
        value=critic(st).squeeze()
        with torch.no_grad():target=rv+(.97 if it%64!=63 else 0.)*critic(ns).squeeze()
        advantage=target-value
        maskfloat=mask_t.float().unsqueeze(1)
        policy_loss=-(logp*maskfloat).sum()/max(int(mask.sum()),1)*advantage.detach()
        value_loss=.5*advantage.pow(2)
        loss=policy_loss+value_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),3);opt.step()
    path=Path(save_path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'algo':'madrl','n_users':n_users,'actor':actor.state_dict(),
                'train_steps':steps,'seed':seed,'load':offered_load,
                'last100_reward':float(np.mean(rewards[-100:]))},path)
    return {'checkpoint':str(path),'steps':steps,'last100_reward':float(np.mean(rewards[-100:]))}


class RLAllocator:
    def __init__(self,cfg:SystemConfig,checkpoint:str|Path,method:str|None=None):
        _require_torch();torch.set_num_threads(1)
        # Checkpoint is created by this package; do not load untrusted .pt files.
        try:data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        except TypeError:data=torch.load(checkpoint,map_location='cpu')
        self.cfg=cfg;self.n_users=int(data['n_users']);self.name=data['algo']
        if method and self.name!=method:raise ValueError('Checkpoint algorithm mismatch')
        self.actor=GlobalActor(self.n_users) if self.name=='ddpg' else SharedAgentActor()
        self.actor.load_state_dict(data['actor']);self.actor.eval()

    def allocate(self,tasks:list[Task])->Allocation:
        tic=time.perf_counter()
        if not tasks:return Allocation(np.zeros(0),np.zeros(0),method=self.name)
        state,mask=encode(tasks,self.n_users,self.cfg)
        with torch.no_grad():
            st=torch.tensor(state,dtype=torch.float32).reshape(1,-1)
            if self.name=='ddpg':logits=self.actor(st).reshape(2,self.n_users).numpy()
            else:
                local=st[:,:self.n_users*N_FEATURES].reshape(1,self.n_users,N_FEATURES)
                logits=self.actor(local).permute(0,2,1).reshape(2,self.n_users).numpy()
        f,b=logits_to_resources(logits[0],logits[1],mask,self.cfg)
        ids=np.array([t.user_id for t in tasks],dtype=int)
        return Allocation(f[ids],b[ids],method=self.name,solve_s=time.perf_counter()-tic)


def train_algorithm(algo,cfg,n_users,load,seed,steps,path,device='auto'):
    if algo=='ddpg':return train_ddpg(cfg,n_users,load,seed,steps,path,device=device)
    if algo=='madrl':return train_madrl(cfg,n_users,load,seed,steps,path,device=device)
    raise ValueError('Train algorithm should be ddpg or madrl')
