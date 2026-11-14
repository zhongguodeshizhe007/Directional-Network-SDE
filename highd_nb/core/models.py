"""New research implementation, NOT the original LaGNA code.
masked: censored Euler transition with observed-neighbor message gating.
latent: Euler-discretized latent network SDE + censored observation likelihood,
        causal amortized Gaussian filtering posterior + analytic transition KL.
No claim of exact continuous-time inference, identifiable diffusion or MNAR correction.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

def censored_log_prob(y,mean,sd):
    sd=sd.clamp_min(1e-6)
    logpdf=-.5*((y-mean)/sd)**2-torch.log(sd)-.5*math.log(2*math.pi)
    logatom=torch.special.log_ndtr(-mean/sd)
    return torch.where(y<=0,logatom,logpdf)

def normal_kl(qm,qs,pm,ps):
    return torch.log(ps/qs)+.5*((qs/ps)**2+((qm-pm)/ps)**2-1)

def mlp(in_dim,hidden,out_dim):
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,out_dim))

class Dynamics(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.self_net=mlp(5,c['hidden'],1)
        self.edge_nets=nn.ModuleList([mlp(10,c['hidden'],1) for _ in range(3)])
        self.diff_net=mlp(5,c['hidden'],1)
    def self_value(self,u): return self.c['drift_limit']*torch.tanh(self.self_net(u)[...,0])
    def edge_value(self,u,delta,k):
        net=self.edge_nets[0 if self.c.get('tie_longitudinal',False) and k in (0,1) else k]
        # Gauge convention G(u,0)=0 removes target-only terms from G.
        return .5*self.c['drift_limit']*(torch.tanh(net(torch.cat([u,delta],-1))[...,0])-
                       torch.tanh(net(torch.cat([u,torch.zeros_like(delta)],-1))[...,0]))
    def sigma_value(self,u):
        q = torch.zeros_like(u) if self.c.get('constant_sigma',False) else u
        return self.c['sigma_min']+(self.c['sigma_max']-self.c['sigma_min'])*torch.sigmoid(self.diff_net(q)[...,0])
    def components(self,x,z,A,availability=None):
        # x [batch,N], z [batch,N,4], A [3,target,source].
        u=torch.cat([x[...,None],z],-1); own=self.self_value(u)
        parts=[]
        for k in range(3):
            out=torch.zeros_like(x)
            if self.c.get('use_edges',True) and k in self.c.get('edge_types',[0,1,2]):
                targets,sources=torch.where(A[k]>0)
                if targets.numel():
                    value=self.edge_value(u[:,targets],u[:,sources]-u[:,targets],k)*A[k,targets,sources]
                    if availability is not None:
                        value=value*availability[:,targets]*availability[:,sources]
                    out.index_add_(1,targets,value)
            parts.append(out)
        sigma=self.sigma_value(u)
        return own,parts,sigma
    def forward(self,x,z,A,availability=None):
        own,edges,sigma=self.components(x,z,A,availability)
        return own+sum(edges),sigma

class FilterPosterior(nn.Module):
    def __init__(self,c):
        super().__init__(); self.net=mlp(16,c['hidden'],3)
    def forward(self,pm,ps,y,my,z,mz,az):
        h=torch.cat([pm[...,None],torch.log(ps.clamp_min(1e-6))[...,None],y[...,None],
                     my.float()[...,None],z,mz.float(),az],-1)
        h=self.net(h)
        gain=torch.sigmoid(h[...,0])
        qm=pm+gain*(y-pm)+.2*ps*torch.tanh(h[...,1])
        qs=ps*(.05+.95*torch.sigmoid(h[...,2]))
        # No risk observation -> q=p exactly. z remains a conditional covariate,
        # not a fabricated observation of x.
        return torch.where(my,qm,pm),torch.where(my,qs,ps)

class NetworkModel(nn.Module):
    def __init__(self,kind,c):
        super().__init__(); self.kind=kind; self.c=c
        self.dynamics=Dynamics(c)
        self.posterior=FilterPosterior(c) if kind=='latent' else None
    def batch_loss(self,b,burn):
        y,my,z=b['y'],b['my'],b['z']; dt=b['dt']; B,L,N=y.shape
        if self.kind=='masked':
            x=y[:,:-1].reshape(-1,N); zz=z[:,:-1].reshape(-1,N,4)
            m=my[:,:-1].reshape(-1,N)
            mu,sigma=self.dynamics(x,zz,b['A'],m)
            target=y[:,1:].reshape(-1,N)
            valid=(my[:,:-1]&my[:,1:]).clone(); valid[:,:burn]=False; valid=valid.reshape(-1,N)
            count=int(valid.sum())
            if not count: return None,0
            lp=censored_log_prob(target,x+mu*dt,sigma*math.sqrt(dt))
            return -lp[valid].sum()/count,count
        x=torch.zeros(B,N,device=y.device); total=torch.zeros((),device=y.device); obs_count=0
        for t in range(L):
            if t==0: pm,ps=torch.zeros_like(x),torch.ones_like(x)
            else:
                drift,sigma=self.dynamics(x,z[:,t-1],b['A'])
                pm,ps=x+drift*dt,sigma*math.sqrt(dt)
            qm,qs=self.posterior(pm,ps,y[:,t],my[:,t],z[:,t],b['mz'][:,t],b['az'][:,t])
            x=qm+qs*torch.randn_like(qm)
            if t>=burn:
                valid=my[:,t]; obs_count+=int(valid.sum())
                lp=censored_log_prob(y[:,t],x,torch.full_like(x,self.c['observation_sd']))
                total=total+normal_kl(qm,qs,pm,ps).sum()-lp[valid].sum()
        if not obs_count: return None,0
        return total/obs_count,obs_count

@torch.no_grad()
def filtered_particles(model,g,anchor,particles,history,device):
    """Only observations with index <= anchor are accessed."""
    N=g['y'].shape[1]; A=g['A'].to(device)
    if model.kind=='masked':
        return g['y'][anchor].to(device).expand(particles,N).clone()
    start=max(0,anchor-history+1)
    x=torch.zeros(particles,N,device=device)
    for t in range(start,anchor+1):
        if t==start: pm,ps=torch.zeros_like(x),torch.ones_like(x)
        else:
            zz=g['z'][t-1].to(device).expand(particles,N,4)
            drift,sigma=model.dynamics(x,zz,A)
            pm,ps=x+drift*g['dt'],sigma*math.sqrt(g['dt'])
        args=[g[k][t].to(device).expand((particles,)+tuple(g[k][t].shape)) for k in ['y','my','z','mz','az']]
        qm,qs=model.posterior(pm,ps,*args); x=qm+qs*torch.randn_like(qm)
    return x
