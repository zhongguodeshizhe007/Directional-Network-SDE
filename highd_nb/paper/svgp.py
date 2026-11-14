"""Small self-contained whitened sparse variational GP for log proximity.
Predictive log likelihood (not the usual E_q log likelihood ELBO):
  mean_i log N(y_i | m_i, v_i + noise_variance) - beta * KL(q(v)||N(0,I))/N.
Jiao et al. use the same type of predictive objective through GPyTorch; this is a
new PyTorch implementation, NOT their repository or a full experimental reproduction.
Only the diagonal predictive distribution is needed for per-frame proximity CDFs.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class WhitenedSVGP(nn.Module):
    def __init__(self,inducing,jitter=1e-4):
        super().__init__()
        if inducing.ndim!=2 or len(inducing)<1:raise ValueError('inducing must be [M,features]')
        self.inducing=nn.Parameter(inducing.detach().clone());m,d=inducing.shape;self.jitter=float(jitter)
        self.log_length=nn.Parameter(torch.zeros(d,device=inducing.device,dtype=inducing.dtype))
        self.log_variance=nn.Parameter(torch.zeros((),device=inducing.device,dtype=inducing.dtype))
        self.log_noise_sd=nn.Parameter(torch.tensor(-1.,device=inducing.device,dtype=inducing.dtype))
        self.mean_constant=nn.Parameter(torch.zeros((),device=inducing.device,dtype=inducing.dtype))
        self.variational_mean=nn.Parameter(torch.zeros(m,device=inducing.device,dtype=inducing.dtype))
        raw=torch.zeros(m,m,device=inducing.device,dtype=inducing.dtype)
        raw.diagonal().fill_(math.log(math.expm1(1.-1e-5)))
        self.raw_cholesky=nn.Parameter(raw)
    def kernel(self,x,z):
        length=self.log_length.clamp(-5,5).exp();amp=self.log_variance.clamp(-8,5).exp()
        delta=x[:,None,:]/length-z[None,:,:]/length
        return amp*torch.exp(-.5*delta.square().sum(-1))
    def variational_cholesky(self):
        return torch.tril(self.raw_cholesky,-1)+torch.diag(F.softplus(self.raw_cholesky.diagonal())+1e-5)
    def kl(self):
        l=self.variational_cholesky();m=len(l)
        return .5*(l.square().sum()+self.variational_mean.square().sum()-m-2*torch.log(l.diagonal()).sum())
    def forward(self,x):
        m=len(self.inducing);kzz=self.kernel(self.inducing,self.inducing)
        L=torch.linalg.cholesky(kzz+self.jitter*torch.eye(m,dtype=x.dtype,device=x.device))
        A=torch.linalg.solve_triangular(L,self.kernel(self.inducing,x),upper=False).T
        qL=self.variational_cholesky();mean=self.mean_constant+A@self.variational_mean
        variance=self.log_variance.clamp(-8,5).exp()+(A@qL).square().sum(-1)-A.square().sum(-1)
        # A tiny numerical floor protects subtraction round-off; it is not learned clipping of risk.
        variance=variance.clamp_min(1e-7)+self.log_noise_sd.clamp(-6,3).exp().square()
        return torch.distributions.Normal(mean,variance.sqrt())


def make_svgp(x,pc):
    ids=torch.randperm(len(x),device=x.device)[:min(len(x),int(pc['context_inducing']))]
    model=WhitenedSVGP(x[ids]);identity=nn.Identity()
    def objective(distribution,target):
        return distribution.log_prob(target).mean()-float(pc['context_beta'])*model.kl()/len(x)
    return model,identity,objective
