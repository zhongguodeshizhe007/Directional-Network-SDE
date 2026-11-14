"""Actual small forward/backward checks on the selected accelerator; artificial data only."""
from __future__ import annotations
import math
import numpy as np
import torch
from .runtime import choose_device,configure_reproducibility,device_summary,synchronize,xgb_parameters,check_xgb_device,xgb_predict
from .core.common import ROOT,read_json


def accelerator_smoke_test(request='auto', include_xgboost=True):
    from .core.models import NetworkModel,censored_log_prob
    from .core.evaluation import score_distribution
    from .paper.baselines import SequenceForecaster
    from .paper.svgp import WhitenedSVGP
    from .paper.hazard import DiscreteHazard,survival_nll,hazard_cdf
    configure_reproducibility();dev=choose_device(request);torch.manual_seed(17);torch.set_num_threads(1)
    checks=[]
    c=read_json(ROOT/'configs/experiment.json')['model'];c=dict(c,hidden=8)
    A=torch.zeros(3,3,3,device=dev);A[0,1,0]=1;A[1,0,1]=1;A[2,1,2]=1
    for kind in ['masked','latent']:
        model=NetworkModel(kind,c).to(dev)
        b={'y':torch.rand(2,5,3,device=dev),'my':torch.ones(2,5,3,dtype=torch.bool,device=dev),
           'z':torch.rand(2,5,3,4,device=dev),'mz':torch.ones(2,5,3,4,dtype=torch.bool,device=dev),
           'az':torch.zeros(2,5,3,4,device=dev),'A':A,'dt':1.}
        loss,n=model.batch_loss(b,1);loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError(f'{kind}: invalid accelerator gradients')
        torch.optim.AdamW(model.parameters(),lr=.001).step()
        checks.append({'name':kind+'_forward_backward_optimizer','status':'PASS','loss':float(loss.detach()),'terms':n})
    for kind in ['lstm','cnn','convlstm','st_transformer']:
        net=SequenceForecaster(kind,3,2,8).to(dev)
        mu,sd=net(torch.rand(2,3,6,14,device=dev),3)
        loss=-censored_log_prob(torch.rand_like(mu),mu,sd).mean();loss.backward()
        if not torch.isfinite(loss):raise FloatingPointError(kind)
        checks.append({'name':kind+'_forward_backward','status':'PASS'})
    gp=WhitenedSVGP(torch.rand(6,3,device=dev));out=gp(torch.rand(12,3,device=dev))
    loss=-out.log_prob(torch.rand(12,device=dev)).mean()+.1*gp.kl();loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in gp.parameters() if p.grad is not None):
        raise FloatingPointError('SVGP Cholesky/likelihood gradients invalid')
    checks.append({'name':'svgp_cholesky_forward_backward','status':'PASS'})
    net=DiscreteHazard(5,4,8).to(dev);logits=net(torch.rand(10,5,device=dev))
    y=torch.zeros_like(logits);y[::2,2]=1;loss=survival_nll(logits,y);loss.backward()
    p=hazard_cdf(logits,1.3)
    if not torch.isfinite(loss) or not bool((p.diff(dim=1)>=-1e-7).all()):raise FloatingPointError('Hazard check failed')
    checks.append({'name':'coherent_hazard_forward_backward','status':'PASS'})
    means=torch.zeros(16,3,device=dev);sds=torch.ones_like(means)
    pr={'samples':torch.randn_like(means).clamp_min(0),'mixture_mean':means,'mixture_sd':sds}
    scores=score_distribution(pr,torch.tensor([0.,.2,2.],device=dev),.2)
    if not all(torch.isfinite(v).all() for v in scores.values()):raise FloatingPointError('Particle scores nonfinite')
    checks.append({'name':'particle_quantile_and_censored_density','status':'PASS'})
    if include_xgboost:
        from xgboost import XGBRegressor,XGBClassifier
        x=np.random.default_rng(17).normal(size=(32,4)).astype('float32');y=(x[:,0]>0).astype(int)
        for name,cls in [('regression',XGBRegressor),('classification',XGBClassifier)]:
            m=cls(n_estimators=3,max_depth=2,**xgb_parameters(dev));m.fit(x,y)
            actual=check_xgb_device(m,dev);pred=xgb_predict(m,x,name=='classification')
            if not np.isfinite(pred).all():raise FloatingPointError('XGBoost nonfinite predictions')
            checks.append({'name':'xgboost_'+name,'status':'PASS',**actual})
    synchronize(dev)
    return {'status':'PASS','data':'SMALL ARTIFICIAL TENSORS ONLY','runtime':device_summary(dev),
            'checks':checks,'cuda_executed':dev.type=='cuda','real_highD_executed':False}
