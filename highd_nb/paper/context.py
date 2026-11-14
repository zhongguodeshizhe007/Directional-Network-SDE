"""Jiao-inspired conditional proximity experiment on the AVAILABLE same-lane pairs.
Implements Eq.(14)/(15) exactly for a fitted lognormal conditional distribution.
Default conditional density estimator is a whitened sparse variational GP with
a predictive log-likelihood objective and observation noise in the prediction.
A heteroscedastic MLP is available as an explicitly named alternative.
Neither is presented as the authors' full 2D lane-change/100-Car reproduction.
"""
from __future__ import annotations
import copy,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import ndtr
from highd_nb.core.common import write_json
from .ssm import conflict_probability,conflict_intensity
from highd_nb.runtime import choose_device,cpu_state_dict,device_summary
FEATURES=['vf_squared','vl_squared','closing_speed','closing_squared','a_f','a_l','length_f','length_l']


def context_matrix(df):
    # Deliberately exclude proximity/positions/TTC/DRAC and future trajectory summaries.
    x=np.column_stack([df.u**2,df.leader_u**2,df.closing_speed,df.closing_speed**2,
                       df.a_long,df.leader_accel,df.width,df.leader_length]).astype('float32')
    good=np.isfinite(x).all(1)&np.isfinite(df.dhw.to_numpy())&(df.dhw.to_numpy()>0)
    return x[good],np.log(df.dhw.to_numpy()[good]).astype('float32'),df.loc[good].reset_index(drop=True)

class LognormalMLP(nn.Module):
    def __init__(self,n,hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,2))
    def forward(self,x):
        q=self.net(x);return q[:,0],.05+2.95*torch.sigmoid(q[:,1])


def _svgp(x,pc):
    from .svgp import make_svgp
    return make_svgp(x,pc)


def fit_context(train,val,pc,device='auto'):
    device=choose_device(device)
    batch=int(pc.get('context_batch_size',1024))
    x,y,train=context_matrix(train);vx,vy,val=context_matrix(val)
    if min(len(x),len(vx))<int(pc['context_min_rows']):raise ValueError('Too few finite train/validation pair observations')
    rng=np.random.default_rng(pc['seed']);cap=int(pc['context_max_train_rows'])
    if len(x)>cap:idx=rng.choice(len(x),cap,replace=False);x,y=x[idx],y[idx]
    # Validation may be sampled ONLY deterministically for numerical cost, not by model errors.
    if len(vx)>cap:idx=np.linspace(0,len(vx)-1,cap,dtype=int);vx,vy=vx[idx],vy[idx]
    mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.
    torch.manual_seed(pc['seed']);xt=torch.tensor((x-mean)/std,device=device);yt=torch.tensor(y,device=device)
    vt=torch.tensor((vx-mean)/std,device=device);vy=torch.tensor(vy,device=device)
    backend=pc['context_backend'];likelihood=mll=None
    if backend=='lognormal_mlp':
        model=LognormalMLP(x.shape[1],pc['hidden']).to(device)
        with torch.no_grad():model.net[-1].bias[0]=yt.mean()
        parameters=model.parameters()
    elif backend=='svgp':
        model,likelihood,mll=_svgp(xt,pc)
        with torch.no_grad():model.mean_constant.copy_(yt.mean())
        parameters=list(model.parameters())+list(likelihood.parameters())
    else:raise ValueError(f'Unknown context_backend {backend}')
    opt=torch.optim.Adam(parameters,lr=float(pc.get('context_learning_rate',.005)))
    best=float('inf');state=None;stale=0;curve=[]
    for epoch in range(pc['context_epochs']):
        model.train()
        if likelihood is not None:likelihood.train()
        order=torch.randperm(len(xt),device=device);total=0.
        for start in range(0,len(xt),batch):
            ii=order[start:start+batch]
            if backend=='svgp':loss=-mll(model(xt[ii]),yt[ii])
            else:
                mu,sd=model(xt[ii]);loss=(.5*((yt[ii]-mu)/sd)**2+torch.log(sd)+.5*math.log(2*math.pi)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contextual proximity likelihood')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();total+=float(loss.detach())*len(ii)
        model.eval()
        if likelihood is not None:likelihood.eval()
        with torch.no_grad():
            val_nll=0.
            for start in range(0,len(vt),batch):
                z=vt[start:start+batch];target=vy[start:start+batch]
                if backend=='svgp':d=likelihood(model(z));mu,sd=d.mean,d.variance.clamp_min(1e-8).sqrt()
                else:mu,sd=model(z)
                val_nll+=float((.5*((target-mu)/sd)**2+torch.log(sd)+.5*math.log(2*math.pi)+target).sum())
            score=val_nll/len(vt)
        curve.append(dict(epoch=epoch+1,training_objective=total/len(xt),validation_proximity_nll=score))
        if score<best:
            best=score;stale=0;state=cpu_state_dict(model);lstate=cpu_state_dict(likelihood) if likelihood else None
        else:stale+=1
        if stale>=pc['patience']:break
    model.load_state_dict(state);model.eval()
    if likelihood is not None:likelihood.load_state_dict(lstate);likelihood.eval()
    return {'model':model,'likelihood':likelihood,'mean':mean,'std':std,'backend':backend,
            'device':device,'features':FEATURES,'train_rows':len(x),'val_rows':len(vt)},pd.DataFrame(curve)


def predict_context(fit,df):
    x,y,df=context_matrix(df);means=[];sds=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            z=torch.tensor((x[start:start+1024]-fit['mean'])/fit['std'],device=fit['device'])
            if fit['backend']=='svgp':d=fit['likelihood'](fit['model'](z));mu,sd=d.mean,d.variance.clamp_min(1e-8).sqrt()
            else:mu,sd=fit['model'](z)
            means.append(mu.cpu().numpy());sds.append(sd.cpu().numpy())
    if not len(x):raise ValueError('No test pairs with complete contextual features')
    return np.concatenate(means),np.concatenate(sds),y,df


def load_pair_tables(graphs):
    out=[]
    for g in graphs:
        p=Path(g['path']).parent/'pairs_sample.csv.gz'
        if not p.exists():raise FileNotFoundError('Pairs missing: reprocess with v4; do not reuse v3 arrays')
        x=pd.read_csv(p,dtype={'recording_id':str});x['location_id']=g['schema']['location_id'];x['graph']=g['name'];out.append(x)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def run_context(train_graphs,val_graphs,test_graphs,pc,dst,device='cpu'):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    tr,va,te=[load_pair_tables(gs) for gs in [train_graphs,val_graphs,test_graphs]]
    sizes=[len(context_matrix(x)[0]) if not x.empty else 0 for x in [tr,va,te]]
    if min(sizes[:2])<int(pc['context_min_rows']) or sizes[2]==0:
        status={'status':'SKIP_INSUFFICIENT_CONTEXT_SUPPORT','train_rows':sizes[0],'validation_rows':sizes[1],'test_rows':sizes[2]}
        write_json(dst/'context_summary.json',status);return status
    fit,curve=fit_context(tr,va,pc,device);curve.to_csv(dst/'context_learning_curve.csv',index=False)
    mu,sd,logg,te=predict_context(fit,te);te['context_mu_log_gap']=mu;te['context_sd_log_gap']=sd
    te['pit']=ndtr((logg-mu)/sd);te['intensity_p50']=conflict_intensity(te.dhw,mu,sd)
    for n in pc['context_intensities']:te[f'C_n{n}']=conflict_probability(te.dhw,mu,sd,n)
    te.to_csv(dst/'context_test_pairs.csv.gz',index=False,compression='gzip')
    nll=.5*((logg-mu)/sd)**2+np.log(sd)+.5*np.log(2*np.pi)+logg
    stats={'status':'FIT','n_test_sampled_frames':len(te),'n_test_pairs':len(te.groupby(['recording_id','direction','id','precedingId'])),
           'mean_proximity_nll':float(nll.mean()),'coverage90':float((np.abs((logg-mu)/sd)<=1.644853627).mean()),
           'pit_mean':float(te.pit.mean()),'pit_variance':float(te.pit.var(ddof=0)),
           'n_train_sampled_frames':fit['train_rows'],'n_validation_sampled_frames':fit['val_rows'],
           'backend':fit['backend'],'engine':'native PyTorch whitened SVGP' if fit['backend']=='svgp' else 'heteroscedastic MLP',
           'nonfinite_intensity_frames':int((~np.isfinite(te.intensity_p50)).sum()),'scope':'same-lane car-following; not 2D lane-change or 100-Car validation',
           'probability_interpretation':'contextual extremeness C(n;s,theta), NOT P(crash in next H seconds)',
           'power_law_claim':False,'test_used_for_selection':False,'features':FEATURES,'device':device_summary(fit['device'])}
    write_json(dst/'context_summary.json',stats)
    torch.save({'model_state_dict':fit['model'].state_dict(),
                'likelihood_state_dict':fit['likelihood'].state_dict() if fit['likelihood'] else None,
                'mean':fit['mean'].tolist(),'std':fit['std'].tolist(),'backend':fit['backend'],'features':FEATURES},dst/'context_checkpoint.pt')
    pairs=te.groupby(['recording_id','direction','id','precedingId'],sort=True).agg(
        sample_count=('frame','size'),mean_intensity=('intensity_p50','mean'),max_intensity=('intensity_p50','max'),
        min_ttc=('ttc_calc','min')).reset_index()
    pairs.to_csv(dst/'context_pair_intensity.csv',index=False)
    # A fixed first test context; surface is a mathematical slice, not a fitted 2D spatial field.
    q=te.sort_values(['recording_id','direction','id','frame']).iloc[0]
    gaps=np.geomspace(max(.02,float(np.exp(q.context_mu_log_gap-3*q.context_sd_log_gap))),
                     float(np.exp(q.context_mu_log_gap+3*q.context_sd_log_gap)),60)
    ns=np.geomspace(1,1000,60);S,NN=np.meshgrid(gaps,ns)
    pd.DataFrame({'gap_m':S.ravel(),'intensity_n':NN.ravel(),
                  'probability':conflict_probability(S, q.context_mu_log_gap,q.context_sd_log_gap,NN).ravel()}).to_csv(dst/'context_spectrum.csv',index=False)
    write_json(dst/'context_slice.json',{'recording_id':q.recording_id,'direction':int(q.direction),
        'id':int(q.id),'precedingId':int(q.precedingId),'frame':int(q.frame),'mu':q.context_mu_log_gap,'sigma':q.context_sd_log_gap,
        'selection':'first test pair/frame in fixed ID order, NOT best prediction'})
    overlap=[]
    for n in pc['context_intensities']:
        for ttc in pc['event_thresholds_s']:
            a=te.ttc_calc<ttc;b=te[f'C_n{n}']>=.5
            overlap.append(dict(intensity=n,ttc_threshold_s=ttc,both=int((a&b).sum()),ttc_only=int((a&~b).sum()),
                                contextual_only=int((~a&b).sum()),neither=int((~a&~b).sum()),
                                note='definition overlap among sampled frames; neither definition is crash ground truth'))
    pd.DataFrame(overlap).to_csv(dst/'context_definition_overlap.csv',index=False)
    return stats
