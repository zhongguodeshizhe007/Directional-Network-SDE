"""Sparse polynomial distillation, training-recording bootstrap, held-out validation fidelity.
This is conditional function compression, not proof of causal or unique physical laws.
Latent state equations remain in the specified latent coordinate, not directly dY.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from .common import *
from .data import prepare_graph,split_paths
from .models import Dynamics,filtered_particles
from .evaluation import load_run,validation_score,validation_scores
VARS=['x','v_std','cv_std','k_std','truck_std']

def library(kind,u,delta=None):
    """Works for both numpy and torch arrays. Last axis enumerates functions."""
    torch_mode=torch.is_tensor(u)
    stack=lambda a: torch.stack(a,-1) if torch_mode else np.stack(a,-1)
    ones=torch.ones_like(u[...,0]) if torch_mode else np.ones_like(u[...,0])
    vals=[]; names=[]
    if kind=='edge':
        for j,n in enumerate(VARS): vals.append(delta[...,j]); names.append('delta_'+n)
        for j,n in enumerate(VARS): vals.append(delta[...,j]**2); names.append('(delta_'+n+')^2')
        for i,ni in enumerate(VARS):
            for j,nj in enumerate(VARS): vals.append(u[...,i]*delta[...,j]); names.append(ni+'*delta_'+nj)
        for i in range(5):
            for j in range(i+1,5):
                vals.append(delta[...,i]*delta[...,j]); names.append('delta_'+VARS[i]+'*delta_'+VARS[j])
    else:
        vals=[ones]; names=['1']
        for j,n in enumerate(VARS): vals.append(u[...,j]); names.append(n)
        for j,n in enumerate(VARS): vals.append(u[...,j]**2); names.append(n+'^2')
        for i in range(5):
            for j in range(i+1,5): vals.append(u[...,i]*u[...,j]); names.append(VARS[i]+'*'+VARS[j])
    return stack(vals),names

def stlsq(X,y,threshold,ridge):
    scale=np.sqrt(np.mean(X**2,axis=0)); scale[scale<1e-10]=1
    z=X/scale; active=np.ones(X.shape[1],bool); coef=np.zeros(X.shape[1])
    for _ in range(15):
        if not active.any(): break
        za=z[:,active]; c=np.linalg.solve(za.T@za+ridge*len(z)*np.eye(active.sum()),za.T@y)
        coef[:]=0; coef[active]=c
        new=active&(np.abs(coef)>=threshold)
        if np.array_equal(new,active): break
        active=new
    coef[~active]=0
    return coef/scale

@torch.no_grad()
def teacher_rows(model,graphs,sc,device,seed):
    rng=np.random.default_rng(seed); buckets={k:[] for k in ['F','G_U','G_D','G_L','log_sigma']}
    for g in graphs:
        T,N=g['y'].shape
        ntime=max(1,int(np.ceil(sc['rows_per_graph']/N)))
        times=np.arange(4,T)
        if len(times)>ntime: times=np.sort(rng.choice(times,ntime,replace=False))
        for t in times:
            x=filtered_particles(model,g,int(t),1,32,device)
            z=g['z'][t].to(device)[None]; u=torch.cat([x[...,None],z],-1)[0]
            my=g['my'][t].to(device); eligible=my
            rec=g['schema']['recording_id']
            for name,values in [('F',model.dynamics.self_value(u)),
                                ('log_sigma',torch.log(model.dynamics.sigma_value(u)))]:
                if eligible.any():
                    uu=u[eligible].cpu().numpy(); yy=values[eligible].cpu().numpy()
                    buckets[name].append((uu,None,yy,np.repeat(rec,len(yy))))
            if model.c.get('use_edges',True):
                for k,name in enumerate(['G_U','G_D','G_L']):
                    if k not in model.c.get('edge_types',[0,1,2]): continue
                    target,source=torch.where(g['A'][k].to(device)>0)
                    ok=my[target]&my[source]
                    if not ok.any(): continue
                    target,source=target[ok],source[ok]
                    uu=u[target]; delta=u[source]-uu
                    values=model.dynamics.edge_value(uu,delta,k)
                    buckets[name].append((uu.cpu().numpy(),delta.cpu().numpy(),values.cpu().numpy(),np.repeat(rec,len(values))))
    output={}
    for name,parts in buckets.items():
        if not parts: continue
        u=np.concatenate([p[0] for p in parts]); d=np.concatenate([p[1] for p in parts]) if parts[0][1] is not None else None
        y=np.concatenate([p[2] for p in parts]); rec=np.concatenate([p[3] for p in parts])
        ix=np.arange(len(y))
        if len(ix)>sc['max_rows_per_component']: ix=rng.choice(ix,sc['max_rows_per_component'],replace=False)
        X,names=library('edge' if name.startswith('G') else 'self',u[ix],d[ix] if d is not None else None)
        output[name]={'X':X.astype(float),'y':y[ix].astype(float),'record':rec[ix],'names':names}
    return output

class SymbolicDynamics(Dynamics):
    def __init__(self,equations,c):
        nn.Module.__init__(self); self.c=c; self.equations=equations
        for name in ['F','G_U','G_D','G_L','log_sigma']:
            self.register_buffer('coef_'+name,torch.tensor(equations['components'][name]['coefficients'],dtype=torch.float32))
    def self_value(self,u):
        X,_=library('self',u); return X@self.coef_F
    def edge_value(self,u,delta,k):
        X,_=library('edge',u,delta); return X@getattr(self,'coef_'+['G_U','G_D','G_L'][k])
    def sigma_value(self,u):
        X,_=library('self',u)
        logs=(X@self.coef_log_sigma).clamp(math.log(self.c['sigma_min']),math.log(self.c['sigma_max']))
        return torch.exp(logs)

def distill(work,run_dir,device):
    work=Path(work); r=Path(run_dir); dst=r/'symbolic'
    model,norm,receipt=load_run(r,device)
    plan=read_json(work/'plan.json')
    if receipt['plan_hash']!=plan['plan_hash'] or plan['code_hash']!=code_hash():
        raise ValueError('Source/plan changed since training.')
    if (dst/'complete.json').exists(): print(f'Cached symbolic: {dst}'); return
    cfg=receipt['config']; sc=cfg['symbolic']; fold=receipt['fold']; scale=receipt['scale_m']; seed=receipt['seed']
    train=[prepare_graph(p,norm,model.c) for p in split_paths(work,scale,fold,'train')]
    val=[prepare_graph(p,norm,model.c) for p in split_paths(work,scale,fold,'validation')]
    seed_all(seed+9000,cfg['training']['threads'])
    tr=teacher_rows(model,train,sc,device,seed); va=teacher_rows(model,val,sc,device,seed+1)
    # Preserve the symmetric hypothesis during equation extraction, not just NN training.
    if model.c.get('tie_longitudinal',False):
        for data in [tr,va]:
            if 'G_U' in data and 'G_D' in data:
                pooled={k:np.concatenate([data['G_U'][k],data['G_D'][k]]) for k in ['X','y','record']}
                pooled['names']=data['G_U']['names'];data['G_U']=pooled;data['G_D']=pooled
    components={}; selection=[]; stability=[]; rng=np.random.default_rng(seed)
    for name in ['F','G_U','G_D','G_L','log_sigma']:
        kind='edge' if name.startswith('G') else 'self'
        dummy=np.zeros((1,5)); _,names=library(kind,dummy,dummy)
        if name not in tr or name not in va:
            if name.startswith('G') and (not model.c.get('use_edges',True) or ['G_U','G_D','G_L'].index(name) not in model.c.get('edge_types',[0,1,2])):
                components[name]={'names':names,'coefficients':[0.]*len(names),'status':'no_edges_local_baseline'}; continue
            raise ValueError(f'Insufficient TRAIN/VALIDATION observed pairs for symbolic {name}. No invented equation.')
        a,b=tr[name],va[name]; trials=[]
        for threshold in sc['thresholds']:
            coef=stlsq(a['X'],a['y'],threshold,sc['ridge'])
            mse=float(np.mean((b['X']@coef-b['y'])**2))
            trials.append((mse,int(np.count_nonzero(coef)),threshold,coef))
        best=min(t[0] for t in trials)
        close=[t for t in trials if t[0]<=best*(1+sc['validation_relative_tolerance'])+1e-10]
        mse,nterms,threshold,coef=min(close,key=lambda t:(t[1],t[0]))
        components[name]={'names':a['names'],'coefficients':coef.tolist(),'threshold':threshold,
            'nonzero_terms':nterms,'validation_teacher_mse':mse,'train_rows':len(a['y']),'validation_rows':len(b['y']),
            'library':'degree<=2 physical standardized state; edges anchored at zero difference'}
        for t in trials: selection.append({'component':name,'threshold':t[2],'validation_mse':t[0],'terms':t[1]})
        recs=np.unique(a['record']); coeffs=[]
        if len(recs)>=2:
            for _ in range(sc['recording_bootstraps']):
                picks=rng.choice(recs,len(recs),replace=True)
                ix=np.concatenate([np.flatnonzero(a['record']==rr) for rr in picks])
                coeffs.append(stlsq(a['X'][ix],a['y'][ix],threshold,sc['ridge']))
        for j,term in enumerate(a['names']):
            bs=np.array(coeffs)[:,j] if coeffs else np.array([])
            stability.append({'component':name,'term':term,'coefficient':coef[j],
                'recording_selection_frequency':float(np.mean(bs!=0)) if len(bs) else None,
                'bootstrap_q025':float(np.quantile(bs,.025)) if len(bs) else None,
                'bootstrap_q975':float(np.quantile(bs,.975)) if len(bs) else None,
                'training_recording_count':len(recs)})
    equations={'version':VERSION,'kind':model.kind,'components':components,'normalizer':norm,
        'state_coordinate':('latent normalized risk potential x; Y=c_y*max(0,x+epsilon)' if model.kind=='latent'
                            else 'x=Y/c_y, censored Euler transition; observed-neighbor gates retained'),
        'time_unit':'seconds','z_order':['speed_mps','speed_cv','density_vehpkm','truck_share'],
        'z_definition':'(physical_z-z_mean)/z_std; forward-only completion, same numeric clipping as neural model',
        'sigma_definition':'exp(polynomial), numerically bounded by trained sigma_min/sigma_max',
        'not_claimed':['unique physical law','causal edge effect','exact continuous-time SDE','pure intrinsic noise'],
        'observation_sd':model.c['observation_sd'],'gauge':'G(u,0)=0','plan_hash':plan['plan_hash']}
    dst.mkdir(parents=True,exist_ok=True); write_json(dst/'equations.json',equations)
    pd.DataFrame(selection).to_csv(dst/'threshold_selection.csv',index=False)
    pd.DataFrame(stability).to_csv(dst/'recording_bootstrap_stability.csv',index=False)
    torch.manual_seed(seed+10000)
    teacher_by_h=validation_scores(model,val,norm,cfg['training'],cfg['evaluation'],device)
    teacher_score=float(np.mean(list(teacher_by_h.values())))
    model.dynamics=SymbolicDynamics(equations,model.c).to(device)
    torch.manual_seed(seed+10000)
    try:
        symbolic_by_h=validation_scores(model,val,norm,cfg['training'],cfg['evaluation'],device)
        symbolic_score=float(np.mean(list(symbolic_by_h.values())))
    except FloatingPointError:
        symbolic_score=float('inf'); symbolic_by_h={h:float('inf') for h in teacher_by_h}
    tolerance=float(sc.get('rollout_relative_tolerance',.20))
    accepted=all(np.isfinite(symbolic_by_h[h]) and symbolic_by_h[h]<=(1+tolerance)*max(v,1e-10)
                 for h,v in teacher_by_h.items())
    write_json(dst/'validation_rollout.json',{'neural_crps':teacher_score,'symbolic_crps':symbolic_score,
        'accepted_for_interpretation_under_engineering_20pct_tolerance':accepted,
        'neural_by_horizon':teacher_by_h,'symbolic_by_horizon':symbolic_by_h,
        'required_horizons_s':list(teacher_by_h),'relative_tolerance':tolerance,
        'acceptance_rule':'ALL prespecified horizons must pass; any unstable rollout is rejected',
        'note':'This tolerance is a study design choice, not a literature standard. No test was used.'})
    text=['# 数据驱动候选方程（不是已证明的物理定律）','',equations['state_coordinate'],'',
          '标准化和观测模型详见 equations.json；latent 模型的 x 不能直接改名成观测风险 Y。','']
    for name,c in components.items():
        expr=' + '.join(f'({coef:.8g})*{term}' for term,coef in zip(c['names'],c['coefficients']) if coef!=0) or '0'
        text.extend([f'## {name}',f'```text\n{expr}\n```',''])
    (dst/'equations.md').write_text('\n'.join(text),encoding='utf-8')
    write_json(dst/'complete.json',{'plan_hash':plan['plan_hash'],'accepted_validation_fidelity':accepted,
                                   'teacher_checkpoint':receipt['checkpoint_sha256'],
                                   'equations_sha256':file_hash(dst/'equations.json')})
    print(f'{dst}: validation symbolic fidelity accepted={accepted}',flush=True)
