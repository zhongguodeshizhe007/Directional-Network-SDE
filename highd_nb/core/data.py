"""Train-only scaling; strictly forward-only covariate completion; recording-safe windows."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .common import *
KEYS=['y','mask_y','z','mask_z','adjacency_by_type','time_s','pair_exposure_s']

def read_model_arrays(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k] for k in KEYS}

def split_paths(work,scale,fold,part):
    locs=fold['train_locations'] if part=='train' else [fold['validation_location'] if part=='validation' else fold['test_location']]
    return [p for p in graph_paths(work,scale) if graph_schema(p)['location_id'] in locs]

def fit_normalizer(paths):
    counts=np.zeros(4); sums=np.zeros(4); squares=np.zeros(4)
    ycount=0; ysquare=0.; diff=[]
    for p in paths:
        a=read_model_arrays(p); z=a['z']; m=a['mask_z']&np.isfinite(z)
        v=np.where(m,z,0); counts+=m.sum((0,1)); sums+=v.sum((0,1),dtype=np.float64)
        squares+=(v.astype(float)**2).sum((0,1))
        y=a['y'][...,0]; my=a['mask_y'][...,0]
        ysquare+=float((y[my].astype(float)**2).sum()); ycount+=int(my.sum())
        pairs=my[:-1]&my[1:]; diff.append((y[1:]-y[:-1])[pairs] / np.sqrt(float(graph_schema(p)['config']['step_s'])))
    if ycount==0 or (counts==0).any(): raise ValueError('No observed training values for one or more inputs.')
    mean=sums/counts; std=np.sqrt(np.maximum(squares/counts-mean**2,0)); std[std<1e-7]=1.
    ys=max(np.sqrt(ysquare/ycount),1e-3) # no risk centering: zero remains the censoring boundary
    delta=np.concatenate(diff)
    ps=max(float(np.std(delta/ys)) if len(delta) else .1,.01)
    return {'z_mean':mean.tolist(),'z_std':std.tolist(),'y_scale':ys,
            'persistence_sigma_per_sqrt_second':ps,'observed_counts':counts.tolist(),
            'fit_graphs':[p.parent.name for p in paths],
            'note':'Training only. y is divided by RMS, not centered. Physical y=scale*normalized risk.'}

def forward_covariates(z,mask,mean,std,dt,max_age,clip):
    """No backward filling/interpolation. Missing normalized defaults are train means (=0)."""
    T,N,F=z.shape; out=np.zeros((T,N,F),np.float32); ages=np.full_like(out,max_age+dt)
    last=np.zeros((N,F),np.float32); age=np.full((N,F),max_age+dt,np.float32)
    for t in range(T):
        ok=mask[t]&np.isfinite(z[t]); val=(np.where(ok,z[t],mean)-mean)/std
        age=np.minimum(age+dt,max_age+dt); age[ok]=0
        last=np.where(ok,val,last); out[t]=np.where(age<=max_age,last,0)
        ages[t]=age
    return np.clip(out,-clip,clip),ages/(max_age+dt)

def prepare_graph(path,normalizer,mcfg):
    a=read_model_arrays(path); s=graph_schema(path); dt=float(s['config']['step_s'])
    zy,zage=forward_covariates(a['z'],a['mask_z'],np.asarray(normalizer['z_mean']),
              np.asarray(normalizer['z_std']),dt,mcfg['max_covariate_carry_s'],mcfg['z_clip_standard_deviations'])
    y=a['y'][...,0]/normalizer['y_scale']; m=a['mask_y'][...,0]&np.isfinite(y)
    return {'path':str(path),'name':Path(path).parent.name,'schema':s,'dt':dt,
        'y':torch.from_numpy(np.where(m,y,0).astype(np.float32)),
        'my':torch.from_numpy(m),'z':torch.from_numpy(zy),
        'mz':torch.from_numpy(a['mask_z']),'az':torch.from_numpy(zage),
        'A':torch.from_numpy(a['adjacency_by_type'].astype(np.float32)),
        'exposure':a['pair_exposure_s'],'time':a['time_s']}

def window_batches(graphs,length,stride,batch_size,shuffle,rng,device):
    order=np.arange(len(graphs));
    if shuffle: rng.shuffle(order)
    for gi in order:
        g=graphs[gi]; T=len(g['y'])
        if T<length: continue
        starts=np.arange(0,T-length+1,stride)
        if shuffle: rng.shuffle(starts)
        for off in range(0,len(starts),batch_size):
            chosen=starts[off:off+batch_size]
            b={k:torch.stack([g[k][i:i+length] for i in chosen]).to(device) for k in ['y','my','z','mz','az']}
            b['A']=g['A'].to(device); b['dt']=g['dt']; yield b
