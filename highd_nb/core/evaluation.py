"""Causal rolling-origin forecasts, censored-distribution scores, recording-cluster CIs."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import *
from .data import prepare_graph,split_paths
from .models import NetworkModel,filtered_particles,censored_log_prob

@torch.no_grad()
def forecast_origin(model,g,anchor,particles,horizons,history,device):
    N=g['y'].shape[1]; dt=g['dt']; A=g['A'].to(device)
    steps={int(round(h/dt)):h for h in horizons}
    if any(abs(k*dt-h)>1e-7 or k<1 for k,h in steps.items()): raise ValueError('Horizons must divide dt.')
    x=filtered_particles(model,g,anchor,particles,history,device)
    # The forecast NEVER reads z[t+1:] or y[t+1:].
    z=g['z'][anchor].to(device).expand(particles,N,4)
    gate=g['my'][anchor].to(device).expand(particles,N) if model.kind=='masked' else None
    answer={}
    for step in range(1,max(steps)+1):
        drift,sigma=model.dynamics(x,z,A,gate)
        pm=x+drift*dt; ps=sigma*math.sqrt(dt)
        if not torch.isfinite(pm).all() or not torch.isfinite(ps).all() or pm.abs().max()>1000:
            raise FloatingPointError('Unstable stochastic rollout. Reject this configuration/equation; do not clip silently.')
        x=pm+ps*torch.randn_like(pm)
        if model.kind=='latent':
            sd=torch.sqrt(ps**2+model.c['observation_sd']**2)
            samples=(x+model.c['observation_sd']*torch.randn_like(x)).clamp_min(0)
        else:
            x=x.clamp_min(0); samples=x; sd=ps
        if step in steps:
            answer[steps[step]]={'samples':samples.clone(),'mixture_mean':pm.clone(),'mixture_sd':sd.clone()}
    return answer

def score_distribution(pred,target,y_scale):
    samples=pred['samples']; pm,ps=pred['mixture_mean'],pred['mixture_sd']
    S=samples.shape[0]; average=samples.mean(0); sorted_s=samples.sort(0).values
    weights=(2*torch.arange(1,S+1,device=samples.device)-S-1).to(samples.dtype)
    crps=(samples-target).abs().mean(0)-(sorted_s*weights[:,None]).sum(0)/(S*S)
    lp=torch.logsumexp(censored_log_prob(target[None],pm,ps),dim=0)-math.log(S)
    # Positive densities transform with 1/y_scale; atom at zero does not.
    nll=-lp+torch.where(target>0,torch.full_like(target,math.log(y_scale)),torch.zeros_like(target))
    qlo,qhi=torch.quantile(samples,torch.tensor([.05,.95],device=samples.device),dim=0)
    pzero=torch.special.ndtr(-pm/ps).mean(0)
    return {'squared_error':((average-target)*y_scale)**2,'absolute_error':(average-target).abs()*y_scale,
            'crps':crps*y_scale,'nll':nll,'coverage90':((target>=qlo)&(target<=qhi)).float(),
            'width90':(qhi-qlo)*y_scale,'zero_brier':(pzero-(target==0).float())**2,
            'standardized_error':(target-average)/samples.std(0,unbiased=False).clamp_min(1e-4),
            'pred_mean':average*y_scale,'pred_sd':samples.std(0,unbiased=False)*y_scale,
            'pred_q05':qlo*y_scale,'pred_q95':qhi*y_scale,'pred_pzero':pzero}

@torch.no_grad()
def evaluate_graph(model,g,normalizer,ec,device,include_persistence=True):
    T,N=g['y'].shape; dt=g['dt']; S=ec['particles']; horizons=ec['horizons_s']
    start=max(int(round(g['schema']['config']['context_window_s']/dt))-1,0)
    anchors=list(range(start,T-int(round(max(horizons)/dt)),ec['origin_stride']))
    cap=ec.get('max_origins_per_graph')
    if cap and len(anchors)>cap: anchors=[anchors[i] for i in np.linspace(0,len(anchors)-1,cap,dtype=int)]
    rows=[]
    for anchor in anchors:
        if not g['my'][anchor].any(): continue
        forecasts=forecast_origin(model,g,anchor,S,horizons,ec['history_steps'],device)
        for h,pred in forecasts.items():
            t=anchor+int(round(h/dt))
            valid=(g['my'][anchor]&g['my'][t]).numpy()
            if not valid.any(): continue
            target=g['y'][t].to(device)
            models={'network':pred}
            if include_persistence:
                pm=g['y'][anchor].to(device).expand(S,N)
                ps=torch.full_like(pm,normalizer['persistence_sigma_per_sqrt_second']*math.sqrt(h))
                models['persistence']={'samples':(pm+ps*torch.randn_like(pm)).clamp_min(0),
                                       'mixture_mean':pm,'mixture_sd':ps}
            A=g['A'].numpy().any(0); my=g['my'][anchor].numpy()
            near_complete=(my.astype(int)@A.T)==A.sum(-1)
            for name,pr in models.items():
                scores={k:v.detach().cpu().numpy() for k,v in score_distribution(pr,target,normalizer['y_scale']).items()}
                target_cpu=target.detach().cpu().numpy()
                for n in np.flatnonzero(valid):
                    row={'graph':g['name'],'recording_id':g['schema']['recording_id'],
                         'location_id':g['schema']['location_id'],'direction':g['schema']['direction'],
                         'anchor_s':float(g['time'][anchor]),'node':int(n),'horizon_s':h,
                         'predictor':name,'neighbor_complete':bool(near_complete[n]),
                         'target_y':float(target_cpu[n])*normalizer['y_scale'],
                         'anchor_y':float(g['y'][anchor,n])*normalizer['y_scale'],
                         'target_s':float(g['time'][t]),'anchor_index':anchor}
                    row.update({key:float(value[n]) for key,value in scores.items()}); rows.append(row)
    return rows

@torch.no_grad()
def validation_scores(model,graphs,normalizer,training,evaluation,device):
    ec=dict(evaluation); ec.update(particles=training['validation_draws'],
                                  origin_stride=training['validation_origin_stride'],
                                  max_origins_per_graph=training['validation_max_origins_per_graph'])
    rows=[]
    for g in graphs: rows.extend(evaluate_graph(model,g,normalizer,ec,device,False))
    if not rows: raise ValueError('No common observed validation targets. Change study design, not test set.')
    scores={float(h):float(np.mean([r['crps'] for r in rows if r['horizon_s']==h])) for h in ec['horizons_s']}
    if not all(np.isfinite(v) for v in scores.values()): raise ValueError('An intended validation horizon lacks support.')
    return scores

@torch.no_grad()
def validation_score(model,graphs,normalizer,training,evaluation,device):
    # Equal horizon weights; NEVER silently replace the user's horizons by one step.
    return float(np.mean(list(validation_scores(model,graphs,normalizer,training,evaluation,device).values())))

def aggregate_scores(df,keys):
    metrics=['squared_error','absolute_error','crps','nll','coverage90','width90','zero_brier']
    out=df.groupby(keys,dropna=False)[metrics].mean().reset_index()
    count=df.groupby(keys,dropna=False).size().reset_index(name='n_correlated_node_time_targets')
    out=out.merge(count,on=keys); out['rmse']=np.sqrt(out.pop('squared_error')); out=out.rename(columns={'absolute_error':'mae'})
    if 'target_y' in df:
        moments=df.groupby(keys,dropna=False)['target_y'].agg(target_mean='mean',target_variance=lambda x: float(np.var(x))).reset_index()
        out=out.merge(moments,on=keys)
        out['r2']=np.where(out.target_variance>1e-12,1-out.rmse**2/out.target_variance,np.nan)
        out['nrmse_mean']=np.where(out.target_mean.abs()>1e-12,out.rmse/out.target_mean.abs(),np.nan)
    return out

def bootstrap_ci(df,n_boot,seed):
    # Resample WHOLE recordings, retaining both directions and all overlapping origins.
    result=[]; rng=np.random.default_rng(seed)
    for (pred,h),g in df.groupby(['predictor','horizon_s']):
        recs=g.recording_id.unique(); records=[]
        for r in recs:
            v=g[g.recording_id==r]; records.append((len(v),v.squared_error.sum(),v.crps.sum(),v.nll.sum()))
        a=np.array(records,float)
        for metric,col in [('rmse',1),('crps',2),('nll',3)]:
            center=a[:,col].sum()/a[:,0].sum(); center=np.sqrt(center) if metric=='rmse' else center
            lo=hi=None
            if len(a)>=2 and n_boot>0:
                picks=rng.integers(len(a),size=(n_boot,len(a))); draws=a[picks]
                values=draws[:,:,col].sum(1)/draws[:,:,0].sum(1)
                if metric=='rmse': values=np.sqrt(values)
                lo,hi=np.quantile(values,[.025,.975])
            result.append({'predictor':pred,'horizon_s':h,'metric':metric,'estimate':center,
                           'ci_low':lo,'ci_high':hi,'independent_resampling_units':len(a),
                           'note':'conditional on trained seed/model; resampling unit=recording, not row'})
    return pd.DataFrame(result)

def load_run(run_dir,device):
    r=Path(run_dir); receipt=read_json(r/'training_receipt.json'); checkpoint=r/'best.pt'
    if file_hash(checkpoint)!=receipt['checkpoint_sha256']: raise ValueError(f'Checkpoint hash mismatch: {r}')
    # Only self-generated checkpoints. Never load an untrusted pickle/checkpoint.
    ck=torch.load(checkpoint,map_location=device,weights_only=True)
    model=NetworkModel(ck['kind'],ck['model_config']).to(device); model.load_state_dict(ck['state_dict']); model.eval()
    return model,read_json(r/'normalizer.json'),receipt

def evaluate_run(work,run_dir,device,symbolic=False):
    work=Path(work); r=Path(run_dir)
    model,norm,receipt=load_run(r,device)
    plan=read_json(work/'plan.json')
    if receipt['plan_hash']!=plan['plan_hash'] or plan['code_hash']!=code_hash():
        raise ValueError('Frozen plan/source differs from training. Use a new work/run.')
    if symbolic:
        from .symbolic import SymbolicDynamics
        status=read_json(r/'symbolic'/'complete.json')
        if not status['accepted_validation_fidelity']:
            print(f'SKIP symbolic test: equation failed validation fidelity at {r}. No test tuning.')
            return
        eq=read_json(r/'symbolic'/'equations.json')
        if file_hash(r/'symbolic'/'equations.json')!=status['equations_sha256']:
            raise ValueError('Symbolic equation changed after validation.')
        model.dynamics=SymbolicDynamics(eq,model.c).to(device)
    ec=receipt['config']['evaluation']; fold=receipt['fold']; scale=receipt['scale_m']
    dst=r/('symbolic_test' if symbolic else 'test')
    if (dst/'complete.json').exists():
        print(f'{dst}: cached evaluation (not re-opened for tuning).'); return
    seed_all(receipt['seed']+100000,receipt['config']['training']['threads'])
    rows=[]
    for p in split_paths(work,scale,fold,'test'):
        if file_hash(p)!=plan['data_fingerprints'][f'L{scale}/{p.parent.name}']:
            raise ValueError(f'Test data hash changed: {p}')
        g=prepare_graph(p,norm,model.c); rows.extend(evaluate_graph(model,g,norm,ec,device))
    if not rows: raise ValueError('No test targets on the prespecified common support.')
    dst.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows)
    df.to_csv(dst/'predictions_scores.csv.gz',index=False,compression='gzip')
    aggregate_scores(df,['predictor','horizon_s']).to_csv(dst/'metrics.csv',index=False)
    aggregate_scores(df,['predictor','horizon_s','recording_id']).to_csv(dst/'recording_metrics.csv',index=False)
    aggregate_scores(df,['predictor','horizon_s','neighbor_complete']).to_csv(dst/'coverage_strata.csv',index=False)
    bootstrap_ci(df,ec['bootstrap_recordings'],receipt['seed']).to_csv(dst/'cluster_bootstrap_ci.csv',index=False)
    # Forecast error ACF, explicitly NOT a proof of Brownian diffusion.
    acf=[]
    for (graph,node,h),v in df[df.predictor=='network'].groupby(['graph','node','horizon_s']):
        v=v.sort_values('anchor_s'); x=v.standardized_error.to_numpy(); times=v.anchor_s.to_numpy()
        ok=np.isclose(np.diff(times),ec['origin_stride']*receipt['config']['base_data']['step_s'])
        if ok.sum()>=5 and np.std(x[:-1][ok])>0 and np.std(x[1:][ok])>0:
            acf.append({'graph':graph,'node':node,'horizon_s':h,'n_pairs':int(ok.sum()),
                        'adjacent_evaluation_origin_correlation':np.corrcoef(x[:-1][ok],x[1:][ok])[0,1]})
    pd.DataFrame(acf).to_csv(dst/'forecast_error_autocorrelation.csv',index=False)
    write_json(dst/'complete.json',{'plan_hash':plan['plan_hash'],'seed':receipt['seed'],'scale_m':scale,
        'fold':fold,'symbolic_dynamics':symbolic,
        'target':'observed future endpoint risk, anchor also observed; not crash probability',
        'covariates':'frozen at forecast anchor; no oracle future traffic variables',
        'symbolic_inference_note':'latent posterior encoder is retained when replacing dynamics by equations' if symbolic else None,
        'n_scored_rows':len(df)})
    print(f'Saved {dst}',flush=True)
