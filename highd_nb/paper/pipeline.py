"""Paper experiment extension of the frozen v4 NetworkSDE workflow.
Fits on training locations, selects/calibrates on validation, scores held-out test once.
Does NOT download data, manufacture empirical results, or silently substitute models.
"""
from __future__ import annotations
import copy,time
from highd_nb.runtime import synchronize,device_summary
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from highd_nb.core.common import (read_json,write_json,code_hash,digest,file_hash,environment,seed_all)
from highd_nb.core.data import split_paths
from highd_nb.core.evaluation import (load_run,forecast_origin,score_distribution,aggregate_scores,bootstrap_ci)
from highd_nb.core.symbolic import SymbolicDynamics
from .data import load_graphs,windows,metadata
from .baselines import train_sequence,predict_sequence,train_tabular
from .events import dataset,fit_event_readout,event_metrics,reliability_table,cluster_ci,warning_episodes,success_failure_table,horizon_coherence
from .context import run_context
from .propagation import event_census,transport_audit,distinct_pair_links,component_samples,negative_controls


def table(dst,name,df):
    """Keep machine-readable source and editable manuscript table side by side."""
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    if not len(df.columns):df=pd.DataFrame({'status':['NO_ELIGIBLE_OBSERVATIONS']})
    df.to_csv(dst/(name+'.csv'),index=False)
    # Escaping protects underscores/IDs. No synthetic values are disguised as paper results.
    (dst/(name+'.tex')).write_text(df.to_latex(index=False,float_format=lambda x:f'{x:.5g}',escape=True),encoding='utf-8')


def _sample_prediction(mu,sd,particles,device):
    mu=torch.as_tensor(mu,dtype=torch.float32,device=device);sd=torch.as_tensor(sd,dtype=torch.float32,device=device)
    pm=mu[None].expand(particles,-1);ps=sd[None].expand_as(pm)
    if not torch.isfinite(pm).all() or not torch.isfinite(ps).all() or (ps<=0).any():
        raise FloatingPointError('Nonfinite baseline distribution')
    return {'samples':(pm+ps*torch.randn_like(pm)).clamp_min(0),'mixture_mean':pm,'mixture_sd':ps}


@torch.no_grad()
def forecast_rows(model_name,model,graphs,ws,norm,horizons,pc,device,score=True):
    """Same causal origins and evaluation support across predictors. No oracle covariates."""
    rows=[];stats={};S=int(pc['particles'])
    batchpred=predict_sequence(model,graphs,ws,horizons,pc,device) if model_name in ['lstm','cnn','convlstm','st_transformer'] else None
    for w in ws:
        g=graphs[w.graph];N=g['y'].shape[1]
        # Common random numbers across neural/ablated/symbolic SDE at each origin.
        seed=int(pc['seed'])+100003*w.graph+97*w.anchor;torch.manual_seed(seed)
        if model_name.startswith('SDE'):
            preds=forecast_origin(model,g,w.anchor,S,horizons,pc['history_steps'],device)
        else:
            if model_name=='persistence':
                mu=np.repeat(g['y'][w.anchor].numpy()[:,None],len(horizons),axis=1)
                sd=np.broadcast_to(norm['persistence_sigma_per_sqrt_second']*np.sqrt(horizons),(N,len(horizons)))
            elif batchpred is not None:mu,sd=batchpred[(w.graph,w.anchor)]
            else:mu,sd=model.predict(g,w.anchor)
            preds={float(h):_sample_prediction(mu[:,j],sd[:,j],S,device) for j,h in enumerate(horizons)}
        for h,pr in preds.items():
            # Distribution summaries do not depend on future y. Only scoring does.
            stats[(w.graph,w.anchor,float(h))]={'mean':pr['samples'].mean(0).cpu().numpy(),
                  'sd':pr['samples'].std(0,unbiased=False).cpu().numpy(),
                  'pzero':torch.special.ndtr(-pr['mixture_mean']/pr['mixture_sd']).mean(0).cpu().numpy()}
            if not score:continue
            t=w.anchor+int(round(h/g['dt']));valid=(g['my'][w.anchor]&g['my'][t]).numpy();ids=np.flatnonzero(valid)
            if not len(ids):continue
            target=g['y'][t].to(device)
            scores={k:v.detach().cpu().numpy() for k,v in score_distribution(pr,target,norm['y_scale']).items()}
            target_cpu=target.detach().cpu().numpy()
            metas=metadata(g,w.anchor,h,ids)
            for n,row in zip(ids,metas):
                row.update(predictor=model_name,target_y=float(target_cpu[n])*norm['y_scale'],
                           anchor_y=float(g['y'][w.anchor,n])*norm['y_scale'])
                row.update({k:float(v[n]) for k,v in scores.items()});rows.append(row)
    return pd.DataFrame(rows),stats


def _save_baseline(model,name,dst,horizons,pc):
    dst.mkdir(parents=True,exist_ok=True)
    if name in ['ridge','xgboost']:
        write_json(dst/'distribution.json',{'kind':name,'history_steps':pc['history_steps'],'horizons':horizons,'residual_sd':model.sds})
        for h,est in zip(horizons,model.models):
            if name=='xgboost':est.save_model(dst/f'h{h:g}.json')
            else:np.savez(dst/f'h{h:g}.npz',coef=est.coef_,intercept=est.intercept_)
        return sum(len(m.coef_)+1 for m in model.models) if name=='ridge' else np.nan
    torch.save({'kind':name,'history':pc['history_steps'],'horizons':list(horizons),'hidden':pc['hidden'],'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},dst/'model.pt')
    return sum(p.numel() for p in model.parameters())


def _support(gs,part):
    rows=[]
    for g in gs:
        a=g['paper_arrays'];names=a['risk_channel_names']
        for j,name in enumerate(names):
            m=a['mask_risk_channels'][...,j];v=a['risk_channels_raw'][...,j][m]
            rows.append(dict(split=part,graph=g['name'],recording_id=str(g['schema']['recording_id']),
                location_id=g['schema']['location_id'],direction=g['schema']['direction'],risk_metric=str(name),
                total_cells_times=int(m.size),observed=int(m.sum()),observed_fraction=float(m.mean()),
                zero_fraction=float((v==0).mean()) if len(v) else np.nan,mean=float(np.mean(v)) if len(v) else np.nan,
                sd=float(np.std(v)) if len(v) else np.nan,median=float(np.median(v)) if len(v) else np.nan,
                p95=float(np.quantile(v,.95)) if len(v) else np.nan,max=float(np.max(v)) if len(v) else np.nan))
    return rows


def run_paper_suite(work,run_tag='main',folds='all',confirm_test=False,device='auto'):
    if not confirm_test:raise PermissionError('Set confirm_test=True only after fixing the paper protocol; this opens held-out test data.')
    from highd_nb.runtime import choose_device
    device=choose_device(device)
    work=Path(work);plan=read_json(work/'plan.json')
    if plan['code_hash']!=code_hash():raise ValueError('v5 source changed after planning. Use a fresh output directory and refit.')
    refs=sorted((work/'models'/run_tag).glob('L*/test_*/*/seed_*/training_receipt.json'))
    wanted=None if folds=='all' else {int(v) for v in str(folds).split(',')}
    groups={}
    for p in refs:
        r=read_json(p)
        if wanted is not None and r['fold']['test_location'] not in wanted:continue
        if r['plan_hash']!=plan['plan_hash']:raise ValueError('Model from a different frozen plan')
        key=(r['scale_m'],r['fold']['test_location'],r['seed'])
        groups.setdefault(key,[]).append(p.parent)
    if not groups:raise ValueError('No completed core training runs. Execute v5 training first.')
    results=[]
    for key,runs in groups.items():
        results.append(_run_group(work,plan,runs,key,device))
    manifest={'status':'COMPLETE','groups':results,'data_scope':'SYNTHETIC SOFTWARE TEST ONLY' if plan['synthetic'] else 'LOCAL USER DATA',
              'claim':'Tables are measured outputs; no guarantee of superiority/significance; no crash ground truth inferred'}
    _summarize_groups(work,results)
    write_json(work/'paper'/'manifest.json',manifest)
    return manifest


def _run_group(work,plan,runs,key,device):
    scale,foldid,seed=key;cfg=copy.deepcopy(plan['config']);pc=cfg['paper'];pc['seed']=seed
    horizons=cfg['evaluation']['horizons_s'];first,norm,receipt=load_run(runs[0],device);fold=receipt['fold'];mc=cfg['model']
    dst=work/'paper'/f'L{scale:03d}'/f'test_{foldid}'/f'seed_{seed}';tables=dst/'tables'
    signature=digest({'plan':plan['plan_hash'],'runs':[(str(r.relative_to(work)),file_hash(r/'training_receipt.json')) for r in runs],
                      'paper':pc,'horizons':horizons})
    if (dst/'complete.json').exists():
        old=read_json(dst/'complete.json')
        if old['signature']!=signature:raise ValueError('Paper output already belongs to a different protocol')
        return old
    if (dst/'protocol.json').exists() and read_json(dst/'protocol.json')['signature']!=signature:
        raise ValueError('Do not change the protocol in a previously opened paper test directory')
    dst.mkdir(parents=True,exist_ok=True);seed_all(seed,1)
    write_json(dst/'protocol.json',{'signature':signature,'plan_hash':plan['plan_hash'],'code_hash':code_hash(),
        'paper_config':pc,'horizons_s':horizons,'fold':fold,'risk_metric':cfg['base_data'].get('risk_metric','inv_ttc'),
        'scope':'SYNTHETIC SOFTWARE TEST ONLY' if plan['synthetic'] else 'LOCAL USER HIGH-D FORMAT DATA',
        'same_origins':True,'test_resampled':False,
        'event_evaluation_support':'anchor observed, not active, complete future pair support through maximum horizon',
        'hazard_interpretation':'discrete-time first-confirmed-onset probability; NOT SDE boundary crossing',
        'device':device_summary(device),'event_reference':'full-rate TTC episode confirmation',
        'context_scope':'same-lane car-following conditional proximity; Jiao-inspired NOT full 2D reproduction',
        'st_transformer_scope':'spatial+temporal attention baseline; NOT the original AAP periodic STTN',
        'nll_comparison':'censored observation densities in the SAME transformed risk units',
        'symbolic_relative_tolerance':cfg['symbolic'].get('rollout_relative_tolerance',.2),
        'environment':environment()})
    gs={};ws={}
    for part in ['train','validation','test']:
        paths=split_paths(work,scale,fold,part)
        for p in paths:
            if file_hash(p)!=plan['data_fingerprints'][f'L{scale}/{p.parent.name}']:raise ValueError(f'Changed data: {p}')
        gs[part]=load_graphs(paths,norm,mc)
        ws[part]=windows(gs[part],pc['history_steps'],horizons,
            pc['train_origin_stride'] if part=='train' else pc['test_origin_stride'],
            pc['max_train_origins_per_graph'] if part=='train' else pc.get('max_test_origins_per_graph'))
    table(tables,'T01_data_support_and_SSMs',pd.DataFrame([r for p in gs for r in _support(gs[p],p)]))
    write_json(dst/'normalizer.json',norm)
    predictions=[];stats_by_model={};protocol=[];symbolic_rows=[];coefficient_frames=[];complexity_frames=[];component_frames=[];control_frames=[]
    for run in runs:
        model,other_norm,rc=load_run(run,device)
        if digest(other_norm)!=digest(norm):raise ValueError('Compared models do not share training-only normalisation')
        name='SDE-'+rc['name'];stats_by_model[name]={}
        test_wall_s=0.
        for part in ['train','validation','test']:
            synchronize(device); started=time.perf_counter()
            print(f'Paper {name}: {part}',flush=True)
            out,st=forecast_rows(name,model,gs[part],ws[part],norm,horizons,pc,device,score=part=='test')
            stats_by_model[name][part]=st
            if part=='test':
                predictions.append(out);synchronize(device);test_wall_s=time.perf_counter()-started
        protocol.append(dict(predictor=name,kind=model.kind,parameters=sum(p.numel() for p in model.parameters()),
                             history_steps=pc['history_steps'],checkpoint=str(run.relative_to(work)),selection='validation multi-horizon CRPS',
                             actual_device=str(device),test_forecast_and_scoring_wall_s=test_wall_s,
                             test_origins=len(ws['test']),
                             training_and_validation_wall_s=read_json(run/'runtime.json').get('training_and_validation_wall_s') if (run/'runtime.json').exists() else None))
        if rc['name'] in ['masked','latent']:
            ctrl=negative_controls(model,gs['test'],ws['test'],norm,horizons,pc,device);ctrl['predictor']=name;control_frames.append(ctrl)
        comp=component_samples(model,gs['test'],ws['test'],pc,device);comp['predictor']=name;component_frames.append(comp)
        sp=run/'symbolic'/'complete.json'
        if sp.exists():
            status=read_json(sp);val=read_json(run/'symbolic'/'validation_rollout.json')
            for filename,bucket in [('recording_bootstrap_stability.csv',coefficient_frames),('threshold_selection.csv',complexity_frames)]:
                frame=pd.read_csv(run/'symbolic'/filename);frame['predictor']=name
                frame['accepted_all_horizons']=status['accepted_validation_fidelity'];bucket.append(frame)
            for h,v in val['neural_by_horizon'].items():
                symbolic_rows.append(dict(predictor=name,horizon_s=float(h),neural_crps=v,
                    symbolic_crps=val['symbolic_by_horizon'][h],accepted_all_horizons=status['accepted_validation_fidelity']))
            if status['accepted_validation_fidelity']:
                eq=read_json(run/'symbolic'/'equations.json')
                if file_hash(run/'symbolic'/'equations.json')!=status['equations_sha256']:raise ValueError('Changed accepted equation')
                model.dynamics=SymbolicDynamics(eq,model.c).to(device)
                out,_=forecast_rows(name+'-symbolic',model,gs['test'],ws['test'],norm,horizons,pc,device);predictions.append(out)
        # Preserve all candidates/rejections for a manuscript, not just attractive formulas.
    out,_=forecast_rows('persistence',None,gs['test'],ws['test'],norm,horizons,pc,device);predictions.append(out)
    protocol.append(dict(predictor='persistence',kind='censored random walk',parameters=0,history_steps=1,selection='training increment variance'))
    for kind in pc['models']:
        print(f'Paper fitting {kind}',flush=True)
        synchronize(device);fit_started=time.perf_counter()
        if kind in ['ridge','xgboost']:
            fitted=train_tabular(kind,gs['train'],ws['train'],gs['validation'],ws['validation'],horizons,pc,device)
        else:
            fitted,curve=train_sequence(kind,gs['train'],ws['train'],gs['validation'],ws['validation'],horizons,pc,device)
            (dst/'learning_curves').mkdir(exist_ok=True);curve.to_csv(dst/'learning_curves'/f'{kind}.csv',index=False)
        synchronize(device);fit_wall_s=time.perf_counter()-fit_started
        npars=_save_baseline(fitted,kind,dst/'baselines'/kind,horizons,pc)
        synchronize(device);prediction_started=time.perf_counter()
        out,_=forecast_rows(kind,fitted,gs['test'],ws['test'],norm,horizons,pc,device);predictions.append(out)
        synchronize(device)
        protocol.append(dict(predictor=kind,kind=kind,parameters=npars,history_steps=pc['history_steps'],selection='validation only',
            actual_device='cpu' if kind=='ridge' else str(device),training_and_validation_wall_s=fit_wall_s,
            test_forecast_and_scoring_wall_s=time.perf_counter()-prediction_started,test_origins=len(ws['test'])))
        del fitted
    pred=pd.concat(predictions,ignore_index=True)
    # Fail rather than compare different samples when a model silently drops difficult targets.
    from .validation import assert_common_prediction_support
    assert_common_prediction_support(pred,'predictor',['graph','anchor_index','node','horizon_s'])
    idcols=['graph','anchor_index','node','horizon_s'];sizes=pred.groupby('predictor').size()
    if sizes.nunique()!=1:raise ValueError('Predictors have unequal scoring support')
    if pred.duplicated(['predictor']+idcols).any():raise ValueError('Duplicate scored targets')
    pred.to_csv(dst/'continuous_predictions.csv.gz',index=False)
    metric=aggregate_scores(pred,['predictor','horizon_s']);base=metric[metric.predictor=='persistence'][['horizon_s','crps','rmse']]
    metric=metric.merge(base.rename(columns={'crps':'persistence_crps','rmse':'persistence_rmse'}),on='horizon_s')
    metric['crps_skill_vs_persistence']=1-metric.crps/metric.persistence_crps
    table(tables,'T02_model_protocol',pd.DataFrame(protocol));table(tables,'T03_continuous_forecasts',metric)
    table(tables,'T10_compute_cost',pd.DataFrame(protocol)[[k for k in ['predictor','actual_device','parameters','training_and_validation_wall_s','test_forecast_and_scoring_wall_s','test_origins'] if k in pd.DataFrame(protocol)]])
    write_json(dst/'device.json',device_summary(device))
    table(tables,'T03b_recording_cluster_CI',bootstrap_ci(pred,pc['bootstrap_recordings'],seed))
    table(tables,'T03c_by_recording',aggregate_scores(pred,['predictor','horizon_s','recording_id']))
    table(tables,'T05_orthogonal_ablations',metric[metric.predictor.str.startswith('SDE')])
    table(tables,'T07_symbolic_acceptance',pd.DataFrame(symbolic_rows))
    if coefficient_frames:
        coef=pd.concat(coefficient_frames,ignore_index=True)
        table(tables,'T07b_symbolic_coefficient_stability',coef)
        table(tables,'T07c_nonzero_candidate_terms',coef[coef.coefficient!=0])
    if complexity_frames:table(tables,'T07d_symbolic_complexity',pd.concat(complexity_frames,ignore_index=True))
    if control_frames:table(tables,'T09c_graph_negative_controls',pd.concat(control_frames,ignore_index=True))
    if component_frames:pd.concat(component_frames,ignore_index=True).to_csv(dst/'dynamics_components.csv.gz',index=False)
    eventpred=[];eventreceipts=[]
    # This readout maps a future risk distribution AND historical information to an event.
    # It is not an analytical transformation of the cell-average inverse TTC.
    for ttc in pc['event_thresholds_s']:
        for h in horizons:
            kinds=['Local-logit','Directional-logit','XGBoost-event','Flow-augmented-logit']+[n+'-readout' for n in stats_by_model if not any(a in n for a in ['local','no_lateral','symmetric','const_sigma'])]
            for name in kinds:
                sd=stats_by_model.get(name.removesuffix('-readout'))
                sets=[dataset(gs[p],ws[p],pc['history_steps'],h,ttc,pc['event_support'],
                              predictions=sd[p] if sd else None,local=name=='Local-logit',flow=name=='Flow-augmented-logit',support_horizon=max(horizons)) for p in ['train','validation','test']]
                ep,er=fit_event_readout(name,*sets,pc,ttc,h,checkpoint_dir=dst/'event_models'/name/f'H{h:g}_TTC{ttc:g}',device=device)
                er.update(model=name,horizon_s=h,ttc_threshold_s=ttc);eventreceipts.append(er)
                if ep is not None:eventpred.append(ep)
    if pc.get('event_hazard_enabled',False):
        from .hazard import run_hazard_comparisons
        hp,hr=run_hazard_comparisons(gs,ws,stats_by_model,horizons,pc,dst/'event_models',device)
        eventpred.extend(hp);eventreceipts.extend(hr)
    write_json(dst/'event_readout_receipts.json',eventreceipts)
    events=event_census(gs['test']);events.to_csv(dst/'test_events.csv',index=False)
    table(tables,'T09_same_pair_transport_audit',transport_audit(events))
    table(tables,'T09b_distinct_pair_adjacency',distinct_pair_links(events,gs['test']))
    if eventpred:
        ep=pd.concat(eventpred,ignore_index=True)
        for _,sub in ep.groupby(['ttc_threshold_s','horizon_s']):
            assert_common_prediction_support(sub,'model',['graph','anchor_index','node'])
        ep.to_csv(dst/'event_predictions.csv.gz',index=False)
        table(tables,'T04_event_probability',event_metrics(ep));table(tables,'T04b_event_cluster_CI',cluster_ci(ep,pc['bootstrap_recordings'],seed))
        table(tables,'T04c_horizon_probability_coherence',horizon_coherence(ep))
        reliability_table(ep).to_csv(dst/'event_reliability.csv',index=False)
        warns=warning_episodes(ep,events,cfg['base_data']['step_s']) if not events.empty else pd.DataFrame()
        warns.to_csv(dst/'event_warnings.csv',index=False)
        if not warns.empty:
            summ=warns.groupby(['model','horizon_s','ttc_threshold_s']).agg(events=('event_id','size'),
                detected_fraction=('detected','mean'),warning_period=('warning_period_fraction','mean'),
                median_first_lead_s=('first_warning_lead_s','median'),median_last_transition_lead_s=('last_warning_transition_lead_s','median')).reset_index()
            table(tables,'T06_warning_episodes',summ);table(tables,'T06b_success_failure_patterns',success_failure_table(warns))
    else:table(tables,'T04_event_probability',pd.DataFrame(eventreceipts))
    print('Paper context-dependent proximity study',flush=True)
    from .context import load_pair_tables
    rawpairs=load_pair_tables(gs['test']);descriptions=[]
    for col in ['ttc_calc','mttc','psd','drac']:
        v=rawpairs[col].to_numpy();finite=v[np.isfinite(v)]
        descriptions.append(dict(metric=col,sampled_frames=len(v),finite_count=len(finite),
            infinite_count=int(np.isinf(v).sum()),missing_count=int(np.isnan(v).sum()),
            mean_finite=float(np.mean(finite)) if len(finite) else np.nan,
            median_finite=float(np.median(finite)) if len(finite) else np.nan,
            p95_finite=float(np.quantile(finite,.95)) if len(finite) else np.nan,
            note='sampled pair-frame descriptive only; no arbitrary truncation, no pooling unlike units'))
    table(tables,'T01b_pair_SSM_descriptives',pd.DataFrame(descriptions))
    ctx=run_context(gs['train'],gs['validation'],gs['test'],pc,dst/'context',device)
    table(tables,'T08_context_distribution',pd.DataFrame([ctx]))
    # Store gridded test observations and node geometry for transparent noninterpolated maps.
    firstg=gs['test'][0];nodes=pd.read_csv(Path(firstg['path']).parent/'nodes.csv');nodes.to_csv(dst/'map_nodes.csv',index=False)
    np.savez_compressed(dst/'map_observations.npz',y=firstg['paper_arrays']['y'][...,0],mask=firstg['my'].numpy(),time=firstg['time'])
    write_json(dst/'map_case.json',{'graph':firstg['name'],'selection':'first held-out recording/direction by ID; NOT lowest error',
                                  'schema':firstg['schema'],'primary_horizon_s':pc['primary_horizon_s']})
    done={'status':'COMPLETE','signature':signature,'output':str(dst),'scale_m':scale,'fold':foldid,'seed':seed,
          'scope':'SYNTHETIC SOFTWARE TEST ONLY' if plan['synthetic'] else 'LOCAL USER DATA',
          'continuous_scored_rows':len(pred),'event_scored_rows':sum(len(q) for q in eventpred),
          'context_status':ctx.get('status'),'plots':'render_paper_report(output) reads these saved results only'}
    write_json(dst/'complete.json',done);print(f'Paper complete: {dst}',flush=True);return done


def _summarize_groups(work,results):
    """Macro over held-out locations within each seed; seed SD is NOT a confidence interval."""
    continuous=[];event=[]
    for r in results:
        for fn,bucket in [('T03_continuous_forecasts',continuous),('T04_event_probability',event)]:
            p=Path(r['output'])/'tables'/(fn+'.csv')
            if not p.exists():continue
            d=pd.read_csv(p)
            if ('predictor' not in d) and ('probability' in fn) and ('average_precision' not in d):continue
            d['scale_m']=r['scale_m'];d['test_location']=r['fold'];d['seed']=r['seed'];bucket.append(d)
    out=Path(work)/'paper'/'summary'
    for name,parts,modelkey,metrics,extra in [
        ('continuous',continuous,'predictor',['rmse','mae','crps','nll','r2','coverage90','crps_skill_vs_persistence'],[]),
        ('event',event,'model',['roc_auc','average_precision','brier','precision','recall','fpr'],['ttc_threshold_s'])]:
        if not parts:continue
        d=pd.concat(parts,ignore_index=True);table(out,name+'_all_folds',d)
        keys=['scale_m',modelkey,'horizon_s']+extra
        macro=d.groupby(keys+['seed'],dropna=False)[metrics].mean().reset_index()
        count=d.groupby(keys+['seed']).test_location.nunique().reset_index(name='heldout_locations')
        macro=macro.merge(count,on=keys+['seed']);table(out,name+'_macro_locations_by_seed',macro)
        summ=macro.groupby(keys)[metrics].agg(['mean','std']).reset_index()
        summ.columns=['_'.join(str(x) for x in c if x) if isinstance(c,tuple) else c for c in summ.columns]
        table(out,name+'_seed_mean_and_SD',summ)
    write_json(out/'interpretation.json',{'location_macro':True,'seed_SD_is_not_CI':True,
        'pooling_different_SSM_targets':False,'pooling_different_spatial_scales':False,
        'note':'Check heldout_locations for each model; absent/unsupported folds are not zero-error cases. Do not rank SSMs by raw RMSE.'})
