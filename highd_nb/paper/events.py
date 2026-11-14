"""Direct future confirmed-conflict occurrence readouts and event-level warnings.
A supervised readout is required: a cell's mean inverse TTC is NOT an event CDF.
Only training rows fit readouts. Validation calibrates scores/selects alarm cutoffs.
Test imbalance is retained. Complete-future support is the primary default.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,
                             brier_score_loss,log_loss)
from .data import tabular,event_targets,metadata
from highd_nb.runtime import choose_device,xgb_parameters,check_xgb_device,xgb_predict


def probability_scores(y,p,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not len(y):return {'n':0,'status':'NO_SUPPORT'}
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Binary observations and finite probabilities in [0,1] required')
    a=p>=threshold;tp=int(((y==1)&a).sum());fp=int(((y==0)&a).sum());fn=int(((y==1)&~a).sum());tn=int(((y==0)&~a).sum())
    two=len(np.unique(y))==2
    return {'n':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()),
            'roc_auc':float(roc_auc_score(y,p)) if two else np.nan,
            'average_precision':float(average_precision_score(y,p)) if two else np.nan,
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
            'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan,
            'specificity':tn/(tn+fp) if tn+fp else np.nan,'fpr':fp/(fp+tn) if fp+tn else np.nan,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,'alarm_threshold':float(threshold),
            'status':'OK' if two else 'UNDEFINED_AUC_AP_SINGLE_CLASS'}


def select_alarm_threshold(y,p,max_fpr=.05):
    """Maximise empirical validation sensitivity subject to a prespecified FPR budget.
    Empty/no-positive/no-negative validation is rejected, not silently assigned 0.5.
    A threshold >1 encodes an explicit no-warning policy when necessary.
    """
    y=np.asarray(y,int);p=np.asarray(p,float)
    if len(np.unique(y))!=2:raise ValueError('Alarm threshold needs both validation classes')
    if not 0<=max_fpr<=1:raise ValueError('max_fpr must lie in [0,1]')
    fpr,tpr,thr=roc_curve(y,p,drop_intermediate=False)
    ok=np.flatnonzero(fpr<=max_fpr+1e-12)
    best=max(ok,key=lambda i:(tpr[i],-fpr[i],thr[i]))
    return float(thr[best]) if np.isfinite(thr[best]) else 1.+1e-8


def calibrate(y,p):
    """Validation-only monotone sigmoid calibration (non-negative slope)."""
    y=np.asarray(y,float);z=logit(np.clip(p,1e-6,1-1e-6))
    def objective(ab):
        a,b=ab;v=a*z+b
        return float(np.mean(np.logaddexp(0,v)-y*v)+1e-6*((a-1)**2+b*b))
    fit=minimize(objective,[1.,0.],method='L-BFGS-B',bounds=[(0,20),(-20,20)])
    if not fit.success:raise RuntimeError(f'Calibration failed: {fit.message}')
    return {'slope':float(fit.x[0]),'intercept':float(fit.x[1]),'fit':'validation only'}


def apply_calibration(p,c):return expit(c['slope']*logit(np.clip(p,1e-6,1-1e-6))+c['intercept'])


def dataset(graphs,ws,history,horizon,ttc,support,predictions=None,local=False,flow=False,support_horizon=None):
    xx=[];ys=[];rows=[]
    for w in ws:
        g=graphs[w.graph];y,m=event_targets(g,w.anchor,horizon,ttc,support)
        if support_horizon is not None:
            _,common=event_targets(g,w.anchor,support_horizon,ttc,support)
            m &= common
        if not m.any():continue
        x=tabular(g,w.anchor,history)
        if local:x=x.reshape(len(x),history,56)[:,:,:14].reshape(len(x),-1)
        if flow:
            # Virtual detector counts and acceleration are strictly from the past window.
            aa=g['paper_arrays'];idx=slice(w.anchor-history+1,w.anchor+1)
            f=np.stack([aa[k][idx] for k in ['q_in_context_veh_per_h','q_out_context_veh_per_h','acceleration_context_mps2']],-1)
            valid=np.isfinite(f);f=np.concatenate([np.where(valid,f,0.),valid.astype(float)],-1)
            f=f.transpose(1,0,2).reshape(len(x),-1);x=np.column_stack([x,f])
        if predictions is not None:
            # These are forecasts generated without accessing future observations.
            stats=predictions[(w.graph,w.anchor,float(horizon))]
            x=np.column_stack([x,stats['mean'],stats['sd'],stats['pzero']])
        ids=np.flatnonzero(m);xx.append(x[ids]);ys.append(y[ids]);rows.extend(metadata(g,w.anchor,horizon,ids))
    if not xx:return np.empty((0,0)),np.empty(0,int),pd.DataFrame()
    return np.concatenate(xx),np.concatenate(ys),pd.DataFrame(rows)


def fit_event_readout(name,train,val,test,pc,ttc,horizon,checkpoint_dir=None,device='auto'):
    device=choose_device(device)
    x,y,_=train;vx,vy,_=val;tx,ty,meta=test
    if len(y)==0 or y.sum()<pc['event_min_train_positives'] or (y==0).sum()<pc['event_min_train_positives']:
        return None,{'status':'SKIP_INSUFFICIENT_TRAIN_EVENT_SUPPORT','train_n':len(y),'train_positives':int(y.sum())}
    if len(vy)==0 or min(int(vy.sum()),int((vy==0).sum()))<pc['event_min_validation_positives']:
        return None,{'status':'SKIP_INSUFFICIENT_VALIDATION_EVENT_SUPPORT','validation_n':len(vy),'validation_positives':int(vy.sum())}
    if not len(ty):return None,{'status':'SKIP_NO_TEST_SUPPORT'}
    if name=='XGBoost-event':
        from xgboost import XGBClassifier
        clf=XGBClassifier(n_estimators=int(pc.get('xgb_estimators',150)),max_depth=3,learning_rate=.05,
                          random_state=pc['seed'],reg_lambda=5.,eval_metric='logloss',**xgb_parameters(device))
    else:clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,solver='lbfgs'))
    # No resampling of evaluation data, and no class weight masquerading as a posterior.
    clf.fit(x,y)
    actual_device=check_xgb_device(clf,device) if name=='XGBoost-event' else {'fitted_device':'cpu'}
    predict=lambda xx: xgb_predict(clf,xx,True)[:,1] if name=='XGBoost-event' else clf.predict_proba(xx)[:,1]
    rawval=predict(vx);c=calibrate(vy,rawval);vp=apply_calibration(rawval,c)
    cutoff=select_alarm_threshold(vy,vp,pc['alarm_max_fpr'])
    if checkpoint_dir is not None:
        from pathlib import Path
        from highd_nb.core.common import write_json
        ck=Path(checkpoint_dir);ck.mkdir(parents=True,exist_ok=True)
        saved={'model':name,'horizon_s':horizon,'ttc_threshold_s':ttc,'calibration':c,'alarm_threshold':cutoff,
               'history_steps':pc['history_steps'],'input_note':'tabular historical directional features, then optional normalized SDE mean/sd/pzero'}
        if name=='XGBoost-event':clf.save_model(ck/'xgboost.json')
        else:
            scaler,logitmodel=clf.steps[0][1],clf.steps[1][1]
            saved.update(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,coef=logitmodel.coef_,intercept=logitmodel.intercept_)
        write_json(ck/'readout.json',saved)
    raw=predict(tx);p=apply_calibration(raw,c)
    out=meta.copy();out['model']=name;out['ttc_threshold_s']=ttc;out['y']=ty;out['probability']=p
    out['uncalibrated_probability']=raw;out['alarm_threshold']=cutoff;out['warning']=p>=cutoff
    receipt={'status':'FIT','model':name,'horizon_s':horizon,'ttc_threshold_s':ttc,
             'train_n':len(y),'train_positives':int(y.sum()),'validation_n':len(vy),
             'validation_positives':int(vy.sum()),'test_n':len(ty),'calibration':c,'cutoff':cutoff,
             'validation_at_selected_cutoff':probability_scores(vy,vp,cutoff),
             'readout_definition':'direct confirmed-onset event probability, not an SDE first-passage or crash probability',
             'support':pc['event_support'],'test_resampled':False,'device':actual_device}
    return out,receipt


def reliability_table(predictions,bins=10):
    records=[]
    for key,g in predictions.groupby(['model','horizon_s','ttc_threshold_s'],sort=False):
        ids=np.minimum((g.probability.to_numpy()*bins).astype(int),bins-1)
        for b in range(bins):
            q=g.iloc[np.flatnonzero(ids==b)]
            if not len(q):continue
            records.append(dict(zip(['model','horizon_s','ttc_threshold_s'],key),bin=b,n=len(q),
                                predicted_probability=float(q.probability.mean()),observed_rate=float(q.y.mean())))
    return pd.DataFrame(records)


def event_metrics(predictions):
    rows=[]
    for key,g in predictions.groupby(['model','horizon_s','ttc_threshold_s'],sort=False):
        rows.append(dict(zip(['model','horizon_s','ttc_threshold_s'],key),
                         **probability_scores(g.y,g.probability,float(g.alarm_threshold.iloc[0]))))
    return pd.DataFrame(rows)


def cluster_ci(predictions,n_boot=500,seed=2026):
    rng=np.random.default_rng(seed);rows=[]
    for key,g in predictions.groupby(['model','horizon_s','ttc_threshold_s'],sort=False):
        recs=g.recording_id.unique();center=probability_scores(g.y,g.probability,g.alarm_threshold.iloc[0])
        vals={k:[] for k in ['average_precision','roc_auc','brier','recall','precision']}
        if len(recs)>=2:
            groups={r:g.loc[g.recording_id==r] for r in recs}
            for _ in range(n_boot):
                b=pd.concat([groups[r] for r in rng.choice(recs,len(recs),replace=True)],ignore_index=True)
                score=probability_scores(b.y,b.probability,b.alarm_threshold.iloc[0])
                for k in vals:
                    if np.isfinite(score[k]):vals[k].append(score[k])
        for metric,v in vals.items():
            lo,hi=np.quantile(v,[.025,.975]) if len(v)>=2 else (np.nan,np.nan)
            rows.append(dict(zip(['model','horizon_s','ttc_threshold_s'],key),metric=metric,estimate=center[metric],
                             ci_low=lo,ci_high=hi,recordings=len(recs),valid_bootstraps=len(v),
                             note='recording-cluster CI conditional on fitted models; not frame-independent or refit CI'))
    return pd.DataFrame(rows)


def warning_episodes(predictions,events,step_s):
    """Each confirmed TTC episode is counted once per model/horizon.
    Reference moment is full-rate confirmation, NOT Jiao's NDS minimum-distance time.
    Warning coverage is relative to available prespecified forecast anchors.
    """
    rows=[]
    for (model,h,thr),g in predictions.groupby(['model','horizon_s','ttc_threshold_s'],sort=False):
        relevant=events[np.isclose(events.threshold_s,thr)]
        for _,e in relevant.iterrows():
            q=g[(g.recording_id.astype(str)==str(e.recording_id))&(g.direction==e.direction)&(g.node==e.confirmed_node)
                &(g.anchor_s>=e.confirmed_s-h)&(g.anchor_s<e.confirmed_s)].sort_values('anchor_s')
            if q.empty:continue
            alarm=q[q.warning];lead=np.nan;first=np.nan
            if len(alarm):
                first=float(e.confirmed_s-alarm.anchor_s.iloc[0])
                # Last safe->unsafe shift among evaluated anchors; breaks across missing anchors.
                a=q.warning.to_numpy(bool);t=q.anchor_s.to_numpy();starts=a&np.r_[True,(~a[:-1])|(np.diff(t)>step_s*1.01)]
                lead=float(e.confirmed_s-t[np.flatnonzero(starts)[-1]])
            rows.append(dict(model=model,horizon_s=h,ttc_threshold_s=thr,event_id=e.event_id,
                             recording_id=str(e.recording_id),direction=e.direction,node=e.confirmed_node,
                             confirmed_s=e.confirmed_s,detected=bool(len(alarm)),available_anchors=len(q),
                             warning_period_fraction=float(q.warning.mean()),first_warning_lead_s=first,
                             last_warning_transition_lead_s=lead))
    return pd.DataFrame(rows)


def success_failure_table(warnings):
    if warnings.empty:return pd.DataFrame()
    p=warnings.pivot_table(index=['horizon_s','ttc_threshold_s','event_id'],columns='model',values='detected',aggfunc='max')
    # Intersection support for the compared event models, never count a missing prediction as a failure.
    p=p.dropna();out=[]
    for key,g in p.groupby(level=[0,1]):
        g=g.droplevel([0,1]);counts=g.reset_index().groupby(list(g.columns),dropna=False).agg(n_events=('event_id','size'),event_ids=('event_id',lambda x:';'.join(x))).reset_index()
        counts.insert(0,'ttc_threshold_s',key[1]);counts.insert(0,'horizon_s',key[0]);out.append(counts)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def horizon_coherence(predictions):
    """Diagnostic only: independently fitted direct readouts need not form a monotone CDF.
    Do not silently repair using test outcomes or label these as SDE first-passage probabilities.
    """
    rows=[]
    for (model,ttc),g in predictions.groupby(['model','ttc_threshold_s']):
        p=g.pivot(index=['graph','anchor_index','node'],columns='horizon_s',values='probability')
        hs=sorted(p.columns)
        for a,b in zip(hs[:-1],hs[1:]):
            q=p[[a,b]].dropna();delta=q[b]-q[a]
            rows.append(dict(model=model,ttc_threshold_s=ttc,short_horizon_s=a,long_horizon_s=b,
                common_origins=len(q),nonmonotone_count=int((delta < -1e-8).sum()),
                nonmonotone_fraction=float((delta < -1e-8).mean()) if len(q) else np.nan,
                note='coherent discrete confirmed-onset CDF; not SDE first passage' if model.endswith('-hazard') else 'direct separately calibrated readouts; not a coherent event CDF'))
    return pd.DataFrame(rows)
