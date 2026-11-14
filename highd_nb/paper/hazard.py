"""Coherent multi-horizon confirmed-conflict early warning, separate from the SDE.

At each forecast origin, a network predicts discrete conditional hazards for the
next K bins. P(T <= h) = 1 - product_{k<=h}(1-hazard_k) is monotone by construction.
Targets are first full-frame-rate confirmed onsets at the fixed road cell, not
collisions or crossings of a boundary by the cell-average SDE.

Complete support through the maximum horizon is shared by every compared event
readout. Partial follow-up is excluded, not filled as a safe observation. Only
training locations fit the scaler/network; validation selects epochs, estimates
one positive cumulative-hazard calibration multiplier and chooses alarm cutoffs.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from highd_nb.runtime import choose_device, cpu_state_dict, device_summary, synchronize
from highd_nb.core.common import write_json
from .data import tabular, event_targets, metadata
from .events import select_alarm_threshold, probability_scores


class DiscreteHazard(nn.Module):
    def __init__(self, inputs: int, steps: int, hidden: int = 24):
        super().__init__()
        if min(inputs, steps, hidden) < 1:
            raise ValueError('Hazard dimensions must be positive.')
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, steps))

    def forward(self, x):
        return self.net(x)


def log_survival(logits):
    return torch.cumsum(F.logsigmoid(-logits), dim=-1)


def hazard_cdf(logits, rate_multiplier: float = 1.0):
    if not np.isfinite(rate_multiplier) or rate_multiplier <= 0:
        raise ValueError('Calibration multiplier must be finite and strictly positive.')
    return -torch.expm1(rate_multiplier * log_survival(logits))


def first_event_targets(onsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All rows have follow-up through K. Subsequent events leave the risk set."""
    onsets = np.asarray(onsets)
    if onsets.ndim != 2 or not np.isin(onsets, [0, 1]).all():
        raise ValueError('onsets must be a finite binary [rows, future_bins] matrix.')
    before = np.cumsum(onsets, axis=1)-onsets
    at_risk = before == 0
    event = (onsets > 0) & at_risk
    return event.astype('float32'), at_risk


def survival_nll(logits, onsets):
    """Likelihood per forecast origin, right censored at the prespecified K."""
    risk = (torch.cumsum(onsets, dim=1)-onsets) == 0
    return (F.binary_cross_entropy_with_logits(logits, onsets, reduction='none')*risk).sum(1).mean()


def hazard_dataset(graphs, windows, horizons, pc, threshold, predictions=None):
    if pc['event_support'] != 'complete_future':
        raise ValueError('The coherent event head currently requires complete_future support.')
    xx, yy, metas = [], [], []
    steps = None
    for w in windows:
        g = graphs[w.graph]
        k = int(round(max(horizons)/g['dt']))
        if steps is not None and k != steps:
            raise ValueError('Hazard graphs must share the same time step.')
        steps = k
        _, mask = event_targets(g, w.anchor, max(horizons), threshold, 'complete_future')
        if not mask.any():
            continue
        ki = int(np.flatnonzero(np.isclose(g['labels']['thresholds_s'], threshold))[0])
        future = g['labels']['onset_event'][w.anchor+1:w.anchor+k+1, :, ki].T.astype('float32')
        if future.shape[1] != k:
            raise ValueError('Hazard window crosses the recording boundary.')
        x = tabular(g, w.anchor, pc['history_steps'])
        if predictions is not None:
            # All future-distribution summaries are forecasts made at the anchor.
            extra = []
            for h in horizons:
                s = predictions[(w.graph, w.anchor, float(h))]
                extra.extend([s['mean'], s['sd'], s['pzero']])
            x = np.column_stack([x]+extra)
        ids = np.flatnonzero(mask)
        xx.append(x[ids]); yy.append(future[ids])
        metas.extend(metadata(g, w.anchor, max(horizons), ids))
    if not xx:
        return np.empty((0, 0), 'float32'), np.empty((0, steps or 0), 'float32'), pd.DataFrame()
    x = np.concatenate(xx).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite causal hazard inputs.')
    return x, np.concatenate(yy), pd.DataFrame(metas)


def fit_rate_calibration(validation_log_survival, validation_events, indices):
    """One positive multiplier across all horizons preserves a coherent event CDF."""
    ls = np.asarray(validation_log_survival, dtype=float)[:, indices]
    y = (np.cumsum(validation_events, axis=1)[:, indices] > 0).astype(float)
    if not len(ls) or not np.isfinite(ls).all() or (ls > 1e-8).any():
        raise ValueError('Invalid validation survival probabilities.')
    objective = lambda a: float(np.mean((-np.expm1(np.exp(a)*ls)-y)**2))
    fitted = minimize_scalar(objective, bounds=(-4., 4.), method='bounded')
    if not fitted.success:
        raise RuntimeError('Shared cumulative-hazard calibration failed.')
    return {'rate_multiplier': float(np.exp(fitted.x)),
            'fit': 'validation-only mean Brier over the prespecified horizons',
            'shared_across_horizons': True,
            'validation_brier_before': objective(0.), 'validation_brier_after': float(fitted.fun)}


def fit_hazard(name: str, train, val, test, horizons, dt: float, pc: dict,
               threshold: float, output: Path, device: Any = 'auto'):
    device = choose_device(device)
    x, y, _ = train; vx, vy, _ = val; tx, ty, meta = test
    receipt = {'model': name, 'ttc_threshold_s': threshold, 'horizons_s': list(horizons),
               'definition': 'first confirmed onset at the fixed cell, discretized in time',
               'not_SDE_first_passage': True, 'not_crash_probability': True,
               'support': 'complete maximum-horizon follow-up, observed non-active anchor',
               'test_resampled': False}
    for part, target, minimum in [('train', y, pc['event_min_train_positives']),
                                 ('validation', vy, pc['event_min_validation_positives'])]:
        counts = (target.sum(1) > 0) if len(target) else np.empty(0, bool)
        if not len(counts) or min(int(counts.sum()), int((~counts).sum())) < minimum:
            return None, {**receipt, 'status': 'SKIP_INSUFFICIENT_'+part.upper()+'_EVENT_SUPPORT'}
    if not len(ty):
        return None, {**receipt, 'status': 'SKIP_NO_TEST_SUPPORT'}
    indices = [int(round(h/dt))-1 for h in horizons]
    if any(abs((i+1)*dt-h)>1e-7 for i,h in zip(indices,horizons)):
        raise ValueError('Horizons must be integer multiples of the data step.')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1.
    xt=torch.from_numpy((x-mean)/scale);yt=torch.from_numpy(y)
    vt=torch.from_numpy((vx-mean)/scale);vyt=torch.from_numpy(vy)
    torch.manual_seed(int(pc['seed']))
    model=DiscreteHazard(x.shape[1],y.shape[1],int(pc.get('event_hazard_hidden',24))).to(device)
    with torch.no_grad():
        events,risk=first_event_targets(y)
        rate=np.clip((events.sum(0)+1)/(risk.sum(0)+2),.001,.999)
        model.net[-1].bias.copy_(torch.as_tensor(np.log(rate/(1-rate)),device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=pc.get('event_hazard_learning_rate',.001),
                          weight_decay=pc.get('event_hazard_l2',1e-4))
    batch=int(pc.get('event_hazard_batch_size',512));epochs=int(pc.get('event_hazard_epochs',60))
    rng=np.random.default_rng(pc['seed']);best=float('inf');state=None;stale=0;curve=[]
    synchronize(device);start=time.perf_counter()
    for epoch in range(epochs):
        order=rng.permutation(len(x));total=0.;model.train()
        for off in range(0,len(x),batch):
            ids=order[off:off+batch];bx=xt[ids].to(device);by=yt[ids].to(device)
            loss=survival_nll(model(bx),by)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hazard training likelihood.')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(ids)
        model.eval();valtotal=0.
        with torch.no_grad():
            for off in range(0,len(vx),batch):
                bx=vt[off:off+batch].to(device);by=vyt[off:off+batch].to(device)
                valtotal+=float(survival_nll(model(bx),by))*len(bx)
        score=valtotal/len(vx);curve.append({'epoch':epoch+1,'training_survival_nll':total/len(x),'validation_survival_nll':score})
        if score<best:best=score;state=cpu_state_dict(model);stale=0
        else:stale+=1
        if stale>=pc['patience']:break
    if state is None:raise RuntimeError('No finite hazard validation checkpoint.')
    model.load_state_dict(state);model.eval()
    def predict_log_survival(features):
        values=[]
        with torch.no_grad():
            for off in range(0,len(features),batch):
                bx=torch.as_tensor((features[off:off+batch]-mean)/scale,device=device)
                values.append(log_survival(model(bx)).double().cpu().numpy())
        return np.concatenate(values)
    vl=predict_log_survival(vx)
    calibration=fit_rate_calibration(vl,vy,indices)
    # Test features are used only after model, calibration and cutoffs are fixed.
    vp=-np.expm1(calibration['rate_multiplier']*vl)
    cutoffs={};skips=[]
    for h,i in zip(horizons,indices):
        vlabel=vy[:,:i+1].sum(1)>0;tlabel=y[:,:i+1].sum(1)>0
        if min(int(vlabel.sum()),int((~vlabel).sum()))<pc['event_min_validation_positives'] or min(int(tlabel.sum()),int((~tlabel).sum()))<pc['event_min_train_positives']:
            skips.append({'horizon_s':h,'reason':'insufficient per-horizon train/validation classes'});continue
        cutoffs[float(h)]=select_alarm_threshold(vlabel,vp[:,i],pc['alarm_max_fpr'])
    tl=predict_log_survival(tx);raw=-np.expm1(tl);pred=-np.expm1(calibration['rate_multiplier']*tl)
    result=[]
    for h,i in zip(horizons,indices):
        if float(h) not in cutoffs:continue
        df=meta.copy();df['horizon_s']=float(h);df['target_s']=df.anchor_s+float(h)
        df['model']=name;df['ttc_threshold_s']=threshold;df['y']=(ty[:,:i+1].sum(1)>0).astype(int)
        df['probability']=pred[:,i];df['uncalibrated_probability']=raw[:,i]
        df['alarm_threshold']=cutoffs[float(h)];df['warning']=df.probability>=df.alarm_threshold
        result.append(df)
    synchronize(device)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(curve).to_csv(output/'learning_curve.csv',index=False)
    torch.save({'state_dict':cpu_state_dict(model),'inputs':x.shape[1],'steps':y.shape[1],
                'hidden':int(pc.get('event_hazard_hidden',24)),'mean':mean.tolist(),'scale':scale.tolist()},output/'model.pt')
    receipt.update(status='FIT',train_n=len(y),validation_n=len(vy),test_n=len(ty),
                   calibration=calibration,cutoffs=cutoffs,skipped_horizons=skips,
                   runtime=device_summary(device),fit_and_calibration_and_prediction_wall_s=time.perf_counter()-start,
                   coherent_before_and_after_calibration=True)
    write_json(output/'readout.json',receipt)
    return pd.concat(result,ignore_index=True) if result else None,receipt


def run_hazard_comparisons(graphs, windows, sde_statistics, horizons, pc, output, device):
    models=['Directional-hazard']+['%s-hazard'%s for s in sde_statistics if s in ['SDE-masked','SDE-latent']]
    results,receipts=[],[]
    for threshold in pc['event_thresholds_s']:
        for name in models:
            source=sde_statistics.get(name.removesuffix('-hazard'))
            sets=[hazard_dataset(graphs[p],windows[p],horizons,pc,threshold,
                                 source[p] if source else None) for p in ['train','validation','test']]
            dt=graphs['train'][0]['dt']
            pred,receipt=fit_hazard(name,*sets,horizons,dt,pc,threshold,
                                   Path(output)/name/f'TTC{threshold:g}',device)
            receipts.append(receipt)
            if pred is not None:results.append(pred)
    return results,receipts
