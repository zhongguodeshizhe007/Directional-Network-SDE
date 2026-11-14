"""Location-held-out training. Validation is used for early stopping, NEVER test."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import *
from .data import fit_normalizer,prepare_graph,split_paths,window_batches
from .models import NetworkModel
from .evaluation import validation_score

from highd_nb.runtime import choose_device, device_summary, synchronize

def model_runs(work,tag='main'):
    return sorted((Path(work)/'models'/tag).glob('L*/test_*/**/training_receipt.json'))

MODEL_NAMES = ['masked','latent','local'] + [f'{k}_{a}' for k in ['masked','latent']
                                            for a in ['local','no_lateral','symmetric','const_sigma']]

def model_spec(name, base):
    """Separate missingness treatment from topology/noise ablations."""
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model {name}; choose {MODEL_NAMES}')
    c=copy.deepcopy(base)
    kind='latent' if name.startswith('latent') else 'masked'
    if name=='local' or name.endswith('_local'): c['use_edges']=False
    if name.endswith('_no_lateral'): c['edge_types']=[0,1]
    if name.endswith('_symmetric'): c['tie_longitudinal']=True
    if name.endswith('_const_sigma'): c['constant_sigma']=True
    return kind,c

def train_one(work,plan,fold,scale,name,seed,tag,device,epochs_override=None):
    cfg=copy.deepcopy(plan['config'])
    if epochs_override is not None: cfg['training']['epochs']=int(epochs_override)
    tc=cfg['training']; mc=copy.deepcopy(cfg['model'])
    kind,mc=model_spec(name,mc)
    r=Path(work)/'models'/tag/f'L{scale:03d}'/f'test_{fold["test_location"]}'/name/f'seed_{seed}'
    expected={'plan_hash':plan['plan_hash'],'config':cfg,'fold':fold,'scale_m':scale,'name':name,'seed':seed,
              'model_config':mc,'kind':kind}
    if (r/'training_receipt.json').exists():
        old=read_json(r/'training_receipt.json')
        if old['training_signature']!=digest(expected):
            raise ValueError(f'Run {r} already exists with a different training config. Use another --run-tag.')
        print(f'Cached training: {r}',flush=True); return
    r.mkdir(parents=True,exist_ok=True); seed_all(seed,tc['threads'])
    trainp=split_paths(work,scale,fold,'train'); valp=split_paths(work,scale,fold,'validation')
    for p in trainp+valp:
        key=f'L{scale}/{p.parent.name}'
        if file_hash(p)!=plan['data_fingerprints'][key]: raise ValueError(f'Data hash changed: {p}')
    norm=fit_normalizer(trainp); write_json(r/'normalizer.json',norm)
    train=[prepare_graph(p,norm,mc) for p in trainp]; val=[prepare_graph(p,norm,mc) for p in valp]
    import time
    synchronize(device); training_started=time.perf_counter()
    model=NetworkModel(kind,mc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    rng=np.random.default_rng(seed); best=np.inf; stale=0; log=[]
    for epoch in range(tc['epochs']):
        model.train(); total=0.; count=0
        for b in window_batches(train,tc['sequence_length'],tc['sequence_stride'],tc['batch_size'],True,rng,device):
            loss,n=model.batch_loss(b,tc['burn_in_steps'])
            if loss is None or n<tc['min_valid_targets_batch']: continue
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; inspect data/scales.')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip']); opt.step()
            total+=float(loss.detach())*n; count+=n
        if not count: raise ValueError('No usable training windows. Check sequence length and risk support.')
        model.eval()
        # Common random numbers across epochs, without perturbing optimizer random state.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed+8000)
            score=validation_score(model,val,norm,tc,cfg['evaluation'],device)
        log.append({'epoch':epoch+1,'objective':total/count,'observed_loss_terms':count,
                    'validation_multihorizon_crps_y_units':score})
        pd.DataFrame(log).to_csv(r/'learning_curve.csv',index=False)
        print(f'L{scale} test={fold["test_location"]} {name} seed={seed} epoch={epoch+1}: '
              f'loss={total/count:.5f}, valCRPS={score:.6f}',flush=True)
        if score<best:
            best=score; stale=0
            torch.save({'state_dict':model.state_dict(),'model_config':mc,'kind':kind},r/'best.pt')
            write_json(r/'best_epoch.json',log[-1])
        else: stale+=1
        if stale>=tc['patience']: break
    synchronize(device)
    write_json(r/'runtime.json',{'device':device_summary(device),
        'training_and_validation_wall_s':time.perf_counter()-training_started,
        'timing_scope':'training plus validation, excludes preprocessing',
        'precision':'float32; no automatic mixed precision for likelihood/Cholesky'})
    write_json(r/'training_receipt.json',{**expected,'training_signature':digest(expected),
        'checkpoint_sha256':file_hash(r/'best.pt'),'best_validation_crps':best,
        'environment':environment(),'n_training_graphs':len(train),'n_validation_graphs':len(val),
        'inference':'Euler discrete state-space approximation; not original LaGNA',
        'data_scope':'SYNTHETIC SOFTWARE TEST' if plan['synthetic'] else 'local user highD files',
        'forecast_covariates':'freeze_at_anchor'})

def train(work,fold_selection='all',scale_selection='all',names='routed',seeds=None,tag='main',device='auto',epochs=None):
    work=Path(work); plan=read_json(work/'plan.json')
    if plan['code_hash']!=code_hash(): raise ValueError('Source changed after planning. Regenerate plan in a fresh work directory.')
    dev=choose_device(device); print(f'Compute device: {dev}',flush=True); attempted=0
    for fold in selected_folds(plan,fold_selection):
        for option in fold['options']:
            scale=int(option['scale_m'])
            if scale_selection=='selected' and scale!=fold['selected_scale_m']: continue
            if scale_selection not in ['all','selected'] and scale not in [int(x) for x in scale_selection.split(',')]: continue
            if not option['eligible']:
                print(f'SKIP test={fold["test_location"]}, L{scale}: {option["reasons"]}',flush=True); continue
            use_names=[option['routed_model']] if names=='routed' else names.split(',')
            for name in use_names:
                if name not in MODEL_NAMES: raise ValueError(f'models: routed or {MODEL_NAMES}')
                for seed in (seeds or plan['config']['training']['seeds']):
                    train_one(work,plan,fold,scale,name,int(seed),tag,dev,epochs)
                    attempted+=1
    if not attempted: raise ValueError('No eligible requested runs. Inspect plan.json; no training was performed.')
