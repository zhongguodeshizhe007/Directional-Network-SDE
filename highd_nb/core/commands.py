from __future__ import annotations
import copy, zipfile
from pathlib import Path
import pandas as pd
from .common import *

def selected_runs(work,tag,folds,scales,models):
    from .training import model_runs
    result=[]
    for p in model_runs(work,tag):
        r=read_json(p)
        if folds!='all' and r['fold']['test_location'] not in [int(v) for v in folds.split(',')]: continue
        if scales=='selected':
            plan=read_json(Path(work)/'plan.json')
            chosen=next(f['selected_scale_m'] for f in plan['folds'] if f['test_location']==r['fold']['test_location'])
            if r['scale_m']!=chosen:continue
        elif scales!='all' and r['scale_m'] not in [int(v) for v in scales.split(',')]: continue
        if models!='all' and r['name'] not in models.split(','): continue
        result.append(p.parent)
    if not result: raise FileNotFoundError('No completed training run matches selection.')
    return result

def summarize(work,tag='main'):
    from .training import model_runs
    work=Path(work); tables=[]
    for p in model_runs(work,tag):
        r=read_json(p)
        for typ,folder in [('neural','test'),('symbolic','symbolic_test')]:
            f=p.parent/folder/'metrics.csv'
            if not f.exists(): continue
            t=pd.read_csv(f); t['scale_m']=r['scale_m']; t['test_location']=r['fold']['test_location']
            t['seed']=r['seed']; t['family']=r['name']; t['dynamics']=typ
            if typ=='symbolic':
                t['symbolic_validation_accepted']=read_json(p.parent/'symbolic'/'complete.json')['accepted_validation_fidelity']
            tables.append(t)
    if not tables: raise FileNotFoundError('No test metrics. Run evaluate first.')
    out=pd.concat(tables,ignore_index=True); dst=work/'summary'/tag; dst.mkdir(parents=True,exist_ok=True)
    out.to_csv(dst/'all_test_metrics.csv',index=False)
    metrics=['rmse','mae','crps','nll','coverage90','width90','zero_brier']
    macro=out.groupby(['scale_m','family','dynamics','predictor','seed','horizon_s'])[metrics].mean().reset_index()
    macro.to_csv(dst/'macro_across_locations_by_seed.csv',index=False)
    macro.groupby(['scale_m','family','dynamics','predictor','horizon_s'])[metrics].agg(['mean','std']).to_csv(dst/'seed_variability.csv')
    write_json(dst/'interpretation.json',{'notes':[
        'Do not compare raw errors across scales as if targets were identical.',
        'Seed standard deviations are not confidence intervals over new locations.',
        'Macro averages weight locations equally. Some model families may occur in fewer routed folds.',
        'Symbolic validation rejection must be disclosed; rejected candidates are not established laws.',
        'All targets are surrogate risk endpoints, not accident frequency or calibrated collision probability.'
    ]})
    print(f'Summary saved to {dst}')

def collect(work,tag='main'):
    work=Path(work); paths=[]
    for pattern in ['plan.json','preprocessing_contract.json','environment_preprocessing.json',
                    'inventory/inventory.csv','reports/*.csv','reports/*.json','summary/*/*.csv','summary/*/*.json',
                    'models/*/L*/test_*/*/seed_*/training_receipt.json',
                    'models/*/L*/test_*/*/seed_*/learning_curve.csv',
                    'models/*/L*/test_*/*/seed_*/test/metrics.csv',
                    'models/*/L*/test_*/*/seed_*/test/cluster_bootstrap_ci.csv',
                    'models/*/L*/test_*/*/seed_*/symbolic/*.json',
                    'models/*/L*/test_*/*/seed_*/symbolic/*.md',
                    'models/*/L*/test_*/*/seed_*/symbolic/recording_bootstrap_stability.csv',
                    'models/*/L*/test_*/*/seed_*/symbolic_test/metrics.csv']:
        paths.extend(work.glob(pattern))
    if not paths: raise FileNotFoundError('Nothing to collect.')
    with zipfile.ZipFile(work/'review_bundle_v3.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)): z.write(p,p.relative_to(work))
    print(f'Wrote {work/"review_bundle_v3.zip"}; no raw trajectories/arrays/checkpoints/per-origin predictions included.')

def smoke(work,device='cpu'):
    from highd_nb.vendor.make_synthetic_fixture import create_fixture
    from .preprocessing import inventory,preprocess
    from .quality import report,decide
    from .training import train,choose_device
    from .symbolic import distill
    from .evaluation import evaluate_run
    work=Path(work)
    if work.exists() and any(work.iterdir()): raise FileExistsError('Smoke directory must be new/empty.')
    raw=work/'synthetic_raw'; data=work/'synthetic_work'
    create_fixture(raw,4)
    c=read_json(ROOT/'configs'/'experiment.json')
    c['gates'].update(min_training_observations=10,min_positive_training_observations=1,
           min_observed_fraction=.01,min_location_observed_fraction=.01,max_p95_missing_gap_s=100)
    c['training'].update(seeds=[7],epochs=2,patience=2,sequence_length=12,sequence_stride=8,
           batch_size=2,burn_in_steps=2,validation_draws=8,validation_origin_stride=8,
           validation_max_origins_per_graph=2)
    c['evaluation'].update(particles=16,origin_stride=8,max_origins_per_graph=2,history_steps=12,bootstrap_recordings=10)
    c['symbolic'].update(max_rows_per_component=500,rows_per_graph=50,recording_bootstraps=3)
    c['model']['hidden']=12
    write_json(work/'synthetic_config.json',c)
    inventory(raw,data,'all',True); preprocess(raw,data,c,[50,75,100],'all',True)
    report(data,c); plan=decide(data,c)
    # Both branches, all THREE scales, one outer fold. Other folds exercised by unit tests.
    train(data,'1','all','masked,latent',[7],'smoke',device)
    dev=choose_device(device)
    for r in selected_runs(data,'smoke','1','all','all'):
        distill(data,r,dev); evaluate_run(data,r,dev,False); evaluate_run(data,r,dev,True)
    summarize(data,'smoke'); collect(data,'smoke')
    write_json(work/'SMOKE_RESULT.json',{'status':'PASS','data':'ARTIFICIAL SOFTWARE FIXTURE, NOT highD research findings',
        'scales':[50,75,100],'branches':['masked','latent'],'outer_test_location':1,'epochs':2,
        'stages':['inventory','preprocess','verify','report','train-only decision','training','symbolic','test','summary'],
        'real_highD_run':False,'environment':environment()})
    print(f'SYNTHETIC end-to-end smoke test completed: {work/"SMOKE_RESULT.json"}')
