"""Coverage summaries and a train-location-only preregistered routing rule."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *

def observation_metrics(a):
    m=a['mask_y'][...,0].astype(bool); z=a['mask_z'].all(-1)
    T,N=m.shape
    adjacency=a['adjacency_by_type'].any(0)
    full_neighbor=(m.astype(int)@adjacency.T)==adjacency.sum(-1)
    neighbor_fraction=(m.astype(float)@adjacency.T)/np.maximum(adjacency.sum(-1),1)
    consecutive=np.zeros_like(m); consecutive[:-1]=m[:-1]&m[1:]
    strict=a['mask_transition_strict'].astype(bool)
    return m,full_neighbor,neighbor_fraction,consecutive,strict

def summarize_selected(a,where):
    m,nfull,nfrac,consecutive,strict=observation_metrics(a)
    count=int(where.sum()); observed=int((where&m).sum())
    y=a['y'][...,0]
    return {'node_time_count':count,'observed_count':observed,
        'positive_count':int((where&m&(y>0)).sum()),
        'zero_count':int((where&m&(y==0)).sum()),
        'empty_count':int((where&(a['vehicle_exposure_s']==0)).sum()),
        'consecutive_count':int((where&consecutive).sum()),
        'strict_count':int((where&strict).sum()),
        'neighbor_complete_given_target_count':int((where&m&nfull).sum()),
        'source_observed_fraction_sum_given_target':float(nfrac[where&m].sum()),
        'observed_fraction':observed/count if count else np.nan,
        'neighbor_complete_given_target':int((where&m&nfull).sum())/observed if observed else np.nan}

def report(work,config,scales=None):
    scales=list(scales or config['scales_m'])
    if not set(scales).issubset(config['scales_m']): raise ValueError('Unknown report scales.')
    work=Path(work); dest=work/'reports'; dest.mkdir(parents=True,exist_ok=True)
    graph_rows=[]; regime_rows=[]; events=[]; labels=[]
    expected=None
    if (work/'inventory'/'inventory.csv').exists():
        inv=pd.read_csv(work/'inventory'/'inventory.csv',dtype={'recording_id':str})
        if (inv.status!='OK_HEADERS').any(): raise ValueError('Inventory has errors.')
        expected=set(inv.recording_id.str.zfill(2))
    for s in scales:
        paths=graph_paths(work,s)
        # Require BOTH directions and completed outputs, not merely one graph per recording.
        got={str(graph_schema(p)['recording_id']).zfill(2) for p in paths}
        missing=sorted((expected or set())-got); unexpected=sorted(got-(expected or got))
        incomplete=[str(p.parent.name) for p in paths if not (p.parent/'complete.json').exists()]
        pairs={(str(graph_schema(p)['recording_id']).zfill(2),int(graph_schema(p)['direction'])) for p in paths}
        missing_directions=[f"{r}/dir{d}" for r in sorted(expected or got) for d in (1,2)
                            if (r,d) not in pairs]
        if not paths or missing or unexpected or incomplete or missing_directions:
            raise ValueError(f'L{s}: preprocessing incomplete. '
                f'missing_recordings={missing}; missing_directions={missing_directions}; '
                f'unexpected_recordings={unexpected}; incomplete_graphs={incomplete}. '
                'Finish preprocessing all audited recordings and both directions before report. '
                'Do not weaken the inventory check.')
        for p in paths:
            a=load_arrays(p); schema=graph_schema(p)
            base={'scale_m':s,'graph':p.parent.name,'recording_id':schema['recording_id'],
                  'location_id':schema['location_id'],'direction':schema['direction']}
            where=np.ones(a['y'].shape[:2],bool)
            r=summarize_selected(a,where); r.update(base)
            m=a['mask_y'][...,0]; dt=float(schema['config']['step_s'])
            gaps=max_missing_runs(m)*dt
            r.update(T=len(m),N=m.shape[1],duration_s=len(m)*dt,
                     missing_gap_p95_s=float(np.quantile(gaps,.95)) if len(gaps) else 0.,
                     arrays_sha256=file_hash(p),
                     roi_low=schema['roi']['analysis_x_low_m'],roi_high=schema['roi']['analysis_x_high_m'])
            graph_rows.append(r)
            # These are pre-specified speed/density bins, NOT validated traffic phases.
            speed=a['z_raw'][...,0]; density=a['z_raw'][...,2]
            sb=np.digitize(np.nan_to_num(speed,nan=-1),[0,15,25,np.inf])-1
            db=np.digitize(np.nan_to_num(density,nan=-1),[0,10,25,np.inf])-1
            for si in range(3):
                for di in range(3):
                    mask=np.isfinite(speed)&np.isfinite(density)&(sb==si)&(db==di)
                    rr={**base,'speed_band_mps':['[0,15)','[15,25)','[25,inf)'][si],
                        'density_band_vehpkm':['[0,10)','[10,25)','[25,inf)'][di]}
                    rr.update(summarize_selected(a,mask)); regime_rows.append(rr)
            ev=pd.read_csv(p.parent/'events.csv')
            if len(ev):
                for threshold,sub in ev.groupby('threshold_s'):
                    events.append({**base,'threshold_s':threshold,'unique_pair_episodes':sub.event_id.nunique()})
            lab=pd.read_csv(p.parent/'label_summary.csv')
            for row in lab.to_dict('records'): labels.append({**base,**row})
    graphs=pd.DataFrame(graph_rows)
    # Explicitly assert matched camera domain for EACH recording and direction.
    for _,g in graphs.groupby(['recording_id','direction']):
        if g.scale_m.nunique()!=len(scales) or np.ptp(g.roi_low)>1e-5 or np.ptp(g.roi_high)>1e-5:
            raise ValueError('Unmatched spatial domain or missing scale; do not compare these outputs.')
    graphs.to_csv(dest/'graph_metrics.csv',index=False)
    regimes=pd.DataFrame(regime_rows); regimes.to_csv(dest/'traffic_regime_metrics.csv',index=False)
    pd.DataFrame(events).to_csv(dest/'event_counts_by_recording_threshold.csv',index=False)
    pd.DataFrame(labels).to_csv(dest/'future_label_support.csv',index=False)
    sums=['node_time_count','observed_count','positive_count','zero_count','empty_count',
          'consecutive_count','strict_count','neighbor_complete_given_target_count',
          'source_observed_fraction_sum_given_target']
    for keys,name,df in [(['scale_m','location_id'],'location_summary',graphs),
                         (['scale_m'],'scale_summary',graphs),
                         (['scale_m','location_id','speed_band_mps','density_band_vehpkm'],'regime_summary',regimes)]:
        out=df.groupby(keys,dropna=False)[sums].sum().reset_index()
        out['observed_fraction']=out.observed_count/out.node_time_count.replace(0,np.nan)
        out['consecutive_fraction']=out.consecutive_count/out.node_time_count.replace(0,np.nan)
        out['strict_fraction']=out.strict_count/out.node_time_count.replace(0,np.nan)
        out['neighbor_complete_given_target']=out.neighbor_complete_given_target_count/out.observed_count.replace(0,np.nan)
        out['edge_source_observed_given_target']=out.source_observed_fraction_sum_given_target/out.observed_count.replace(0,np.nan)
        out.to_csv(dest/f'{name}.csv',index=False)
    write_json(dest/'report_contract.json',{'version':VERSION,'config_hash':digest(config),
                'graph_count':len(graphs),'recordings':sorted(graphs.recording_id.unique().tolist()),
                'scales':scales,'synthetic':(work/'SYNTHETIC_NOT_EMPIRICAL.txt').exists(),
                'notes':['Weighted denominators, not means of percentages.',
                         'node-time samples are correlated, NOT independent vehicles/events.',
                         'Regimes are descriptive bins, not inferred traffic-flow phases.',
                         'No model selection may use held-out test-location outcomes.',
                         'Counts at different TTC thresholds must not be added together.']})
    print(pd.read_csv(dest/'scale_summary.csv').to_string(index=False),flush=True)

def decide(work,config):
    work=Path(work); reports=work/'reports'
    rc=read_json(reports/'report_contract.json')
    if rc['config_hash']!=digest(config): raise ValueError('Config changed since report. Re-run report.')
    if set(rc.get('scales',config['scales_m']))!=set(config['scales_m']):
        raise ValueError('Final decision requires all three scales. Re-run report without --scales.')
    table=pd.read_csv(reports/'graph_metrics.csv',dtype={'recording_id':str})
    locs=sorted(int(x) for x in table.location_id.unique()); gates=config['gates']
    if len(locs)<3: raise ValueError('Need at least 3 locations for disjoint train/validation/test.')
    folds=[]
    for i,test in enumerate(locs):
        val=locs[(i+1)%len(locs)]; train=[x for x in locs if x not in [test,val]]
        options=[]
        for s in config['scales_m']:
            g=table[(table.scale_m==s)&table.location_id.isin(train)]
            loc=g.groupby('location_id')[['observed_count','node_time_count']].sum()
            coverage=g.observed_count.sum()/g.node_time_count.sum()
            minloc=(loc.observed_count/loc.node_time_count).min()
            consecutive=g.consecutive_count.sum()/g.node_time_count.sum()
            neighbors=g.neighbor_complete_given_target_count.sum()/max(g.observed_count.sum(),1)
            gap=g.missing_gap_p95_s.max()  # conservative maximum over training recordings
            reasons=[]
            if g.observed_count.sum()<gates['min_training_observations']: reasons.append('too_few_observations')
            if g.positive_count.sum()<gates['min_positive_training_observations']: reasons.append('too_few_positive_states')
            if coverage<gates['min_observed_fraction']: reasons.append('low_overall_coverage')
            if minloc<gates['min_location_observed_fraction']: reasons.append('low_site_coverage')
            if gap>gates['max_p95_missing_gap_s']: reasons.append('long_missing_runs')
            route=('masked' if consecutive>=gates['masked_min_consecutive_fraction'] and
                   neighbors>=gates['masked_min_neighbor_complete_given_target'] else 'latent')
            options.append({'scale_m':s,'eligible':not reasons,'reasons':reasons,'routed_model':route,
                'train_observations':int(g.observed_count.sum()),'train_positive_states':int(g.positive_count.sum()),
                'train_coverage':float(coverage),'minimum_site_coverage':float(minloc),
                'consecutive_fraction':float(consecutive),'neighbor_complete_given_target':float(neighbors),
                'maximum_recording_gap_p95_s':float(gap)})
        eligible=[o for o in options if o['eligible']]
        folds.append({'test_location':test,'validation_location':val,'train_locations':train,
            'selected_scale_m':min(o['scale_m'] for o in eligible) if eligible else None,
            'status':'READY_FOR_RESEARCH_TRAINING' if eligible else 'BLOCKED_REVIEW_DATA_SUPPORT',
            'options':options})
    plan={'version':VERSION,'config':config,'config_hash':digest(config),'gates_are_research_design_not_literature_standards':True,
          'selection_rule':'Finest eligible scale using TRAIN locations only. Do not compare raw RMSE across different-scale targets.',
          'model_rule':'routed_model is a preregistered engineering branch, not proof of SDE/causal validity.',
          'synthetic':rc['synthetic'],'code_hash':code_hash(),
          'data_fingerprints':{f'L{r.scale_m}/{r.graph}':r.arrays_sha256 for r in table.itertuples()},'folds':folds}
    plan['plan_hash']=digest(plan)
    target=work/'plan.json'
    if target.exists() and read_json(target)['plan_hash']!=plan['plan_hash'] and (work/'models').exists():
        raise ValueError('Training outputs exist under a different plan. Use a NEW work directory.')
    write_json(target,plan)
    for f in folds:
        print(f'Outer test={f["test_location"]}, validation={f["validation_location"]}, '
              f'train={f["train_locations"]}, selected scale={f["selected_scale_m"]}: {f["status"]}')
    return plan
