"""Publication-oriented plots from saved experiment outputs, with traceable source tables.
Each panel is an independent vector PDF and 180-dpi PNG. No model fitting in this file.
No interpolating an empty road cell into a measured safe state. No invented p-values.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from highd_nb.core.common import read_json,write_json


def _read(path):
    if not Path(path).exists():return pd.DataFrame()
    try:return pd.read_csv(path,dtype={'recording_id':str})
    except pd.errors.EmptyDataError:return pd.DataFrame()


def render_paper_report(directory):
    root=Path(directory);receipt=read_json(root/'complete.json');protocol=read_json(root/'protocol.json')
    pc=protocol['paper_config'];figdir=root/'figures';figdir.mkdir(exist_ok=True);entries=[];skips=[]
    synthetic='SYNTHETIC' in protocol['scope'];tag='SYNTHETIC SOFTWARE TEST - NOT EMPIRICAL RESULTS' if synthetic else 'Held-out data; see frozen protocol'
    def save(fig,name,title,source,reference,note=''):
        fig.suptitle(title,fontsize=12,y=.99)
        fig.text(.5,.012,tag,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.96))
        for ext in ['png','pdf']:fig.savefig(figdir/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        entries.append({'file':name,'title':title,'source':source,'reference':reference,'note':note,'scope':protocol['scope']})
    def figure():return plt.subplots(figsize=(8.2,4.8))
    def skip(name,reason):skips.append(dict(file=name,reason=reason))
    metric=_read(root/'tables/T03_continuous_forecasts.csv');pred=_read(root/'continuous_predictions.csv.gz')
    main=metric[~metric.predictor.str.contains('local|no_lateral|symmetric|const_sigma|symbolic',regex=True)]
    for key,label in [('rmse','RMSE (transformed risk units)'),('crps','CRPS (transformed risk units)'),('coverage90','90% interval coverage'),('r2','R-squared')]:
        fig,ax=figure()
        for model,g in main.groupby('predictor',sort=False):
            g=g.sort_values('horizon_s');ax.plot(g.horizon_s,g[key],marker='o',label=model)
        if key=='coverage90':ax.axhline(.9,linestyle='--',label='Nominal coverage')
        ax.set(xlabel='Forecast horizon (s)',ylabel=label);ax.legend(fontsize=8,ncol=2)
        save(fig,'F02_'+key,'Multi-horizon forecast comparison','tables/T03_continuous_forecasts.csv','AAP-2 Table 5 and Figures 16-17',
             'Same target, split, origins and masks. Different SSMs are not ranked by unscaled RMSE.')
    abl=metric[metric.predictor.str.startswith('SDE')]
    if len(abl):
        fig,ax=figure()
        for name,g in abl.groupby('predictor',sort=False):
            g=g.sort_values('horizon_s');ax.plot(g.horizon_s,g.crps,marker='o',label=name)
        ax.set(xlabel='Horizon (s)',ylabel='CRPS');ax.legend(fontsize=7,ncol=2)
        save(fig,'F03_ablation','Graph, direction and diffusion ablations','tables/T05_orthogonal_ablations.csv','AAP-2 ablation comparison',
             'Latent-local retains the latent observation model. No-lateral, symmetric and constant-sigma are separate hypotheses.')
    case=read_json(root/'map_case.json');gname=case['graph'];nodes=_read(root/'map_nodes.csv')
    primary='SDE-latent' if 'SDE-latent' in pred.predictor.unique() else next(x for x in pred.predictor.unique() if x.startswith('SDE'))
    q=pred[(pred.graph==gname)&(pred.predictor==primary)];a=int(q.anchor_index.min());M=int(case['schema']['roi']['n_cells']);N=len(nodes)
    snap=q[q.anchor_index==a];vmax=max(float(snap.target_y.max()),float(snap.pred_mean.max()),1e-5)
    hvals=sorted(snap.horizon_s.unique())
    for h in hvals:
        v=snap[np.isclose(snap.horizon_s,h)]
        for mode,col in [('observed','target_y'),('forecast','pred_mean'),('uncertainty','pred_sd')]:
            values=np.full(N,np.nan);values[v.node.to_numpy(int)]=v[col]
            fig,ax=figure();im=ax.imshow(np.ma.masked_invalid(values.reshape(-1,M)),origin='lower',aspect='auto',
                extent=(0,M*protocol.get('risk_cell_m',case['schema']['roi']['cell_length_m']),0,N//M),
                vmin=0,vmax=vmax if mode!='uncertainty' else max(float(snap.pred_sd.max()),1e-5),interpolation='nearest')
            ax.set(xlabel='Distance along travel direction (m)',ylabel='Lane rank from median',
                   yticks=np.arange(N//M)+.5,yticklabels=[str(i) for i in range(N//M)]);fig.colorbar(im,ax=ax,label='Transformed risk Y' if mode!='uncertainty' else 'Predictive standard deviation')
            save(fig,f'F04_map_h{h:g}_{mode}',f'{mode.title()} risk map | {primary} | +{h:g} s',
                 'continuous_predictions.csv.gz + map_nodes.csv','AAP-2 Figures 13-15 and 18-20',
                 'Fixed first eligible anchor. Observed/forecast share a color range across all horizons; missing is masked. No road-image interpolation.')
    horizon=float(pc['primary_horizon_s']);line=q[np.isclose(q.horizon_s,horizon)]
    if len(line):
        node=int(line.node.min());line=line[line.node==node].sort_values('target_s');fig,ax=figure()
        ax.plot(line.target_s,line.target_y,label='Observed');ax.plot(line.target_s,line.pred_mean,label='Forecast mean')
        ax.fill_between(line.target_s.to_numpy(),line.pred_q05.to_numpy(),line.pred_q95.to_numpy(),alpha=.2,label='90% prediction interval')
        ax.set(xlabel='Target time (s)',ylabel='Transformed risk Y');ax.legend()
        save(fig,'F05_temporal_interval',f'Risk trajectory | node {node} | horizon {horizon:g} s','continuous_predictions.csv.gz',
             'AAP-2 Figures 10-12; extended with predictive uncertainty','First eligible node by ID, not selected for good fit.')
    obs=np.load(root/'map_observations.npz',allow_pickle=False)
    for mode in ['observed','forecast']:
        data=np.full_like(obs['y'],np.nan,dtype=float)
        if mode=='observed':data=np.where(obs['mask'],obs['y'],np.nan)
        else:
            for r in q[np.isclose(q.horizon_s,horizon)].itertuples():
                t=int(r.anchor_index+round(horizon/case['schema']['config']['step_s']));data[t,int(r.node)]=r.pred_mean
        # Fixed median-side lane. Time gaps remain gaps rather than padded/smoothed risk.
        data=data[:,:M];fig,ax=figure();im=ax.imshow(np.ma.masked_invalid(data.T),origin='lower',aspect='auto',interpolation='nearest',
             extent=(float(obs['time'].min())-.5*case['schema']['config']['step_s'],float(obs['time'].max())+.5*case['schema']['config']['step_s'],0,M*case['schema']['roi']['cell_length_m']),
             vmin=0,vmax=max(float(np.nanmax(obs['y'])),float(q.pred_mean.max()),1e-5))
        ax.set(xlabel='Time (s)',ylabel='Distance along travel direction (m)');fig.colorbar(im,ax=ax,label='Transformed risk Y')
        save(fig,'F06_spacetime_'+mode,f'{mode.title()} space-time risk | median-side lane','map_observations.npz / continuous_predictions.csv.gz',
             'AAP-2 propagation map design','Space-time pattern alone is not causal evidence; see same-pair transport audit.')
    ep=_read(root/'event_predictions.csv.gz')
    if not ep.empty:
        use=ep[np.isclose(ep.horizon_s,horizon)&np.isclose(ep.ttc_threshold_s,pc['primary_ttc_s'])]
        for curve in ['ROC','PR']:
            fig,ax=figure();plotted=0
            for name,g in use.groupby('model',sort=False):
                if g.y.nunique()<2:continue
                if curve=='ROC':x,y,_=roc_curve(g.y,g.probability)
                else:y,x,_=precision_recall_curve(g.y,g.probability)
                ax.plot(x,y,label=name);plotted+=1
            if plotted:
                if curve=='ROC':ax.plot([0,1],[0,1],linestyle='--',label='Chance');ax.set(xlabel='False-positive rate',ylabel='Sensitivity')
                else:
                    ax.axhline(use.y.mean(),linestyle='--',label='Natural prevalence');ax.set(xlabel='Recall',ylabel='Precision')
                ax.legend(fontsize=8)
                save(fig,'F07_'+curve,curve+' | future confirmed-conflict onset','event_predictions.csv.gz',
                     'Unified Figure 6 and Yuan Figure 3; direct future-event task is different from current detection')
            else:plt.close(fig);skip('F07_'+curve,'No two-class support at the prespecified primary horizon/threshold')
        reliability=_read(root/'event_reliability.csv');r=reliability[np.isclose(reliability.horizon_s,horizon)&np.isclose(reliability.ttc_threshold_s,pc['primary_ttc_s'])]
        if not r.empty:
            fig,ax=figure();ax.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
            for name,g in r.groupby('model',sort=False):ax.plot(g.predicted_probability,g.observed_rate,marker='o',label=name)
            ax.set(xlabel='Predicted event probability',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
            save(fig,'F08_reliability','Event probability calibration','event_reliability.csv','Extension to Unified/Yuan probability evaluation',
                 'Bin counts are in the source CSV. Cutoff and calibration fitted on validation only.')
        else:skip('F08_reliability','No eligible primary-horizon event predictions')
        em=_read(root/'tables/T04_event_probability.csv');em=em[np.isclose(em.horizon_s,horizon)]
        if not em.empty and em.average_precision.notna().any():
            fig,ax=figure()
            for name,g in em.groupby('model',sort=False):ax.plot(g.ttc_threshold_s,g.average_precision,marker='o',label=name)
            ax.set(xlabel='TTC threshold (s)',ylabel='Average precision (AP)');ax.legend(fontsize=8)
            save(fig,'F09_threshold_sensitivity','Conflict-definition sensitivity','tables/T04_event_probability.csv','Yuan Figure 3',
                 'AP is non-interpolated average precision, not trapezoidal PR-AUC; undefined single-class values are left missing.')
        else:skip('F09_threshold_sensitivity','No defined primary-horizon AP')
    else:skip('F07_F09','Event support gate failed; consult event_readout_receipts.json')
    warns=_read(root/'event_warnings.csv')
    if not warns.empty:
        warns=warns[np.isclose(warns.horizon_s,horizon)&np.isclose(warns.ttc_threshold_s,pc['primary_ttc_s'])]
        for col,label in [('warning_period_fraction','Fraction of available anchors warned'),('last_warning_transition_lead_s','Last warning-transition lead time (s)')]:
            pairs=[(name,g[col].dropna().to_numpy()) for name,g in warns.groupby('model',sort=False)]
            pairs=[(n,v) for n,v in pairs if len(v)]
            if not pairs:skip('F10_'+col,'No eligible detected events');continue
            fig,ax=figure();ax.boxplot([v for _,v in pairs],tick_labels=[n for n,_ in pairs],showfliers=True)
            ax.tick_params(axis='x',rotation=20,labelsize=8);ax.set(ylabel=label)
            save(fig,'F10_'+col,'Event warning effectiveness', 'event_warnings.csv','Unified Figure 6 and Table 5',
                 'Adaptation: reference is full-rate TTC confirmation, not NDS minimum distance. Undetected events remain in recall but have no lead time.')
    ctx=_read(root/'context/context_test_pairs.csv.gz')
    if not ctx.empty:
        curve=_read(root/'context/context_learning_curve.csv');fig,ax=figure();ax.plot(curve.epoch,curve.validation_proximity_nll)
        ax.set(xlabel='Epoch',ylabel='Validation proximity NLL');save(fig,'F11_context_learning','Conditional proximity validation','context/context_learning_curve.csv','Unified Figure 5')
        fig,ax=figure();ax.hist(ctx.pit,bins=20,density=True);ax.axhline(1,linestyle='--');ax.set(xlabel='Conditional proximity PIT',ylabel='Density')
        save(fig,'F12_context_PIT','Proximity distribution diagnostic','context/context_test_pairs.csv.gz','Extension: verify the inferred conditional distribution',
             'Deviation from uniformity diagnoses lack of distribution calibration, not necessarily conflict misclassification.')
        for source,label in [(ctx,'sampled pair-frames'),(_read(root/'context/context_pair_intensity.csv'),'pair summaries')]:
            column='intensity_p50' if source is ctx else 'max_intensity';x=np.sort(source[column].to_numpy());x=x[np.isfinite(x)&(x>0)]
            fig,ax=figure();ax.loglog(x,(len(x)-np.arange(len(x)))/max(1,len(x)));ax.set(xlabel='Contextual intensity n at p=0.5',ylabel='Empirical survival probability')
            save(fig,'F13_CCDF_'+('frames' if source is ctx else 'pairs'),'Intensity spectrum | '+label,
                 'context/context_test_pairs.csv.gz / context_pair_intensity.csv','Unified Figure 9',
                 'Descriptive CCDF only. A visually straight log-log segment is NOT a power-law test; pair summaries are not full lane-change events.')
        spectrum=_read(root/'context/context_spectrum.csv');gap=np.sort(spectrum.gap_m.unique());intensity=np.sort(spectrum.intensity_n.unique())
        Z=spectrum.pivot(index='intensity_n',columns='gap_m',values='probability').reindex(index=intensity,columns=gap).to_numpy()
        fig,ax=figure();im=ax.pcolormesh(gap,intensity,Z,shading='auto',vmin=0,vmax=1);ax.set_xscale('log');ax.set_yscale('log')
        ax.set(xlabel='Gap (m)',ylabel='Conflict intensity n');fig.colorbar(im,ax=ax,label='Contextual extremeness C(n;s,theta)')
        save(fig,'F14_context_spectrum','Context-dependent interaction spectrum','context/context_spectrum.csv','Unified Figures 3 and 10',
             'A gap-intensity slice, NOT a spatial lane-change field, future-event probability or empirical crash rate.')
        row=ctx.sort_values(['recording_id','direction','id','frame']).iloc[0]
        casepair=ctx[(ctx.recording_id==row.recording_id)&(ctx.direction==row.direction)&(ctx.id==row.id)&(ctx.precedingId==row.precedingId)].sort_values('time_s')
        for col,label in [('u','Speed (m/s)'),('a_long','Longitudinal acceleration (m/s^2)'),('intensity_p50','Contextual intensity n'),('ttc_calc','TTC (s)')]:
            fig,ax=figure();vv=np.where(np.isfinite(casepair[col]),casepair[col],np.nan);ax.plot(casepair.time_s,vv,label='Follower' if col in ['u','a_long'] else col)
            if col in ['u','a_long']:ax.plot(casepair.time_s,casepair['leader_u' if col=='u' else 'leader_accel'],label='Leader')
            if col=='intensity_p50':ax.set_yscale('log')
            ax.set(xlabel='Time (s)',ylabel=label);ax.legend()
            save(fig,'F15_pair_'+col,'Same-lane interaction case','context/context_test_pairs.csv.gz','Unified Figures 7 and 10',
                 'First test pair by ID. Sampled trajectories, not a reconstructed NDS near-crash.')
    comp=_read(root/'dynamics_components.csv.gz')
    if not comp.empty:
        v=comp[comp.predictor==primary]
        if len(v):
            fig,ax=figure();ax.boxplot([v[c] for c in ['F','G_U','G_D','G_L']],tick_labels=['F','G_U','G_D','G_L'])
            ax.set(ylabel='Drift contribution (normalized state / s)')
            save(fig,'F16_drift_components','Local and directional drift contributions','dynamics_components.csv.gz','NetworkSDE-specific mechanistic evidence',
                 'State-supported observational decomposition; magnitudes do not establish causal effects.')
            fig,ax=figure();sc=ax.scatter(v.cv_std,v.density_std,c=v.sigma,s=20,alpha=.7);fig.colorbar(sc,ax=ax,label='Sigma (normalized state / sqrt(s))')
            ax.set(xlabel='Speed CV (train-standardized)',ylabel='Density (train-standardized)')
            save(fig,'F17_diffusion_support','State-dependent predictive variability','dynamics_components.csv.gz','NetworkSDE-specific uncertainty evidence',
                 'Only sampled supported states shown; not a pure intrinsic-driver-noise estimate or interpolated physical field.')
    controls=_read(root/'tables/T09c_graph_negative_controls.csv')
    if 'delta_crps' in controls:
        controls=controls[(controls.predictor==primary)&(controls.control!='factual')]
        fig,ax=figure()
        for name,g in controls.groupby('control',sort=False):
            v=g.groupby('horizon_s').delta_crps.mean();ax.plot(v.index,v.values,marker='o',label=name)
        ax.axhline(0,linestyle='--');ax.set(xlabel='Horizon (s)',ylabel='CRPS change from factual graph');ax.legend(fontsize=8)
        save(fig,'F19_graph_placebos','Fixed-model graph falsification','tables/T09c_graph_negative_controls.csv',
             'Extension: propagation-specific negative controls',
             'Negative values can occur and are retained. A worse placebo does not prove causality; see retrained ablations too.')
    symbolic=_read(root/'tables/T07_symbolic_acceptance.csv')
    if 'neural_crps' in symbolic:
        fig,ax=figure()
        for name,g in symbolic.groupby('predictor',sort=False):
            g=g.sort_values('horizon_s');ratio=g.symbolic_crps/g.neural_crps;ax.plot(g.horizon_s,ratio,marker='o',label=name)
        ax.axhline(1+protocol.get('symbolic_relative_tolerance',.2),linestyle='--',label='Prespecified engineering tolerance')
        ax.set(xlabel='Validation horizon (s)',ylabel='Symbolic CRPS / neural CRPS');ax.legend(fontsize=7,ncol=2)
        save(fig,'F18_symbolic_fidelity','Every validation horizon must pass','tables/T07_symbolic_acceptance.csv','Equation-discovery validation',
             'All candidates shown, including rejected ones. Exact tolerance is recorded in the symbolic receipt; default 20% is not a literature standard.')
    costs=_read(root/'tables/T10_compute_cost.csv')
    if 'training_and_validation_wall_s' in costs:
        v=costs.dropna(subset=['training_and_validation_wall_s']).sort_values('predictor')
        if len(v):
            fig,ax=figure();ax.barh(v.predictor,v.training_and_validation_wall_s)
            ax.set(xlabel='Training + validation wall time (s)',ylabel='Model')
            save(fig,'F20_compute_cost','Measured computation cost','tables/T10_compute_cost.csv','Extension: reproducible deployment cost',
                 'Measured on the recorded device and configuration. SDE/sequence training objectives differ; no cross-device speed-up is inferred.')
    coherent=_read(root/'tables/T04c_horizon_probability_coherence.csv')
    if 'nonmonotone_fraction' in coherent:
        v=coherent[np.isclose(coherent.ttc_threshold_s,pc['primary_ttc_s'])].groupby('model').nonmonotone_fraction.max().sort_index()
        if len(v):
            fig,ax=figure();ax.barh(v.index,v.values);ax.set(xlabel='Maximum adjacent-horizon violation fraction',ylabel='Model')
            save(fig,'F21_horizon_coherence','Do cumulative event probabilities decrease with horizon?',
                 'tables/T04c_horizon_probability_coherence.csv','Extension: multi-horizon probability coherence',
                 'Shared positive cumulative-hazard calibration preserves monotonicity. Zero violations are a mathematical constraint, not proof of empirical accuracy.')
    if not ep.empty:
        v=ep[np.isclose(ep.ttc_threshold_s,pc['primary_ttc_s']) & ep.model.str.endswith('-hazard')]
        if len(v):
            row=v.sort_values(['graph','anchor_index','node']).iloc[0]
            v=v[(v.graph==row.graph)&(v.anchor_index==row.anchor_index)&(v.node==row.node)]
            fig,ax=figure()
            for name,g in v.groupby('model'):
                g=g.sort_values('horizon_s');ax.plot(g.horizon_s,g.probability,marker='o',label=name)
            ax.set(xlabel='Forecast horizon (s)',ylabel='Probability of a new confirmed onset',ylim=(0,1));ax.legend(fontsize=8)
            save(fig,'F22_hazard_curve','Coherent multi-horizon early warning','event_predictions.csv.gz',
                 'Extension: future confirmed-onset probability','First eligible test origin by ID, not selected by predicted risk or label. Not collision probability.')
    write_json(figdir/'figure_manifest.json',{'scope':protocol['scope'],'generated':entries,'not_generated':skips,
              'automatic_powerlaw_claim':False,'empirical_superiority_claim':False})
    # Lightweight browsable gallery: references actual files, never generated placeholder figures.
    html=['<!doctype html><meta charset="utf-8"><title>Paper figure gallery</title>',
          '<style>body{font-family:system-ui;max-width:1000px;margin:30px auto}img{max-width:100%}article{margin:35px 0}small{display:block}</style>',
          '<h1>NetworkSDE paper outputs</h1>',f'<h2>{tag}</h2>']
    for e in entries:html.append(f'<article><h3>{e["file"]}: {e["title"]}</h3><img src="{e["file"]}.png"><p>{e["note"]}</p><small>Source: {e["source"]}; design reference: {e["reference"]}</small></article>')
    (figdir/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print(f'Rendered {len(entries)} figures to {figdir}; {len(skips)} explicitly skipped.')
    return {'figure_directory':str(figdir),'generated_figures':len(entries),'skipped':skips,'scope':protocol['scope']}
