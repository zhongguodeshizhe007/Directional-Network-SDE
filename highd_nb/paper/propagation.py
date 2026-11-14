"""Descriptive propagation evidence, not identification of intervention effects."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from highd_nb.core.models import filtered_particles


def event_census(graphs):
    frames=[]
    for g in graphs:
        p=Path(g['path']).parent/'events.csv'
        try:q=pd.read_csv(p,dtype={'recording_id':str})
        except pd.errors.EmptyDataError:continue
        if q.empty:continue
        q['graph']=g['name'];q['location_id']=g['schema']['location_id'];frames.append(q)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def transport_audit(events):
    """One event visiting >1 cell is migration of a pair, NOT additional events."""
    if events.empty:return pd.DataFrame()
    rows=[]
    for key,g in events.groupby(['recording_id','direction','threshold_s']):
        visited=g.visited_nodes.fillna('').astype(str).map(lambda s:len(set(x for x in s.split(';') if x)))
        rows.append(dict(zip(['recording_id','direction','ttc_threshold_s'],key),
                         unique_pair_episodes=len(g),episodes_visiting_multiple_cells=int((visited>1).sum()),
                         total_episode_cell_visits=int(visited.sum()),
                         not_additional_events=int((visited-1).clip(lower=0).sum()),
                         interpretation='same-pair transport excluded from new-event counts'))
    return pd.DataFrame(rows)


def distinct_pair_links(events,graphs,max_lag_s=5.):
    """Neighbouring *different* pair onsets after an event. Descriptive candidates only.
    No event-tree/causal assignment is made; a destination can have multiple candidates.
    """
    if events.empty:return pd.DataFrame()
    am={g['name']:g['A'].numpy() for g in graphs};rows=[]
    for (graph,thr),g in events.groupby(['graph','threshold_s']):
        g=g.sort_values('confirmed_s');a=am[graph]
        for _,src in g.iterrows():
            q=g[(g.confirmed_s>src.confirmed_s)&(g.confirmed_s<=src.confirmed_s+max_lag_s)]
            for _,dst in q.iterrows():
                # Deduplicate oriented follower-leader identities; event_id alone is insufficient.
                if src.follower_id==dst.follower_id and src.leader_id==dst.leader_id:continue
                i,j=int(dst.confirmed_node),int(src.confirmed_node)
                if i<0 or j<0:continue
                k=np.flatnonzero(a[:,i,j]>0)
                if len(k):rows.append(dict(graph=graph,ttc_threshold_s=thr,source_event=src.event_id,
                    destination_event=dst.event_id,source_node=j,target_node=i,source_type=['U','D','L'][int(k[0])],
                    lag_s=float(dst.confirmed_s-src.confirmed_s),interpretation='distinct-pair temporal adjacency; not causal'))
    return pd.DataFrame(rows)


@torch.no_grad()
def component_samples(model,graphs,ws,pc,device='cpu'):
    rows=[];limit=int(pc.get('component_origins_per_graph',20))
    by={}
    for w in ws:by.setdefault(w.graph,[]).append(w)
    for gi,group in by.items():
        g=graphs[gi]
        ix=np.linspace(0,len(group)-1,min(limit,len(group)),dtype=int)
        for w in [group[i] for i in ix]:
            torch.manual_seed(pc['seed']+w.anchor)
            x=filtered_particles(model,g,w.anchor,1,pc['history_steps'],device)
            z=g['z'][w.anchor].to(device)[None];gate=g['my'][w.anchor].to(device)[None] if model.kind=='masked' else None
            F,G,sigma=model.dynamics.components(x,z,g['A'].to(device),gate)
            for n in np.flatnonzero(g['my'][w.anchor].numpy()):
                rows.append(dict(graph=g['name'],recording_id=str(g['schema']['recording_id']),node=int(n),
                    anchor_s=float(g['time'][w.anchor]),x=float(x[0,n]),cv_std=float(z[0,n,1]),
                    density_std=float(z[0,n,2]),F=float(F[0,n]),G_U=float(G[0][0,n]),G_D=float(G[1][0,n]),
                    G_L=float(G[2][0,n]),sigma=float(sigma[0,n]),coordinate='normalized model state'))
    return pd.DataFrame(rows)


def negative_controls(model,graphs,ws,norm,horizons,pc,device='cpu'):
    """Fixed-model graph placebos; distinct from retrained topology ablations."""
    import copy
    from highd_nb.core.evaluation import forecast_origin,score_distribution
    rows=[]
    by={}
    for w in ws:by.setdefault(w.graph,[]).append(w)
    for gi,group in by.items():
        g=graphs[gi];indices=np.linspace(0,len(group)-1,min(3,len(group)),dtype=int)
        for wi in indices:
            w=group[wi];N=g['y'].shape[1];rng=np.random.default_rng(pc['seed']+gi)
            permutation=torch.tensor(rng.permutation(N),dtype=torch.long)
            matrices={'factual':g['A'],'reverse_longitudinal':g['A'][[1,0,2]],
                      'shuffle_sources':g['A'][:,:,permutation], 'remove_edges':torch.zeros_like(g['A'])}
            for name,A in matrices.items():
                changed=dict(g,A=A);torch.manual_seed(pc['seed']+w.anchor+gi*100003)
                with torch.no_grad():forecasts=forecast_origin(model,changed,w.anchor,pc['particles'],horizons,pc['history_steps'],device)
                for h,pr in forecasts.items():
                    target=w.anchor+int(round(h/g['dt']));mask=(g['my'][w.anchor]&g['my'][target]).numpy()
                    if not mask.any():continue
                    scores=score_distribution(pr,g['y'][target].to(device),norm['y_scale'])
                    rows.append(dict(graph=g['name'],recording_id=str(g['schema']['recording_id']),anchor_index=w.anchor,
                        horizon_s=h,control=name,crps=float(scores['crps'].cpu().numpy()[mask].mean()),
                        n_targets=int(mask.sum()),interpretation='fixed-model graph falsification, NOT retrained comparison or causal effect'))
    df=pd.DataFrame(rows)
    if len(df):
        factual=df[df.control=='factual'][['graph','anchor_index','horizon_s','crps']].rename(columns={'crps':'factual_crps'})
        df=df.merge(factual,on=['graph','anchor_index','horizon_s']);df['delta_crps']=df.crps-df.factual_crps
    return df
