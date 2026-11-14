"""Same recording-safe historical windows for every paper baseline.
No time shuffling between train/validation/test. The frozen outer plan is authoritative.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from highd_nb.core.data import prepare_graph
from highd_nb.core.common import load_arrays

@dataclass(frozen=True)
class Window:
    graph: int
    anchor: int


def load_graphs(paths,norm,mc):
    result=[]
    for p in paths:
        g=prepare_graph(p,norm,mc);a=load_arrays(p)
        g['labels']={k:a[k] for k in ['future_onset','future_onset_mask','active_event','onset_event','horizons_s','thresholds_s','mask_pair_support']}
        g['paper_arrays']=a
        # Fully observed y is NOT reconstructed through these masks: missing y stays unavailable.
        g['features']=torch.cat([g['y'][...,None],g['my'][...,None].float(),g['z'],g['mz'].float(),g['az']],-1)
        # Features: y, risk observed flag, four z, four observed flags, four forward-age values = 14.
        result.append(g)
    return result


def windows(graphs,history,horizons,stride,cap=None):
    if history<1 or stride<1: raise ValueError('history and stride must be positive')
    out=[]
    for gi,g in enumerate(graphs):
        dt=g['dt'];h=max(int(round(x/dt)) for x in horizons)
        start=max(history-1,int(round(g['schema']['config']['context_window_s']/dt))-1)
        anchors=list(range(start,len(g['y'])-h,stride))
        if cap and len(anchors)>cap:
            anchors=[anchors[i] for i in np.linspace(0,len(anchors)-1,int(cap),dtype=int)]
        out.extend(Window(gi,a) for a in anchors)
    return out


def history_features(g,anchor,history):
    if anchor<history-1: raise ValueError('insufficient history; no future/backward filling allowed')
    return g['features'][anchor-history+1:anchor+1]


def directional_features(seq,A):
    """[L,N,14] -> [L,N,56]. Three gated neighbour-average differences.
    Counts/availability are carried as features; no neighbour is silently a measured zero.
    This is a tabular baseline representation, not a learned propagation coefficient.
    """
    base=seq;out=[base]
    for k in range(3):
        weights=A[k][None]*seq[...,1][:,None,:]
        den=weights.sum(-1)
        avg=torch.einsum('lij,ljf->lif',weights,seq)/den.clamp_min(1)[...,None]
        delta=torch.where((den>0)[...,None],avg-base,torch.zeros_like(avg))
        delta[...,1]=(den>0).float()
        out.append(delta)
    return torch.cat(out,-1)


def tabular(g,anchor,history):
    x=directional_features(history_features(g,anchor,history),g['A'])
    return x.permute(1,0,2).reshape(x.shape[1],-1).numpy()


def targets(g,anchor,horizons):
    ids=[anchor+int(round(h/g['dt'])) for h in horizons]
    y=g['y'][ids].T.numpy();m=(g['my'][ids]&g['my'][anchor]).T.numpy()
    return y,m


def event_targets(g,anchor,horizon,threshold,support='complete_future'):
    a=g['labels'];hi=np.flatnonzero(np.isclose(a['horizons_s'],horizon));ki=np.flatnonzero(np.isclose(a['thresholds_s'],threshold))
    if not len(hi) or not len(ki): raise ValueError('Requested horizon/threshold was not preprocessed')
    y=a['future_onset'][anchor,:,hi[0],ki[0]]
    m=a['future_onset_mask'][anchor,:,hi[0],ki[0]].copy()
    m &= g['my'][anchor].numpy()
    if support=='complete_future':
        h=int(round(horizon/g['dt']))
        m &= a['mask_pair_support'][anchor+1:anchor+h+1].all(0)
    elif support!='observed_labels': raise ValueError('Unknown event support policy')
    return np.nan_to_num(y,nan=0).astype(int),m


def metadata(g,anchor,horizon,nodes):
    s=g['schema']
    return [dict(graph=g['name'],recording_id=str(s['recording_id']),location_id=s['location_id'],direction=s['direction'],
                 anchor_index=anchor,anchor_s=float(g['time'][anchor]),target_s=float(g['time'][anchor])+horizon,
                 node=int(n),horizon_s=float(horizon)) for n in nodes]
