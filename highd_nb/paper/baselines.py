"""Fair, compact baselines using the same historical channels and held-out locations.
STTransformer is a NEW spatial/temporal adaptation, NOT an exact reproduction of
AAP's STTN (which has periodic/global-local branches). No fictional daily cycles.
ConvLSTM implements convolution inside its gates, not an LSTM merely named ConvLSTM.
"""
from __future__ import annotations
import copy,math
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from highd_nb.core.models import censored_log_prob
from .data import history_features,tabular,targets
from highd_nb.runtime import choose_device, xgb_parameters, check_xgb_device, xgb_predict, cpu_state_dict

class ConvLSTMCell(nn.Module):
    def __init__(self,n_in,hidden):
        super().__init__();self.hidden=hidden;self.gates=nn.Conv2d(n_in+hidden,4*hidden,3,padding=1)
    def forward(self,x,h,c):
        i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1)
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c),c

class SequenceForecaster(nn.Module):
    def __init__(self,kind,history,n_horizons,hidden=32):
        super().__init__();self.kind=kind;self.H=n_horizons;self.hidden=hidden;self.history=history
        if kind=='lstm':
            self.temporal=nn.LSTM(14,hidden,batch_first=True)
        elif kind=='cnn':
            self.conv=nn.Sequential(nn.Conv2d(14*history,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.ReLU())
        elif kind=='convlstm':self.cell=ConvLSTMCell(14,hidden)
        elif kind=='st_transformer':
            # Geometry is relative median-side lane rank and distance in the travel direction.
            self.project=nn.Linear(16,hidden)
            heads=4 if hidden%4==0 else 1
            self.spatial=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,heads,2*hidden,dropout=0.,batch_first=True),1,enable_nested_tensor=False)
            self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,heads,2*hidden,dropout=0.,batch_first=True),1,enable_nested_tensor=False)
        else: raise ValueError(f'Unknown sequence baseline {kind}')
        self.head=nn.Linear(hidden,2*n_horizons)
    def forward(self,x,n_cells):
        B,L,N,F=x.shape
        if L!=self.history or F!=14 or N%n_cells: raise ValueError('Invalid historical grid shape')
        if self.kind=='lstm':
            seq=x.permute(0,2,1,3).reshape(B*N,L,F);v,_=self.temporal(seq);h=v[:,-1].reshape(B,N,-1)
        elif self.kind in ['cnn','convlstm']:
            lanes=N//n_cells;grid=x.permute(0,1,3,2).reshape(B,L,F,lanes,n_cells)
            if self.kind=='cnn':h=self.conv(grid.reshape(B,L*F,lanes,n_cells))
            else:
                h=torch.zeros(B,self.hidden,lanes,n_cells,device=x.device);c=torch.zeros_like(h)
                for t in range(L):h,c=self.cell(grid[:,t],h,c)
            h=h.flatten(2).transpose(1,2)
        else:
            idx=torch.arange(N,device=x.device);pos=torch.stack([(idx//n_cells).float()/max(N//n_cells-1,1),
                                                          (idx%n_cells).float()/max(n_cells-1,1)],-1)
            h=self.project(torch.cat([x,pos[None,None].expand(B,L,N,2)],-1))
            h=self.spatial(h.reshape(B*L,N,-1)).reshape(B,L,N,-1).permute(0,2,1,3).reshape(B*N,L,-1)
            # Ordered past windows, with a nonlearned sinusoidal time encoding.
            t=torch.arange(L,device=x.device).float()[:,None];freq=torch.exp(-math.log(10000)*torch.arange(0,self.hidden,2,device=x.device)/self.hidden)
            pe=torch.zeros(L,self.hidden,device=x.device);pe[:,0::2]=torch.sin(t*freq);pe[:,1::2]=torch.cos(t*freq)[:,:pe[:,1::2].shape[1]]
            h=self.temporal(h+pe[None])[:,-1].reshape(B,N,-1)
        q=self.head(h).reshape(B,N,self.H,2)
        return q[...,0],(.005+torch.nn.functional.softplus(q[...,1])).clamp_max(20)


def _grid_cells(g):return int(g['schema']['roi']['n_cells'])


def _batches(graphs,ws,history,horizons,batch,rng=None,device='cpu'):
    # Each batch comes from one graph. No padding-induced invisible vehicles.
    by_graph={}
    for w in ws:by_graph.setdefault(w.graph,[]).append(w)
    order=list(by_graph)
    if rng is not None:rng.shuffle(order)
    for gi in order:
        seq=list(by_graph[gi]);g=graphs[gi]
        if rng is not None:rng.shuffle(seq)
        for start in range(0,len(seq),batch):
            chunk=seq[start:start+batch]
            x=torch.stack([history_features(g,w.anchor,history) for w in chunk]).to(device)
            a=[targets(g,w.anchor,horizons) for w in chunk]
            y=torch.tensor(np.stack([q[0] for q in a]),device=device)
            m=torch.tensor(np.stack([q[1] for q in a]),device=device)
            yield chunk,x,y,m,_grid_cells(g)


def train_sequence(kind,graphs,ws,val_graphs,val_ws,horizons,pc,device='auto'):
    device=choose_device(device)
    torch.manual_seed(int(pc['seed']))
    model=SequenceForecaster(kind,pc['history_steps'],len(horizons),pc['hidden']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=pc['learning_rate'],weight_decay=1e-5)
    best=float('inf');state=None;stale=0;rows=[];rng=np.random.default_rng(pc['seed'])
    for epoch in range(pc['epochs']):
        model.train();total=n=0
        for _,x,y,m,cells in _batches(graphs,ws,pc['history_steps'],horizons,pc['batch_size'],rng,device):
            if not m.any():continue
            mu,sd=model(x,cells);loss=-censored_log_prob(y,mu,sd)[m].mean()
            if not torch.isfinite(loss):raise FloatingPointError(f'{kind}: nonfinite training loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())*int(m.sum());n+=int(m.sum())
        model.eval();vtotal=vn=0
        with torch.no_grad():
            for _,x,y,m,cells in _batches(val_graphs,val_ws,pc['history_steps'],horizons,pc['batch_size'],None,device):
                if not m.any():continue
                mu,sd=model(x,cells);vtotal+=float(-censored_log_prob(y,mu,sd)[m].sum());vn+=int(m.sum())
        if not n or not vn:raise ValueError(f'{kind}: no training or validation targets')
        score=vtotal/vn;rows.append(dict(epoch=epoch+1,training_nll=total/n,validation_nll=score))
        if score<best:best=score;state=cpu_state_dict(model);stale=0
        else:stale+=1
        if stale>=pc['patience']:break
    model.load_state_dict(state);model.eval()
    return model,pd.DataFrame(rows)


def predict_sequence(model,graphs,ws,horizons,pc,device='cpu'):
    answer={}
    with torch.no_grad():
        for chunk,x,_,_,cells in _batches(graphs,ws,pc['history_steps'],horizons,pc['batch_size'],None,device):
            mu,sd=model(x,cells)
            for w,a,b in zip(chunk,mu.cpu().numpy(),sd.cpu().numpy()):answer[(w.graph,w.anchor)]=(a,b)
    return answer

class TabularForecaster:
    def __init__(self,kind,models,sds,history):self.kind=kind;self.models=models;self.sds=sds;self.history=history
    def predict(self,g,anchor):
        x=tabular(g,anchor,self.history)
        mu=np.column_stack([xgb_predict(m,x) if self.kind=='xgboost' else m.predict(x) for m in self.models]);sd=np.broadcast_to(np.asarray(self.sds)[None],mu.shape).copy()
        return mu.astype('float32'),sd.astype('float32')


def train_tabular(kind,graphs,ws,val_graphs,val_ws,horizons,pc,device='auto'):
    device=choose_device(device)
    def matrix(gs,windows):
        xx=[];yy=[];mm=[]
        for w in windows:
            g=gs[w.graph];x=tabular(g,w.anchor,pc['history_steps']);y,m=targets(g,w.anchor,horizons)
            xx.append(x);yy.append(y);mm.append(m)
        if not xx:raise ValueError('No historical windows')
        return np.concatenate(xx),np.concatenate(yy),np.concatenate(mm)
    x,y,m=matrix(graphs,ws);vx,vy,vm=matrix(val_graphs,val_ws);models=[];sds=[]
    for h in range(len(horizons)):
        if not m[:,h].any() or not vm[:,h].any():raise ValueError('No comparable support for tabular model')
        if kind=='ridge':est=Ridge(alpha=10.)
        elif kind=='xgboost':
            from xgboost import XGBRegressor
            est=XGBRegressor(n_estimators=int(pc.get('xgb_estimators',150)),max_depth=3,learning_rate=.05,
                             objective='reg:squarederror',random_state=pc['seed'],reg_lambda=5.,**xgb_parameters(device))
        else:raise ValueError(kind)
        est.fit(x[m[:,h]],y[m[:,h],h])
        if kind=='xgboost':check_xgb_device(est,device)
        pred=xgb_predict(est,vx[vm[:,h]]) if kind=='xgboost' else est.predict(vx[vm[:,h]])
        # Validation-only residual scale; no test residuals. This is a simple benchmark,
        # not a claim of a fitted stochastic process or calibrated tails.
        sd=max(float(np.sqrt(np.mean((pred-vy[vm[:,h],h])**2))),.005)
        models.append(est);sds.append(sd)
    return TabularForecaster(kind,models,sds,pc['history_steps'])
