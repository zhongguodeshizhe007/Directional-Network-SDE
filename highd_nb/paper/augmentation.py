"""Extra arrays/paired trajectories created during the existing full CSV read.
They are descriptive/target data, NEVER automatically injected into predictors.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .ssm import following_metrics

METRIC_NAMES=['inv_ttc','drac','inv_psd','inv_mttc']
METRIC_UNITS={'inv_ttc':'1/s','drac':'m/s^2','inv_psd':'dimensionless','inv_mttc':'1/s'}


def attach_pair_metrics(d,leader,c):
    for name,field in [('leader_u','u'),('leader_accel','a_long'),('leader_length','width'),
                       ('leader_xc','xc'),('leader_yc','yc')]:
        d[name]=leader[field].to_numpy()
    metrics=following_metrics(d.dhw,d.u,d.leader_u,d.a_long,d.leader_accel,
                               braking=float(c.get('psd_braking_mps2',3.924)))
    for name in ['inv_mttc','inv_psd','mttc','psd']:
        d[name]=np.where(d.pair_valid,metrics[name],np.nan)


def risk_channels(valid_pairs,T,N,vehicle_exposure,time_support,c,fps):
    means=[];masks=[];exposures=[]
    for name in METRIC_NAMES:
        v=valid_pairs.loc[np.isfinite(valid_pairs[name]),['bin','node',name]]
        g=v.groupby(['bin','node'],sort=False)[name]
        avg=np.full((T,N),np.nan); expo=np.zeros((T,N))
        if len(v):
            m=g.mean();count=g.count();i=m.index.get_level_values(0).to_numpy(int);j=m.index.get_level_values(1).to_numpy(int)
            avg[i,j]=m.to_numpy();expo[i,j]=count.to_numpy()/fps
        with np.errstate(divide='ignore',invalid='ignore'):coverage=expo/vehicle_exposure
        ok=(np.isfinite(avg)&time_support[:,None]&(expo>=c['min_pair_exposure_s'])&(coverage>=c['min_pair_coverage']))
        means.append(avg);masks.append(ok);exposures.append(expo)
    raw=np.stack(means,-1);mask=np.stack(masks,-1)
    # Dimensional scales: 1 s for inverse times, 1 m/s^2 for DRAC, 1 for inverse PSD.
    scales=np.array([float(c['risk_time_scale_s']),1.,1.,float(c['risk_time_scale_s'])])
    transformed=np.log1p(raw*scales)
    return {'risk_channel_names':np.array(METRIC_NAMES),'risk_channels_raw':raw.astype('float32'),
            'risk_channels_y':transformed.astype('float32'),'mask_risk_channels':mask,
            'risk_channels_exposure_s':np.stack(exposures,-1).astype('float32'),
            'risk_channel_multiplier':scales.astype('float32')}


def save_pair_sample(base,graph_dir,rec,direction,origin,fps,c):
    """Fixed 5 Hz time sampling; optional deterministic cap. Export is NOT an event census.
    ID-aware event census still comes from full-rate events.csv, not from this sample.
    """
    hz=float(c.get('pair_sample_hz',5.))
    if hz<=0 or hz>fps: raise ValueError('pair_sample_hz must be in (0,frameRate]')
    stride=max(1,int(round(fps/hz)))
    x=base.loc[base.pair_valid&(((base.frame-origin)%stride)==0)].copy()
    cols=['frame','id','precedingId','laneId','node','bin','xc','yc','u','leader_u','a_long','leader_accel',
          'width','height','leader_length','dhw','closing_speed','ttc_calc','drac','mttc','psd','inv_ttc',
          'inv_psd','inv_mttc','leader_xc','leader_yc']
    x=x[cols]
    # Sort keys before sampling so CSV order cannot influence which observations are retained.
    x=x.sort_values(['id','precedingId','frame'],kind='stable')
    cap=c.get('pair_sample_max_rows',100000)
    if cap and len(x)>int(cap):
        h=pd.util.hash_pandas_object(x[['frame','id','precedingId']],index=False).to_numpy()
        x=x.iloc[np.argsort(h)[:int(cap)]]
    x=x.sort_values(['id','frame']);x['time_s']=(x.frame-origin)/fps
    x['recording_id']=str(rec);x['direction']=direction
    x.to_csv(Path(graph_dir)/'pairs_sample.csv.gz',index=False,compression='gzip')
