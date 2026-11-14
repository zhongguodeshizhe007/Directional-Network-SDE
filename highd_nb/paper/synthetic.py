"""HighD-shaped analytical moving-platoon fixture for SOFTWARE validation only.
Positive oscillating gaps, consistent position/velocity/acceleration, complete metadata.
NOT a behavioural traffic model, NOT empirical highD, NOT a known-NetworkSDE recovery test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from highd_nb.core.common import write_json
from highd_nb.testing import synthetic_config


def create_dense_fixture(root,n_recordings=8,duration=40):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Synthetic fixture directory must be new/empty')
    root.mkdir(parents=True,exist_ok=True)
    (root/'SYNTHETIC_NOT_HIGHD.txt').write_text('SYNTHETIC SOFTWARE TEST. Not natural traffic. Not paper evidence of accuracy.\n',encoding='utf-8')
    fps=25;road=400.;length=4.;t=np.arange(duration*fps)/fps
    for rec in range(1,n_recordings+1):
        parts=[];metas=[]
        for direction in [1,2]:
            sign=-1 if direction==1 else 1
            for lane in ([2,3] if direction==1 else [5,6]):
                base=direction*10000+lane*1000;yc={2:2.,3:6.,5:14.,6:18.}[lane]
                speed=12.+.25*(rec%3)+.4*(lane%2);omega=.73+.03*(lane%2)
                phase=.23*rec+.4*direction+.7*lane
                # Consecutive gaps alternate phase, preventing unbounded velocity sums.
                count=85;pos=np.empty((count,len(t)));vel=np.empty_like(pos);acc=np.empty_like(pos)
                pos[0]=road+100+speed*t;vel[0]=speed;acc[0]=0
                for j in range(1,count):
                    phi=phase+(j%2)*np.pi
                    # Long safe intervals make both event classes available at 1/3/5/10 s.
                    # This is intentional software branch coverage, NOT the natural prevalence.
                    envelope=.5*(1+np.cos(.21*t+.27*rec))
                    edot=-.105*np.sin(.21*t+.27*rec);eddot=-.02205*np.cos(.21*t+.27*rec)
                    ss=np.sin(omega*t+phi);cc=np.cos(omega*t+phi)
                    gap=11.+9.5*envelope*ss
                    dg=9.5*(edot*ss+envelope*omega*cc)
                    ddg=9.5*(eddot*ss+2*edot*omega*cc-envelope*omega**2*ss)
                    pos[j]=pos[j-1]-length-gap;vel[j]=vel[j-1]-dg
                    acc[j]=acc[j-1]-ddg
                for j in range(count):
                    ok=(pos[j]>=0)&(pos[j]<=road);ix=np.flatnonzero(ok)
                    if not len(ix):continue
                    ident=base+j+1;s=pos[j,ix];u=vel[j,ix]
                    hasleader=(j>0)&((pos[max(0,j-1),ix]>=0)&(pos[max(0,j-1),ix]<=road))
                    gap=np.where(hasleader,pos[max(0,j-1),ix]-s-length,0.)
                    pu=vel[max(0,j-1),ix];dv=u-pu;ttc=np.zeros(len(ix));closing=hasleader&(dv>1e-8);ttc[closing]=gap[closing]/dv[closing]
                    xc=road-s if direction==1 else s
                    d=pd.DataFrame(dict(frame=ix+1,id=ident,x=xc-2,y=yc-1,width=4.,height=2.,
                        xVelocity=sign*u,yVelocity=0.,xAcceleration=sign*acc[j,ix],yAcceleration=0.,
                        frontSightDistance=road-s,backSightDistance=s,dhw=gap,thw=np.where(hasleader,gap/u,0),ttc=ttc,
                        precedingXVelocity=np.where(hasleader,sign*pu,0),precedingId=np.where(hasleader,base+j,0),
                        followingId=0,laneId=lane))
                    parts.append(d);metas.append(dict(id=ident,width=4.,height=2.,initialFrame=int(ix.min()+1),
                        finalFrame=int(ix.max()+1),numFrames=len(ix),**{'class':'Truck' if j%7==0 else 'Car'},
                        drivingDirection=direction,meanXVelocity=float(u.mean()),minTTC=float(ttc[ttc>0].min()) if (ttc>0).any() else 0.,numLaneChanges=0))
        pd.concat(parts,ignore_index=True).sort_values(['id','frame']).to_csv(root/f'{rec:02d}_tracks.csv',index=False)
        pd.DataFrame(metas).to_csv(root/f'{rec:02d}_tracksMeta.csv',index=False)
        pd.DataFrame([dict(id=rec,frameRate=fps,locationId=(rec-1)//2+1,duration=duration,
            upperLaneMarkings='0;4;8',lowerLaneMarkings='12;16;20',numVehicles=len(metas),speedLimit=33.3333,
            month='SYNTHETIC',weekDay='SYNTHETIC',startTime='00:00')]).to_csv(root/f'{rec:02d}_recordingMeta.csv',index=False)
    write_json(root/'FIXTURE_PROVENANCE.json',{'synthetic':True,'not_highD':True,'recordings':n_recordings,
        'locations':(n_recordings+1)//2,'duration_s':duration,'analytical_derivatives':True,
        'purpose':'Exercise pipeline, masks, multiple events and plotting. NOT evaluate scientific predictive performance.'})
    return root


def paper_test_config():
    c=synthetic_config();c['scales_m']=[50];c['training'].update(epochs=3,sequence_stride=6,batch_size=8,
        validation_draws=12,validation_origin_stride=12,validation_max_origins_per_graph=2)
    c['symbolic'].update(max_rows_per_component=800,rows_per_graph=100,recording_bootstraps=4)
    c['paper'].update(history_steps=4,epochs=4,patience=3,hidden=12,batch_size=16,train_origin_stride=2,
        max_train_origins_per_graph=12,max_test_origins_per_graph=None,particles=24,bootstrap_recordings=30,
        event_min_train_positives=2,event_min_validation_positives=1,context_epochs=5,
        event_hazard_epochs=5,event_hazard_hidden=12,
        context_max_train_rows=2000,context_inducing=24,context_min_rows=30,xgb_estimators=30,component_origins_per_graph=5)
    return c
