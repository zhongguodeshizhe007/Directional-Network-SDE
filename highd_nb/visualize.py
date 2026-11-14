"""Small Matplotlib diagnostics; figures reflect local outputs, not fitted conclusions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .core.common import read_json


def geometry_plot(report_path: str | Path):
    import matplotlib.pyplot as plt
    r=read_json(report_path)
    fig,ax=plt.subplots(figsize=(10,3.6))
    names=[]
    ranges=[]
    for label,left,right in [
        ('Median inferred view','median_recorded_x_low_m','median_recorded_x_high_m'),
        ('All-row common view','all_row_intersection_low_m','all_row_intersection_high_m'),
        ('After boundary margins','safe_analysis_low_m','safe_analysis_high_m'),
        ('Selected analysis region','common_analysis_low_m','common_analysis_high_m')]:
        if left in r and right in r:
            names.append(label); ranges.append((r[left],r[right]))
    for j,(lo,hi) in enumerate(ranges): ax.plot([lo,hi],[j,j],marker='o',linewidth=3)
    ax.set_yticks(range(len(names)),names); ax.set_xlabel('Image x coordinate (m)')
    ax.set_title(f"Recording {r['recording_id']} direction {r['direction']} | {r['status']}")
    ax.grid(axis='x',alpha=.25); fig.tight_layout()
    return fig


def coverage_plot(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    cols=['observed_fraction','consecutive_fraction','neighbor_complete_given_target']
    fig,ax=plt.subplots(figsize=(8,4))
    for name in cols: ax.plot(table.scale_m,table[name],marker='o',label=name)
    ax.set_xlabel('Cell length (m)'); ax.set_ylabel('Fraction'); ax.set_ylim(0,1)
    ax.set_xticks(table.scale_m); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    return fig


def risk_plot(array_path: str | Path):
    import matplotlib.pyplot as plt
    with np.load(array_path,allow_pickle=False) as a:
        y=np.asarray(a['y'][...,0]); time=np.asarray(a['time_s'])
    fig,ax=plt.subplots(figsize=(10,4))
    img=ax.imshow(np.ma.masked_invalid(y.T),aspect='auto',origin='lower',
                  extent=[time[0],time[-1],-.5,y.shape[1]-.5])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Road-cell node'); ax.set_title('Observed continuous risk; missing entries masked')
    fig.colorbar(img,ax=ax,label='log1p(mean inverse TTC × 1 s)'); fig.tight_layout()
    return fig
