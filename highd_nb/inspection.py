"""Local real-file inspection: schema preview, complete-track audit, and ROI evidence.

Unlike displaying head(), full_check scans ALL rows of one recording in chunks.
It compares counts and frame ranges with tracksMeta to identify truncated samples.
This cannot certify the distributor/authenticity of a file whose metadata was also
rewritten; source authorization and completeness remain the researcher's duty.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .csvio import columns, read_csv
from .vendor import prepare_highd as base
from .core.roi_geometry import inspect_roi
from .core.common import write_json, read_json, file_hash, digest


def select_records(raw: Path, records='all') -> list[str]:
    if records is None or records == 'all':
        selection='all'
    elif isinstance(records, (str, int)):
        selection=str(records)
    else:
        selection=','.join(str(r).zfill(2) for r in records)
    return base.recordings(Path(raw), selection)


def preview(raw: Path, recording: str, nrows: int = 6) -> dict[str, pd.DataFrame]:
    rec=str(recording).zfill(2)
    if not 1 <= nrows <= 10000:
        raise ValueError('预览行数应在 1..10000；完整性检查请用 preflight。')
    return {kind: read_csv(path, nrows=nrows) for kind,path in base.input_paths(Path(raw),rec).items()}


def fingerprints(paths: dict[str,Path]) -> dict[str,str]:
    return {k:file_hash(v) for k,v in paths.items()}


def full_check(raw: Path, rec: str, config: dict) -> dict[str, Any]:
    """Complete-recording check + exact all-row common ROI. Reads no trajectories into the model."""
    paths=base.input_paths(Path(raw),rec)
    result={'recording_id':rec, 'status':'BLOCKED', 'errors':[], 'warnings':[],
            'inspection':'all CSV rows; metadata count/frame comparison, not head() sampling'}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'同一录像的三个 CSV 必须存在：{path}')
    st=read_csv(paths['static']); meta_frame=read_csv(paths['recording'])
    hdr=columns(paths['tracks'])
    base.require_columns(pd.DataFrame(columns=hdr),base.TRACK_REQUIRED,paths['tracks'].name)
    base.require_columns(st,base.STATIC_REQUIRED,paths['static'].name)
    base.require_columns(meta_frame,base.META_REQUIRED,paths['recording'].name)
    if len(meta_frame)!=1:
        raise ValueError('recordingMeta 必须有且只有一行')
    meta=meta_frame.iloc[0]; geom=base.lane_geometry(meta)
    if int(meta['id'])!=int(rec):
        raise ValueError(f'文件编号 {rec} 与 recordingMeta.id 不一致')
    if st.id.duplicated().any():
        raise ValueError('tracksMeta 中存在重复车辆 ID')
    if not st.drivingDirection.isin([1,2]).all():
        raise ValueError('drivingDirection 必须为 1/2')
    fps=float(meta['frameRate'])
    if not np.isfinite(fps) or fps<=0 or not np.isfinite(float(meta['duration'])) or float(meta['duration'])<=0:
        raise ValueError('frameRate 和 duration 必须为正的有限值')
    static_extra=['initialFrame','finalFrame','numFrames']
    full_metadata=set(static_extra).issubset(st.columns)
    if not full_metadata:
        result['errors'].append('缺少 tracksMeta 完整性字段 initialFrame/finalFrame/numFrames；可预览，不可直接认定为完整录像。')
    directions=st.set_index('id').drivingDirection
    counts:dict[int,int]={}; first:dict[int,int]={}; last:dict[int,int]={}
    nrows=0; duplicate_within_chunks=0
    geom_parts={1:[],2:[]}; unknown_rows=0
    wanted=['id','frame','x','width','frontSightDistance','backSightDistance']
    for chunk in read_csv(paths['tracks'],usecols=wanted,
                          chunksize=int(config['preflight']['chunksize'])):
        for k in wanted:
            chunk[k]=pd.to_numeric(chunk[k],errors='raise')
        for k in ['id','frame']:
            vals=chunk[k].to_numpy(float)
            if not np.isfinite(vals).all() or (vals!=np.floor(vals)).any():
                raise ValueError(f'{rec} 的 {k} 含缺失/非整数')
        nrows+=len(chunk)
        duplicate_within_chunks+=int(chunk.duplicated(['id','frame']).sum())
        stats=chunk.groupby('id',sort=False).frame.agg(['size','min','max'])
        for ident,row in stats.iterrows():
            ident=int(ident)
            counts[ident]=counts.get(ident,0)+int(row['size'])
            first[ident]=min(first.get(ident,int(row['min'])),int(row['min']))
            last[ident]=max(last.get(ident,int(row['max'])),int(row['max']))
        d=chunk.id.map(directions)
        unknown_rows+=int(d.isna().sum())
        for direction in [1,2]:
            g=chunk.loc[d==direction,['x','width','frontSightDistance','backSightDistance']]
            if len(g): geom_parts[direction].append(g.to_numpy(float))
    if not nrows:
        raise ValueError('轨迹 CSV 只有表头，没有数据')
    result.update(n_rows=nrows,n_observed_tracks=len(counts),n_metadata_tracks=len(st),
        unknown_track_rows=unknown_rows, duplicate_rows_within_read_chunks=duplicate_within_chunks,
        frame_min=min(first.values()),frame_max=max(last.values()),frame_rate=fps,
        location_id=int(meta.locationId), metadata_duration_s=float(meta.duration),
        present_directions=sorted(set(int(x) for x in st.drivingDirection)),
        lane_counts={str(k):len(v) for k,v in geom.items()},track_columns=hdr,
        metadata_columns=st.columns.tolist(),has_complete_track_metadata=full_metadata)
    if unknown_rows: result['errors'].append(f'{unknown_rows} 行轨迹无法找到 tracksMeta')
    if duplicate_within_chunks: result['errors'].append('发现重复 id/frame；不自动删行')
    missing=set(map(int,st.id))-set(counts)
    result['metadata_tracks_missing_in_csv']=len(missing)
    if missing: result['errors'].append(f'{len(missing)} 个 tracksMeta 车辆在 CSV 中完全缺失，疑似片段/筛选数据')
    if full_metadata:
        bad=[]
        for row in st.itertuples(index=False):
            ident=int(row.id)
            if (counts.get(ident,0)!=int(row.numFrames) or first.get(ident)!=int(row.initialFrame)
                or last.get(ident)!=int(row.finalFrame)):
                bad.append(ident)
        result['tracks_with_metadata_count_or_span_mismatch']=len(bad)
        result['example_mismatch_track_ids']=bad[:10]
        result['expected_rows_from_tracksMeta']=int(st.numFrames.sum())
        if bad: result['errors'].append(f'{len(bad)} 个车辆的行数/起止帧不符合 tracksMeta；片段只能做结构调试，不能当成完整实证数据')
        gaps=[i for i in counts if counts[i] != last[i]-first[i]+1]
        result['tracks_with_noncontiguous_count']=len(gaps)
        if gaps: result['errors'].append('存在车辆帧数不等于连续跨度；完整配对检查仍将在预处理时执行')
    roi={}
    for direction in [1,2]:
        if not geom_parts[direction]:
            result['errors'].append(f'方向 {direction} 无轨迹；本研究预设两个方向，不补造数据')
            continue
        arr=np.concatenate(geom_parts[direction],axis=0)
        data=pd.DataFrame(arr,columns=['x','width','frontSightDistance','backSightDistance'])
        r=inspect_roi(data,rec,direction,float(config['common_roi_m']),
                      float(config['base_data']['boundary_margin_m']))
        roi[str(direction)]=r
        if r['status']!='PASS_COMMON_ROI':
            result['errors'].append(f'dir{direction}: '+ '; '.join(r['reasons']))
        result['warnings'].extend(f'dir{direction}: {w}' for w in r.get('warnings',[]))
        del data,arr
    result['roi']=roi
    result['status']='PASS_COMPLETE_AND_ROI' if not result['errors'] else 'BLOCKED'
    result['input_sha256']=fingerprints(paths)
    result['schema_notes']=[
        'Same-frame duplicate keys spanning read chunks are checked again by full preprocessing.',
        'Metadata-count agreement is not a guarantee of source authenticity; edited metadata can conceal filtering.',
        'No intrinsic-noise/causal/SDE-validity claim follows from this structural check.'
    ]
    return result


def preflight(raw: Path, work: Path, config: dict, records='all', force=False) -> pd.DataFrame:
    """Save every recording's failure before raising anything; safe to rerun after kernel restart."""
    dest=Path(work)/'preflight'; dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    contract=digest({'preflight':config['preflight'],'width':config['common_roi_m'],
                     'margin':config['base_data']['boundary_margin_m'],
                     'reader':file_hash(Path(__file__).with_name('csvio.py')),
                     'inspector':file_hash(Path(__file__))})
    recs=select_records(raw,records)
    for index,rec in enumerate(recs,1):
        print(f'[{index}/{len(recs)}] 全行结构与共同区域预检：{rec}',flush=True)
        path=dest/f'{rec}.json'; start=time.perf_counter()
        try:
            inputs=base.input_paths(Path(raw),rec)
            hashes=fingerprints(inputs)
            if path.exists() and not force:
                old=read_json(path)
                if old.get('contract')==contract and old.get('input_sha256')==hashes:
                    record=old
                    print('  使用已核验的预检缓存',flush=True)
                else:
                    record=full_check(raw,rec,config)
            else:
                record=full_check(raw,rec,config)
            record['contract']=contract
        except Exception as exc:
            record={'recording_id':rec,'status':'BLOCKED','errors':[f'{type(exc).__name__}: {exc}'],
                    'contract':contract,'exception':type(exc).__name__}
        record['elapsed_seconds_last_call']=round(time.perf_counter()-start,3)
        write_json(path,record)
        for direction,roi in record.get('roi',{}).items():
            write_json(dest/f'{rec}_dir{direction}_roi.json',roi)
        rows.append({'recording_id':rec,'status':record['status'],
                     'location_id':record.get('location_id'),'rows':record.get('n_rows'),
                     'tracks':record.get('n_observed_tracks'),
                     'issues':' | '.join(record.get('errors',[])),
                     'report':str(path)})
        print(f'  {record["status"]}'+('；'+rows[-1]['issues'] if rows[-1]['issues'] else ''),flush=True)
    # Include all current reports, not just the most recent subset call.
    known=[]
    for path in sorted(dest.glob('[0-9][0-9].json')):
        r=read_json(path)
        known.append({'recording_id':r['recording_id'],'status':r['status'],
                      'location_id':r.get('location_id'),'rows':r.get('n_rows'),
                      'tracks':r.get('n_observed_tracks'),'issues':' | '.join(r.get('errors',[])),
                      'report':str(path)})
    pd.DataFrame(known).to_csv(dest/'summary.csv',index=False,encoding='utf-8-sig')
    return pd.DataFrame(rows)
