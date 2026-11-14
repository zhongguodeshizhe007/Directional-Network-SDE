"""Read each recording once, build matched-domain spatial resolutions, safe resume."""
from __future__ import annotations
import copy, shutil, tempfile
from pathlib import Path
import pandas as pd
from highd_nb.vendor import prepare_highd as base
from .common import *
from .roi_geometry import inspect_roi, ROI_POLICY

def inventory(raw,work,selection='all',allow_synthetic=False):
    guard_raw(raw,work)
    if (Path(raw)/'SYNTHETIC_NOT_HIGHD.txt').exists() and not allow_synthetic:
        raise ValueError('Synthetic input detected. Use --allow-synthetic ONLY for a software test.')
    base.audit(Path(raw),Path(work)/'inventory',selection)

def preprocess(raw,work,config,scales,selection='all',allow_synthetic=False):
    raw,work=Path(raw),Path(work); guard_raw(raw,work)
    synthetic=(raw/'SYNTHETIC_NOT_HIGHD.txt').exists()
    if synthetic and not allow_synthetic: raise ValueError('Synthetic fixture requires --allow-synthetic.')
    scales=[int(s) for s in scales]
    width=float(config['common_roi_m'])
    for s in scales:
        if s not in config['scales_m'] or width/s<2 or abs(round(width/s)-width/s)>1e-8:
            raise ValueError('Every scale must evenly divide common_roi_m with at least 2 cells.')
    basecfg=copy.deepcopy(config['base_data'])
    if basecfg.get('roi_overrides'):
        raise ValueError('base_data.roi_overrides is not supported by this matched-domain wrapper. '
                         'Run tools/diagnose_roi.py; do not invent camera endpoints.')
    basecfg['hash_inputs']=True
    signature={'version':VERSION,'common_roi_m':width,'base_data':basecfg,
               'raw_root':str(raw.resolve()),'synthetic':synthetic,
               'vendor_sha256':file_hash(ROOT/'highd_nb'/'vendor'/'prepare_highd.py'),
               'raw_csv_reader_sha256':file_hash(ROOT/'highd_nb'/'csvio.py'),
               'roi_policy':ROI_POLICY,
               'roi_geometry_sha256':file_hash(ROOT/'highd_nb'/'core'/'roi_geometry.py'),
               'preprocessing_sha256':file_hash(ROOT/'highd_nb'/'core'/'preprocessing.py'),
               'ssm_sha256':file_hash(ROOT/'highd_nb'/'paper'/'ssm.py'),
               'augmentation_sha256':file_hash(ROOT/'highd_nb'/'paper'/'augmentation.py')}
    receipt=work/'preprocessing_contract.json'
    if receipt.exists() and read_json(receipt)!=signature:
        raise ValueError('Work directory has a different data contract. Use a NEW work directory.')
    write_json(receipt,signature)
    if synthetic: (work/'SYNTHETIC_NOT_EMPIRICAL.txt').write_text('SOFTWARE TEST ONLY',encoding='utf-8')
    for rec in base.recordings(raw,selection):
        inputs=base.input_paths(raw,rec)
        fingerprints={k:file_hash(p) for k,p in inputs.items()}
        root_receipt=work/'receipts'/f'recording_{rec}.json'
        if root_receipt.exists() and read_json(root_receipt)['input_sha256']!=fingerprints:
            raise ValueError(f'Raw input changed for {rec}. Use a new work directory.')
        all_done=all((work/'scales'/f'L{s:03d}'/f'rec_{rec}_dir{d}'/'complete.json').exists()
                     for s in scales for d in (1,2))
        if all_done:
            for s in scales:
                for direction in (1,2):
                    gd=work/'scales'/f'L{s:03d}'/f'rec_{rec}_dir{direction}'
                    done=read_json(gd/'complete.json')
                    for name, checksum in done.get('artifacts_sha256',{}).items():
                        if not (gd/name).is_file() or file_hash(gd/name)!=checksum:
                            raise ValueError(f'Corrupted cached artifact: {gd/name}. Use intact outputs.')
                    if file_hash(gd/'arrays.npz')!=done['arrays_sha256']:
                        raise ValueError(f'Corrupted cached arrays: {gd}. Remove this graph folder and rerun.')
            print(f'{rec}: verified cached arrays for scales {scales}; skipped.',flush=True); continue
        print(f'{rec}: reading full trajectory CSV once for scales {scales}...',flush=True)
        d,meta,qc=base.read_recording(raw,rec,basecfg)
        # Derive the same center/bounds independently of scale, with physical margin preserved.
        overrides={}; roi_evidence={}
        for direction in sorted(d.direction.unique()):
            dc=d.loc[d.direction==direction]
            probe=inspect_roi(dc,rec,int(direction),width,float(basecfg['boundary_margin_m']))
            diagnostic_path=work/'roi_diagnostics'/f'{rec}_dir{int(direction)}.json'
            write_json(diagnostic_path,probe)
            if probe['status'] != 'PASS_COMMON_ROI':
                raise ValueError(f"{rec}/dir{direction}: ROI BLOCKED. " + '; '.join(probe['reasons'])
                                 + f" See {diagnostic_path}")
            overrides[f'{rec}_dir{direction}']=probe['builder_envelope_override_m']
            roi_evidence[str(direction)]=probe
            print(f"  ROI {rec}/dir{direction}: p95={probe['old_boundary_p95_deviation_m']:.2f}m, "
                  f"common=[{probe['common_analysis_low_m']:.3f},"
                  f"{probe['common_analysis_high_m']:.3f}], all-row support=100%, "
                  f"minimum margins=({probe['min_left_margin_m']:.3f},"
                  f"{probe['min_right_margin_m']:.3f})m",flush=True)
        write_json(root_receipt,{'recording':rec,'input_sha256':fingerprints,'roi_evidence':roi_evidence})
        for s in scales:
            sd=work/'scales'/f'L{s:03d}'; sd.mkdir(parents=True,exist_ok=True)
            c=copy.deepcopy(basecfg); c['cell_length_m']=float(s); c['roi_overrides']=overrides
            write_json(sd/f'{rec}_recording_qc.json',qc)
            for direction in sorted(d.direction.unique()):
                gd=sd/f'rec_{rec}_dir{int(direction)}'
                if (gd/'complete.json').exists():
                    done=read_json(gd/'complete.json')
                    if file_hash(gd/'arrays.npz') != done['arrays_sha256']:
                        raise ValueError(f'Corrupted cached arrays: {gd}. Do not mix altered cached data.')
                    continue
                tmp=Path(tempfile.mkdtemp(prefix='.building_',dir=sd))
                try:
                    q=base.build_graph(d,meta,rec,int(direction),c,tmp,qc)
                    # Preserve unambiguous original camera bounds alongside matched ROI.
                    write_json(tmp/'matched_roi_provenance.json',roi_evidence[str(direction)])
                    write_json(tmp/'complete.json',{'input_sha256':fingerprints,
                        'arrays_sha256':file_hash(tmp/'arrays.npz'),'scale_m':s,'synthetic':synthetic,
                        'artifacts_sha256':{p.name:file_hash(p) for p in tmp.iterdir() if p.is_file()}})
                    if gd.exists(): raise FileExistsError(f'Incomplete graph directory {gd}; inspect/remove explicitly.')
                    tmp.rename(gd)
                    print(f'  L{s} {gd.name}: N={q["N"]}, valid={q["valid_risk_fraction"]:.3f}, '
                          f'strict={q["strict_transition_count"]}',flush=True)
                finally:
                    if tmp.exists(): shutil.rmtree(tmp)
            base.refresh_manifest(sd)
        del d
    for s in scales:
        sd=work/'scales'/f'L{s:03d}'
        base.refresh_manifest(sd); base.verify(sd)
    write_json(work/'environment_preprocessing.json',environment())
