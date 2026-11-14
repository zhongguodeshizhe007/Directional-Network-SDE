"""Shared I/O, provenance and path guards. Python 3.10+."""
from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
VERSION = 'scheme-a-notebook-5.0.0'
ROOT = Path(__file__).resolve().parents[2]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean_json(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def digest(obj):
    return hashlib.sha256(json.dumps(clean_json(obj),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def code_hash():
    paths=sorted((ROOT/'highd_nb').rglob('*.py'))
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in paths})

def guard_raw(raw,work):
    raw,work=Path(raw).resolve(),Path(work).resolve()
    if raw==work or raw in work.parents or work in raw.parents:
        raise ValueError('输出目录与原始数据目录不能相同或互为祖先；请用 outputs/nb_v5_paper，与 highd-dataset-v1.0/data 分开。')
    if not raw.is_dir(): raise FileNotFoundError(raw)
    work.mkdir(parents=True,exist_ok=True)

def seed_all(seed,threads=1):
    from highd_nb.runtime import configure_reproducibility
    import torch
    configure_reproducibility()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def environment():
    out={'python':sys.version,'platform':platform.platform(),'version':VERSION,'code_hash':code_hash()}
    for name in ['numpy','pandas','scipy','sklearn','torch']:
        try: out[name]=__import__(name).__version__
        except ImportError: out[name]='not installed'
    return out

def graph_paths(work,scale):
    return sorted((Path(work)/'scales'/f'L{int(scale):03d}').glob('rec_*_dir*/arrays.npz'))

def graph_schema(path): return read_json(Path(path).parent/'schema.json')

def load_arrays(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k] for k in a.files}

def max_missing_runs(mask):
    runs=[]
    for n in range(mask.shape[1]):
        c=0
        for ok in mask[:,n]:
            if not ok: c+=1
            elif c: runs.append(c); c=0
        if c: runs.append(c)
    return np.asarray(runs,dtype=int)

def selected_folds(plan,selection):
    folds=plan['folds']
    if str(selection).lower()=='all': return folds
    want=set(int(v) for v in str(selection).split(','))
    result=[f for f in folds if f['test_location'] in want]
    if len(result)!=len(want): raise ValueError(f'Unknown fold(s): {selection}')
    return result
