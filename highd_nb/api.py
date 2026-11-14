"""Notebook-first Python API. No shell variables, subprocess CLI or hidden global state.

Run one stage per Jupyter cell. Data paths are explicit pathlib.Paths. A fresh
kernel can construct a Workflow again and reuse verified on-disk stage receipts.
"""
from __future__ import annotations
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Iterable
import numpy as np
import pandas as pd
from .core.common import (ROOT, VERSION, read_json, write_json, file_hash, digest,
                          guard_raw, environment, graph_paths, graph_schema, load_arrays)
from . import inspection


class StageBlocked(RuntimeError):
    """A research/data prerequisite failed; inspect diagnostics before continuing."""


class _Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return False


def _selection(records):
    if records is None or records=='all': return 'all'
    if isinstance(records,(str,int)): return str(records)
    return ','.join(str(r).zfill(2) for r in records)


def validate_config(c: dict) -> None:
    """Fail in the configuration cell, not halfway through 60 recordings."""
    from .vendor.prepare_highd import integral_ratio
    for section in ['base_data','gates','model','training','evaluation','symbolic','preflight']:
        if section not in c: raise ValueError(f'缺少配置节：{section}')
    width=float(c['common_roi_m']); scales=c['scales_m']; data=c['base_data']
    if not np.isfinite(width) or width<=0: raise ValueError('common_roi_m 必须为正有限值')
    if len(set(scales))!=len(scales) or not scales: raise ValueError('尺度不能为空或重复')
    for s in scales:
        if s<=0 or integral_ratio(width,float(s),'common_roi_m/cell')<2:
            raise ValueError('所有尺度必须整除共同 ROI 且每车道至少有2格')
    step=float(data['step_s']); history=float(data['context_window_s'])
    if not np.isfinite([step,history]).all() or step<=0 or history<step:
        raise ValueError('step/history 必须为正且 history>=step')
    integral_ratio(history,step,'history/step')
    for h in data['forecast_horizons_s']+data['diagnostic_lags_s']+c['evaluation']['horizons_s']:
        integral_ratio(float(h),step,'horizon/step')
    if data['boundary_margin_m']<0 or not 0<=data['min_pair_coverage']<=1:
        raise ValueError('边界缓冲/配对覆盖率非法')
    if data.get('roi_overrides'):
        raise ValueError('不接受未经验证的人工 ROI 覆盖；请查看 preflight 几何证据。')
    if c['preflight'].get('allow_partial_recordings',False):
        raise ValueError('本版正式训练禁止把截断片段当成完整录像；片段用 preview 检查即可。')
    t=c['training']
    for key in ['sequence_length','sequence_stride','batch_size','epochs','patience']:
        if int(t[key])<1: raise ValueError(f'training.{key} 必须>=1')
    if t['burn_in_steps']<0 or t['burn_in_steps']>=t['sequence_length']-1:
        raise ValueError('burn_in_steps 必须小于 sequence_length-1')
    if c['model']['sigma_min']<=0 or c['model']['sigma_max']<=c['model']['sigma_min']:
        raise ValueError('sigma 上下界非法')
    if c['evaluation']['forecast_covariates']!='freeze_at_anchor':
        raise ValueError('本版预测只实现 freeze_at_anchor，不可偷用未来交通特征。')


    if 'paper' in c:
        p=c['paper'];hs=c['evaluation']['horizons_s']
        if sorted(set(hs))!=hs or any(float(h)<=0 for h in hs):
            raise ValueError('论文预测跨度必须为严格递增的正值。')
        if not set(hs).issubset(data['forecast_horizons_s']):
            raise ValueError('论文跨度必须已经在预处理标签中定义。')
        if p['primary_horizon_s'] not in hs:
            raise ValueError('primary_horizon_s 不在评价跨度中。')
        if p['primary_ttc_s'] not in p['event_thresholds_s']:
            raise ValueError('primary_ttc_s 不在事件阈值中。')
        for key in ['event_hazard_epochs','event_hazard_batch_size','event_hazard_hidden','context_batch_size']:
            if int(p.get(key,1))<1:raise ValueError(f'paper.{key} 必须>=1')
        if p.get('event_hazard_enabled',False) and p['event_support']!='complete_future':
            raise ValueError('本版 hazard 仅支持 complete_future；不能将未知未来标为无冲突。')
        if not p.get('event_hazard_exclude_active',True):
            raise ValueError('本版预警要求预测起点没有正在进行的冲突。')


class Workflow:
    """Main notebook interface.

    Parameters
    ----------
    data_dir : actual highd-dataset-v1.0/data folder, never modified.
    work_dir : NEW outputs folder, outside raw data and outside all old v1/v2 outputs.
    config_path : optional JSON protocol; defaults to packaged experiment.json.
    allow_synthetic : True only for isolated software fixtures; all outputs marked.
    """
    def __init__(self, data_dir: str | Path, work_dir: str | Path,
                 config_path: str | Path | None = None, *, config: dict | None = None,
                 allow_synthetic: bool = False):
        for value in [data_dir, work_dir]:
            text=str(value)
            if '%RAW%' in text or '%WORK%' in text or '$RAW' in text or '$WORK' in text:
                raise ValueError('Notebook 中使用 Path 对象，不能传 %RAW% 或 $RAW 字符串。')
        self.data_dir=Path(data_dir).expanduser().resolve()
        self.work_dir=Path(work_dir).expanduser().resolve()
        guard_raw(self.data_dir,self.work_dir)
        self.config=copy.deepcopy(config if config is not None else read_json(config_path or ROOT/'configs/experiment.json'))
        validate_config(self.config)
        self.allow_synthetic=bool(allow_synthetic)
        synthetic=(self.data_dir/'SYNTHETIC_NOT_HIGHD.txt').exists()
        if synthetic and not self.allow_synthetic:
            raise StageBlocked('检测到合成数据。真实 Notebook 不会自动改用示例数据。')
        self.synthetic=synthetic
        self._lock_path=self.work_dir/'.workflow_running.lock'
        current={'version':VERSION,'raw_root':str(self.data_dir),
                 'config':self.config,'synthetic':synthetic}
        protocol=self.work_dir/'notebook_protocol.json'
        if protocol.exists() and read_json(protocol)!=current:
            raise StageBlocked('输出目录已冻结另一配置/数据路径。请使用新的 work_dir，不能混用旧结果。')
        write_json(protocol,current)
        if synthetic:
            (self.work_dir/'SYNTHETIC_NOT_EMPIRICAL.txt').write_text('ARTIFICIAL SOFTWARE TEST ONLY',encoding='utf-8')

    @contextlib.contextmanager
    def _stage(self,name):
        self.work_dir.mkdir(parents=True,exist_ok=True)
        try:
            fd=os.open(str(self._lock_path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            raise StageBlocked(f'发现运行锁 {self._lock_path}。另一 Notebook 可能仍在运行；确认无任务后才手动删除失效锁。') from None
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump({'pid':os.getpid(),'stage':name,'utc':datetime.now(timezone.utc).isoformat()},f)
        logdir=self.work_dir/'logs'; logdir.mkdir(exist_ok=True)
        log=logdir/f'{name}.log'
        try:
            with log.open('a',encoding='utf-8') as fp, \
                 contextlib.redirect_stdout(_Tee(sys.stdout,fp)), \
                 contextlib.redirect_stderr(_Tee(sys.stderr,fp)):
                print(f'\n=== {name}: {datetime.now(timezone.utc).isoformat()} ===',flush=True)
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    print(f'阶段中断；日志：{log}',flush=True)
                    raise
                else:
                    print(f'阶段完成：{name}',flush=True)
        finally:
            self._lock_path.unlink(missing_ok=True)

    @property
    def recordings(self) -> list[str]:
        return inspection.select_records(self.data_dir)

    def environment(self) -> dict:
        out=environment()
        for name in ['nbformat','nbclient','ipykernel','jupyterlab','matplotlib']:
            try: out[name]=__import__(name).__version__
            except ImportError: out[name]='not installed'
        out['python_executable']=sys.executable
        out['data_dir']=str(self.data_dir); out['work_dir']=str(self.work_dir)
        out['real_csv_available_in_this_local_run']=not self.synthetic
        try:
            import torch
            bridge=torch.from_numpy(np.zeros(2,dtype=np.float32)).numpy()
            out['numpy_torch_bridge']='PASS' if bridge.shape==(2,) else 'FAIL'
            out['cuda_available']=bool(torch.cuda.is_available())
        except ImportError:
            out['numpy_torch_bridge']='torch 未安装；仅数据处理仍可运行'
        except Exception as exc:
            out['numpy_torch_bridge']=f'FAIL: {exc}'
        write_json(self.work_dir/'environment.json',out)
        return out

    def preview(self, recording: str | None = None, nrows: int = 6):
        rec=recording or self.recordings[0]
        return inspection.preview(self.data_dir,str(rec),nrows)

    def audit(self, expected_recordings: int | None = None) -> pd.DataFrame:
        from .core.preprocessing import inventory
        with self._stage('01_audit'):
            inventory(self.data_dir,self.work_dir,'all',self.allow_synthetic)
            frame=pd.read_csv(self.work_dir/'inventory/inventory.csv',dtype={'recording_id':str})
            if expected_recordings is not None and len(frame)!=expected_recordings:
                raise StageBlocked(f'发现 {len(frame)} 段，预期 {expected_recordings} 段。检查是否指向片段目录或漏解压文件。')
            return frame

    def preflight(self, recordings='all', *, force: bool = False) -> pd.DataFrame:
        with self._stage('02_preflight'):
            return inspection.preflight(self.data_dir,self.work_dir,self.config,recordings,force)

    def require_ready(self, recordings='all') -> None:
        failures=[]
        for rec in inspection.select_records(self.data_dir,recordings):
            p=self.work_dir/'preflight'/f'{rec}.json'
            if not p.exists(): failures.append(f'{rec}: 尚未运行全行预检'); continue
            r=read_json(p)
            if r.get('status')!='PASS_COMPLETE_AND_ROI':
                failures.append(f'{rec}: '+'; '.join(r.get('errors',[])))
            elif inspection.fingerprints(inspection.base.input_paths(self.data_dir,rec))!=r.get('input_sha256'):
                failures.append(f'{rec}: 预检后 CSV 已改变，请重新预检')
        if failures:
            raise StageBlocked('不能开始正式处理：\n'+'\n'.join(failures[:20])+'\n查看 outputs/preflight/summary.csv。')

    def preprocess(self, recordings='all', scales: Iterable[int] | None = None) -> pd.DataFrame:
        from .core.preprocessing import preprocess
        scales=list(scales or self.config['scales_m'])
        with self._stage('03_preprocess'):
            if not (self.work_dir/'inventory/inventory.csv').exists():
                raise StageBlocked('请先运行 audit()，完整录像清单不能由单段试跑覆盖。')
            self.require_ready(recordings)
            preprocess(self.data_dir,self.work_dir,self.config,scales,_selection(recordings),self.allow_synthetic)
            return self.completion(scales)

    def completion(self, scales: Iterable[int] | None = None) -> pd.DataFrame:
        result=[]
        for s in list(scales or self.config['scales_m']):
            for rec in self.recordings:
                for direction in [1,2]:
                    folder=self.work_dir/'scales'/f'L{int(s):03d}'/f'rec_{rec}_dir{direction}'
                    ok=(folder/'complete.json').is_file() and (folder/'arrays.npz').is_file()
                    result.append({'scale_m':s,'recording_id':rec,'direction':direction,'completed':ok,'folder':str(folder)})
        return pd.DataFrame(result)

    def report(self, scales: Iterable[int] | None = None) -> pd.DataFrame:
        from .core.quality import report
        selected=list(scales or self.config['scales_m'])
        with self._stage('04_coverage_report'):
            status=self.completion(selected); missing=status.loc[~status.completed]
            if len(missing):
                example=missing[['scale_m','recording_id','direction']].head(15).to_string(index=False)
                raise StageBlocked(f'还有 {len(missing)} 张图未完成，不生成不完整的总体报告：\n{example}\n请重新运行 preprocess()，已完成图会校验后跳过。')
            report(self.work_dir,self.config,selected)
            return self.table('reports/scale_summary.csv')

    def table(self, relative_path: str) -> pd.DataFrame:
        path=self.work_dir/relative_path
        if not path.is_file(): raise FileNotFoundError(f'尚未生成：{path}；请先执行对应阶段。')
        try: return pd.read_csv(path)
        except pd.errors.EmptyDataError: return pd.DataFrame()

    def plan(self) -> dict:
        from .core.quality import decide
        with self._stage('05_training_plan'):
            result=decide(self.work_dir,self.config)
            return result

    def _check_torch(self):
        try:
            import torch
            torch.from_numpy(np.zeros(2,np.float32)).numpy()
        except Exception as exc:
            raise StageBlocked('PyTorch/NumPy 桥接不可用。不要直接升级整个环境；先运行 environment() 查看版本。'+str(exc)) from exc

    @staticmethod
    def _safe_tag(tag):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):
            raise ValueError('run_tag 只能包含英文字母、数字、下划线、连字符')

    def train(self, *, folds='all', scales='selected', models='routed', seeds=None,
              epochs=None, run_tag='main', device='auto') -> pd.DataFrame:
        self._safe_tag(run_tag); self._check_torch()
        from .core.training import train, model_runs
        with self._stage(f'06_train_{run_tag}'):
            if not (self.work_dir/'plan.json').exists(): raise StageBlocked('先生成 plan.json')
            train(self.work_dir,str(folds),str(scales),str(models),seeds,run_tag,device,epochs)
            rows=[]
            for p in model_runs(self.work_dir,run_tag):
                r=read_json(p); rows.append({'run':str(p.parent),'scale_m':r['scale_m'],
                    'test_location':r['fold']['test_location'],'model':r['name'],'seed':r['seed']})
            return pd.DataFrame(rows)

    def symbolic(self, *, run_tag='main', folds='all', scales='all', models='all', device='auto'):
        self._safe_tag(run_tag); self._check_torch()
        from .core.commands import selected_runs
        from .core.training import choose_device
        from .core.symbolic import distill
        with self._stage(f'07_symbolic_{run_tag}'):
            rows=[]
            for folder in selected_runs(self.work_dir,run_tag,str(folds),str(scales),str(models)):
                distill(self.work_dir,folder,choose_device(device))
                status=read_json(folder/'symbolic/complete.json')
                rows.append({'run':str(folder),'accepted':status['accepted_validation_fidelity']})
            return pd.DataFrame(rows)

    def evaluate(self, *, run_tag='main', folds='all', scales='all', models='all',
                 symbolic=False, device='auto', confirm_protocol_frozen=False):
        if not confirm_protocol_frozen:
            raise StageBlocked('测试集解锁被阻止。完成训练/验证选择后，明确设置 confirm_protocol_frozen=True。')
        self._safe_tag(run_tag); self._check_torch()
        from .core.commands import selected_runs
        from .core.training import choose_device
        from .core.evaluation import evaluate_run
        with self._stage(f'08_test_{run_tag}_'+('symbolic' if symbolic else 'neural')):
            for folder in selected_runs(self.work_dir,run_tag,str(folds),str(scales),str(models)):
                evaluate_run(self.work_dir,folder,choose_device(device),symbolic)

    def summarize(self, run_tag='main') -> pd.DataFrame:
        self._safe_tag(run_tag)
        from .core.commands import summarize
        with self._stage(f'09_summary_{run_tag}'):
            summarize(self.work_dir,run_tag)
            return self.table(f'summary/{run_tag}/all_test_metrics.csv')

    def collect(self) -> Path:
        import zipfile
        from .core.commands import collect
        with self._stage('10_debug_bundle'):
            collect(self.work_dir)
            path=self.work_dir/'review_bundle_v3.zip'
            # Contains diagnostics/logs, never raw rows, weights or per-origin predictions.
            with zipfile.ZipFile(path,'a',compression=zipfile.ZIP_DEFLATED) as z:
                included=set(z.namelist())
                for pattern in ['preflight/*.json','preflight/summary.csv','logs/*.log',
                                'environment.json','notebook_protocol.json','roi_diagnostics/*.json']:
                    for p in sorted(self.work_dir.glob(pattern)):
                        name=str(p.relative_to(self.work_dir)).replace('\\','/')
                        if name not in included: z.write(p,name); included.add(name)
            print(path)
            return path
