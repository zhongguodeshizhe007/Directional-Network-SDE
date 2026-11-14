"""Isolated synthetic fixtures and reproducible test execution. NEVER real highD.

No implicit fallback: the main workflow never calls these functions unless the
user explicitly opens 00_SelfTest.ipynb or the notebook validation harness does.
"""
from __future__ import annotations
import copy
import subprocess
import sys
from pathlib import Path
import pandas as pd
from .core.common import ROOT, read_json, write_json
from .vendor.make_synthetic_fixture import create_fixture


def synthetic_config() -> dict:
    c=read_json(ROOT/'configs/experiment.json')
    c['gates'].update(min_training_observations=10,min_positive_training_observations=1,
                     min_observed_fraction=.01,min_location_observed_fraction=.01,max_p95_missing_gap_s=100)
    c['training'].update(seeds=[7],epochs=2,patience=2,sequence_length=12,sequence_stride=8,
                     batch_size=2,burn_in_steps=2,validation_draws=8,
                     validation_origin_stride=8,validation_max_origins_per_graph=2)
    c['evaluation'].update(particles=16,origin_stride=8,max_origins_per_graph=2,
                      history_steps=12,bootstrap_recordings=10)
    c['symbolic'].update(max_rows_per_component=500,rows_per_graph=50,recording_bootstraps=3)
    c['model']['hidden']=12
    return c


def make_test_data(directory: Path, n_recordings=4, reproduce_roi_5_07=True) -> Path:
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'测试目录必须为空/不存在：{directory}。不覆盖原始数据。')
    create_fixture(directory,n_recordings)
    if reproduce_roi_5_07 and n_recordings>=2:
        path=directory/'02_tracks.csv'; d=pd.read_csv(path)
        meta=pd.read_csv(directory/'02_tracksMeta.csv').set_index('id')
        idx=d.index[d.id.map(meta.drivingDirection)==1]
        idx=idx[:max(1,len(idx)//10)]
        # Shift the derived left/right camera endpoints together; physical trajectory untouched.
        d.loc[idx,'frontSightDistance']-=5.07
        d.loc[idx,'backSightDistance']+=5.07
        d.to_csv(path,index=False)
    write_json(directory/'FIXTURE_PROVENANCE.json',{'synthetic':True,'not_highD':True,
        'purpose':'Software regression, includes previous p95=5.07 m boundary condition',
        'recordings':n_recordings})
    return directory


def run_unit_tests(output_dir: Path | None = None) -> dict:
    """Run in this kernel: avoids an extra torch process and subprocess pipe deadlocks.

    Random state/thread counts may change during tests. Training subsequently
    reseeds itself explicitly. Test modules do not call unittest.main on import.
    """
    import contextlib
    import io
    import unittest
    dest=Path(output_dir or ROOT/'outputs/software_tests'); dest.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    class ProgressResult(unittest.TextTestResult):
        def startTest(self, test):
            super().startTest(test)
            with (dest/'unit_progress.log').open('a',encoding='utf-8') as f:
                f.write(str(test)+'\n')
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        executed=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=ProgressResult).run(suite)
    text=stream.getvalue()
    log=dest/'unit_tests.log'; log.write_text(text,encoding='utf-8')
    print(text)
    result={'status':'PASS' if executed.wasSuccessful() else 'FAIL',
            'tests_run':executed.testsRun, 'failures':len(executed.failures),
            'errors':len(executed.errors), 'skipped':len(executed.skipped), 'log':str(log), 'python':sys.executable,
            'execution':'same Jupyter kernel, no shell subprocess', 'real_data_tested':False}
    write_json(dest/'unit_result.json',result)
    if not executed.wasSuccessful(): raise RuntimeError(f'自动测试失败，请查看 {log}')
    return result
