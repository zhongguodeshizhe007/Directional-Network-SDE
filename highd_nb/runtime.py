"""One accelerator policy for notebooks, CLI, core SDE, SVGP and baselines.

Windows production backend: NVIDIA CUDA when PyTorch can execute on it; otherwise
CPU, explicitly reported. No unsupported GPU backend is silently claimed.
Likelihoods, sparse-GP Cholesky and particle scoring remain float32.
"""
from __future__ import annotations
import json
import os
import re
import warnings
from typing import Any

# Must precede CUDA context creation for deterministic cuBLAS operations.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def configure_reproducibility() -> None:
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # float32 reference math; do not change rare-tail calculations to float16.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def choose_device(request: Any = 'auto'):
    """Resolve auto/cpu/cuda/cuda:N; explicit invalid CUDA requests never fall back.

    Auto checks that a real kernel executes, not merely that nvidia-smi exists.
    GPU OOM during training is NOT caught here or silently retried on another device.
    """
    import torch
    text = str(request).strip().lower()
    if text == 'cpu':
        return torch.device('cpu')
    if text != 'auto' and not re.fullmatch(r'cuda(?::\d+)?', text):
        raise ValueError(f'Unsupported device {request!r}. Use auto, cpu, cuda or cuda:N.')
    if not torch.cuda.is_available():
        if text != 'auto':
            raise ValueError('CUDA requested but unavailable in this Python kernel. '
                             'Run tools/doctor.py; check the NVIDIA driver and CUDA PyTorch wheel.')
        return torch.device('cpu')
    index = int(text.split(':')[1]) if ':' in text else 0
    if index >= torch.cuda.device_count():
        raise ValueError(f'CUDA index {index} outside visible device count {torch.cuda.device_count()}.')
    dev = torch.device(f'cuda:{index}')
    try:
        # Actual kernel launch catches an incompatible wheel/driver/GPU architecture.
        with torch.cuda.device(dev):
            x = torch.ones(2, 2, device=dev)
            finite = bool(torch.isfinite(x @ x).all().item())
            torch.cuda.synchronize(dev)
        if not finite:
            raise RuntimeError('Nonfinite CUDA smoke result')
    except Exception as exc:
        if text != 'auto':
            raise RuntimeError(f'CUDA device {dev} is detected but unusable: {exc}') from exc
        warnings.warn(f'CUDA detected but a kernel failed; AUTO selected CPU: {exc}', RuntimeWarning)
        return torch.device('cpu')
    return dev


def synchronize(device: Any) -> None:
    import torch
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def device_summary(device: Any) -> dict:
    import torch
    dev = torch.device(device)
    out = {'selected_device': str(dev), 'torch_version': str(torch.__version__),
           'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
           'precision': 'float32', 'mixed_precision': False,
           'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'cpu_stages': ['CSV preprocessing', 'NumPy symbolic sparse regression',
                          'scikit-learn logit/Ridge', 'bootstrap and plotting']}
    if dev.type == 'cuda':
        p = torch.cuda.get_device_properties(dev)
        out.update(device_name=p.name, total_memory_GiB=p.total_memory/(1024**3),
                   compute_capability=f'{p.major}.{p.minor}',
                   peak_allocated_MiB=torch.cuda.max_memory_allocated(dev)/(1024**2))
    return out


def cpu_state_dict(model) -> dict:
    """Portable best-epoch snapshots; does not mutate a model running on the GPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def xgb_parameters(device: Any = 'auto') -> dict:
    """XGBoost uses a separately packaged CUDA runtime; verify its fitted device too."""
    dev = choose_device(device)
    return {'tree_method': 'hist', 'device': str(dev), 'n_jobs': 1}


def check_xgb_device(estimator, requested: Any) -> dict:
    """XGBoost can warn and silently switch to CPU; this pipeline does not allow that.

    With --device cpu a CPU fit is intended. With a resolved CUDA device, a silent
    library-level fallback is a setup error, not successful GPU execution.
    """
    import torch
    expected = str(torch.device(requested))
    config = json.loads(estimator.get_booster().save_config())
    actual = config['learner']['generic_param'].get('device', 'unknown')
    if expected.startswith('cuda') and not actual.startswith('cuda'):
        raise RuntimeError('PyTorch CUDA is working but XGBoost fitted on CPU. '
                           'Check the XGBoost CUDA build/driver or explicitly rerun with --device cpu.')
    return {'requested_device': expected, 'fitted_device': actual}


def xgb_predict(estimator, x, probabilities: bool = False):
    """Use DMatrix prediction for host NumPy inputs to avoid inplace device mismatch.

    Booster inference follows its configured CPU/CUDA device. Returned arrays are
    host-side; uploading a training tensor does not move pandas/NumPy I/O to CUDA.
    """
    import numpy as np
    import xgboost as xgb
    result = estimator.get_booster().predict(xgb.DMatrix(x))
    return np.column_stack([1-result, result]) if probabilities else result
