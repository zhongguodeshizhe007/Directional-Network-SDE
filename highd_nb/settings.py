"""Deployment paths shared by command-line tools. No path is created on import."""
from __future__ import annotations
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PROJECT_ROOT = r'F:\PythonFiles\Gemini\NetworkSDE'


def default_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    if project_root is None:
        project_root = os.environ.get('NETWORKSDE_PROJECT_ROOT')
    if project_root is None:
        project_root = WINDOWS_PROJECT_ROOT if os.name == 'nt' else PACKAGE_ROOT
    root = Path(project_root).expanduser().resolve()
    return {'project_root': root,
            'data_dir': root / 'highd-dataset-v1.0' / 'data',
            'work_dir': root / 'outputs' / 'nb_v5_paper',
            'config': root / 'configs' / 'experiment.json'}
