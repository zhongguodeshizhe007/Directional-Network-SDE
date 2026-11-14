"""CSV reader: UTF-8 BOM/outer header whitespace only; never reinterpret units/IDs.

Column names are checked with csv.reader before pandas to catch normalized
collisions. Original highD files stay read-only. Comma is deliberately required;
a spreadsheet export with changed delimiters must be inspected, not guessed.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Any
import pandas as pd


def canonical_column(value: str) -> str:
    return str(value).lstrip('\ufeff').strip()


def columns(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        try:
            raw = next(csv.reader(f))
        except StopIteration:
            raise ValueError(f'空 CSV：{path}') from None
    names = [canonical_column(x) for x in raw]
    if len(set(names)) != len(names):
        raise ValueError(f'去除表头空白后存在重名字段：{path.name} {names}')
    if len(names) < 2:
        raise ValueError(f'{path.name} 没有解析到逗号分隔的多列；请勿用 Excel 改写分隔符。')
    return names


def read_csv(path: str | Path, *, usecols: Iterable[str] | None = None,
             **kwargs: Any):
    """Return DataFrame or a lazy iterator of normalized chunks."""
    names = columns(path)
    selected = None if usecols is None else set(usecols)
    if selected is not None:
        missing = sorted(selected - set(names))
        if missing:
            raise ValueError(f'{Path(path).name} 缺少字段：{missing}; 实际字段：{names}')
    kwargs.setdefault('encoding', 'utf-8-sig')
    kwargs.setdefault('low_memory', False)
    if selected is not None:
        kwargs['usecols'] = lambda name: canonical_column(name) in selected
    result = pd.read_csv(path, **kwargs)
    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(columns=canonical_column)
    if kwargs.get('chunksize') is not None:
        def iterate():
            try:
                for frame in result:
                    yield normalize(frame)
            finally:
                result.close()
        return iterate()
    return normalize(result)
