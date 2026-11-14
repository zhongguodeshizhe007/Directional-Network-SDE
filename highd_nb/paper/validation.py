"""Small strict checks for comparable held-out results; no selection using scores."""
from __future__ import annotations
import pandas as pd


def assert_common_prediction_support(frame: pd.DataFrame, model_column: str, keys: list[str]) -> None:
    """Equal row counts are insufficient: compare the exact scored identifiers."""
    if frame.empty:
        raise ValueError('No prediction rows to compare.')
    if frame.duplicated([model_column]+keys).any():
        raise ValueError('Duplicate scored target identifiers.')
    reference = None
    reference_name = None
    for name, group in frame.groupby(model_column, sort=False):
        support = pd.MultiIndex.from_frame(group[keys]).sort_values()
        if reference is None:
            reference, reference_name = support, name
        elif not support.equals(reference):
            raise ValueError(f'Unequal exact scoring support: {name} vs {reference_name}. '
                             'Do not compare scores on different targets.')
