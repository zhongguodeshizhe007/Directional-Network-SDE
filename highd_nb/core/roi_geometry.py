"""Conservative, matched highD ROI inference without the old arbitrary p95 cutoff.

highD-Format.pdf, page 4: x is the bounding-box left edge; front/back sight
are measured from the vehicle center. We keep that documented convention.

All finite, nonnegative sight observations must support the same fixed ROI,
including the configured margin. No row is trimmed to manufacture acceptance.
The original p95 dispersion is retained as a diagnostic, not a pass criterion.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ROI_POLICY = 'center_preferred_all_row_intersection_v1'
FORMAT_SOURCE = 'https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf'
TOLERANCE_M = 1e-7


def _summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {name: float(np.quantile(a, q)) for name, q in
            [('min', 0), ('p01', .01), ('median', .5), ('p99', .99), ('max', 1)]}


def inspect_roi(d: pd.DataFrame, rec: str, direction: int,
                analysis_length_m: float, margin_m: float) -> dict:
    """Return JSON-safe decision/evidence; never silently choose an alternative reference.

    BLOCKED is a diagnostic return value. select_roi() converts it to an error,
    allowing callers to save the evidence before stopping.
    """
    width, margin = float(analysis_length_m), float(margin_m)
    if direction not in (1, 2):
        raise ValueError('direction must be 1 or 2')
    if not np.isfinite([width, margin]).all() or width <= 0 or margin < 0:
        raise ValueError('Positive finite analysis length and nonnegative margin required')
    out = {'policy': ROI_POLICY, 'recording_id': str(rec).zfill(2),
           'direction': int(direction), 'format_source': FORMAT_SOURCE,
           'analysis_length_m': width, 'boundary_margin_m': margin,
           'n_rows': int(len(d)), 'status': 'BLOCKED', 'reasons': [],
           'note': 'Geometry evidence only. It does not prove trajectory or risk-model validity.'}
    needed = ['x', 'width', 'frontSightDistance', 'backSightDistance']
    absent = [k for k in needed if k not in d]
    if absent:
        out['reasons'].append('Missing geometry columns: ' + ', '.join(absent))
        return out
    x, length, front, back = (d[k].to_numpy(dtype=float) for k in needed)
    xc = x + length / 2.
    if 'xc' in d:
        given = d['xc'].to_numpy(dtype=float)
        same = np.isfinite(given) & np.isfinite(xc)
        if same.any() and float(np.max(np.abs(given[same] - xc[same]))) > 1e-6:
            out['reasons'].append('Precomputed xc disagrees with x + width/2')
    ok = (np.isfinite(x) & np.isfinite(length) & (length > 0) &
          np.isfinite(front) & np.isfinite(back))
    # Non-finite/invalid input is reported, not quietly removed for acceptance.
    # Vehicle centers crossing camera entry/exit lines have negative front or back
    # sight distance while front + back (total view length) remains positive.
    out['n_usable_sight_rows'] = int(ok.sum())
    out['n_invalid_sight_rows'] = int((~ok).sum())
    out['n_negative_sight_rows'] = int((ok & ((front < 0) | (back < 0))).sum())
    if not len(d) or not ok.any():
        out['reasons'].append('No usable sight observations')
        return out
    if not ok.all():
        out['reasons'].append('Invalid sight/geometry rows exist; inspect before overriding camera bounds')
    low = xc[ok] - (front[ok] if direction == 1 else back[ok])
    high = xc[ok] + (back[ok] if direction == 1 else front[ok])
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        out['reasons'].append('Non-finite derived endpoints')
        return out
    if np.any(high <= low):
        out['reasons'].append('Nonpositive sight-derived view lengths exist')
    lo, hi = float(np.median(low)), float(np.median(high))
    p95 = float(max(np.quantile(np.abs(low-lo), .95), np.quantile(np.abs(high-hi), .95)))
    inner_lo, inner_hi = float(np.max(low)), float(np.min(high))
    safe_lo, safe_hi = inner_lo + margin, inner_hi - margin
    nominal_center = (lo+hi) / 2.
    out.update(median_recorded_x_low_m=lo, median_recorded_x_high_m=hi,
               old_boundary_p95_deviation_m=p95, old_p95_guard_would_fail=bool(p95 > 5),
               low_endpoint_summary_m=_summary(low), high_endpoint_summary_m=_summary(high),
               view_length_summary_m=_summary(high-low), vehicle_length_summary_m=_summary(length),
               all_row_intersection_low_m=inner_lo, all_row_intersection_high_m=inner_hi,
               safe_analysis_low_m=safe_lo, safe_analysis_high_m=safe_hi,
               safe_available_length_m=safe_hi-safe_lo, nominal_center_m=nominal_center)
    if safe_hi - safe_lo < width - TOLERANCE_M:
        out['reasons'].append(
            f'All-row common view supports only {safe_hi-safe_lo:.6f} m after margins; '
            f'{width:g} m is required. Do not shrink ROI or skip recording silently.')
        return out
    if out['reasons']:
        return out
    # Choose the closest feasible center to the original median-centered ROI.
    cmin, cmax = safe_lo+width/2., safe_hi-width/2.
    if cmin > cmax:  # floating-point equality within tolerance
        center = (cmin+cmax)/2.
    else:
        center = float(np.clip(nominal_center, cmin, cmax))
    target_lo, target_hi = center-width/2., center+width/2.
    supported = (target_lo-low >= margin-TOLERANCE_M) & (high-target_hi >= margin-TOLERANCE_M)
    out.update(common_analysis_low_m=target_lo, common_analysis_high_m=target_hi,
               selected_center_m=center, center_shift_m=center-nominal_center,
               min_left_margin_m=float(np.min(target_lo-low)),
               min_right_margin_m=float(np.min(high-target_hi)),
               supported_rows=int(supported.sum()), support_fraction=float(supported.mean()))
    if not supported.all():
        out['reasons'].append('Internal containment check failed; no ROI emitted')
        return out
    # For the legacy builder, supply ROI endpoints BEFORE its own margin step.
    # They are NOT estimates of actual camera boundaries.
    out['builder_envelope_override_m'] = [target_lo-margin, target_hi+margin]
    out['status'] = 'PASS_COMMON_ROI'
    out['warnings'] = []
    if p95 > 5:
        out['warnings'].append(
            'Old 5 m dispersion guard fails, but every row supports the selected ROI and margin. '
            'Boundary variation remains unexplained; do not relabel this as a camera calibration.')
    if abs(out['center_shift_m']) > TOLERANCE_M:
        out['warnings'].append('ROI center was shifted inside the common view; inspect center_shift_m.')
    return out


def select_roi(d, rec, direction, analysis_length_m, margin_m):
    result = inspect_roi(d, rec, direction, analysis_length_m, margin_m)
    if result['status'] != 'PASS_COMMON_ROI':
        raise ValueError(f'{rec}/dir{direction}: ROI BLOCKED: ' + '; '.join(result['reasons']))
    return result
