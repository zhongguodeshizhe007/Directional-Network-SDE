#!/usr/bin/env python3
"""Scheme A / highD data layer v1.0.0.

No model training, downloading, or editing of source data. See README_中文.md.
Sources: official highD Format.pdf and RobertKrajewski/highD-dataset.
All aggregation, masking and forecast-target choices are research design choices,
not claims about the original highD or LaGNA implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from highd_nb.csvio import read_csv as read_raw_csv

VERSION = 'scheme-a-highd-1.0.0'
DEFAULT_CONFIG = Path(__file__).with_name('config.json')
TRACK_REQUIRED = [
    'frame', 'id', 'x', 'y', 'width', 'height', 'xVelocity', 'laneId',
    'dhw', 'ttc', 'precedingId', 'precedingXVelocity',
    'frontSightDistance', 'backSightDistance',
]
TRACK_OPTIONAL = ['xAcceleration', 'yVelocity', 'yAcceleration']
STATIC_REQUIRED = ['id', 'class', 'drivingDirection']
META_REQUIRED = ['id', 'frameRate', 'locationId', 'duration',
                 'upperLaneMarkings', 'lowerLaneMarkings']
CORE_FEATURES = ['speed_mean_mps', 'speed_cv', 'density_veh_per_km', 'truck_time_share']
EVENT_COLUMNS = ['event_id', 'recording_id', 'direction', 'follower_id', 'leader_id',
                 'lane_id', 'threshold_s', 'start_frame', 'end_frame',
                 'confirmed_frame', 'start_s', 'end_s_exclusive', 'confirmed_s',
                 'duration_s', 'min_ttc_s', 'start_node', 'confirmed_node',
                 'end_node', 'visited_nodes']
EDGE_NAMES = ['UPSTREAM_SOURCE', 'DOWNSTREAM_SOURCE', 'LATERAL_SOURCE']


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def finite_or_none(x: Any) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def ratio(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'{where}: missing required columns: {missing}')


def load_config(path: Path | None) -> dict:
    c = json.loads((path or DEFAULT_CONFIG).read_text(encoding='utf-8-sig'))
    for key in ['cell_length_m', 'step_s', 'context_window_s', 'risk_time_scale_s']:
        if c[key] <= 0:
            raise ValueError(f'{key} must be > 0')
    for key in ['boundary_margin_m', 'gap_min_m', 'min_pair_exposure_s', 'min_event_duration_s']:
        if c[key] < 0:
            raise ValueError(f'{key} must be >= 0')
    if not 0 <= c['min_pair_coverage'] <= 1:
        raise ValueError('min_pair_coverage must be in [0,1]')
    if c['context_window_s'] < c['step_s']:
        raise ValueError('context_window_s must be >= step_s')
    integral_ratio(c['context_window_s'], c['step_s'], 'context/step')
    for h in c['forecast_horizons_s'] + c['diagnostic_lags_s']:
        integral_ratio(h, c['step_s'], 'horizon or diagnostic lag / step')
    if any(t <= 0 for t in c['conflict_thresholds_s']):
        raise ValueError('TTC thresholds must be positive')
    return c


def integral_ratio(a: float, b: float, name: str) -> int:
    k = round(a / b)
    if k < 1 or not np.isclose(a / b, k, atol=1e-8):
        raise ValueError(f'{name} must be a positive integer; got {a}/{b}')
    return k


def recordings(data_dir: Path, selection: str) -> list[str]:
    available = sorted(p.name[:-len('_tracks.csv')] for p in data_dir.glob('*_tracks.csv'))
    if not available:
        raise FileNotFoundError(f'No XX_tracks.csv under {data_dir}. Pass the actual data folder.')
    if selection.lower() == 'all':
        return available
    requested = [s.strip().zfill(2) for s in selection.split(',') if s.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f'Missing recordings: {missing}; available: {available}')
    return sorted(set(requested))


def input_paths(data_dir: Path, rec: str) -> dict[str, Path]:
    return {'tracks': data_dir / f'{rec}_tracks.csv',
            'static': data_dir / f'{rec}_tracksMeta.csv',
            'recording': data_dir / f'{rec}_recordingMeta.csv'}


def check_output(data_dir: Path, out_dir: Path) -> None:
    src, dst = data_dir.resolve(), out_dir.resolve()
    if src == dst or src in dst.parents:
        raise ValueError('Output must be outside the raw data directory. Raw files are read-only.')
    out_dir.mkdir(parents=True, exist_ok=True)


def marks(value: Any) -> np.ndarray:
    a = np.array([float(v.strip()) for v in str(value).split(';') if v.strip()], dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError(f'Invalid lane markings: {value}')
    return a


def lane_geometry(meta: pd.Series) -> dict[int, dict[int, tuple[float, float]]]:
    upper, lower = marks(meta['upperLaneMarkings']), marks(meta['lowerLaneMarkings'])
    if upper[-1] >= lower[0]:
        raise ValueError('Upper and lower lane markings overlap or are reversed')
    return {
        1: {k + 2: (float(upper[k]), float(upper[k + 1])) for k in range(len(upper) - 1)},
        2: {len(upper) + k + 2: (float(lower[k]), float(lower[k + 1]))
            for k in range(len(lower) - 1)}
    }


def audit(data_dir: Path, out_dir: Path, selection: str) -> None:
    check_output(data_dir, out_dir)
    rows, details = [], []
    for rec in recordings(data_dir, selection):
        paths = input_paths(data_dir, rec)
        row: dict[str, Any] = {'recording_id': rec}
        detail: dict[str, Any] = {'recording_id': rec, 'errors': []}
        for kind, p in paths.items():
            row[f'{kind}_exists'] = p.exists()
            row[f'{kind}_size_bytes'] = p.stat().st_size if p.exists() else 0
        try:
            header = read_raw_csv(paths['tracks'], nrows=0)
            static = read_raw_csv(paths['static'])
            m = read_raw_csv(paths['recording'])
            require_columns(header, TRACK_REQUIRED, paths['tracks'].name)
            require_columns(static, STATIC_REQUIRED, paths['static'].name)
            require_columns(m, META_REQUIRED, paths['recording'].name)
            if len(m) != 1:
                raise ValueError('recordingMeta must contain exactly one row')
            geom = lane_geometry(m.iloc[0])
            row.update(location_id=int(m.iloc[0]['locationId']),
                       frame_rate=float(m.iloc[0]['frameRate']),
                       duration_s=float(m.iloc[0]['duration']),
                       n_tracks_meta=len(static),
                       dir1_lanes=len(geom[1]), dir2_lanes=len(geom[2]), status='OK_HEADERS')
            detail['track_columns'] = header.columns.tolist()
            detail['static_columns'] = static.columns.tolist()
            detail['directions_in_meta'] = sorted(static['drivingDirection'].unique().tolist())
            detail['lane_geometry'] = geom
            detail['note'] = 'Header/meta audit only. No trajectory-level QC has been run.'
        except Exception as exc:
            row['status'] = 'ERROR'
            detail['errors'].append(str(exc))
        rows.append(row)
        details.append(detail)
    pd.DataFrame(rows).to_csv(out_dir / 'inventory.csv', index=False, encoding='utf-8-sig')
    dump_json(out_dir / 'inventory.json', {'version': VERSION, 'records': details})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nSaved: {out_dir / "inventory.csv"}')
    if any(r['status'] == 'ERROR' for r in rows):
        raise ValueError('Inventory contains errors; inspect inventory.json before building.')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_recording(data_dir: Path, rec: str, c: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    paths = input_paths(data_dir, rec)
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(str(p))
    header = read_raw_csv(paths['tracks'], nrows=0)
    require_columns(header, TRACK_REQUIRED, paths['tracks'].name)
    usecols = [k for k in TRACK_REQUIRED + TRACK_OPTIONAL if k in header.columns]
    d = read_raw_csv(paths['tracks'], usecols=usecols)
    static = read_raw_csv(paths['static'])
    m = read_raw_csv(paths['recording'])
    require_columns(static, STATIC_REQUIRED, paths['static'].name)
    require_columns(m, META_REQUIRED, paths['recording'].name)
    if len(m) != 1 or int(m.iloc[0]['id']) != int(rec):
        raise ValueError('recordingMeta row count or recording ID mismatch')
    for col in [k for k in usecols if k not in []]:
        try:
            d[col] = pd.to_numeric(d[col], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{paths["tracks"].name}: {col} 含非数值；请检查 CSV 片段。') from exc
    for col in ['id', 'drivingDirection']:
        static[col] = pd.to_numeric(static[col], errors='raise')
    for col in ['frameRate', 'duration', 'locationId', 'id']:
        m[col] = pd.to_numeric(m[col], errors='raise')
    if not np.isfinite(m[['frameRate','duration']].to_numpy()).all() or (m[['frameRate','duration']] <= 0).any().any():
        raise ValueError('recordingMeta frameRate/duration must be positive finite values')
    if static['id'].duplicated().any():
        raise ValueError('Duplicate IDs in tracksMeta')
    if d.duplicated(['id', 'frame']).any():
        raise ValueError('Duplicate (id, frame) keys in tracks.csv; do not silently drop duplicates')
    if not d['id'].isin(static['id']).all():
        raise ValueError('Some track IDs have no tracksMeta row')
    for col in ['frame', 'id', 'laneId', 'precedingId']:
        a = d[col].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
            raise ValueError(f'Non-integer or missing values in {col}')
        d[col] = d[col].astype(np.int64)
    if (d['id'] <= 0).any() or (d['frame'] < 0).any():
        raise ValueError('Invalid ID or negative frame')
    # Only static class/direction are joined. Full-track summary predictors are prohibited.
    d = d.merge(static[STATIC_REQUIRED], on='id', how='left', validate='many_to_one')
    if not d['drivingDirection'].isin([1, 2]).all():
        raise ValueError('Unknown drivingDirection (expected 1 or 2)')
    if not d['class'].isin(['Car', 'Truck']).all():
        raise ValueError('Unknown vehicle class; expected Car or Truck')
    d = d.sort_values(['id', 'frame'], kind='stable').reset_index(drop=True)
    d['direction'] = d['drivingDirection'].astype(int)
    d['sign'] = np.where(d['direction'] == 1, -1.0, 1.0)
    d['xc'] = d['x'] + d['width'] / 2.0
    d['yc'] = d['y'] + d['height'] / 2.0
    signed_speed = d['sign'] * d['xVelocity']
    bad_speed = signed_speed < -c['negative_speed_tolerance_mps']
    if bad_speed.mean() > .01:
        raise ValueError('More than 1% of rows oppose metadata drivingDirection; inspect signs')
    d['u'] = signed_speed.clip(lower=0).where(~bad_speed)
    d['is_truck'] = (d['class'] == 'Truck').astype(float)
    if 'xAcceleration' in d:
        d['a_long'] = d['sign'] * d['xAcceleration']
    else:
        d['a_long'] = np.nan
    basic = (np.isfinite(d[['xc', 'yc', 'width', 'height', 'u']]).all(axis=1)
             & (d['width'] > 0) & (d['height'] > 0))
    geom = lane_geometry(m.iloc[0])
    lane_valid = np.zeros(len(d), dtype=bool)
    center_consistent = np.zeros(len(d), dtype=bool)
    for direction, lane_map in geom.items():
        for lane_id, (yl, yu) in lane_map.items():
            sel = (d['direction'].to_numpy() == direction) & (d['laneId'].to_numpy() == lane_id)
            lane_valid |= sel
            center_consistent |= sel & (d['yc'].to_numpy() >= yl) & (d['yc'].to_numpy() <= yu)
    d['obs_valid'] = basic & lane_valid
    # Resolve the actual leader at the same frame BEFORE any ROI crop.
    indexed = d.set_index(['frame', 'id'])
    keys = pd.MultiIndex.from_arrays([d['frame'], d['precedingId']], names=['frame', 'id'])
    leader = indexed[['u', 'xc', 'yc', 'width', 'a_long', 'direction', 'laneId', 'obs_valid']].reindex(keys)
    leader.index = d.index
    d['leader_found'] = leader['direction'].notna()
    d['same_lane_leader'] = (leader['laneId'].to_numpy() == d['laneId'].to_numpy())
    d['same_dir_leader'] = (leader['direction'].to_numpy() == d['direction'].to_numpy())
    d['gap_geom'] = d['sign'] * (leader['xc'].to_numpy() - d['xc']) - (
        d['width'] + leader['width'].to_numpy()) / 2.0
    d['closing_speed'] = d['u'] - leader['u'].to_numpy()
    d['pair_valid'] = (
        d['obs_valid'] & (d['precedingId'] > 0) & (d['precedingId'] != d['id'])
        & d['leader_found'] & d['same_lane_leader'] & d['same_dir_leader']
        & leader['obs_valid'].eq(True).to_numpy(bool)
        & np.isfinite(d['dhw']) & (d['dhw'] > c['gap_min_m'])
        & (d['gap_geom'] > 0) & np.isfinite(d['closing_speed']))
    rho = np.maximum(d['closing_speed'].to_numpy(), 0) / np.where(d['dhw'] > 0, d['dhw'], np.nan)
    d['inv_ttc'] = np.where(d['pair_valid'], rho, np.nan)
    d['ttc_calc'] = np.where(d['pair_valid'] & (d['closing_speed'] > 0),
                             np.divide(1.0, rho, out=np.full(len(d), np.inf), where=np.asarray(rho) > 0), np.inf)
    d['drac'] = np.where(d['pair_valid'],
                          np.maximum(d['closing_speed'], 0) ** 2 / (2 * d['dhw'].where(d['dhw'] > 0)),
                          np.nan)
    from highd_nb.paper.augmentation import attach_pair_metrics
    attach_pair_metrics(d,leader,c)
    both = d['pair_valid'] & (d['closing_speed'] > 0) & (d['ttc'] > 0)
    gap_errors = (d.loc[d['pair_valid'], 'gap_geom'] - d.loc[d['pair_valid'], 'dhw']).abs()
    inv_errors = (d.loc[both, 'inv_ttc'] - 1 / d.loc[both, 'ttc']).abs()
    vel_errors = (leader.loc[d['leader_found'], 'u'].to_numpy()
                  - (d.loc[d['leader_found'], 'sign'] * d.loc[d['leader_found'], 'precedingXVelocity']).to_numpy())
    same_track = d['id'].eq(d['id'].shift())
    frame_jumps = d['frame'].diff().where(same_track)
    qc = {
        'n_rows': len(d), 'n_tracks': int(d['id'].nunique()),
        'metadata_tracks_not_in_csv': int((~static['id'].isin(d['id'].unique())).sum()),
        'invalid_observation_rows': int((~d['obs_valid']).sum()),
        'invalid_lane_or_direction_rows': int((~lane_valid).sum()),
        'lane_center_mismatch_fraction': ratio(int((lane_valid & ~center_consistent).sum()), int(lane_valid.sum())),
        'negative_direction_speed_rows': int(bad_speed.sum()),
        'missing_internal_track_frames': int(np.maximum(frame_jumps.fillna(1).to_numpy() - 1, 0).sum()),
        'raw_ttc_zero_rows': int((d['ttc'] == 0).sum()),
        'raw_ttc_negative_rows': int((d['ttc'] < 0).sum()),
        'no_predecessor_rows': int((d['precedingId'] == 0).sum()),
        'unresolved_predecessor_rows': int(((d['precedingId'] > 0) & ~d['leader_found']).sum()),
        'leader_lane_or_direction_mismatch_rows': int(((d['precedingId'] > 0) & d['leader_found'] &
            (~d['same_lane_leader'] | ~d['same_dir_leader'])).sum()),
        'valid_pair_rows': int(d['pair_valid'].sum()),
        'nonclosing_valid_pair_rows': int((d['pair_valid'] & (d['closing_speed'] <= 0)).sum()),
        'closing_valid_pair_rows': int((d['pair_valid'] & (d['closing_speed'] > 0)).sum()),
        'gap_geometry_vs_dhw_abs_median_m': finite_or_none(gap_errors.median()),
        'gap_geometry_vs_dhw_abs_p95_m': finite_or_none(gap_errors.quantile(.95)),
        'positive_ttc_inverse_abs_error_median': finite_or_none(inv_errors.median()),
        'positive_ttc_comparable_rows': int(both.sum()),
        'leader_velocity_abs_error_median_mps': finite_or_none(np.median(np.abs(vel_errors))) if len(vel_errors) else None,
        'checksums': {k: sha256(p) for k, p in paths.items()} if c['hash_inputs'] else {},
        'warnings': []}
    if qc['missing_internal_track_frames']:
        qc['warnings'].append('Internal track gaps detected; no interpolation was applied.')
    if (qc['gap_geometry_vs_dhw_abs_p95_m'] or 0) > 1:
        qc['warnings'].append('DHW/geometry mismatch exceeds 1 m at p95. Audit gap convention before analysis.')
    if (qc['lane_center_mismatch_fraction'] or 0) > .02:
        qc['warnings'].append('More than 2% lane-center mismatches. Inspect raw lane markings and lane-change frames.')
    return d, m.iloc[0], qc


def infer_roi(d: pd.DataFrame, rec: str, direction: int, c: dict) -> dict:
    override = c['roi_overrides'].get(f'{rec}_dir{direction}')
    if override is not None:
        lo, hi = map(float, override)
        method, spread = 'manual_override', None
    else:
        front, back, xc = (d[k].to_numpy(float) for k in ['frontSightDistance', 'backSightDistance', 'xc'])
        ok = np.isfinite(front + back + xc) & (front >= 0) & (back >= 0)
        if not ok.any():
            raise ValueError('No usable sight distances; specify roi_overrides for this recording/direction')
        low = xc[ok] - (front[ok] if direction == 1 else back[ok])
        high = xc[ok] + (back[ok] if direction == 1 else front[ok])
        lo, hi = float(np.median(low)), float(np.median(high))
        spread = float(max(np.quantile(np.abs(low - lo), .95), np.quantile(np.abs(high - hi), .95)))
        if spread > 5:
            raise ValueError(f'Sight-derived boundaries vary by p95={spread:.2f} m. Set roi_overrides.')
        method = 'median_center_plus_sight_distance'
    usable = hi - lo - 2 * c['boundary_margin_m']
    n = int(math.floor((usable + 1e-9) / c['cell_length_m']))
    if n < 2:
        raise ValueError('ROI has fewer than 2 longitudinal cells; check geometry/margin/cell size')
    start = lo + c['boundary_margin_m'] + (usable - n * c['cell_length_m']) / 2
    return {'recorded_x_low_m': lo, 'recorded_x_high_m': hi,
            'analysis_x_low_m': start, 'analysis_x_high_m': start + n * c['cell_length_m'],
            'n_cells': n, 'cell_length_m': c['cell_length_m'], 'method': method,
            'boundary_p95_deviation_m': spread}


def build_nodes_edges(meta: pd.Series, direction: int, roi: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Physical leftmost/median-side lane first for both directions.
    lane_map = lane_geometry(meta)[direction]
    lane_ids = sorted(lane_map, reverse=(direction == 1))
    M, L = roi['n_cells'], roi['cell_length_m']
    rows = []
    for rank, lane_id in enumerate(lane_ids):
        for m in range(M):
            x0 = (roi['analysis_x_low_m'] + m * L if direction == 2
                  else roi['analysis_x_high_m'] - (m + 1) * L)
            rows.append({'node_id': rank * M + m, 'direction': direction, 'lane_id': lane_id,
                         'lane_rank_from_median': rank, 'cell_index': m,
                         's_start_m': m * L, 's_end_m': (m + 1) * L,
                         'x_low_m': x0, 'x_high_m': x0 + L,
                         'y_low_m': lane_map[lane_id][0], 'y_high_m': lane_map[lane_id][1],
                         'boundary_node': m in [0, M - 1]})
    nodes = pd.DataFrame(rows)
    edges = []
    for rank in range(len(lane_ids)):
        for m in range(M):
            target = rank * M + m
            neighbors = []
            if m > 0:
                neighbors.append((rank * M + m - 1, 0))
            if m + 1 < M:
                neighbors.append((rank * M + m + 1, 1))
            for other in [rank - 1, rank + 1]:
                if 0 <= other < len(lane_ids):
                    neighbors.append((other * M + m, 2))
            for source, etype in neighbors:
                a, b = nodes.iloc[source], nodes.iloc[target]
                ds = (a.s_start_m + a.s_end_m - b.s_start_m - b.s_end_m) / 2
                dy = (a.y_low_m + a.y_high_m - b.y_low_m - b.y_high_m) / 2
                edges.append({'source': source, 'target': target, 'edge_type': etype,
                              'edge_name': EDGE_NAMES[etype], 'delta_s_m': ds,
                              'delta_y_image_m': dy, 'weight': 1.0})
    return nodes, pd.DataFrame(edges)


def grouped_array(g: pd.Series, T: int, N: int, fill: float = 0) -> np.ndarray:
    a = np.full((T, N), fill, dtype=float)
    if len(g):
        b = g.index.get_level_values(0).to_numpy(int)
        n = g.index.get_level_values(1).to_numpy(int)
        a[b, n] = g.to_numpy(float)
    return a


def rolling_sum(a: np.ndarray, W: int) -> np.ndarray:
    cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)], axis=0)
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) >= W:
        out[W - 1:] = cs[W:] - cs[:-W]
    return out


def unique_vehicle_counts(rows: pd.DataFrame, T: int, N: int, W: int = 1) -> np.ndarray:
    triples = rows[['bin', 'node', 'id']].drop_duplicates()
    if W > 1:
        expanded = [triples.assign(bin=triples['bin'] + j) for j in range(W)]
        triples = pd.concat(expanded, ignore_index=True)
        triples = triples.loc[triples['bin'] < T].drop_duplicates()
    vals = triples.groupby(['bin', 'node'], observed=True).size()
    a = grouped_array(vals, T, N)
    if W > 1:
        a[:W - 1] = np.nan
    return a


def traffic_features(sums: dict[str, np.ndarray], unique: np.ndarray,
                     duration: float, L: float, c: dict) -> dict[str, np.ndarray]:
    n = sums['n']
    with np.errstate(divide='ignore', invalid='ignore'):
        v = sums['v'] / n
        var = np.maximum(sums['v2'] / n - v ** 2, 0)
        sd = np.sqrt(var)
        cv = sd / v
        truck = sums['truck'] / n
        acc = sums['a'] / sums['a_n']
    cv[(unique < c['min_distinct_vehicles_cv']) | (v < c['min_speed_for_cv_mps'])] = np.nan
    return {'speed_mean_mps': v, 'speed_std_mps': sd, 'speed_cv': cv,
            'density_veh_per_km': 1000 * n / (L * duration),
            'edie_flow_veh_per_h': 3600 * sums['v'] / (L * duration),
            'truck_time_share': truck, 'accel_mean_mps2': acc,
            'unique_vehicles': unique, 'vehicle_exposure_s': n}


def detector_crossings(d: pd.DataFrame, T: int, N: int, roi: dict,
                       origin: int, fps: float, step: float) -> tuple[np.ndarray, np.ndarray, int]:
    # Continuous x crossing, only consecutive raw frames with unchanged valid lane.
    # These counts are optional descriptors; Edie flow is the principal flow measure.
    cin, cout = np.zeros((T, N)), np.zeros((T, N))
    prev = d.shift(1)
    valid = (d['id'].eq(prev['id']) & d['frame'].eq(prev['frame'] + 1)
             & d['laneId'].eq(prev['laneId']) & d['obs_valid']
             & prev['obs_valid'].eq(True)
             & d['lane_rank'].notna() & (d['s'] >= prev['s']))
    changes = int((d['id'].eq(prev['id']) & d['frame'].eq(prev['frame'] + 1)
                   & ~d['laneId'].eq(prev['laneId'])).sum())
    s0, s1 = prev.loc[valid, 's'].to_numpy(), d.loc[valid, 's'].to_numpy()
    f0 = prev.loc[valid, 'frame'].to_numpy(float)
    ranks = d.loc[valid, 'lane_rank'].to_numpy(int)
    M, L = roi['n_cells'], roi['cell_length_m']
    for boundary in range(M + 1):
        x = boundary * L
        hit = (s0 < x) & (s1 >= x) & (s1 > s0)
        if not hit.any():
            continue
        at = ((f0[hit] - origin) + (x - s0[hit]) / (s1[hit] - s0[hit])) / fps
        bins = np.floor(at / step + 1e-10).astype(int)
        good = (bins >= 0) & (bins < T)
        bins, rank = bins[good], ranks[hit][good]
        if boundary < M:
            np.add.at(cin, (bins, rank * M + boundary), 1)
        if boundary > 0:
            np.add.at(cout, (bins, rank * M + boundary - 1), 1)
    return cin, cout, changes


def conflict_events(d: pd.DataFrame, T: int, N: int, rec: str, direction: int,
                    origin: int, fps: float, c: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Identify pair episodes on full raw trajectories; never retroactively flag pre-confirmation frames."""
    K = len(c['conflict_thresholds_s'])
    active, onset = np.zeros((T, N, K), dtype=bool), np.zeros((T, N, K), dtype=bool)
    event_rows = []
    min_frames = max(1, int(math.ceil(c['min_event_duration_s'] * fps - 1e-9)))
    pair_boundary = (d['id'].ne(d['id'].shift()) | d['precedingId'].ne(d['precedingId'].shift())
                     | d['laneId'].ne(d['laneId'].shift()) | d['frame'].ne(d['frame'].shift() + 1))
    for k, threshold in enumerate(c['conflict_thresholds_s']):
        eligible = d['pair_valid'] & (d['closing_speed'] > 0) & (d['ttc_calc'] < threshold)
        starts = eligible & (pair_boundary | ~eligible.shift(1, fill_value=False))
        run_ids = starts.cumsum()
        work = d.loc[eligible].copy()
        if work.empty:
            continue
        work['run'] = run_ids[eligible].to_numpy()
        sizes = work.groupby('run', sort=False)['frame'].transform('size')
        work = work.loc[sizes >= min_frames]
        if work.empty:
            continue
        work['position'] = work.groupby('run', sort=False).cumcount()
        confirmed = work.loc[work['position'] >= min_frames - 1]
        in_roi = (confirmed['node'] >= 0) & (confirmed['bin'] >= 0) & (confirmed['bin'] < T)
        w = confirmed.loc[in_roi]
        active[w['bin'].to_numpy(int), w['node'].to_numpy(int), k] = True
        on = w.loc[w['position'] == min_frames - 1]
        onset[on['bin'].to_numpy(int), on['node'].to_numpy(int), k] = True
        for run, event in work.groupby('run', sort=False):
            visited = event.loc[(event['node'] >= 0) & (event['bin'] < T), 'node'].unique().astype(int)
            if not len(visited):
                continue
            a, b, conf = event.iloc[0], event.iloc[-1], event.iloc[min_frames - 1]
            event_rows.append({
                'event_id': f'{rec}_d{direction}_ttc{threshold:g}_r{run}',
                'recording_id': rec, 'direction': direction, 'follower_id': int(a['id']),
                'leader_id': int(a['precedingId']), 'lane_id': int(a['laneId']),
                'threshold_s': float(threshold), 'start_frame': int(a['frame']), 'end_frame': int(b['frame']),
                'confirmed_frame': int(conf['frame']), 'start_s': (int(a['frame']) - origin) / fps,
                'end_s_exclusive': (int(b['frame']) - origin + 1) / fps,
                'confirmed_s': (int(conf['frame']) - origin) / fps,
                'duration_s': len(event) / fps, 'min_ttc_s': float(event['ttc_calc'].min()),
                'start_node': int(a['node']), 'confirmed_node': int(conf['node']), 'end_node': int(b['node']),
                'visited_nodes': ';'.join(map(str, visited))})
    return pd.DataFrame(event_rows, columns=EVENT_COLUMNS), active, onset


def future_labels(active: np.ndarray, onset: np.ndarray, risk_mask: np.ndarray,
                  c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, N, K = active.shape
    H = len(c['forecast_horizons_s'])
    ya, yo = np.full((T, N, H, K), np.nan, np.float32), np.full((T, N, H, K), np.nan, np.float32)
    ma, mo = np.zeros_like(ya, bool), np.zeros_like(yo, bool)
    ca = np.concatenate([np.zeros((1, N, K), dtype=int), active.cumsum(axis=0)], axis=0)
    co = np.concatenate([np.zeros((1, N, K), dtype=int), onset.cumsum(axis=0)], axis=0)
    cm = np.concatenate([np.zeros((1, N), dtype=int), risk_mask.cumsum(axis=0)], axis=0)
    for h_idx, seconds in enumerate(c['forecast_horizons_s']):
        h = integral_ratio(seconds, c['step_s'], 'forecast horizon')
        stop = T - h
        if stop <= 0:
            continue
        # Anchor b is at the right edge of bin b. Only b+1,...,b+h are future.
        pa = ca[h + 1:] - ca[1:T - h + 1] > 0
        po = co[h + 1:] - co[1:T - h + 1] > 0
        full_pair_observation = cm[h + 1:] - cm[1:T - h + 1] == h
        known_a = pa | full_pair_observation[..., None]
        # New-onset evaluation excludes nodes already active in the anchor bin.
        known_o = (po | full_pair_observation[..., None]) & ~active[:stop]
        ya[:stop, :, h_idx] = np.where(known_a, pa.astype(float), np.nan)
        yo[:stop, :, h_idx] = np.where(known_o, po.astype(float), np.nan)
        ma[:stop, :, h_idx], mo[:stop, :, h_idx] = known_a, known_o
    return ya, ma, yo, mo


def increment_diagnostics(y: np.ndarray, mask: np.ndarray, c: dict) -> pd.DataFrame:
    rows = []
    for seconds in c['diagnostic_lags_s']:
        lag = integral_ratio(seconds, c['step_s'], 'diagnostic lag')
        if lag >= len(y):
            continue
        valid = mask[lag:] & mask[:-lag]
        # Require all intermediate bins observed, not just matching endpoints.
        for j in range(1, lag):
            valid &= mask[j:len(mask) - lag + j]
        delta = (y[lag:] - y[:-lag])[valid]
        variance = float(np.var(delta)) if len(delta) > 1 else None
        rows.append({'lag_s': seconds, 'n_valid_increments': len(delta),
                     'increment_mean': finite_or_none(np.mean(delta)) if len(delta) else None,
                     'increment_variance': variance,
                     'variance_per_second': variance / seconds if variance is not None else None})
    return pd.DataFrame(rows)


def build_graph(dall: pd.DataFrame, meta: pd.Series, rec: str, direction: int,
                c: dict, graph_dir: Path, recording_qc: dict) -> dict:
    d = dall.loc[dall['direction'] == direction].copy().reset_index(drop=True)
    if d.empty:
        raise ValueError(f'No rows for {rec}, direction {direction}')
    fps, step = float(meta['frameRate']), c['step_s']
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid frame rate')
    frames_per_bin = integral_ratio(step, 1 / fps, 'step_s * frameRate')
    W = integral_ratio(c['context_window_s'], step, 'context/step')
    # Frame 1 corresponds to 0 s; frame 0 variants are accepted but recorded.
    origin = 0 if int(dall['frame'].min()) == 0 else 1
    # Never invent observations beyond the last observed frame or metadata duration.
    observed_end = (int(dall['frame'].max()) - origin + 1) / fps
    duration = min(float(meta['duration']), observed_end)
    T = int(math.floor(duration / step + 1e-7))
    observed_start = (int(dall['frame'].min()) - origin) / fps
    time_support = np.arange(T) * step >= observed_start - 1e-9
    if T < W + 2:
        raise ValueError('Recording too short for configured history/step')
    roi = infer_roi(d, rec, direction, c)
    nodes, edges = build_nodes_edges(meta, direction, roi)
    N, M = len(nodes), roi['n_cells']
    ranks = dict(nodes[['lane_id', 'lane_rank_from_median']].drop_duplicates().itertuples(index=False, name=None))
    d['lane_rank'] = d['laneId'].map(ranks)
    d['s'] = (d['xc'] - roi['analysis_x_low_m'] if direction == 2 else roi['analysis_x_high_m'] - d['xc'])
    d['bin'] = ((d['frame'] - origin) // frames_per_bin).astype(int)
    inside = (d['obs_valid'] & d['lane_rank'].notna()
              & (d['s'] >= 0) & (d['s'] < M * c['cell_length_m']))
    d['node'] = -1
    d.loc[inside, 'node'] = (d.loc[inside, 'lane_rank'] * M + np.floor(d.loc[inside, 's'] / c['cell_length_m'])).astype(int)
    base = d.loc[inside & (d['bin'] >= 0) & (d['bin'] < T)].copy()
    if base.empty:
        raise ValueError('No valid trajectory rows inside ROI')
    key = ['bin', 'node']
    group = base.groupby(key, observed=True, sort=False)
    unit = 1 / fps
    sums = {'n': grouped_array(group.size(), T, N) * unit,
            'v': grouped_array(group['u'].sum(), T, N) * unit,
            'v2': grouped_array(base.assign(v2=base['u'] ** 2).groupby(key, observed=True)['v2'].sum(), T, N) * unit,
            'truck': grouped_array(group['is_truck'].sum(), T, N) * unit,
            'a': grouped_array(group['a_long'].sum(), T, N) * unit,
            'a_n': grouped_array(group['a_long'].count(), T, N) * unit}
    raw_features = traffic_features(sums, unique_vehicle_counts(base, T, N), step, c['cell_length_m'], c)
    context_features = traffic_features({k: rolling_sum(v, W) for k, v in sums.items()},
        unique_vehicle_counts(base, T, N, W), c['context_window_s'], c['cell_length_m'], c)
    context_support = rolling_sum(time_support[:, None].astype(float), W)[:, 0] == W
    for a in raw_features.values():
        a[~time_support] = np.nan
    for a in context_features.values():
        a[~context_support] = np.nan
    valid_pairs = base.loc[base['pair_valid']]
    pg = valid_pairs.groupby(key, observed=True, sort=False)
    pair_exposure = grouped_array(pg.size(), T, N) * unit
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_coverage = pair_exposure / sums['n']
        risk_mean = grouped_array(pg['inv_ttc'].sum(), T, N) * unit / pair_exposure
        mean_drac = grouped_array(pg['drac'].sum(), T, N) * unit / pair_exposure
    risk_max = grouped_array(pg['inv_ttc'].max(), T, N, np.nan)
    max_drac = grouped_array(pg['drac'].max(), T, N, np.nan)
    from highd_nb.paper.augmentation import risk_channels, METRIC_NAMES, save_pair_sample
    channels=risk_channels(valid_pairs,T,N,sums['n'],time_support,c,fps)
    metric=c.get('risk_metric','inv_ttc')
    if metric not in METRIC_NAMES: raise ValueError(f'Unknown risk_metric {metric}')
    kmetric=METRIC_NAMES.index(metric)
    y_observed=channels['risk_channels_y'][...,kmetric].astype(float)
    y_max = np.log1p(c['risk_time_scale_s'] * risk_max)
    risk_mask=channels['mask_risk_channels'][...,kmetric]
    pair_support=channels['mask_risk_channels'][...,0]
    y = np.where(risk_mask, y_observed, np.nan)
    cin, cout, lc_changes = detector_crossings(d, T, N, roi, origin, fps, step)
    events, active, onset = conflict_events(d, T, N, rec, direction, origin, fps, c)
    fa, fma, fo, fmo = future_labels(active, onset, pair_support, c)
    z = np.stack([context_features[k] for k in CORE_FEATURES], axis=-1)
    zraw = np.stack([raw_features[k] for k in CORE_FEATURES], axis=-1)
    mask_z = np.isfinite(z)
    mask_raw_z = np.isfinite(zraw)
    t = (np.arange(T) + 1) * step
    edge_index = edges[['source', 'target']].to_numpy(np.int64).T
    edge_type = edges['edge_type'].to_numpy(np.int64)
    A = np.zeros((3, N, N), np.float32)
    A[edge_type, edge_index[1], edge_index[0]] = 1  # A[type, target, source]
    adjacency_mask = A.astype(bool).any(axis=0)
    neighbor_valid = (risk_mask.astype(int) @ adjacency_mask.T) == adjacency_mask.sum(axis=1)
    current_complete = risk_mask & mask_z.all(axis=-1)
    neighborhood_complete = current_complete & ((current_complete.astype(int) @ adjacency_mask.T) == adjacency_mask.sum(axis=1))
    transition_mask = np.zeros_like(risk_mask)
    transition_mask[:-1] = neighborhood_complete[:-1] & risk_mask[1:]
    graph_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        'time_s': t.astype(np.float64), 'mask_time_support': time_support, 'y': y.astype(np.float32)[..., None],
        'y_observed_unfiltered': y_observed.astype(np.float32)[..., None],
        'y_max_observed': y_max.astype(np.float32)[..., None],
        'z': z.astype(np.float32), 'z_raw': zraw.astype(np.float32),
        'mask_y': risk_mask[..., None], 'mask_z': mask_z, 'mask_z_raw': mask_raw_z,
        'mask_transition_strict': transition_mask,
        'vehicle_exposure_s': sums['n'].astype(np.float32),
        'pair_exposure_s': pair_exposure.astype(np.float32),
        'pair_coverage': pair_coverage.astype(np.float32),
        'active_event': active, 'onset_event': onset,
        'future_active': fa, 'future_active_mask': fma,
        'future_onset': fo, 'future_onset_mask': fmo,
        'edge_index': edge_index, 'edge_type': edge_type,
        'edge_attr': edges[['delta_s_m', 'delta_y_image_m', 'weight']].to_numpy(np.float32),
        'adjacency_by_type': A, 'node_ids': nodes['node_id'].to_numpy(np.int64),
        'core_feature_names': np.array(CORE_FEATURES),
        'horizons_s': np.array(c['forecast_horizons_s'], dtype=np.float64),
        'thresholds_s': np.array(c['conflict_thresholds_s'], dtype=np.float64)}
    arrays.update(channels)
    arrays['mask_pair_support']=pair_support
    arrays['q_in_context_veh_per_h']=(rolling_sum(cin,W)*3600/c['context_window_s']).astype('float32')
    arrays['q_out_context_veh_per_h']=(rolling_sum(cout,W)*3600/c['context_window_s']).astype('float32')
    arrays['acceleration_context_mps2']=context_features['accel_mean_mps2'].astype('float32')
    np.savez_compressed(graph_dir / 'arrays.npz', **arrays)
    if c.get('export_paper_arrays',True): save_pair_sample(base,graph_dir,rec,direction,origin,fps,c)
    nodes.to_csv(graph_dir / 'nodes.csv', index=False, encoding='utf-8-sig')
    edges.to_csv(graph_dir / 'edges.csv', index=False, encoding='utf-8-sig')
    events.to_csv(graph_dir / 'events.csv', index=False, encoding='utf-8-sig')
    panel = {'bin_index': np.repeat(np.arange(T), N), 'time_s': np.repeat(t, N),
             'node_id': np.tile(np.arange(N), T), 'risk_y': y.ravel(),
             'risk_y_observed_unfiltered': y_observed.ravel(), 'risk_y_max': y_max.ravel(),
             'mean_inverse_ttc_s_inv': risk_mean.ravel(), 'max_inverse_ttc_s_inv': risk_max.ravel(),
             'mean_drac_mps2': mean_drac.ravel(), 'max_drac_mps2': max_drac.ravel(),
             'risk_valid': risk_mask.ravel(), 'pair_exposure_s': pair_exposure.ravel(),
             'pair_coverage': pair_coverage.ravel(),
             'strict_transition_valid': transition_mask.ravel(),
             'q_in_veh_per_h': (cin * 3600 / step).ravel(),
             'q_out_veh_per_h': (cout * 3600 / step).ravel(),
             'q_in_context_veh_per_h': (rolling_sum(cin, W) * 3600 / c['context_window_s']).ravel(),
             'q_out_context_veh_per_h': (rolling_sum(cout, W) * 3600 / c['context_window_s']).ravel()}
    for name, a in raw_features.items():
        panel[f'raw_{name}'] = a.ravel()
    for name, a in context_features.items():
        panel[f'context_{name}'] = a.ravel()
    if c['write_panel_csv']:
        pd.DataFrame(panel).to_csv(graph_dir / 'panel.csv.gz', index=False, compression='gzip')
    increment_diagnostics(y, risk_mask, c).to_csv(graph_dir / 'increment_diagnostics.csv', index=False)
    label_rows = []
    for h, seconds in enumerate(c['forecast_horizons_s']):
        for k, threshold in enumerate(c['conflict_thresholds_s']):
            for label_name, label, known in [('active', fa, fma), ('onset', fo, fmo)]:
                v = label[:, :, h, k][known[:, :, h, k]]
                label_rows.append({'task': label_name, 'horizon_s': seconds, 'ttc_threshold_s': threshold,
                                   'known_labels': len(v), 'positives': int(v.sum()) if len(v) else 0,
                                   'positive_fraction': float(v.mean()) if len(v) else None})
    pd.DataFrame(label_rows).to_csv(graph_dir / 'label_summary.csv', index=False)
    qc = {
        'version': VERSION, 'recording_id': rec, 'location_id': int(meta['locationId']), 'direction': direction,
        'frame_rate': fps, 'frame_origin': origin, 'duration_used_s': T * step,
        'observed_start_s': observed_start, 'supported_full_time_bins': int(time_support.sum()),
        'trailing_partial_seconds_discarded': max(0, duration - T * step),
        'T': T, 'N': N, 'n_edges': len(edges), 'roi': roi,
        'n_trajectory_rows_in_roi': len(base),
        'valid_risk_cells': int(risk_mask.sum()), 'valid_risk_fraction': float(risk_mask.mean()),
        'empty_cell_fraction': float((sums['n'] == 0).mean()),
        'observed_zero_risk_fraction': ratio(int(((risk_mean == 0) & risk_mask).sum()), int(risk_mask.sum())),
        'strict_transition_count': int(transition_mask.sum()),
        'strict_transition_fraction': float(transition_mask.mean()),
        'mean_neighbor_risk_completeness': float(neighbor_valid.mean()),
        'event_rows_across_thresholds': len(events),
        'lane_id_changes_raw': lc_changes,
        'q_equals_kv_max_abs_veh_per_h': finite_or_none(np.nanmax(np.abs(
            raw_features['edie_flow_veh_per_h'] - raw_features['density_veh_per_km'] * raw_features['speed_mean_mps'] * 3.6))),
        'recording_qc_file': f'../{rec}_recording_qc.json',
        'warnings': ['Event rows at different thresholds are alternative definitions; do not add them together.',
                     'Mean inverse TTC is a surrogate score, not a collision probability.',
                     'Brownian/Markov/nonnegative-state assumptions remain to be tested at model stage.']}
    if origin != 1:
        qc['warnings'].append('Zero-based frame numbering detected; source differs from the usual highD release.')
    if qc['valid_risk_fraction'] < .2:
        qc['warnings'].append('Low risk-state coverage: inspect 75/100 m cells and raw 2 s bins using separate output folders.')
    if qc['strict_transition_fraction'] < .01:
        qc['warnings'].append('Less than 1% strict usable transitions. Do not feed zero-filled states into unmodified LaGNA.')
    dump_json(graph_dir / 'qc.json', qc)
    schema = {'version': VERSION, 'recording_id': rec, 'location_id': int(meta['locationId']),
              'direction': direction, 'T': T, 'N': N, 'core_feature_names': CORE_FEATURES,
              'config': c, 'roi': roi,
              'adjacency_convention': 'adjacency_by_type[type,target,source]; edge_index=[source,target]',
              'edge_type_names': EDGE_NAMES,
              'time_convention': 'time_s[b]=(b+1)*step; raw state uses [time-step,time); context uses [time-W,time)',
              'missing_convention': 'NaN in saved observations; masks are mandatory. Density/flow may legitimately be 0.',
              'risk_definition': f'log1p(multiplier * exposure-weighted mean {metric}); see risk_channel_multiplier',
              'risk_metric':metric,
              'ssm_assumptions':'same-lane; MTTC constant accelerations until either predicted stop; PSD fixed available braking',
              'pair_sample_note':'fixed time sample for distribution/cases; event census uses full-rate episodes',
              'risk_scope': 'Observed, valid same-lane following pairs. Lateral edges are contextual, not side-swipe labels.',
              'future_negative_condition': 'All future bins must have adequate pair coverage; observable positives retained.',
              'no_target_leakage': 'No future covariates, full-track summaries, or event summaries are in z.',
              'not_an_original_lagna_input_file': True}
    dump_json(graph_dir / 'schema.json', schema)
    return qc


def refresh_manifest(out_dir: Path) -> None:
    rows = []
    for p in sorted(out_dir.glob('rec_*_dir*/schema.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        q = json.loads((p.parent / 'qc.json').read_text(encoding='utf-8'))
        rows.append({'graph_dir': p.parent.name, 'recording_id': s['recording_id'],
                     'location_id': s['location_id'], 'direction': s['direction'],
                     'T': s['T'], 'N': s['N'], 'valid_risk_fraction': q['valid_risk_fraction'],
                     'strict_transition_count': q['strict_transition_count']})
    pd.DataFrame(rows).to_csv(out_dir / 'manifest.csv', index=False, encoding='utf-8-sig')


def build(data_dir: Path, out_dir: Path, selection: str, c: dict, overwrite: bool) -> None:
    check_output(data_dir, out_dir)
    config_path = out_dir / 'build_config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing != {'version': VERSION, 'config': c}:
            raise ValueError('Output already contains a different config/version. Use a NEW output folder.')
    dump_json(config_path, {'version': VERSION, 'config': c})
    for rec in recordings(data_dir, selection):
        print(f'Reading recording {rec}...', flush=True)
        d, meta, qc = read_recording(data_dir, rec, c)
        dump_json(out_dir / f'{rec}_recording_qc.json', qc)
        for direction in sorted(d['direction'].unique()):
            graph_dir = out_dir / f'rec_{rec}_dir{direction}'
            if graph_dir.exists():
                if not overwrite:
                    raise FileExistsError(f'{graph_dir} exists. Use --overwrite explicitly to rebuild generated outputs.')
                if not (graph_dir / 'schema.json').exists():
                    raise ValueError(f'Refusing to remove unrecognized folder {graph_dir}')
            temporary = Path(tempfile.mkdtemp(prefix=f'.{graph_dir.name}_', dir=out_dir))
            try:
                q = build_graph(d, meta, rec, int(direction), c, temporary, qc)
                if graph_dir.exists():
                    shutil.rmtree(graph_dir)
                temporary.rename(graph_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            print(f'  {graph_dir.name}: T={q["T"]}, N={q["N"]}, '
                  f'risk coverage={q["valid_risk_fraction"]:.3f}, '
                  f'strict transitions={q["strict_transition_count"]}', flush=True)
        refresh_manifest(out_dir)
    print(f'Finished. Run verify next: {out_dir}')


def create_splits(out_dir: Path) -> None:
    manifest = pd.read_csv(out_dir / 'manifest.csv', dtype={'recording_id': str})
    locs = sorted(manifest['location_id'].unique().tolist())
    if len(locs) < 3:
        raise ValueError('At least 3 locations required for disjoint train/validation/test locations. Build more recordings.')
    dst = out_dir / 'splits'
    dst.mkdir(parents=True, exist_ok=True)
    for i, test in enumerate(locs):
        val = locs[(i + 1) % len(locs)]
        train = [x for x in locs if x not in [test, val]]
        fold = {'version': VERSION, 'train_locations': train, 'validation_locations': [val],
                'test_locations': [test],
                'train_graphs': manifest.loc[manifest.location_id.isin(train), 'graph_dir'].tolist(),
                'validation_graphs': manifest.loc[manifest.location_id == val, 'graph_dir'].tolist(),
                'test_graphs': manifest.loc[manifest.location_id == test, 'graph_dir'].tolist(),
                'note': 'Outer location holdout; deterministic next-location validation. Hyperparameters use validation only.'}
        dump_json(dst / f'leave_location_{test}.json', fold)
    print(f'{len(locs)} leakage-safe location folds saved to {dst}')


def fit_scaler(out_dir: Path, fold_path: Path) -> None:
    fold = json.loads(fold_path.read_text(encoding='utf-8'))
    if not fold['train_graphs']:
        raise ValueError('Empty training fold')
    # Train-only streaming sufficient statistics, using observation masks.
    F = len(CORE_FEATURES)
    count, total, square = np.zeros(F + 1), np.zeros(F + 1), np.zeros(F + 1)
    for name in fold['train_graphs']:
        with np.load(out_dir / name / 'arrays.npz', allow_pickle=False) as a:
            values = np.concatenate([a['y'], a['z']], axis=-1).reshape(-1, F + 1).astype(float)
            masks = np.concatenate([a['mask_y'], a['mask_z']], axis=-1).reshape(-1, F + 1)
            masks &= np.isfinite(values)
            clean = np.where(masks, values, 0)
            count += masks.sum(axis=0)
            total += clean.sum(axis=0)
            square += (clean ** 2).sum(axis=0)
    if (count == 0).any():
        raise ValueError(f'No observed training values for some features. Counts: {count.tolist()}')
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 0))
    std[std < 1e-8] = 1
    dest = fold_path.with_name(fold_path.stem + '_scaler.json')
    dump_json(dest, {'version': VERSION, 'features': ['risk_y'] + CORE_FEATURES,
                     'mean': mean.tolist(), 'std': std.tolist(), 'count': count.astype(int).tolist(),
                     'fit_graphs': fold['train_graphs'], 'fit_fold': fold_path.name,
                     'note': 'Train only; do not refit on test. Keep masks after filling normalized NaN with zero.'})
    print(f'Saved train-only scaler: {dest}')


def verify(out_dir: Path) -> None:
    records = []
    for path in sorted(out_dir.glob('rec_*_dir*/arrays.npz')):
        s = json.loads((path.parent / 'schema.json').read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as a:
            T, N = s['T'], s['N']
            assert a['y'].shape == (T, N, 1)
            assert a['z'].shape == (T, N, len(CORE_FEATURES))
            assert np.array_equal(a['mask_y'], np.isfinite(a['y']))
            assert np.array_equal(a['mask_z'], np.isfinite(a['z']))
            assert np.all(a['y'][a['mask_y']] >= 0)
            assert np.all(np.diff(a['time_s']) > 0)
            e, et = a['edge_index'], a['edge_type']
            assert e.shape[0] == 2 and len(et) == e.shape[1]
            assert np.all((e >= 0) & (e < N)) and np.all(e[0] != e[1])
            A = a['adjacency_by_type']
            assert np.all(A[et, e[1], e[0]] == 1) and A.sum() == e.shape[1]
            nodes = pd.read_csv(path.parent / 'nodes.csv')
            for source, target, k in zip(e[0], e[1], et):
                sn, tn = nodes.iloc[source], nodes.iloc[target]
                assert sn.direction == tn.direction
                if k == 0:
                    assert sn.lane_id == tn.lane_id and sn.cell_index + 1 == tn.cell_index
                elif k == 1:
                    assert sn.lane_id == tn.lane_id and sn.cell_index - 1 == tn.cell_index
                else:
                    assert sn.cell_index == tn.cell_index and abs(sn.lane_rank_from_median - tn.lane_rank_from_median) == 1
            for name in ['future_active', 'future_onset']:
                label, mask = a[name], a[name + '_mask']
                assert np.array_equal(mask, np.isfinite(label))
                assert np.isin(label[mask], [0, 1]).all()
                for i, seconds in enumerate(a['horizons_s']):
                    h = integral_ratio(float(seconds), s['config']['step_s'], 'horizon')
                    assert not mask[max(0, T - h):, :, i].any()
            assert not a['mask_transition_strict'][-1].any()
            assert not a['mask_z'][:integral_ratio(s['config']['context_window_s'], s['config']['step_s'], 'context') - 1].any()
            records.append({'graph': path.parent.name, 'status': 'PASS', 'T': T, 'N': N})
    if not records:
        raise FileNotFoundError('No arrays.npz found; run build first')
    pd.DataFrame(records).to_csv(out_dir / 'verification.csv', index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print('PASS validates internal data contracts, NOT empirical model accuracy or scientific validity.')



def collect_qc(out_dir: Path) -> None:
    """Collect small diagnostics only. Never include raw trajectories, panels or model arrays."""
    selected = []
    for pattern in ['inventory.csv', 'inventory.json', 'manifest.csv', 'verification.csv',
                    'build_config.json', '*_recording_qc.json', 'rec_*_dir*/qc.json',
                    'rec_*_dir*/schema.json', 'rec_*_dir*/nodes.csv', 'rec_*_dir*/edges.csv',
                    'rec_*_dir*/label_summary.csv', 'rec_*_dir*/increment_diagnostics.csv']:
        selected.extend(out_dir.glob(pattern))
    selected = sorted(set(selected))
    if not selected:
        raise FileNotFoundError('No QC files found; run audit/build first')
    destination = out_dir / 'qc_bundle.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(out_dir).as_posix())
    print(f'Saved {len(selected)} small QC files: {destination}')
    print('No raw tracks, event-level rows, arrays.npz or panel.csv.gz are included.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ['audit', 'build']:
        a = sub.add_parser(cmd)
        a.add_argument('--data-dir', type=Path, required=True)
        a.add_argument('--out-dir', type=Path, required=True)
        a.add_argument('--recordings', default='all', help='all or comma-separated IDs such as 01,02')
        if cmd == 'build':
            a.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
            a.add_argument('--overwrite', action='store_true')
    for cmd in ['verify', 'splits', 'fit-scaler', 'collect-qc']:
        a = sub.add_parser(cmd)
        a.add_argument('--out-dir', type=Path, required=True)
        if cmd == 'fit-scaler':
            a.add_argument('--fold', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == 'audit':
            audit(args.data_dir, args.out_dir, args.recordings)
        elif args.command == 'build':
            build(args.data_dir, args.out_dir, args.recordings, load_config(args.config), args.overwrite)
        elif args.command == 'verify':
            verify(args.out_dir)
        elif args.command == 'splits':
            create_splits(args.out_dir)
        elif args.command == 'collect-qc':
            collect_qc(args.out_dir)
        else:
            fit_scaler(args.out_dir, args.fold)
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
