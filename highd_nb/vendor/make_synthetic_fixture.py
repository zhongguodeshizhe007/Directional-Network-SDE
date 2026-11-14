#!/usr/bin/env python3
"""Create ARTIFICIAL, tiny highD-shaped CSVs for software tests only.
These are not highD records and must never be used as empirical paper results.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def create_fixture(root: Path, n_recordings: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'SYNTHETIC_NOT_HIGHD.txt').write_text(
        'ARTIFICIAL SOFTWARE TEST FIXTURE. Not real highD data. Not research results.\n', encoding='utf-8')
    fps, duration, road_length = 25, 24, 400.0
    for record in range(1, n_recordings + 1):
        data = []
        offset_v = (record - 1) * .2
        for direction in [1, 2]:
            sign = -1 if direction == 1 else 1
            # Separate carriageways: lane IDs follow the official global marking convention.
            for lane in ([2, 3] if direction == 1 else [5, 6]):
                yc = {2: 2, 3: 6, 5: 14, 6: 18}[lane]
                base_id = direction * 100 + lane * 10
                # Median-side lane has one closing episode; the other lane has equal-speed followers.
                critical = lane in [3, 5]
                specifications = ([(80, 6 + offset_v, duration), (60, 7 + offset_v, 14.8),
                                   (200, 4 + offset_v, duration), (250, 4 + offset_v, duration)]
                                  if critical else [(40, 5 + offset_v, duration),
                                                     (100, 5 + offset_v, duration),
                                                     (200, 5 + offset_v, duration)])
                for f in range(fps * duration):
                    t = f / fps
                    visible = [(base_id + j + 1, s0 + speed * t, speed, j)
                               for j, (s0, speed, end) in enumerate(specifications)
                               if t < end and 0 <= s0 + speed * t <= road_length]
                    visible.sort(key=lambda x: x[1], reverse=True)
                    for order, (ident, s, u, j) in enumerate(visible):
                        pred = visible[order - 1] if order else None
                        gap = pred[1] - s - 4 if pred else 0
                        dv = u - pred[2] if pred else 0
                        ttc = gap / dv if pred and abs(dv) > 1e-8 else 0
                        xc = road_length - s if direction == 1 else s
                        data.append({'frame': f + 1, 'id': ident, 'x': xc - 2, 'y': yc - 1,
                                     'width': 4., 'height': 2., 'xVelocity': sign * u, 'yVelocity': 0.,
                                     'xAcceleration': 0., 'yAcceleration': 0.,
                                     'frontSightDistance': road_length - s, 'backSightDistance': s,
                                     'dhw': gap, 'thw': gap / u if pred else 0., 'ttc': ttc,
                                     'precedingXVelocity': sign * pred[2] if pred else 0.,
                                     'precedingId': pred[0] if pred else 0, 'followingId': 0,
                                     'laneId': lane, '_direction': direction,
                                     '_class': 'Truck' if j == 2 else 'Car'})
        d = pd.DataFrame(data).sort_values(['id', 'frame']).reset_index(drop=True)
        meta_rows = []
        for ident, rows in d.groupby('id'):
            meta_rows.append({'id': ident, 'width': 4., 'height': 2.,
                              'initialFrame': int(rows.frame.min()), 'finalFrame': int(rows.frame.max()),
                              'numFrames': len(rows), 'class': rows['_class'].iloc[0],
                              'drivingDirection': int(rows['_direction'].iloc[0]),
                              'meanXVelocity': rows.xVelocity.abs().mean(),
                              'minTTC': rows.ttc.min(), 'numLaneChanges': 0})
        tag = f'{record:02d}'
        d.drop(columns=['_direction', '_class']).to_csv(root / f'{tag}_tracks.csv', index=False)
        pd.DataFrame(meta_rows).to_csv(root / f'{tag}_tracksMeta.csv', index=False)
        pd.DataFrame([{'id': record, 'frameRate': fps, 'locationId': record,
                       'duration': duration, 'upperLaneMarkings': '0;4;8', 'lowerLaneMarkings': '12;16;20',
                       'numVehicles': len(meta_rows), 'speedLimit': 33.3333,
                       'month': 'SYNTHETIC', 'weekDay': 'SYNTHETIC', 'startTime': '00:00'}]).to_csv(
            root / f'{tag}_recordingMeta.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--recordings', type=int, default=3)
    args = parser.parse_args()
    create_fixture(args.out_dir, args.recordings)
    print(f'Artificial fixtures created in {args.out_dir}. NOT real highD data.')
