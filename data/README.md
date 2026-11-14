# Dataset Directory

Please place the raw highD vehicle trajectory CSV files in this directory or in `highd-dataset-v1.0/data/`.

- Official highD website: https://highd-dataset.com/
- Expected files per recording `XX`:
  - `XX_tracks.csv`
  - `XX_tracksMeta.csv`
  - `XX_recordingMeta.csv`
  - `XX_highway.png`

For immediate software verification without downloading the multi-gigabyte dataset, run `00_SelfTest.ipynb` or `python tools/doctor.py --device auto`, which use built-in synthetic fixtures.
