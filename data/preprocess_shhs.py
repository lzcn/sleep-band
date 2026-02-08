#!/usr/bin/env python
# %%
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mne.set_log_level(verbose=False)
# %%
# ---- Configuration ----
data_root = Path("~/datasets/sleep-dataset/shhs").expanduser()
save_root = Path("processed/shhs1").expanduser()

psg_dir = data_root / "files/polysomnography" / "edfs" / "shhs1"
ann_dir = data_root / "files/polysomnography" / "annotations-events-profusion" / "shhs1"


ch_names = ["EEG", "EOG(L)"]  # used channels
sfreq = 100  # signal sampling frequency
seq_len = 20  # number of epochs in a sequence
epoch_sec = 30  # length of an epoch in seconds
# %%
# ---- Match PSG and annotation files ----
psg_files = sorted(psg_dir.glob("*.edf"))
ann_files = sorted(ann_dir.glob("*.xml"))
files = pd.merge(
    pd.DataFrame({"nssr_id": [f.name[:12] for f in psg_files], "psg_file": psg_files}),
    pd.DataFrame({"nssr_id": [f.name[:12] for f in ann_files], "ann_file": ann_files}),
    on="nssr_id",
    how="inner",
)
print(f"✅ Matched EDF-XML file pairs: {len(files):,}")
files.head()

# %%
n_total_seq = 0
n_samples_epoch = epoch_sec * sfreq
for _, row in tqdm(files.iterrows(), total=len(files), desc="Processing subjects"):
    nssr_id, psg_file, ann_file = row.nssr_id, row.psg_file, row.ann_file

    # --- Load and preprocess PSG signal ---
    with warnings.catch_warnings():
        # disable RuntimeWarnings from loading raw PSG files
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
        raw = read_raw_edf(row.psg_file, preload=True, verbose=False)  # Load raw PSG file
    raw.pick(picks=ch_names, verbose=False)
    raw.resample(sfreq)
    raw.filter(0.3, 35, fir_design="firwin", verbose=False)

    # extract and standardize signal (num_epochs * n_samples_epoch, n_channels)
    signal = raw.get_data(picks=ch_names).T
    signal = StandardScaler().fit_transform(signal)

    # Extract sleep stage labels (num_epochs,)
    stages = np.array([int(e.text) for e in ET.parse(row.ann_file).iter("SleepStage")])
    # all should be matched
    if len(stages) * n_samples_epoch != signal.shape[0]:
        print(f"❌ {nssr_id} length mismatch: {len(stages) * n_samples_epoch} != {signal.shape[0]}")

    # merge and reassign sleep stages
    stages[stages == 4] = 3
    stages[stages == 5] = 4

    valid_mask = (stages >= 0) & (stages <= 4)
    valid_stage = stages[valid_mask]

    # Truncate the signal to a multiple of epochs
    n_samples = (signal.shape[0] // n_samples_epoch) * n_samples_epoch
    signal = signal[:n_samples].reshape(-1, n_samples_epoch, len(ch_names))
    signal = signal[valid_mask[: n_samples // n_samples_epoch]]

    # Reshape to [n_seq, 20, channels, epoch_len]
    n_epochs = (signal.shape[0] // seq_len) * seq_len
    signal = signal[:n_epochs].reshape(-1, seq_len, n_samples_epoch, len(ch_names))
    data_seq = signal.transpose(0, 1, 3, 2)

    # Extract sleep stage labels
    stages = valid_stage[:n_epochs].reshape(-1, seq_len)

    # Create save paths
    seq_path = save_root / "data" / nssr_id
    ann_path = save_root / "label" / nssr_id
    seq_path.mkdir(parents=True, exist_ok=True)
    ann_path.mkdir(parents=True, exist_ok=True)
    # Save each sequence
    for idx, (x, y) in enumerate(zip(data_seq, stages, strict=False)):
        np.save(seq_path / f"{nssr_id}-{idx}.npy", x)
        np.save(ann_path / f"{nssr_id}-{idx}.npy", y)
        n_total_seq += 1
print(f"✅ Total saved sequences: {n_total_seq:,}")

# %%
