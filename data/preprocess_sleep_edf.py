#!/usr/bin/env python
# %%
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mne.set_log_level(verbose=False)
# %% [markdown]
# # Sleep-EDF Preprocessing Script
# This notebook processes Sleep Cassette and Sleep Telemetry data from the Sleep-EDF Expanded database (downloaded from PhysioNet).
# It reads raw PSG and hypnogram EDF files, extracts EEG and EOG signals, segments into 30s epochs, removes wake edges,
# applies standardization, and saves them into fixed-length sequences (20 epochs per sequence) for downstream modeling.
#
# > Used version: sleep-edf-database-expanded-1.0.0 downloaded from PhysioNet.
#
# ## Dataset Description:
# |Subset|Prefix|File Example|Setting|Nights|
# |------|------|-------------|--------|-------|
# |Sleep Cassette|SC|SC4ssNE0-PSG.edf|Home sleep, healthy subjects|153|
# |Sleep Telemetry|ST|ST7ssNJ0-PSG.edf|Clinical environment, drug trial|44|
#
# ss is the subject number, and N is the night.
#
# ## Preprocessing Rules:
# - Selected channels: `EEG Fpz-Cz`, `EOG horizontal`
# - Bandpass filter: `0.3–35 Hz` (FIR)
# - Epoch length: `30s` (100 Hz → 3000 samples)
# - Label mapping (R&K standard): W=0, 1=1, 2=2, 3/4=3, R=4
# - Remove initial and tailing `Wake` stages (extend ±30min)
# - Drop remaining epochs to make total divisible by 20
# - Output format: `(n_seq, 20, 2, 3000)` and `(n_seq, 20)`
# %%
# Sleep Cassette
data_root_sc = Path("~/datasets/sleep-dataset/sleep-edf/raw/sleep-cassette").expanduser()
psg_files_sc = sorted(data_root_sc.glob("*PSG.edf"))
ann_files_sc = sorted(data_root_sc.glob("*Hypnogram.edf"))
files_sc = pd.merge(
    pd.DataFrame({"idx": [f.name[:6] for f in psg_files_sc], "psg_file": psg_files_sc}),
    pd.DataFrame({"idx": [f.name[:6] for f in ann_files_sc], "ann_file": ann_files_sc}),
    on="idx",
    how="inner",
)

# Sleep Telemetry
data_root_st = Path("~/datasets/sleep-dataset/sleep-edf/raw/sleep-telemetry").expanduser()
psg_files_st = sorted(data_root_st.glob("*PSG.edf"))
ann_files_st = sorted(data_root_st.glob("*Hypnogram.edf"))
files_st = pd.merge(
    pd.DataFrame({"idx": [f.name[:6] for f in psg_files_st], "psg_file": psg_files_st}),
    pd.DataFrame({"idx": [f.name[:6] for f in ann_files_st], "ann_file": ann_files_st}),
    on="idx",
    how="inner",
)

# Merge
files = pd.concat([files_sc, files_st], ignore_index=True)
print(f"✅ Matched total EDF-Hypnogram file pairs: {len(files)}")
# %% [markdown]
# ## Output Configuration
# - Save to structured directories: `seq/subject-id/` and `ann/subject-id/`

save_root = Path("processed/sleep-edf").expanduser()
save_root.mkdir(parents=True, exist_ok=True)

ch_names = ["EEG Fpz-Cz", "EOG horizontal"]
epoch_sec = 30
seq_len = 20
sample_rate = 100

label2id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

n_total_seq = 0

# %% [markdown]
# ## Processing Loop: For each subject-night, segment, normalize, and save

for _, row in tqdm(files.iterrows(), total=len(files), desc="Processing subjects"):
    with warnings.catch_warnings():
        # disable RuntimeWarnings from loading raw PSG files
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
        raw = read_raw_edf(row.psg_file, preload=True, verbose=False)  # Load raw PSG file
    raw.pick(picks=ch_names, verbose=False)  # Select EEG and EOG
    raw.filter(0.3, 35, fir_design="firwin", verbose=False)  # Bandpass filter

    annotation = mne.read_annotations(row.ann_file)  # Load hypnogram
    raw.set_annotations(annotation, emit_warning=False)  # Attach annotations

    # get 30-second epochs and remove unknown and movement sleep events
    # events_train: [sample_index, 0, event_id], event_id: {name: id}
    events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0)
    event_id.pop("Sleep stage ?", None)
    event_id.pop("Movement time", None)

    # Epochs use closed interval, so we need to subtract 1/sfreq from tmax
    tmax = 30.0 - 1.0 / raw.info["sfreq"]
    # each epoch contains 30 seconds of data [n_chs, n_samples]
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None)
    epoch_annots = epochs.get_annotations_per_epoch()
    # sleep stage labels
    labels = []
    for annot in epoch_annots:
        # non-overlapping annotation has only one element
        onset, duration, description = annot[0]
        labels.append(label2id[description])

    # only # keep 30 minutes (60 epochs) before and after the first and last sleep stage (i.e., non-Wake)
    labels = np.array(labels)
    non_wake_indices = np.where(labels > 0)[0]
    start = max(0, non_wake_indices[0] - 60)
    end = min(len(labels), non_wake_indices[-1] + 60)
    # clip epochs and labels
    epochs_data = np.array([epoch for epoch in epochs[start:end]])
    labels_data = labels[start:end]

    # clip to a multiple of 20 epochs
    index = len(epochs_data)
    while index % 20 != 0:
        index -= 1
    epochs_data = epochs_data[:index]
    labels_data = labels_data[:index]

    # z-score standardization
    epochs_data = epochs_data.transpose(0, 2, 1)
    epochs_data = epochs_data.reshape(-1, 2)
    std = StandardScaler()
    epochs_data = std.fit_transform(epochs_data)

    epochs_data = epochs_data.reshape(-1, 3000, 2)
    epochs_data = epochs_data.transpose(0, 2, 1)

    # n_seq, n_epoch, n_ch, n_sample
    epochs_seq = epochs_data.reshape(-1, 20, 2, 3000)
    # n_seq, n_epoch
    labels_seq = labels_data.reshape(-1, 20)

    # Create save paths
    epoch_path = save_root / "data" / row.idx
    label_path = save_root / "label" / row.idx
    epoch_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    # Save each sequence
    for seq_idx, (x, y) in enumerate(zip(epochs_seq, labels_seq, strict=False)):
        np.save(epoch_path / f"{row.idx}-{seq_idx}.npy", x)
        np.save(label_path / f"{row.idx}-{seq_idx}.npy", y)
        n_total_seq += 1
print(f"✅ Total saved sequences: {n_total_seq:,}")
