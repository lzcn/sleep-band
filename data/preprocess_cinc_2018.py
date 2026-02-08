#!/usr/bin/env python

# %%
import warnings
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from wfdb.processing import resample_multichan

warnings.filterwarnings("ignore")


# %%
# Define paths: change these to your local dataset and desired save location
data_root = Path("~/datasets/sleep-dataset/challenge-2018/1.0.0/training").expanduser()
save_root = Path("processed/cinc-2018").expanduser()


records = [(line.rstrip("/")) for line in (data_root / "RECORDS").read_text().splitlines()]
record_names = [str(data_root / f"{r}/{r}") for r in records]

n_total_seq = 0
index_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
# %% Save each record as a sequence of 20 epochs
for record, record_name in tqdm(
    zip(records, record_names, strict=False), total=len(records), desc="Processing subjects"
):
    signals, fields = wfdb.rdsamp(record_name, channel_names=["C3-M2", "E1-M2"])
    ann = wfdb.rdann(record_name, "arousal")

    # Downsample from 200Hz to 100Hz
    signals, ann = resample_multichan(signals, ann, fs=fields["fs"], fs_target=100)

    # Bandpass filter: 0.3 - 35Hz
    b, a = signal.butter(8, [0.006, 0.7], "bandpass")
    signals = signal.filtfilt(b, a, signals, axis=0)

    # Trim from first annotation onward
    signals = signals[ann.sample[0] :, :]
    signals = signals[: signals.shape[0] - signals.shape[0] % 3000]
    # Normalize signals
    signals = StandardScaler().fit_transform(signals)
    signals = signals.reshape(-1, 3000, 2).transpose(0, 2, 1)
    num_epochs = signals.shape[0]

    events = [(ann.sample[i] - ann.sample[0], label) for i, label in enumerate(ann.aux_note) if label in index_map]
    if events:
        indices, labels = zip(*events)
        indices = np.array(indices)
        labels = np.array(labels)
    else:
        indices, labels = np.array([]), np.array([])
    # Generate epoch-level labels
    epoch_labels = []
    for k in range(len(events) - 1):
        n = (indices[k + 1] - indices[k]) // 3000
        epoch_labels.extend([index_map[labels[k]]] * n)
    # Append last label to fill the remaining epochs
    epoch_labels.extend([index_map[labels[-1]]] * (num_epochs - len(epoch_labels)))
    epoch_labels = np.array(epoch_labels[:num_epochs])

    # Drop incomplete last sequence
    total_epochs = signals.shape[0] - signals.shape[0] % 20
    signals = signals[:total_epochs]
    epoch_labels = epoch_labels[:total_epochs]

    # Reshape into sequences of 20 epochs
    epochs_seq = signals.reshape(-1, 20, 2, 3000)
    labels_seq = epoch_labels.reshape(-1, 20)
    # Create save paths
    seq_path = save_root / "data" / record
    ann_path = save_root / "label" / record
    seq_path.mkdir(parents=True, exist_ok=True)
    ann_path.mkdir(parents=True, exist_ok=True)

    # Save each sequence
    for indices, (x, labels) in enumerate(zip(epochs_seq, labels_seq, strict=False)):
        np.save(seq_path / f"{record}-{indices}.npy", x)
        np.save(ann_path / f"{record}-{indices}.npy", labels)
        n_total_seq += 1
print(f"✅ Total saved sequences: {n_total_seq:,}")


# %%
