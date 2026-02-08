#!/usr/bin/env python
# %% Preprocessing for HMC-Sleep-Staging dataset
import warnings
from pathlib import Path

import mne
import numpy as np
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mne.set_log_level(verbose=False)
# %%
data_root = Path("~/datasets/sleep-dataset/hmc-sleep-staging/1.1/").expanduser()
save_root = Path("processed/hmc")

# load lines
records = (data_root / "RECORDS").read_text().splitlines()  # recordings/SNxxx.edf
data_files = [data_root / rec for rec in records]
record_names = [f.stem for f in data_files]
label_files = [f.with_name(f"{s}_sleepscoring.edf") for f, s in zip(data_files, record_names, strict=False)]

# Label map
label2id = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
    "Lights off@@EEG F4-A1": 0,
}
# %%
n_total_seq = 0
ch_names = ["EEG F4-M1", "EOG E1-M2"]
for subject_id, data_file, label_file in tqdm(
    zip(record_names, data_files, label_files, strict=False), total=len(record_names), desc="Processing subjects"
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
        raw = read_raw_edf(data_file, preload=True)
    raw.pick(picks=ch_names, verbose=False)  # Select EEG and EOG
    raw.resample(sfreq=100)

    raw.filter(0.3, 35, fir_design="firwin", verbose=False)
    annotation = mne.read_annotations(label_file)
    raw.set_annotations(annotation, emit_warning=False)

    events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0)
    event_id = {k: v for k, v in event_id.items() if "Light" not in k}

    tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None)

    epoch_annots = epochs.get_annotations_per_epoch()
    epoch_data = np.array([epoch for epoch in epochs])
    epoch_label = np.array([label2id[annot[0][2]] for annot in epoch_annots])

    # Trim to multiple of 20 epochs
    trim_len = len(epoch_data) - len(epoch_data) % 20
    epoch_data = epoch_data[:trim_len]
    epoch_label = epoch_label[:trim_len]

    epoch_data = epoch_data.transpose(0, 2, 1).reshape(-1, 2)
    epoch_data = StandardScaler().fit_transform(epoch_data).reshape(-1, 3000, 2).transpose(0, 2, 1)

    epochs_seq = epoch_data.reshape(-1, 20, 2, 3000)
    labels_seq = epoch_label.reshape(-1, 20)

    seq_path = save_root / "data" / subject_id
    ann_path = save_root / "label" / subject_id
    seq_path.mkdir(parents=True, exist_ok=True)
    ann_path.mkdir(parents=True, exist_ok=True)

    # Save each sequence
    for idx, (x, y) in enumerate(zip(epochs_seq, labels_seq, strict=False)):
        np.save(seq_path / f"{subject_id}-{idx}.npy", x)
        np.save(ann_path / f"{subject_id}-{idx}.npy", y)
        n_total_seq += 1

print(f"✅ Total saved sequences: {n_total_seq:,}")
