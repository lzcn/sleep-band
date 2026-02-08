#!/usr/bin/env python
# %%
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mne.set_log_level(verbose=False)


# %%
def create_edf_symlinks_from_psg(psg_files: list[Path]) -> list[Path]:
    """Create .edf symlinks pointing to given .rec files.

    Parameters
    ----------
    psg_files : list of Path
        List of original .rec file paths.

    Returns:
    -------
    edf_links : list of Path
        List of created or existing .edf symlink paths.
    """
    edf_links = []
    for rec_path in psg_files:
        edf_link = rec_path.with_suffix(".edf")
        if not edf_link.exists():
            edf_link.symlink_to(rec_path)
        edf_links.append(edf_link)
    print(f"Created {len(edf_links)} .edf symlinks.")
    return edf_links


def clean_symlinks(link_paths: list[Path]):
    count = 0
    for path in link_paths:
        if path.is_symlink():
            path.unlink()
            count += 1
    print(f"Cleaned up {count} symlinks.")


# %%
# ---- Configuration ----
# saved format: subject-xx-yyy-zz, where xx: subgroup, yyy: subject id, zz: session id
# file format: subject-xx-yyy-zz-idx.npy, where idx: sequence index
data_root = Path("~/datasets/sleep-dataset/ISRUC-SLEEP/").expanduser()
save_root = Path("processed/isruc").expanduser()

# Subgroup I: data from 100 subjects, one recording session per subject
data_root_1 = data_root / "Subgroup-I"
subject_ids_1 = [f"subject-01-{i:03d}-01" for i in range(1, 101)]
psg_files_1 = [data_root_1 / f"subject-{i:03d}" / f"{i}" / f"{i}.rec" for i in range(1, 101)]
ana_files_1 = [data_root_1 / f"subject-{i:03d}" / f"{i}" / f"{i}_1.txt" for i in range(1, 101)]

# Subgroup II: data from 8 subjects, two recording sessions were performed per subject
data_root_2 = data_root / "Subgroup-II"
subject_ids_2_1 = [f"subject-02-{i:03d}-01" for i in range(1, 9)]
subject_ids_2_2 = [f"subject-02-{i:03d}-02" for i in range(1, 9)]
psg_files_2_1 = [data_root_2 / f"subject-{i:03d}" / f"{i}/1" / "1.rec" for i in range(1, 9)]
ana_files_2_1 = [data_root_2 / f"subject-{i:03d}" / f"{i}/1" / "1_1.txt" for i in range(1, 9)]
psg_files_2_2 = [data_root_2 / f"subject-{i:03d}" / f"{i}/2" / "2.rec" for i in range(1, 9)]
ana_files_2_2 = [data_root_2 / f"subject-{i:03d}" / f"{i}/2" / "2_1.txt" for i in range(1, 9)]

# Subgroup III: data collected from one recording session related to 10 healthy subjects
data_root_3 = data_root / "Subgroup-III"
subject_ids_3 = [f"subject-03-{i:03d}-01" for i in range(1, 11)]
psg_files_3 = [data_root_3 / f"subject-{i:03d}" / f"{i}" / f"{i}.rec" for i in range(1, 11)]
ana_files_3 = [data_root_3 / f"subject-{i:03d}" / f"{i}" / f"{i}_1.txt" for i in range(1, 11)]

subject_ids = subject_ids_1 + subject_ids_2_1 + subject_ids_2_2 + subject_ids_3
psg_files = psg_files_1 + psg_files_2_1 + psg_files_2_2 + psg_files_3
ana_files = ana_files_1 + ana_files_2_1 + ana_files_2_2 + ana_files_3

assert all([f.exists() for f in psg_files]), "Some PSG files do not exist"
assert all([f.exists() for f in ana_files]), "Some ANN files do not exist"

# create symlinks for .rec files to make read_raw_edf work
psg_files = create_edf_symlinks_from_psg(psg_files)
df = pd.DataFrame({"subject_id": subject_ids, "psg_file": psg_files, "ana_file": ana_files})
pair_files = [(a, b) for a, b in zip(psg_files, ana_files, strict=False) if a.exists() and b.exists()]
# %%
label2id = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}
# %%
ch_names = ["F4-A1", "LOC-A2"]  # EEG and EOG channels used in SleepDG paper
# ch_names = ["C4-A1", "LOC-A2"]  # NOTE: changed to SHHS EEG
# fallback = {"EEG": ["C4-A1", "C3-A2", "F4-A1", "F3-A2", "O1-A2", "O2-A1"], "EOG": ["LOC-A2", "ROC-A1"]}
n_total_seq = 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    # load channels and filter
    raw = read_raw_edf(row.psg_file, preload=True, verbose=False)
    # inplace resampling and filtering
    raw.resample(sfreq=100)
    raw.filter(0.3, 35, fir_design="firwin")
    # raw.pick(ch_names)
    # psg_array = raw.get_data().T
    psg_array = raw.to_data_frame().values[:, 1:]  # remove time column
    psg_array = psg_array[:, [5, 0]]  # NOTE: use the default channels in SleepDG paper
    psg_array = StandardScaler().fit_transform(psg_array)

    # truncate to 30s epochs
    i = psg_array.shape[0] % (30 * 100)
    if i > 0:
        psg_array = psg_array[:-i, :]
    psg_array = psg_array.reshape(-1, 30 * 100, 2)

    # truncate to multiple 20 epochs
    a = psg_array.shape[0] % 20
    if a > 0:
        psg_array = psg_array[:-a, :, :]
    psg_array = psg_array.reshape(-1, 20, 30 * 100, 2)

    # n_seq, n_epoch, n_ch, n_sample
    epochs_seq = psg_array.transpose(0, 1, 3, 2)

    labels_list = []
    for line in row.ana_file.read_text(encoding="utf-8").splitlines():
        line_str = line.strip()
        if line_str != "":
            labels_list.append(label2id[line_str])
    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    labels_seq = labels_array.reshape(-1, 20)

    unique_id = row.subject_id

    ann_dir = save_root / "label" / unique_id
    seq_dir = save_root / "data" / unique_id
    ann_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    for idx, (x, y) in enumerate(zip(epochs_seq, labels_seq, strict=False)):
        np.save(seq_dir / f"{unique_id}-{idx}.npy", x)
        np.save(ann_dir / f"{unique_id}-{idx}.npy", y)
        n_total_seq += 1

# %%
clean_symlinks(psg_files)
