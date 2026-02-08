import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset

import utils

logger = utils.logger.get_logger(__name__)

class SleepRecords(Dataset):
    """
    Dataset for loading sleep EEG/EOG recordings from a DataFrame.

    Supports two modes:
        - 'segment' mode (default): Each sample is a single short segment (e.g., 30s)
        - 'subject' mode: All segments from a subject are concatenated into one sample,
                          and optionally truncated or zero-padded to `num_epochs`

    Args:
        df (pd.DataFrame):
            A DataFrame containing the following columns:
                - 'data_file': Path to the .npy EEG/EOG data file (relative to `data_root`)
                - 'label_file': Path to the corresponding label .npy file
                - 'domain_index': Integer domain/class index (e.g., for domain adaptation)
                - 'subject_name': Identifier used to group segments into full-night samples
                - 'dataset_name': Optional dataset identifier (e.g., SleepEDF, ISRUC)

        data_root (str | Path):
            Root directory that all relative paths in `df` are joined with.

        mode (str):
            Either 'segment' (default) or 'subject'.
            - 'segment': each item is a single epoch (signal, label, domain).
            - 'subject': each item is a full-night recording assembled from multiple segments,
                         returned as (signal, label, domain, length), with zero-padding if needed.

        full_night (bool):
            Deprecated (use `mode='subject'` instead). Kept for backward compatibility.

        num_epochs (int):
            Maximum number of epochs to return per subject (used in 'subject' mode only).
            If a subject has fewer than this, the sample is zero-padded; if more, it's truncated.

    Returns:
        In 'segment' mode:
            signal: np.ndarray of shape (T, C, F)
            label: np.ndarray of shape (T,)
            domain_index: int

        In 'subject' mode:
            signal: np.ndarray of shape (num_epochs, C, F)
            label: np.ndarray of shape (num_epochs,)
            domain_index: int
            actual_len: int — number of original (pre-pad) epochs for this subject
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path = Path("data/SleepDG-20"),
        mode: str = "segment",
        full_night: bool = False,
        num_epochs: int = 1200,
    ) -> None:
        assert mode in {"segment", "subject"}, f"Unsupported mode: {mode}"
        self.df = df
        self.mode = mode
        self.full_night = full_night
        self.num_epochs = num_epochs
        self.data_root = Path(data_root).expanduser()

        if self.mode == "subject":
            self.groups = self.df.groupby("subject_name")
            self.subjects = list(self.groups.groups)
            logger.info(f"Subject mode: {len(self.subjects):,} subjects")
        else:
            self.groups = self.subjects = None
            logger.info(f"Segment mode: {len(self.df):,} samples")

    def __len__(self) -> int:
        return len(self.subjects) if self.mode == "subject" else len(self.df)

    def _load_subject(self, idx: int):
        subject = self.subjects[idx]
        group_df = self.groups.get_group(subject).copy()

        def extract_index(path: str) -> int:
            try:
                return int(Path(path).stem.split("-")[-1])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid filename format: {path}") from e

        group_df["epoch_index"] = group_df["data_file"].apply(extract_index)
        group_df = group_df.sort_values("epoch_index")

        signals = [np.load(self.data_root / row["data_file"]).astype(np.float32) for _, row in group_df.iterrows()]
        labels = [np.load(self.data_root / row["label_file"]).astype(np.int64) for _, row in group_df.iterrows()]

        signal = np.concatenate(signals, axis=0)
        label = np.concatenate(labels, axis=0)
        domain = int(group_df["domain_index"].iloc[0]) if "domain_index" in group_df.columns else 0

        length = signal.shape[0]

        if length > self.num_epochs:
            signal, label = signal[: self.num_epochs], label[: self.num_epochs]
        elif length < self.num_epochs:
            pad = self.num_epochs - length
            signal = np.pad(signal, ((0, pad), (0, 0), (0, 0)), constant_values=0)
            label = np.pad(label, (0, pad), constant_values=0)

        return signal, label, domain, length

    def _load_segment(self, idx: int):
        row = self.df.iloc[idx]
        signal = np.load(self.data_root / row["data_file"]).astype(np.float32)
        label = np.load(self.data_root / row["label_file"]).astype(np.int64)
        domain = row.get("domain_index", 0)
        return signal, label, domain

    def __getitem__(self, idx: int):
        return self._load_subject(idx) if self.mode == "subject" else self._load_segment(idx)


def limit_subjects_per_dataset(df: pd.DataFrame, limits: dict[str, int], legacy_sort: bool = True) -> pd.DataFrame:
    """Limit the number of unique subjects per dataset.

    Args:
        df (pd.DataFrame): The full records DataFrame.
        limits (Dict[str, int]): Dictionary mapping dataset name to subject limit.
            e.g., {"shhs1": 150, "cinc-2018": 150}
        legacy_sort (bool): If True, use string dictionary order (legacy, like psg_f_names.sort()).
                            If False, use natural sort (e.g., SC2 < SC10).

    Returns:
        pd.DataFrame: Filtered DataFrame with limited subjects per specified dataset.
    """
    filtered_records = []
    for name, group in df.groupby("dataset_name"):
        if name in limits:
            n = limits[name]
            unique_subjects = group["subject_name"].drop_duplicates()
            if legacy_sort:
                selected_subjects = unique_subjects.sort_values().iloc[:n]
            else:
                selected_subjects = unique_subjects.iloc[:n]
            group = group[group["subject_name"].isin(selected_subjects)]
        filtered_records.append(group)
    return pd.concat(filtered_records).reset_index(drop=True)


class SleepDataModule:
    """Utility class for loading multi-domain sleep datasets with train/val/test splits.

    Supports domain generalization setup with configurable source and target domains.

    Args:
        data_root (Path): Root directory where all domain folders are stored.
        target_domains (str): The domain used for testing (target).
        source_domains (List[str], optional): List of domains used for training/validation.
            If None, all other domains are treated as source.
        batch_size (int): Batch size for all DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        split_ratio (float): Proportion of data used for training/validation vs. testing.
        domain_generation (bool): If True, set up for domain generalization (DG) with
            source and target domains. If False, all selected target domains are used for
            standard supervised learning.
    """

    all_domains = ["sleep-edf", "hmc", "isruc", "shhs1", "cinc-2018"]

    def __init__(
        self,
        data_root: Path = Path("data/SleepDG-20"),
        target_domains: str | list[str] = "cinc-2018",
        source_domains: list[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio=0.8,
        domain_generation=True,
        **kwargs: dict[str, any],
    ):
        # logging unused kwargs
        if kwargs:
            logger.warning(f"Unused kwargs: {kwargs.keys()}")
        self.data_root = Path(data_root).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 5

        target_domains = [target_domains] if isinstance(target_domains, str) else target_domains

        assert all(
            d.lower() in self.all_domains for d in target_domains
        ), f"Invalid target domains: {target_domains}. Must be one of {self.all_domains}"
        source_domains = source_domains or [d for d in self.all_domains if d not in target_domains]
        assert all(
            d.lower() in self.all_domains for d in source_domains
        ), f"Invalid source domains: {source_domains}. Must be one of {self.all_domains}"
        self.target_domains = target_domains

        if domain_generation:
            all_selected = target_domains + source_domains
            self.source_domains = source_domains
        else:
            all_selected = target_domains
            self.source_domains = []

        self.num_source_domains = len(self.source_domains)
        self.num_target_domains = len(self.target_domains)
        self.num_domains = len(all_selected)

        # csv_file = self.data_root / "records_matched.csv"
        csv_file = self.data_root / "records.csv"
        self.records = pd.read_csv(csv_file)

        # only keep limited subjects for the following datasets
        subject_limits = {"shhs1": 150, "cinc-2018": 150}
        self.records = limit_subjects_per_dataset(self.records, subject_limits, legacy_sort=False)

        # only index source domains, others (target and unused domains) are -1
        domain_mapping = {d: (source_domains.index(d) if d in source_domains else -1) for d in self.all_domains}
        self.records["domain_index"] = self.records["dataset_name"].map(domain_mapping)

        source_df = self.records[self.records["dataset_name"].isin(source_domains)]
        source_df = source_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # compute the distribution of each source domain
        source_df["label_dist"] = source_df["label_dist"].apply(json.loads)
        source_group = source_df.groupby("dataset_name")
        source_dist_group = source_group["label_dist"].apply(lambda dists: np.mean(np.vstack(dists), axis=0))
        domain_counts_group = source_group.size()

        self.domain_dist = np.zeros((len(self.source_domains), self.num_classes), dtype=np.float32)
        self.source_dist = np.zeros((self.num_classes,), dtype=np.float32)
        self.num_source_samples = np.zeros((len(self.source_domains), self.num_classes), dtype=np.float32)

        for domain in self.source_domains:
            idx = domain_mapping[domain]
            self.domain_dist[idx] = source_dist_group[domain]
            self.source_dist += source_dist_group[domain] * domain_counts_group[domain]
            self.num_source_samples[idx] = source_dist_group[domain] * domain_counts_group[domain]
        self.source_dist /= domain_counts_group.sum()

        if domain_generation:
            df = self.records[self.records["dataset_name"].isin(self.source_domains)]
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            split = int(len(df) * split_ratio)
            self.train_df, self.val_df = df[:split], df[split:]
            self.test_df = self.records[self.records["dataset_name"].isin(self.target_domains)]
            logger.info(f"Transfer from {source_domains} to {self.target_domains}")
        else:
            df = self.records[self.records["dataset_name"].isin(self.target_domains)]
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            split_tv = int(len(df) * split_ratio)
            self.train_df, self.test_df = df[:split_tv], df[split_tv:]
            split_v = int(len(self.train_df) * 0.1)
            self.val_df, self.train_df = self.train_df[:split_v], self.train_df[split_v:]
            logger.info(f"Learning from {self.target_domains} with no source domains")

    def get_data_loader(self) -> tuple[dict[str, DataLoader], int]:

        def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
            return DataLoader(
                SleepRecords(df.reset_index(drop=True), data_root=self.data_root),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=True,
            )

        logger.info(f"Data Loaders: train={len(self.train_df):,}, val={len(self.val_df):,}, test={len(self.test_df):,}")
        loaders = {
            "train": make_loader(self.train_df, shuffle=True),
            "val": make_loader(self.val_df, shuffle=False),
            "test": make_loader(self.test_df, shuffle=False),
        }

        return loaders

