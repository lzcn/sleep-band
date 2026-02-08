from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def extract_records(data_root: Path = Path("processed")):
    """Helper function for saveing dataset information to records.csv."""

    def get_label_distribution(path, n_classes=5):
        labels = np.load(data_root / path)
        counts = np.bincount(labels, minlength=n_classes)
        dist = counts / counts.sum()
        return dist.tolist()  # Convert to list for DataFrame compatibility

    all_domains = ["sleep-edf", "hmc", "isruc", "shhs1", "cinc-2018"]
    num_classes = 5
    rows = []

    for name in all_domains:
        dataset_path = data_root / name
        data_dirs = sorted((dataset_path / "data").iterdir())
        label_dirs = sorted((dataset_path / "label").iterdir())

        for data_dir, label_dir in zip(data_dirs, label_dirs, strict=False):
            subject_name = data_dir.name
            data_files = sorted(data_dir.glob("*.npy"), key=lambda f: int(f.stem.split("-")[-1]))
            label_files = sorted(label_dir.glob("*.npy"), key=lambda f: int(f.stem.split("-")[-1]))

            rows.extend(
                {
                    "dataset_name": name,
                    "subject_name": subject_name,
                    "data_file": str(df.relative_to(dataset_path)),
                    "label_file": str(lf.relative_to(dataset_path)),
                }
                for df, lf in zip(data_files, label_files, strict=False)
            )

    df = pd.DataFrame(rows)

    # === Parallel label_dist computation ===
    print("Computing label distributions in parallel...")
    label_paths = df["label_file"].tolist()
    label_dists = Parallel(n_jobs=-1)(delayed(get_label_distribution)(p, num_classes) for p in tqdm(label_paths))
    df["label_dist"] = label_dists

    # === Save ===
    save_path = data_root / "records.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} samples to {save_path}")


if __name__ == "__main__":
    extract_records(data_root=Path("processed"))
