# SleepBand

Official implementation of the SleepBand paper.

## Data Download and Preparation

### Download raw datasets

| Name        | Dataset                      | Link                                                     |
| ----------- | ---------------------------- | -------------------------------------------------------- |
| `cinc-2018` | PhysioNet Challenge 2018     | [Link](https://physionet.org/content/challenge-2018/)    |
| `hmc`       | HMC Sleep Staging Dataset    | [Link](https://physionet.org/content/hmc-sleep-staging/) |
| `shhs1`     | SHHS Visit 1                 | [Link](https://sleepdata.org/datasets/shhs)              |
| `sleep-edf` | Sleep-EDF Expanded           | [Link](https://physionet.org/content/sleep-edfx/)        |
| `isruc`     | ISRUC-SLEEP (Subgroup I-III) | [Link](https://sleeptight.isr.uc.pt/ISRUC_Sleep/)        |

### Preprocess each dataset

Run the preprocessing scripts in `data/`:

- `data/preprocess_sleep_edf.py`
- `data/preprocess_hmc.py`
- `data/preprocess_isruc.py`
- `data/preprocess_shhs.py`
- `data/preprocess_cinc_2018.py`

Then extract all preprocessed data into record files:

- `data/extract_records.py`

### Preprocessed Sleep Data

#### Directory Structure

```bash
data-root/
├── <dataset-name>/
│   ├── data/                   # preprocessed signal sequences
│   │   └── unique-id/          # signals for signal sequences
│   │       └── unique-id-0.npy # first 10-min segment, shape = (20, 2, 3000)
│   └── label/                  # corresponding label sequences
│       └── unique-id/          # labels for signal sequences
│           └── unique-id-0.npy # shape = (20,)
```

#### File Contents

Each file represents a 10-minute segment composed of 20 consecutive epochs from a single overnight recording.
- data: shape `(20, 2, 3000)`, containing `20` epochs (`30` seconds each), `2` channels (EEG and EOG), and `3000` samples per epoch (`100 Hz × 30 s`).
- label: shape `(20,)`, containing `20` sleep-stage labels aligned with the `20` epochs.

## Reproduce SleepBand Experiments

Core config: `experiment=sleep_band`.

### Single-source DG

Train on one source dataset and test on the other four (sweep all sources):

```bash
python main.py experiment=sleep_band run_name=single-source sweep_all_domains=source
```

### Multi-source DG

Train on four source datasets and test on one target dataset (sweep all targets):

```bash
python main.py experiment=sleep_band run_name=multi-source sweep_all_domains=target
```

## Citation

If you use this repository, please cite the SleepBand paper.

```bibtex
@article{sleepband,
  title={SleepBand: ...},
  author={...},
  journal={...},
  year={...}
}
```
## Acknowledgements
The data processing scripts and base model architecture are developed based on the codebase of [SleepDG](https://github.com/wjq-learning/SleepDG).