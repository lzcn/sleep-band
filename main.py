import argparse
import os
import random
import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import utils.logger
from utils.registry import get_trainer

logger = utils.logger.get_logger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🔒 Seed set to: {seed}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/band.yaml", help="Path to YAML config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config options from command line")
    return parser.parse_args()


def as_list(x):
    return None if x is None else [x] if isinstance(x, str) else list(x)


def run(cfg, run_dir):
    log_file = os.path.join(run_dir, "train.log")
    os.makedirs(run_dir, exist_ok=True)
    utils.logger.config_logger(log_file=log_file)

    logger.info(f"🔍 Getting trainer: {cfg.trainer.name}")
    trainer_cls = get_trainer(cfg.trainer.name)
    trainer = trainer_cls(cfg, run_dir)
    logger.info(f"🚀 Training from {cfg.data.source_domains} -> {cfg.data.target_domains}")
    logger.info("🏁 Start training")
    val_acc, val_f1, test_acc, test_f1 = trainer.train()
    logger.info(
        f"Source domains: {cfg.data.source_domains} | Target domains: {cfg.data.target_domains}\n"
        f"✅ Validation  | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}\n"
        f"✅ Test        | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}"
    )
    return val_acc, val_f1, test_acc, test_f1


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # args = parse_args()
    OmegaConf.set_struct(cfg, False)

    run_dir = HydraConfig.get().run.dir
    logger.info(f"📁 Experiment root: {run_dir}")
    logger.info("🧾 Base config:\n" + OmegaConf.to_yaml(cfg))

    # === Load config ===
    # cfg = OmegaConf.load(args.config)
    # if args.overrides:
    #     logger.info(f"⚙️  Applying CLI overrides: {args.overrides}")
    #     cli_cfg = OmegaConf.from_dotlist(args.overrides)
    #     cfg = OmegaConf.merge(cfg, cli_cfg)

    # base_dir is outputs/project_name/run_name
    # run_dir = os.path.join("outputs", cfg.project_name, cfg.run_name)
    # os.makedirs(run_dir, exist_ok=True)
    # utils.logger.config_logger(log_file=os.path.join(run_dir, "train.log"))
    # OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    # logger.info(f"💾 Config saved to: {os.path.join(run_dir, 'config.yaml')}")
    # logger.info(f"🧾 Final Config:\n{OmegaConf.to_yaml(cfg)}")

    # === Setup ===
    setup_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Using device: {device}")
    torch.cuda.set_device(cfg.cuda)

    # === Run training ===
    datasets = ["sleep-edf", "hmc", "isruc", "shhs1", "cinc-2018"]
    sweep_all_domains = cfg.get("sweep_all_domains", None)
    if sweep_all_domains is None:
        source = as_list(cfg.data.get("source_domains"))
        target = as_list(cfg.data.get("target_domains"))

        if source is None and target is None:
            raise ValueError("Must specify data.source_domains or data.target_domains")

        cfg.data.source_domains = source or [d for d in datasets if d not in target]
        cfg.data.target_domains = target or [d for d in datasets if d not in source]

        run(cfg, run_dir)
    else:
        test_accs, test_f1s, val_accs, val_f1s = [], [], [], []
        base_run_name = cfg.run_name

        summary_path = os.path.join(run_dir, "performance_summary.csv")
        for idx, domain in enumerate(datasets):
            if sweep_all_domains == "target":  # sweep multi-domain training over all target domains
                cfg.data.source_domains = [d for d in datasets if d != domain]
                cfg.data.target_domains = [domain]
            elif sweep_all_domains == "source":  # sweep single-domain training over all source domains
                cfg.data.source_domains = [domain]
                cfg.data.target_domains = [d for d in datasets if d != domain]
            else:
                raise ValueError(f"Invalid sweep_all_domains: {sweep_all_domains}, must be None, 'source', or 'target'")

            cfg.run_name = f"{base_run_name}/{domain}"

            val_acc, val_f1, test_acc, test_f1 = run(cfg, os.path.join(run_dir, domain))

            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            # Save CSV after each domain
            df = pd.DataFrame(
                {
                    "val_accuracy": val_accs,
                    "val_f1_score": val_f1s,
                    "test_accuracy": test_accs,
                    "test_f1_score": test_f1s,
                },
                index=datasets[: idx + 1],
            )
            df.loc["average"] = df.mean()
            df.to_csv(summary_path)
        utils.logger.config_logger(log_file=os.path.join(run_dir, "train.log"))
        logger.info(f"📊 Summary Table:\n" f"{df.map(lambda x: f'{x*100:.2f}' if isinstance(x, float) else x)}")

    logger.info("🏁 Training finished")


if __name__ == "__main__":
    main()
