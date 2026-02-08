import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tabulate import tabulate
from tqdm import tqdm
from methods import DomainContext
import utils


class BaseTrainer:
    def __init__(self, params, run_dir):
        self.params = params
        self.step_count = 1
        self.epoch_count = 1

        # Logging setup
        self.log_dir = Path(run_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = utils.logger.get_logger()

        # Dataset & Evaluator
        self.class_names = ["W", "N1", "N2", "N3", "REM"]
        self.num_classes = len(self.class_names)
        self.build_data()

        # Model & Optimizer
        self.build_model()
        self.configure_optimizer()

        # Logging & Saving
        self.writer = utils.logger.WriterAdapter.from_params(params, log_dir=self.log_dir)
        self.model_saver = utils.saver.ModelSaver(
            dirname=os.path.join(self.log_dir, "checkpoints"),
            score_name="acc",
            filename_prefix="model",
            save_best=True,
            save_latest=True,
        )

        self.logger.info(f"✅ Model initialized {self.model}")

    def build_data(self):
        from datasets.dataset import SleepDataModule

        data_module = SleepDataModule(**self.params.data)
        self.dataloaders = data_module.get_data_loader()
        self.steps_per_epoch = len(self.dataloaders["train"])
        self.total_steps = self.steps_per_epoch * self.params.epochs

        self.data_module = data_module

        # Save splits for reproducibility
        save_folder = self.log_dir / "split"
        save_folder.mkdir(parents=True, exist_ok=True)
        data_module.train_df.to_csv(save_folder / "train.csv", index=False)
        data_module.val_df.to_csv(save_folder / "val.csv", index=False)
        data_module.test_df.to_csv(save_folder / "test.csv", index=False)

    def train(self):
        best_val_acc, best_val_f1, best_epoch = 0, 0, 0
        best_test_acc, best_test_f1 = 0, 0

        for epoch in range(1, self.params.epochs + 1):
            self.model.train()
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)

            self.model.eval()
            val_metrics = self._evaluate("val", epoch)
            test_metrics = self._evaluate("test", epoch)

            val_acc, val_f1 = val_metrics["acc"], val_metrics["f1"]
            test_acc, test_f1 = test_metrics["acc"], test_metrics["f1"]

            # Log metrics
            self.writer.log_metrics(
                {
                    "Loss/train": train_loss,
                    "Accuracy/val": val_acc,
                    "F1/val": val_f1,
                    "Accuracy/test": test_acc,
                    "F1/test": test_f1,
                },
                step=epoch,
            )

            self.model_saver.save(self.model, val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epoch
                best_test_acc, best_test_f1 = test_acc, test_f1
                self.logger.info(
                    f"🌟 New Best @ Epoch {best_epoch} | "
                    f"Val Acc: {val_acc:.5f} | Val F1: {val_f1:.5f} | "
                    f"Test Acc: {test_acc:.5f} | Test F1: {test_f1:.5f}"
                )

            elapsed = (time.time() - start_time) / 60
            self.logger.info(
                f"Epoch [{epoch:>2}] | Loss: {train_loss:.5f} | "
                f"Val Acc: {val_acc:.5f} | Val F1: {val_f1:.5f} | "
                f"Test Acc: {test_acc:.5f} | Test F1: {test_f1:.5f} | "
                f"Time: {elapsed:.2f} mins"
            )

        self.logger.info(
            f"🏁 Best Model @ Epoch {best_epoch} | "
            f"Val Acc: {best_val_acc:.5f} | Val F1: {best_val_f1:.5f} | "
            f"Test Acc: {best_test_acc:.5f} | Test F1: {best_test_f1:.5f}"
        )
        test_acc, test_f1 = self.test()
        return best_val_acc, best_val_f1, test_acc, test_f1

    @torch.no_grad()
    def test(self):
        best_model = self.model_saver.best_checkpoint
        self.logger.info(f"🔍 Loading best model from {best_model} for final evaluation")
        self.model.load_state_dict(torch.load(best_model, map_location="cuda"))
        test_metrics = self._evaluate("test", epoch=None)
        return test_metrics["acc"], test_metrics["f1"]

    @torch.no_grad()
    def _evaluate(self, phase="val", epoch=None):
        self.model.eval()
        data_loader = self.dataloaders[phase.lower()]
        y_true, y_pred = [], []

        for inputs, labels, _ in tqdm(data_loader, ncols=88, desc=f"Evaluating {phase}"):
            inputs = inputs.cuda(non_blocking=True)
            logits = self.model.inference(inputs)
            logits = logits.view(-1, self.num_classes)
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.view(-1).numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

        # Print evaluation results
        self.logger.info(f"🔍 {f'Epoch [{epoch}] -' if epoch else '[Final] -'} Performance on {phase.upper()} set")
        cm_table = tabulate(cm, headers=self.class_names, showindex=self.class_names, tablefmt="fancy_grid")
        self.logger.info(f"📊 Confusion Matrix:\n{cm_table}")

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        df = pd.DataFrame(report).T.drop(columns=["support"])
        report_table = tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".4f")
        self.logger.info(f"📋 Classification Report:\n{report_table}")

        if self.writer and epoch:
            self.writer.log_confusion_matrix(
                cm=cm,
                class_names=self.class_names,
                tag=f"ConfusionMatrix/{phase.lower()}",
                step=epoch,
            )
        return {"acc": acc, "f1": f1}

    def build_model(self):
        # Build model from registry
        params = self.params.model.copy()
        name = params.pop("name", None)
        model_class = utils.registry.get_model(name)
        if model_class is None:
            raise ValueError(f"❌ Model '{name}' not found. Available: {list(utils.registry.MODEL_REGISTRY.keys())}")
        self.model = model_class(**params, data_module=self.data_module).cuda()

    def configure_optimizer(self):
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_steps)

    def train_one_epoch(self, epoch):
        """Train for one epoch, return average loss"""
        self.model.train()

        epoch_loss = 0.0

        for inputs, labels, domains in tqdm(
            self.dataloaders["train"], desc=f"Epoch {epoch}/{self.params.epochs}", ncols=88
        ):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            domains = domains.cuda(non_blocking=True)

            context = DomainContext(step=self.step_count, epoch=epoch, total_steps=self.total_steps, training=True)

            # Forward + Backward + Optimize
            loss_value, metrics = self.train_one_step(inputs, labels, domains, context)

            epoch_loss += loss_value

            # Post step hook
            if hasattr(self.model, "post_step"):
                self.model.post_step()

            if self.step_count % 100 == 0:
                self.writer.log_metrics(metrics, step=self.step_count)

            self.step_count += 1

        epoch_loss /= self.steps_per_epoch

        return epoch_loss

    def train_one_step(self, x, y, domains, context):

        self.optimizer.zero_grad()

        loss, metrics = self.model(x, y, domains, context=context)

        loss.backward()

        if self.params.clip_value > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), metrics
