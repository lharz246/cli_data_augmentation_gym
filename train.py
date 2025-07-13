import json
import time
from pathlib import Path
from typing import Tuple

from trainer import *
from torch.utils.data import DataLoader
from utils.config import build_parser, load_config
from utils.logger import init_logging, MetricsLogger
from utils.metrics import *
from utils.plot import plot_training
from utils.helper import save_training_history

torch.manual_seed(42)


class TrainLoop:
    """Generic training loop that works with any BaseTrainer implementation."""

    def __init__(
        self,
        trainer: BaseTrainer,
        config: Any,
        logger: MetricsLogger,
        epochs: int = 10,
    ):
        self.trainer = trainer
        self.config = config
        self.logger = logger
        self.history: Dict[str, list] = {"train": [], "val": [], "test": []}
        self.start_time = time.time()
        self.num_epochs = epochs

    def train(self) -> Tuple[Dict[str, Any], Any]:
        train_start = time.time()
        self.trainer.on_training_start()
        for epoch in range(1, self.num_epochs + 1):
            t_epoch_start = time.time()
            train_loader, val_loader = self.trainer.prepare_epoch(epoch)
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader)
            if getattr(self.trainer, "scheduler", None):
                self.trainer.scheduler.step()
            stats = Metrics.get_metrics(val_metrics, mode="val")
            self.logger.log_val(
                epoch=epoch,
                total_epochs=self.num_epochs,
                val_time=time.time() - t_epoch_start,
                phase="val",
                **stats,
            )

            self.history["train"].append(train_metrics)
            self.history["val"].append(stats)

        test_metrics, model = self.test()
        test_stats = Metrics.get_metrics(test_metrics, mode="test")
        self.logger.log_val(
            epoch=self.num_epochs,
            total_epochs=self.num_epochs,
            val_time=time.time() - train_start,
            phase="test",
            **test_stats,
        )
        self.history["test"].append(test_stats)
        return self.history, model

    def test(self) -> Tuple[List[Dict[str, Any]], Any]:
        test_loader, model = self.trainer.on_training_end()
        batch_metrics = []
        with torch.no_grad():
            for batch in test_loader:
                metrics = self.trainer.validation_step(batch)
                batch_metrics.append(metrics)
        return batch_metrics, model

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> List[Dict[str, Any]]:
        batch_metrics = []
        for idx, batch in enumerate(dataloader, 1):
            t_batch_start = time.time()
            try:
                metrics = self.trainer.train_step(batch)
            except Exception as e:
                self.logger.error(
                    f"Error in train_step batch {idx}, epoch {epoch}: {e}"
                )
                raise
            batch_time = time.time() - t_batch_start
            stats = Metrics.get_metrics([metrics], "train")
            self.logger.log_train(
                epoch=epoch,
                total_epochs=self.num_epochs,
                batch=idx,
                total_batches=len(dataloader),
                time=batch_time,
                **stats,
            )
            batch_metrics.append(metrics)

        return batch_metrics

    def _validate_epoch(self, dataloader: DataLoader) -> List[Any]:
        batch_metrics = []
        with torch.no_grad():
            for batch in dataloader:
                raw = self.trainer.validation_step(batch)
                batch_metrics.append(raw)
        return batch_metrics


def create_trainer(config: Any) -> BaseTrainer:
    tname = config["train_cfg"].get("trainer_type")
    if not tname:
        available = BaseTrainer.list_available()
        raise ValueError(f"No trainer_type in Config. Available: {available}")
    return BaseTrainer.create(tname, config)


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    logger = init_logging(
        debug=True,
        wandb_enabled=(config.get("wandb_cfg", None)).get("use_wandb", False),
        wandb_config=config.get("wandb_cfg", None),
        configs=config,
    )
    trainer = create_trainer(config)
    loop = TrainLoop(trainer, config, logger, config["train_cfg"].get("epochs", None))
    history, model = loop.train()
    plots = plot_training(history)
    augment_str = "augmented" if config["train_cfg"]["use_augmentation"] else "normal"
    augmenter = (
        config["data_cfg"]["augmentation_cfg"].keys()
        if config["train_cfg"]["use_augmentation"]
        else "none"
    )
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(config["train_cfg"]["data_path"])
    last_part = path.name
    save_dir = (
        f"{config['data_cfg']['output_dir']}/{timestamp}_{config['model_cfg']['name']}"
        f"_{augment_str}_{last_part}"
    )
    if augmenter != "none":
        save_dir += f"_{augmenter}_{config['train_cfg'].get('augment_factor', None)}"
    # save_dir += f"target_class_[{config['train_cfg'].get('target_class', None)}]_{config['loss_cfg'].keys()}"
    logger.log_artifact(
        name=config["model_cfg"]["name"],
        files=plots,
        save_dir=f"{save_dir}/plots",
    )
    if hasattr(model, "teacher_model"):
        model.teacher_model = None
    torch.save(model, f"{save_dir}/model")
    return history


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args)
    print(json.dumps(cfg["model_cfg"], indent=4))
    history = train_model(cfg)
    print("Training successful!")
    try:
        saved_path = save_training_history(history, cfg)
        print(f"Training History successfully saved in: {saved_path}")
    except Exception as e:
        print(f"Error while saving the history: {e}")


if __name__ == "__main__":
    main()
