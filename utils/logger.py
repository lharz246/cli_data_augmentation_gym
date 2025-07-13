from __future__ import annotations
import logging
import time as _time
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, List, Union, Dict, Optional

import torch
import wandb
from matplotlib.figure import Figure


class ETACalculator:
    WINDOW = 20

    def __init__(self, window: int | None = None) -> None:
        self.window = window or self.WINDOW
        self._times: List[float] = []

    def _push(self, t: float) -> None:
        self._times.append(t)
        if len(self._times) > self.window:
            self._times.pop(0)

    def get_eta(
        self,
        *,
        completed: int,
        total: int,
        time: float,
        batch_size: int | None = 1,
    ) -> str:
        self._push(time)
        if not self._times or completed == 0:
            return "N/A"
        avg = sum(self._times) / len(self._times)
        remain = ((total / batch_size) - completed) * avg
        return _time.strftime("%H:%M:%S", _time.gmtime(remain))


class _Colours:
    GREY = "\x1b[38;5;250m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[1;31m"
    GREEN = "\x1b[32m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"


class CustomFormatter(logging.Formatter):
    DATEFMT = "%d-%m-%Y %H:%M:%S"
    _TRAIN_TPL = "[%(epoch)d/%(total_epochs)d][Batch %(batch)d/%(total_batches)d|%(batch_time)s] "
    _VAL_TPL = "[%(epoch)d/%(total_epochs)d][ETA %(eta_time)s][Val %(val_time)s] "
    _LEVEL_MAP = {
        logging.DEBUG: _Colours.CYAN,
        logging.INFO: _Colours.GREY,
        logging.WARNING: _Colours.YELLOW,
        logging.ERROR: _Colours.RED,
        logging.CRITICAL: _Colours.BOLD_RED,
    }

    def _is_val_record(self, rec: logging.LogRecord) -> bool:
        return (
            any(k.startswith("val_") for k in rec.__dict__)
            or "val_loss" in rec.__dict__
        )

    def _build_metric_string(
        self, rec: logging.LogRecord, *, prefix: str | None = None
    ) -> str:
        parts: list[str] = []

        def _fmt(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.3e}" if abs(v) < 1e-3 else f"{v:.3f}"
            return str(v)

        hidden = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "global_step",
            "epoch",
            "total_epochs",
            "batch",
            "total_batches",
            "eta_time",
            "batch_time",
            "val_time",
        }
        for k, v in rec.__dict__.items():
            if k in hidden or k.startswith("_"):
                continue
            if prefix and not k.startswith(prefix):
                continue
            if prefix is None and k.startswith("val_"):
                continue
            parts.append(f"{k}: {_fmt(v)}")
        return " ".join(parts)

    def format(self, record: logging.LogRecord) -> str:
        if "batch" in record.__dict__ and "total_batches" in record.__dict__:
            prefix_colour = _Colours.GREEN
            body_colour = _Colours.GREY
            body_tpl = self._TRAIN_TPL + self._build_metric_string(record)
            level_name = "TRAIN"
        elif self._is_val_record(record):
            prefix_colour = _Colours.BLUE
            body_colour = _Colours.GREY
            body_tpl = self._VAL_TPL + self._build_metric_string(record, prefix="val_")
            level_name = "VAL"
        else:
            prefix_colour = self._LEVEL_MAP.get(record.levelno, _Colours.GREY)
            body_colour = prefix_colour
            body_tpl = " %(message)s"
            level_name = record.levelname
        fmt = (
            f"{body_colour}[%(asctime)s]{_Colours.RESET}"
            f"{prefix_colour}[{level_name}]{_Colours.RESET}"
            f"{body_colour}{body_tpl}{_Colours.RESET}"
        )
        return logging.Formatter(fmt, datefmt=self.DATEFMT).format(record)


class MetricsLogger(logging.LoggerAdapter):

    def __init__(
        self,
        logger: logging.Logger,
        *,
        wandb_enabled: bool = False,
        colour_cli: bool = True,
    ) -> None:
        super().__init__(logger, {})
        self.wandb_enabled = wandb_enabled and wandb is not None
        self.global_step = 0
        self.epoch_eta = ETACalculator(10)
        self.batch_eta = ETACalculator(100)
        self.colour_cli = colour_cli

    @staticmethod
    def _format_metrics(
        metrics: Dict[str, Union[int, float]],
        prefix: Optional[str] = None,
        low_thr: float = 0.49,
        high_thr: float = 0.75,
    ) -> str:
        lines = []
        for k, v in metrics.items():
            if k in ("loss", "hamming_loss"):
                if v <= low_thr:
                    color = _Colours.GREEN
                elif v <= high_thr:
                    color = _Colours.YELLOW
                else:
                    color = _Colours.RED
            else:
                if v >= high_thr:
                    color = _Colours.GREEN
                elif v >= low_thr:
                    color = _Colours.YELLOW
                else:
                    color = _Colours.RED
            name = f"{prefix}/{k}" if prefix else k
            lines.append(f"{color}{name:<30} {v:>7.4f}{_Colours.RESET}")
        border = "-" * 50
        return "\n".join([border] + lines + [border])

    def log_train(
        self,
        *,
        epoch: int,
        batch: int,
        total_batches: int,
        time: float,
        total_epochs: Optional[int] = None,
        **metrics: Union[int, float],
    ) -> None:
        extra: Dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": total_epochs or epoch,
            "batch": batch,
            "total_batches": total_batches,
            "batch_time": _format_time(time),
            "global_step": self.global_step,
            **metrics,
        }
        if self.wandb_enabled:
            wandb.log(
                {
                    f"train/{k}": float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
                step=self.global_step,
            )
        sh = None
        for handler in self.logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                sh = handler
                break
        if sh:
            record = logging.LogRecord(
                name=self.logger.name,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Train batch",
                args=(),
                exc_info=None,
            )
            for k, v in extra.items():
                if isinstance(v, torch.Tensor):
                    setattr(record, k, v.item())
                else:
                    setattr(record, k, v)
            formatted_msg = (
                sh.formatter.format(record)
                if sh.formatter
                else str(record.getMessage())
            )
            # print(f"\r{formatted_msg}", end="", flush=True) # locally
            print(f"\r{formatted_msg}")
            if batch == total_batches:
                print()
        else:
            self.logger.log(logging.INFO, "Train batch", extra=extra)
        self.global_step += 1

    def log_val(
        self,
        *,
        epoch: int,
        total_epochs: int,
        val_time: float,
        metrics: Optional[Dict[str, Union[int, float]]] = None,
        phase: str = "val",
        **metrics_kw: Union[int, float],
    ) -> None:
        metrics = metrics or metrics_kw
        eta_time = self.epoch_eta.get_eta(
            completed=epoch, total=total_epochs, time=val_time
        )
        simple_metrics = {}
        complex_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                simple_metrics[k] = v
            else:
                complex_metrics[k] = v
        prefix = f"{phase}_"
        extra_header: Dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "eta_time": eta_time,
            "val_time": _format_time(val_time),
            "phase_label": phase.capitalize(),
        }
        for k, v in simple_metrics.items():
            if k.startswith(f"{phase}_"):
                extra_header[k] = v
            else:
                extra_header[f"{prefix}{k}"] = v
        if self.wandb_enabled:
            wandb.log(
                {
                    f"val/{k.lstrip(f'{phase}_')}": float(v)
                    for k, v in simple_metrics.items()
                },
                step=self.global_step,
            )
        log_message = f"{phase.capitalize()} epoch"
        self.logger.log(logging.INFO, log_message, extra=extra_header)
        formatted_metrics = self._format_metrics(simple_metrics)
        print(formatted_metrics)
        if phase.capitalize() == "Test":
            print(f"\nAdditional {phase.capitalize()} Results:")
            for key, value in complex_metrics.items():
                if key == "conf_matrix_classes":
                    print(f"Confusion Matrix:\n{value}")
                elif key == "label_classification_breakdown":
                    print("Label Classification Breakdown:")
                    for label, stats in value.items():
                        print(
                            f"  {label}: {stats['correct']}/{stats['total']} "
                            f"(accuracy: {stats['accuracy']:.4f})"
                        )
        self.global_step += 1

    def log_artifact(
        self,
        name: str,
        files: Union[
            str,
            Path,
            Figure,
            Dict[str, Figure],
            List[Union[str, Path, Figure]],
        ],
        *,
        save_dir: Union[str, Path] = "plots",
        img_format: str = "png",
        dpi: int = 150,
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def _save_fig(fig: Figure, stem: str) -> str:
            fp = save_dir / f"{stem}.{img_format}"
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            return str(fp)

        local_paths: List[str] = []
        if isinstance(files, dict):
            for stem, fig in files.items():
                local_paths.append(_save_fig(fig, stem))
        elif isinstance(files, Figure):
            local_paths.append(_save_fig(files, stem=name))
        elif isinstance(files, (str, Path)):
            local_paths.append(str(files))
        elif isinstance(files, list):
            for item in files:
                if isinstance(item, Figure):
                    stem = f"{name}_{len(local_paths)}"
                    local_paths.append(_save_fig(item, stem))
                else:
                    local_paths.append(str(item))
        else:
            raise TypeError(
                "`files` have to be str | Path | Figure | Dict[str, Figure] | List[…]."
            )
        if self.wandb_enabled:
            log_payload = {
                f"{name}/{Path(fp).stem}": wandb.Image(fp) for fp in local_paths
            }
            wandb.log(log_payload)
            self.logger.info(
                f"Logged {len(local_paths)} image file(s) to W&B via wandb.log (no artifact)."
            )
        else:
            self.logger.info(
                f"[W&B disabled] Saved {len(local_paths)} file(s) locally in "
                f"'{save_dir}'. Artifact '{name}' was *not* uploaded."
            )


def _format_time(seconds: float) -> str:
    if seconds >= 1:
        return time.strftime("%H:%M:%S", time.gmtime(seconds))
    return f"{seconds*1000:5.1f} ms"


def init_logging(
    *,
    debug: bool = False,
    logfile: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    wandb_enabled: bool = False,
    wandb_config: Optional[dict] = None,
    configs=None,
) -> MetricsLogger:
    logger = logging.getLogger("advanced_logger")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = CustomFormatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logger.level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        fh = logging.handlers.RotatingFileHandler(
            filename=logfile,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if wandb_enabled:
        try:
            if wandb_config:
                cfg = configs["model_cfg"]
                cfg["train_cfg"] = configs["train_cfg"]
                cfg["loss_cfg"] = configs["loss_cfg"]
                wandb.init(
                    project=wandb_config.get("wandb_project", None),
                    entity=wandb_config.get("wandb_entity", None),
                    name=wandb_config.get("wandb_run_name", None),
                    reinit=True,
                    config=cfg,
                )
                logger.info(
                    f"Wandb successfully initialized with project: {wandb_config.get('project', 'default')}"
                )
            else:
                wandb.init()
                logger.info("Wandb successfully initialized")

        except ImportError:
            logger.warning(
                "Wandb not installed but wandb_enabled=True. Install with: pip install wandb"
            )
            wandb_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            wandb_enabled = False
    logger = MetricsLogger(logger, wandb_enabled=wandb_enabled)
    set_global_logger(logger)
    return logger


_global_logger: Optional[MetricsLogger] = None


def get_logger() -> MetricsLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = init_logging()
    return _global_logger


def set_global_logger(logger: MetricsLogger) -> None:
    global _global_logger
    _global_logger = logger
