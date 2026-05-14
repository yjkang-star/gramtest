from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, TypeGuard, cast

import gram_sdk
import yaml

from model import build_model

DEFAULT_CONFIG_PATH = Path("config.yaml")
ConfigDict = dict[str, object]
ConvertibleToFloat = int | float | str


class TrainerArgsLike(Protocol):
    epochs: int


class TrainerLike(Protocol):
    epoch: int
    args: TrainerArgsLike
    tloss: object
    lr: object
    metrics: object

    def label_loss_items(
        self, tloss: object, prefix: str = ""
    ) -> Mapping[str, object]: ...


class ResultsLike(Protocol):
    save_dir: Path
    results_dict: Mapping[str, object]


class CallbackModelLike(Protocol):
    def add_callback(self, event: str, func: Callable[[TrainerLike], None]) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with Ultralytics.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config yaml."
    )
    parser.add_argument(
        "--model", type=str, help="YOLO model yaml or weights file. ex) yolo11n.pt"
    )
    parser.add_argument("--data", type=str, help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, help="Image size.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--device", type=str, help='Device. ex) "0", "0,1", "cpu"')
    parser.add_argument("--project", type=str, help="Output project directory.")
    parser.add_argument("--name", type=str, help="Run name.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument(
        "--optimizer", type=str, help='Optimizer. ex) "auto", "SGD", "AdamW"'
    )
    parser.add_argument("--lr0", type=float, help="Initial learning rate.")
    parser.add_argument("--resume", action="store_true", help="Resume previous run.")
    parser.add_argument(
        "--task",
        type=str,
        help='Task type. ex) "detect", "segment", "classify", "pose"',
    )
    return parser.parse_args()


def load_yaml(path: Path) -> ConfigDict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return cast(ConfigDict, data)


def merge_train_args(
    config: ConfigDict, cli_args: argparse.Namespace
) -> dict[str, object]:
    train_section = config.get("train", {})
    if not isinstance(train_section, dict):
        raise ValueError("`train` in config.yaml must be a mapping.")

    train_config: dict[str, object] = dict(train_section)
    cli_values = vars(cli_args)

    override_keys = (
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "project",
        "name",
        "workers",
        "patience",
        "optimizer",
        "lr0",
    )
    for key in override_keys:
        if cli_values.get(key) is not None:
            train_config[key] = cli_values[key]

    if cli_args.resume:
        train_config["resume"] = True

    return train_config


def validate_config(model_source: str | None, data_source: str | None) -> None:
    if not model_source:
        raise ValueError(
            "Model is required. Set `model` in config.yaml or pass `--model`."
        )
    if not data_source:
        raise ValueError(
            "Dataset yaml is required. Set `train.data` in config.yaml or pass `--data`."
        )


def init_gram_tracking(config: ConfigDict, train_args: Mapping[str, object]) -> str:
    run_name = str(
        train_args.get("name")
        or config.get("run_name")
        or Path(str(config.get("model", "yolo"))).stem
    )
    params = {
        "model": config.get("model"),
        "task": config.get("task"),
        "train": train_args,
    }
    gram_sdk.tracking_init(experiment=run_name, params=params)
    return run_name


def is_convertible_to_float(value: object) -> TypeGuard[ConvertibleToFloat]:
    return isinstance(value, (int, float, str))


def to_float(value: object) -> float | None:
    if not is_convertible_to_float(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def require_str(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value


def is_results_like(value: object) -> TypeGuard[ResultsLike]:
    if value is None:
        return False
    save_dir = getattr(value, "save_dir", None)
    results_dict = getattr(value, "results_dict", None)
    return isinstance(save_dir, Path) and isinstance(results_dict, Mapping)


def require_results_like(value: object) -> ResultsLike:
    if not is_results_like(value):
        raise ValueError("Training did not return a valid results object.")
    return cast(ResultsLike, value)


def normalize_metrics(raw_metrics: object) -> dict[str, float]:
    if isinstance(raw_metrics, Mapping):
        normalized: dict[str, float] = {}
        for key, value in raw_metrics.items():
            converted = to_float(value)
            if converted is not None:
                normalized[str(key)] = converted
        return normalized
    return {}


def register_gram_callbacks(model: CallbackModelLike) -> None:
    def on_train_epoch_end(trainer: TrainerLike) -> None:
        epoch = trainer.epoch
        total_epochs = trainer.args.epochs

        metrics: dict[str, float] = {}
        try:
            loss_metrics = trainer.label_loss_items(trainer.tloss, prefix="train")
            metrics.update(normalize_metrics(loss_metrics))
        except Exception:
            pass

        metrics.update(normalize_metrics(trainer.lr))

        for key, value in metrics.items():
            try:
                gram_sdk.log_metric(key, value, step=epoch + 1)
            except Exception:
                continue

        gram_sdk.log_progress(epoch + 1, total_epochs)

    def on_fit_epoch_end(trainer: TrainerLike) -> None:
        epoch = trainer.epoch
        metrics = normalize_metrics(trainer.metrics)
        if not metrics:
            return

        for key, value in metrics.items():
            try:
                gram_sdk.log_metric(key, value, step=epoch + 1)
            except Exception:
                continue

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def extract_best_metric(results_dict: Mapping[str, object]) -> tuple[str, float] | None:
    for metric_name in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "fitness"):
        metric_value = results_dict.get(metric_name)
        if metric_value is None:
            continue
        converted = to_float(metric_value)
        if converted is not None:
            return metric_name, converted
    return None


def extract_final_metrics(results_dict: Mapping[str, object]) -> dict[str, float]:
    final_metrics: dict[str, float] = {}
    for key, value in results_dict.items():
        converted = to_float(value)
        if converted is not None:
            final_metrics[f"final/{key}"] = converted
    return final_metrics


def find_weight_path(save_dir: Path) -> str | None:
    candidate_paths = (
        save_dir / "weights" / "best.pt",
        save_dir / "best.pt",
        save_dir / "weights" / "last.pt",
        save_dir / "last.pt",
        save_dir / "model.onnx",
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path)
    return None


def finalize_gram_run(
    model: object, results: ResultsLike, train_args: Mapping[str, object]
) -> None:
    results_dict = results.results_dict
    final_metrics = extract_final_metrics(results_dict)
    for key, value in final_metrics.items():
        try:
            gram_sdk.log_metric(key, value)
        except Exception:
            continue

    epoch_value = train_args.get("epochs")
    epoch = int(epoch_value) if isinstance(epoch_value, int) else 0
    best_metric = extract_best_metric(results_dict)
    if best_metric is not None:
        score_name, score = best_metric
        try:
            gram_sdk.save_best(model, str(results.save_dir), score_name, score, epoch)
        except Exception:
            pass

    try:
        gram_sdk.tracking_end(
            status="completed",
            metrics=final_metrics or None,
            weight_path=find_weight_path(results.save_dir),
        )
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    model_source = args.model or optional_str(config.get("model"))
    task = args.task or optional_str(config.get("task"))
    train_args = merge_train_args(config, args)
    data_source = optional_str(train_args.get("data"))
    validate_config(model_source=model_source, data_source=data_source)
    resolved_model_source = require_str(model_source, "model")

    init_gram_tracking(config=config, train_args=train_args)
    try:
        model = build_model(model_source=resolved_model_source, task=task)
        register_gram_callbacks(model)
        raw_results = model.train(**train_args)
        results = require_results_like(raw_results)
        finalize_gram_run(model=model, results=results, train_args=train_args)
    except Exception:
        try:
            gram_sdk.tracking_end(status="failed")
        except Exception:
            pass
        raise

    print("Training finished.")
    print(f"Save dir: {results.save_dir}")


if __name__ == "__main__":
    main()
