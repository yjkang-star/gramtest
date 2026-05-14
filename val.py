from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeGuard, cast

import gram_sdk
import yaml

from model import build_model

DEFAULT_CONFIG_PATH = Path("config.yaml")
ConfigDict = dict[str, object]
ConvertibleToFloat = int | float | str


class ResultsLike(Protocol):
    save_dir: Path
    results_dict: Mapping[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO with Ultralytics.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config yaml."
    )
    parser.add_argument(
        "--model", type=str, help="YOLO model yaml or weights file. ex) best.pt"
    )
    parser.add_argument("--data", type=str, help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, help="Image size.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--device", type=str, help='Device. ex) "0", "0,1", "cpu"')
    parser.add_argument("--split", type=str, help='Dataset split. ex) "val", "test"')
    parser.add_argument("--project", type=str, help="Output project directory.")
    parser.add_argument("--name", type=str, help="Run name.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections.")
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


def merge_val_args(
    config: ConfigDict, cli_args: argparse.Namespace
) -> dict[str, object]:
    val_section = config.get("val", {})
    if not isinstance(val_section, dict):
        raise ValueError("`val` in config.yaml must be a mapping.")

    val_config: dict[str, object] = dict(val_section)
    cli_values = vars(cli_args)
    override_keys = (
        "data",
        "imgsz",
        "batch",
        "device",
        "split",
        "project",
        "name",
        "workers",
        "conf",
        "iou",
        "max_det",
    )
    for key in override_keys:
        if cli_values.get(key) is not None:
            val_config[key] = cli_values[key]

    return val_config


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def require_str(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value


def is_convertible_to_float(value: object) -> TypeGuard[ConvertibleToFloat]:
    return isinstance(value, (int, float, str))


def to_float(value: object) -> float | None:
    if not is_convertible_to_float(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_results_like(value: object) -> TypeGuard[ResultsLike]:
    if value is None:
        return False
    save_dir = getattr(value, "save_dir", None)
    results_dict = getattr(value, "results_dict", None)
    return isinstance(save_dir, Path) and isinstance(results_dict, Mapping)


def require_results_like(value: object) -> ResultsLike:
    if not is_results_like(value):
        raise ValueError("Validation did not return a valid results object.")
    return cast(ResultsLike, value)


def validate_config(model_source: str | None, data_source: str | None) -> None:
    if not model_source:
        raise ValueError(
            "Model is required. Set `model` in config.yaml or pass `--model`."
        )
    if not data_source:
        raise ValueError(
            "Dataset yaml is required. Set `val.data` in config.yaml or pass `--data`."
        )


def init_gram_tracking(config: ConfigDict, val_args: Mapping[str, object]) -> str:
    run_name = str(
        val_args.get("name")
        or config.get("val_run_name")
        or f"{Path(str(config.get('model', 'yolo'))).stem}-val"
    )
    params = {
        "mode": "val",
        "model": config.get("model"),
        "task": config.get("task"),
        "val": val_args,
    }
    gram_sdk.tracking_init(experiment=run_name, params=params)
    return run_name


def log_validation_metrics(results: ResultsLike) -> None:
    for key, value in results.results_dict.items():
        converted = to_float(value)
        if converted is not None:
            try:
                gram_sdk.log_metric(key, converted)
            except Exception:
                continue


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    model_source = args.model or optional_str(config.get("model"))
    task = args.task or optional_str(config.get("task"))
    val_args = merge_val_args(config, args)
    data_source = optional_str(val_args.get("data"))
    validate_config(model_source=model_source, data_source=data_source)
    resolved_model_source = require_str(model_source, "model")

    init_gram_tracking(config=config, val_args=val_args)
    model = build_model(model_source=resolved_model_source, task=task)
    raw_results = model.val(**val_args)
    results = require_results_like(raw_results)
    log_validation_metrics(results)

    print("Validation finished.")
    print(f"Save dir: {results.save_dir}")


if __name__ == "__main__":
    main()
