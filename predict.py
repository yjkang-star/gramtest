from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import gram_sdk
import yaml

from model import build_model

DEFAULT_CONFIG_PATH = Path("config.yaml")
ConfigDict = dict[str, object]


class PredictResultLike(Protocol):
    save_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction with Ultralytics."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config yaml."
    )
    parser.add_argument(
        "--model", type=str, help="YOLO model yaml or weights file. ex) best.pt"
    )
    parser.add_argument(
        "--source",
        type=str,
        help='Input source. ex) "./images", "./image.jpg", "./video.mp4"',
    )
    parser.add_argument("--imgsz", type=int, help="Image size.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, help="IoU threshold.")
    parser.add_argument("--device", type=str, help='Device. ex) "0", "0,1", "cpu"')
    parser.add_argument("--project", type=str, help="Output project directory.")
    parser.add_argument("--name", type=str, help="Run name.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections.")
    parser.add_argument(
        "--vid-stride", dest="vid_stride", type=int, help="Video stride."
    )
    parser.add_argument(
        "--task", type=str, help='Task type. ex) "detect", "segment", "pose"'
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


def merge_predict_args(
    config: ConfigDict, cli_args: argparse.Namespace
) -> dict[str, object]:
    predict_section = config.get("predict", {})
    if not isinstance(predict_section, dict):
        raise ValueError("`predict` in config.yaml must be a mapping.")

    predict_config: dict[str, object] = dict(predict_section)
    cli_values = vars(cli_args)
    override_keys = (
        "source",
        "imgsz",
        "conf",
        "iou",
        "device",
        "project",
        "name",
        "max_det",
        "vid_stride",
    )
    for key in override_keys:
        if cli_values.get(key) is not None:
            predict_config[key] = cli_values[key]

    return predict_config


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def require_str(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def validate_config(model_source: str | None, source: str | None) -> None:
    if not model_source:
        raise ValueError(
            "Model is required. Set `model` in config.yaml or pass `--model`."
        )
    if not source:
        raise ValueError(
            "Prediction source is required. Set `predict.source` in config.yaml or pass `--source`."
        )


def init_gram_tracking(config: ConfigDict, predict_args: Mapping[str, object]) -> str:
    run_name = str(
        predict_args.get("name")
        or config.get("predict_run_name")
        or f"{Path(str(config.get('model', 'yolo'))).stem}-predict"
    )
    params = {
        "mode": "predict",
        "model": config.get("model"),
        "task": config.get("task"),
        "predict": predict_args,
    }
    gram_sdk.tracking_init(experiment=run_name, params=params)
    return run_name


def is_predict_result_like(value: object) -> bool:
    return isinstance(getattr(value, "save_dir", None), Path)


def extract_save_dir(results: Sequence[object]) -> Path | None:
    if not results:
        return None
    first_result = results[0]
    if not is_predict_result_like(first_result):
        return None
    return cast(PredictResultLike, first_result).save_dir


def require_prediction_results(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Prediction did not return a valid results sequence.")
    return list(value)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    model_source = args.model or optional_str(config.get("model"))
    task = args.task or optional_str(config.get("task"))
    predict_args = merge_predict_args(config, args)
    source = optional_str(predict_args.get("source"))
    validate_config(model_source=model_source, source=source)
    resolved_model_source = require_str(model_source, "model")
    resolved_source = require_str(source, "source")
    imgsz = optional_int(predict_args.get("imgsz"))
    conf = optional_float(predict_args.get("conf"))
    iou = optional_float(predict_args.get("iou"))
    device = optional_str(predict_args.get("device"))
    project = optional_str(predict_args.get("project"))
    name = optional_str(predict_args.get("name"))
    max_det = optional_int(predict_args.get("max_det"))
    vid_stride = optional_int(predict_args.get("vid_stride"))
    save = optional_bool(predict_args.get("save"))
    verbose = optional_bool(predict_args.get("verbose"))
    stream = optional_bool(predict_args.get("stream")) or False

    init_gram_tracking(config=config, predict_args=predict_args)
    model = build_model(model_source=resolved_model_source, task=task)
    raw_results = model.predict(
        source=resolved_source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        project=project,
        name=name,
        max_det=max_det,
        vid_stride=vid_stride,
        save=save,
        verbose=verbose,
        stream=stream,
    )
    results = require_prediction_results(raw_results)
    save_dir = extract_save_dir(results)

    gram_sdk.log_metric("prediction/result_count", float(len(results)))

    print("Prediction finished.")
    print(f"Result count: {len(results)}")
    if save_dir is not None:
        print(f"Save dir: {save_dir}")


if __name__ == "__main__":
    main()
