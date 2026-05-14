from __future__ import annotations

import argparse
import random
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import gram_sdk
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")
ConfigDict = dict[str, object]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a YOLO dataset and create train/val/test splits."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config yaml."
    )
    parser.add_argument("--images-dir", type=str, help="Source images directory.")
    parser.add_argument("--labels-dir", type=str, help="Source labels directory.")
    parser.add_argument("--output-dir", type=str, help="Output dataset directory.")
    parser.add_argument(
        "--data-yaml", type=str, help="Path to generated Ultralytics data yaml."
    )
    parser.add_argument("--train-ratio", type=float, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, help="Test split ratio.")
    parser.add_argument("--seed", type=int, help="Random seed for split.")
    parser.add_argument(
        "--copy-files",
        dest="copy_files",
        action="store_true",
        help="Copy files instead of hard-linking when possible.",
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


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def require_str(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    return value


def get_class_names(config: ConfigDict) -> list[str]:
    names_value = config.get("class_names", config.get("names"))
    if not isinstance(names_value, Sequence) or isinstance(names_value, (str, bytes)):
        raise ValueError("`class_names` in config.yaml must be a list of class names.")

    class_names: list[str] = []
    for name in names_value:
        if not isinstance(name, str):
            raise ValueError("Each class name must be a string.")
        class_names.append(name)

    if not class_names:
        raise ValueError("`class_names` must not be empty.")
    return class_names


def merge_preprocess_args(
    config: ConfigDict, cli_args: argparse.Namespace
) -> dict[str, object]:
    preprocess_section = config.get("preprocess", {})
    if not isinstance(preprocess_section, dict):
        raise ValueError("`preprocess` in config.yaml must be a mapping.")

    preprocess_config: dict[str, object] = dict(preprocess_section)
    cli_values = vars(cli_args)
    override_keys = (
        "images_dir",
        "labels_dir",
        "output_dir",
        "data_yaml",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
        "copy_files",
    )
    for key in override_keys:
        if cli_values.get(key) is not None:
            preprocess_config[key] = cli_values[key]

    return preprocess_config


def init_gram_tracking(
    config: ConfigDict, preprocess_args: Mapping[str, object]
) -> str:
    run_name = str(preprocess_args.get("name") or config.get("preprocess_run_name") or "preprocess")
    params = {
        "mode": "preprocess",
        "preprocess": preprocess_args,
    }
    gram_sdk.tracking_init(experiment=run_name, params=params)
    return run_name


def list_image_files(images_dir: Path) -> list[Path]:
    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_files.sort()
    return image_files


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def parse_label_line(line: str, class_count: int) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return False

    if class_id < 0 or class_id >= class_count:
        return False
    return all(0.0 <= value <= 1.0 for value in coords)


def is_valid_label_file(label_path: Path, class_count: int) -> bool:
    if not label_path.exists():
        return False

    with label_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if not parse_label_line(line, class_count):
                return False
    return True


def collect_pairs(
    images_dir: Path, labels_dir: Path, class_count: int
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in list_image_files(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if is_valid_label_file(label_path, class_count):
            pairs.append((image_path, label_path))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled_pairs = pairs[:]
    random.Random(seed).shuffle(shuffled_pairs)

    total_count = len(shuffled_pairs)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    return {
        "train": shuffled_pairs[:train_end],
        "val": shuffled_pairs[train_end:val_end],
        "test": shuffled_pairs[val_end:],
    }


def ensure_output_dirs(output_dir: Path) -> None:
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def transfer_file(source: Path, destination: Path, copy_files: bool) -> None:
    if destination.exists():
        destination.unlink()

    if copy_files:
        shutil.copy2(source, destination)
        return

    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def write_split_files(
    split_pairs_map: Mapping[str, Sequence[tuple[Path, Path]]],
    output_dir: Path,
    copy_files: bool,
) -> None:
    ensure_output_dirs(output_dir)

    for split, pairs in split_pairs_map.items():
        for image_path, label_path in pairs:
            transfer_file(
                image_path, output_dir / "images" / split / image_path.name, copy_files
            )
            transfer_file(
                label_path, output_dir / "labels" / split / label_path.name, copy_files
            )


def write_data_yaml(
    output_dir: Path,
    data_yaml_path: Path,
    class_names: Sequence[str],
) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(class_names),
        "nc": len(class_names),
    }
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=False)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    preprocess_args = merge_preprocess_args(config, args)

    images_dir = Path(
        require_str(optional_str(preprocess_args.get("images_dir")), "images_dir")
    )
    labels_dir = Path(
        require_str(optional_str(preprocess_args.get("labels_dir")), "labels_dir")
    )
    output_dir = Path(
        require_str(optional_str(preprocess_args.get("output_dir")), "output_dir")
    )
    data_yaml_value = optional_str(preprocess_args.get("data_yaml"))
    data_yaml_path = Path(data_yaml_value) if data_yaml_value else output_dir / "data.yaml"

    train_ratio = optional_float(preprocess_args.get("train_ratio")) or 0.8
    val_ratio = optional_float(preprocess_args.get("val_ratio")) or 0.1
    test_ratio = optional_float(preprocess_args.get("test_ratio")) or 0.1
    seed = optional_int(preprocess_args.get("seed")) or 42
    copy_files = optional_bool(preprocess_args.get("copy_files")) or False
    class_names = get_class_names(config)

    validate_ratios(train_ratio, val_ratio, test_ratio)
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images_dir does not exist: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise ValueError(f"labels_dir does not exist: {labels_dir}")

    init_gram_tracking(config=config, preprocess_args=preprocess_args)

    pairs = collect_pairs(images_dir, labels_dir, len(class_names))
    if not pairs:
        raise ValueError("No valid image/label pairs found for preprocessing.")

    split_pairs_map = split_pairs(
        pairs=pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    write_split_files(split_pairs_map, output_dir=output_dir, copy_files=copy_files)
    write_data_yaml(output_dir, data_yaml_path=data_yaml_path, class_names=class_names)

    gram_sdk.log_metric("preprocess/total_pairs", float(len(pairs)))
    for split in SPLITS:
        gram_sdk.log_metric(
            f"preprocess/{split}_count", float(len(split_pairs_map[split]))
        )

    print("Preprocessing finished.")
    print(f"Output dir: {output_dir}")
    print(f"Data yaml: {data_yaml_path}")


if __name__ == "__main__":
    main()
