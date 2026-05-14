from __future__ import annotations

from ultralytics import YOLO


def build_model(
    model_source: str, task: str | None = None, verbose: bool = True
) -> YOLO:
    """Create a YOLO model from a `.pt` weight or a model yaml file."""
    if task:
        return YOLO(model_source, task=task, verbose=verbose)
    return YOLO(model_source, verbose=verbose)
