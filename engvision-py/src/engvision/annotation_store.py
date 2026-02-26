"""In-memory annotation store with JSON file persistence."""

from __future__ import annotations

import json
import os
from typing import Any

from .models import Annotation, BoundingBox, DetectedRegion


class AnnotationStore:
    def __init__(self, output_directory: str) -> None:
        self._manual: dict[str, list[dict[str, Any]]] = {}
        self._auto: dict[str, list[dict[str, Any]]] = {}
        self._persist_dir = os.path.join(output_directory, "annotations")
        os.makedirs(self._persist_dir, exist_ok=True)
        self._load_from_disk()

    def get_manual_annotations(self, doc_key: str, page_num: int) -> list[dict[str, Any]]:
        return self._manual.get(f"{doc_key}:{page_num}", [])

    def get_auto_detections(self, doc_key: str, page_num: int) -> list[dict[str, Any]]:
        return self._auto.get(f"{doc_key}:{page_num}", [])

    def set_auto_detections(self, doc_key: str, page_num: int, regions: list[dict[str, Any]]) -> None:
        self._auto[f"{doc_key}:{page_num}"] = regions

    def add_manual_annotation(self, doc_key: str, page_num: int, annotation: dict[str, Any]) -> None:
        key = f"{doc_key}:{page_num}"
        if key not in self._manual:
            self._manual[key] = []
        self._manual[key].append(annotation)
        self._save_to_disk()

    def update_manual_annotation(
        self, doc_key: str, page_num: int, ann_id: str, annotation: dict[str, Any]
    ) -> None:
        key = f"{doc_key}:{page_num}"
        if key not in self._manual:
            return
        for i, a in enumerate(self._manual[key]):
            if a.get("id") == ann_id:
                self._manual[key][i] = annotation
                break
        self._save_to_disk()

    def clear_manual_annotations(self, doc_key: str, page_num: int) -> None:
        key = f"{doc_key}:{page_num}"
        self._manual.pop(key, None)
        self._save_to_disk()

    def delete_manual_annotation(self, doc_key: str, page_num: int, ann_id: str) -> None:
        key = f"{doc_key}:{page_num}"
        if key not in self._manual:
            return
        self._manual[key] = [a for a in self._manual[key] if a.get("id") != ann_id]
        self._save_to_disk()

    def export_all(self, doc_key: str) -> dict[str, list[dict[str, Any]]]:
        return {k: v for k, v in self._manual.items() if k.startswith(f"{doc_key}:")}

    def _save_to_disk(self) -> None:
        path = os.path.join(self._persist_dir, "manual_annotations.json")
        with open(path, "w") as f:
            json.dump(self._manual, f, indent=2)

    def _load_from_disk(self) -> None:
        path = os.path.join(self._persist_dir, "manual_annotations.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._manual = data
        except Exception:
            pass
