"""Validate and load the OCR upload gate."""
from __future__ import annotations

import yaml

from backend.schemas.ecg_gate import GATE_DESCRIPTOR
from backend.services.ecg_gate_ocr import OcrLeadGate, OcrUnavailable, load_ocr_gate
from backend.settings import settings

EcgGate = OcrLeadGate


class GateLoadError(Exception):
    pass


def load_gate(gate_id: str) -> EcgGate:
    path = settings.models_dir / "gates" / f"{gate_id}.yaml"
    if not path.is_file():
        raise GateLoadError(f"ECG gate descriptor not found: {path}")
    try:
        descriptor = GATE_DESCRIPTOR.validate_python(
            yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        raise GateLoadError(f"{path.name}: invalid descriptor: {exc}") from exc
    if descriptor.id != gate_id:
        raise GateLoadError(f"{path.name}: descriptor id does not match the file name")
    try:
        return load_ocr_gate(descriptor)
    except OcrUnavailable as exc:
        raise GateLoadError(f"{path.name}: {exc}") from exc
