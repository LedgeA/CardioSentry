"""Prepare rectified ECG images for the configured classifier.

Dimensions use (height, width). Resizing uses the descriptor's PIL kernel;
height_pad preserves aspect ratio and uses the corner paper colour for padding.
The saved model_input.png contains the exact pixels normalized for inference.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def as_hw(size) -> tuple[int, int]:
    """Normalize a descriptor `input_size` / view size to (height, width)."""
    if isinstance(size, int):
        return (size, size)
    h, w = size
    return (int(h), int(w))


_RESAMPLE = {"pil_bicubic": Image.BICUBIC, "pil_lanczos": Image.LANCZOS}


def resize_hw(img: Image.Image, size, resample: str = "pil_bicubic") -> Image.Image:
    """THE (h, w) -> PIL (w, h) swap. The only place in the app that reorders
    the pair, so a transposed input can only ever come from here."""
    h, w = as_hw(size)
    if (img.width, img.height) == (w, h):
        return img
    return img.resize((w, h), _RESAMPLE[resample])


def paper_colour_p90(img: Image.Image) -> tuple[int, int, int]:
    """Estimate paper colour using the
    90th percentile of the four corner patches (a median is dragged toward the
    grid tint by antialiased halo pixels; p90 recovers the paper white)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        arr[:k, :k].reshape(-1, 3), arr[:k, -k:].reshape(-1, 3),
        arr[-k:, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)])
    return tuple(int(v) for v in np.percentile(patches, 90, axis=0))


def fit_height_pad(img: Image.Image, size, resample: str = "pil_lanczos") -> Image.Image:
    """Aspect-preserving fit into (h, w) plus a centred paper-colour pad — the
    configured target dimensions. The scale is
    min(h/H, w/W) so a sheet wider than the box cannot overflow it; for every
    sheet at the training aspect that is the height fit the worker used."""
    h, w = as_hw(size)
    scale = min(h / img.height, w / img.width)
    new_w = min(w, round(img.width * scale))
    new_h = min(h, round(img.height * scale))
    small = img.convert("RGB").resize((new_w, new_h), _RESAMPLE[resample])
    if (new_w, new_h) == (w, h):
        return small
    canvas = Image.new("RGB", (w, h), paper_colour_p90(small))
    canvas.paste(small, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def pil_from_rgb_array(arr_rgb: np.ndarray) -> Image.Image:
    """ndarray (H,W,3) uint8 RGB -> PIL RGB. THE named BGR->RGB conversion point:
    callers are responsible for having converted OpenCV BGR output to RGB before
    this line (see rectify.py, which produces RGB throughout)."""
    return Image.fromarray(np.ascontiguousarray(arr_rgb), mode="RGB")


def jpeg95_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode->decode through a JPEG in memory, reproducing the training
    pipeline's second JPEG generation (prepare_dataset saved quality=95,
    optimize=True, and eval re-opened that file)."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        return im.convert("RGB")


def normalize_tensor(pil_img: Image.Image, mean, std) -> torch.Tensor:
    tf = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return tf(pil_img)


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


class PreprocessResult:
    def __init__(self, tensor: torch.Tensor, model_input_png: bytes,
                 tensor_sha: str, ms: float, input_hw: tuple[int, int]):
        self.tensor = tensor                  # [1,3,H,W] float32, normalized
        self.model_input_png = model_input_png
        self.tensor_sha256 = tensor_sha
        self.ms = ms
        self.input_hw = input_hw              # (height, width)

    @property
    def input_size(self) -> list[int]:
        """[height, width] — what the API and the capture record store."""
        return list(self.input_hw)


def preprocess_corrected(corrected_rgb: np.ndarray, *,
                         input_size,
                         match_training_jpeg: bool = True,
                         jpeg_quality: int = 95,
                         mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225),
                         fit: str = "stretch",
                         resample: str = "pil_bicubic",
                         margin_px: int = 0) -> PreprocessResult:
    """Run the full canonical chain on the rectified sheet (RGB ndarray).

    `input_size` is an int (square) or (height, width). `fit="stretch"` resizes
    the whole rectified sheet, margin included, to the input size (every
    stretch-fit model). `fit="height_pad"` crops the `margin_px` rectify border
    and runs the aspect-preserving fit-and-pad (module docstring). Returns the normalized
    [1,3,H,W] tensor, the exact post-JPEG pixels as lossless PNG bytes, the
    tensor sha256, and elapsed ms.
    """
    t0 = time.perf_counter()
    hw = as_hw(input_size)

    img = pil_from_rgb_array(corrected_rgb)
    if fit == "stretch":
        img = resize_hw(img, hw, resample)
    elif fit == "height_pad":
        m = int(margin_px)
        if m < 0 or 2 * m >= min(img.width, img.height):
            raise ValueError(f"margin_px={m} does not fit a {img.width}x{img.height} sheet")
        if m:
            img = img.crop((m, m, img.width - m, img.height - m))
        img = fit_height_pad(img, hw, resample)
    else:
        raise ValueError(f"unknown fit {fit!r}")

    if match_training_jpeg:
        img = jpeg95_roundtrip(img, jpeg_quality)

    png_buf = io.BytesIO()
    img.save(png_buf, "PNG")
    model_input_png = png_buf.getvalue()

    tensor = normalize_tensor(img, mean, std).unsqueeze(0)
    ms = (time.perf_counter() - t0) * 1000.0
    return PreprocessResult(tensor, model_input_png, tensor_sha256(tensor), ms, hw)


def tensor_from_model_input(path: Path, *, mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """Rebuild the exact tensor from a stored model_input.png (no re-encoding,
    no resize — the stored pixels ARE the tensorized pixels). Used by
    offline processing so re-runs score bit-identical inputs."""
    with Image.open(path) as im:
        im.load()
        img = im.convert("RGB")
    return normalize_tensor(img, mean, std).unsqueeze(0)


def load_corrected_rgb(path: Path) -> np.ndarray:
    """Decode a stored corrected.png into the RGB grid the chain consumes."""
    with Image.open(path) as im:
        im.load()
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
