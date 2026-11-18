"""Build the architecture required by the serving checkpoint."""
from __future__ import annotations

REGISTRY = {
    "convnext_tiny_rfpn_clean12": {
        "timm": "convnext_tiny.fb_in22k_ft_in1k_384",
        "num_classes": 12,
        "drop_path_rate": 0.1,
        "head_init_scale": 0.001,
    },
}


def build(name: str, pretrained: bool = False):
    import timm

    from .fusion_necks import ConvNeXtMultiScaleFusion

    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}")
    spec = REGISTRY[name]
    backbone = timm.create_model(
        spec["timm"], pretrained=pretrained, num_classes=spec["num_classes"],
        drop_path_rate=spec["drop_path_rate"], head_init_scale=spec["head_init_scale"])
    model = ConvNeXtMultiScaleFusion(backbone, spec["num_classes"])
    return model, {**spec, "timm_version": timm.__version__}
