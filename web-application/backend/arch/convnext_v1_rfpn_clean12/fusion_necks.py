"""Residual feature-pyramid layers used by the serving classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

FUSION_STAGE_INDICES = (1, 2, 3)


def _gn(channels):
    return nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)


def _conv_gn_act(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        _gn(out_channels), nn.GELU(),
    )


class FPNNeck(nn.Module):
    '''Top-down FPN: 1x1 laterals, nearest upsampling + add, 3x3 conv-GN-GELU per level.'''

    def __init__(self, in_channels, channels):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([_conv_gn_act(channels, channels) for _ in in_channels])

    def top_down(self, features):
        lateral = [layer(f) for layer, f in zip(self.lateral, features)]
        outputs = [lateral[-1]]
        for i in range(len(lateral) - 2, -1, -1):
            outputs.insert(0, lateral[i] + F.interpolate(
                outputs[0], size=lateral[i].shape[-2:], mode="nearest"))
        return [smooth(o) for smooth, o in zip(self.smooth, outputs)]

    def forward(self, features):
        return self.top_down(features)


class ConvNeXtMultiScaleFusion(nn.Module):
    """ConvNeXt logits plus a pooled residual feature-pyramid head."""

    def __init__(self, backbone, num_classes: int, channels: int = 256):
        super().__init__()
        self.backbone = backbone
        in_channels = [backbone.feature_info[i]["num_chs"] for i in FUSION_STAGE_INDICES]
        self.neck = FPNNeck(in_channels, channels)
        self.fuse_norm = nn.LayerNorm(channels * len(in_channels))
        self.fuse_fc = nn.Linear(channels * len(in_channels), num_classes)

    @property
    def default_cfg(self):
        return self.backbone.default_cfg

    @property
    def pretrained_cfg(self):
        return self.backbone.pretrained_cfg

    def get_classifier(self):
        return self.backbone.get_classifier()

    def forward(self, x):
        backbone = self.backbone
        x = backbone.stem(x)
        features = []
        for stage in backbone.stages:
            x = stage(x)
            features.append(x)
        maps = self.neck([features[i] for i in FUSION_STAGE_INDICES])
        pooled = torch.cat([torch.mean(m, dim=(2, 3), dtype=torch.float32) for m in maps], dim=1)
        fused_logits = self.fuse_fc(self.fuse_norm(pooled))
        return backbone.forward_head(backbone.norm_pre(x)) + fused_logits
