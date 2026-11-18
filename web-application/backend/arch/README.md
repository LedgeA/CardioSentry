# Serving architecture

`convnext_v1_rfpn_clean12` builds the ConvNeXt Tiny and residual feature-pyramid
modules required by the configured checkpoint. Module names are part of the
checkpoint contract; strict loading detects mismatches.

ConvNeXt is provided by Meta Research and implemented through `timm`.
The application adds a feature-pyramid branch and uses the final backbone
stage as the descriptor's Grad-CAM target. Runtime preprocessing and thresholds
are configured in `models/convnext_tiny_rfpn_clean12_aug.yaml`.

The `timm` version is pinned in `pyproject.toml`. Verify checkpoint loading,
synthetic reference scores, and Grad-CAM behavior before changing dependencies.
