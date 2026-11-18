# Inference model

`active.yaml` selects `convnext_tiny_rfpn_clean12_aug.yaml`. The application
serves a ConvNeXt Tiny classifier with residual feature-pyramid fusion and
twelve independent ECG labels. Its Python implementation is in
`backend/arch/convnext_v1_rfpn_clean12/`.

## Checkpoint

Supply `models/weights/convnext_tiny_rfpn_clean12_aug_inference.pt` separately.
The descriptor pins its SHA-256 and protocol fingerprint. Startup validates
the descriptor, checks the checkpoint, rebuilds the architecture, loads the
weights strictly, and warms up inference. A mismatch prevents startup.

The descriptor also defines label order, thresholds, preprocessing dimensions,
normalization, and the Grad-CAM target. These values form the serving contract.
Changing label order or preprocessing changes the meaning of the model output.

```bash
python scripts/verify_model.py --fixture /path/to/rectified-sheet.jpg
```

The verification script checks inference on the supplied image. A local
expected score vector enables an additional consistency comparison. The
application's optional startup fixture check is skipped when local fixture
images are absent. This does not disable checkpoint validation or inference.

## Upload gate

`gates/ocr_lead_labels_v1.yaml` configures Tesseract OCR. It counts distinct
printed lead labels and requires at least four of the configured labels.
Tesseract and English language data must be installed locally. The gate has
no learned checkpoint and is independent of the classifier.

## Runtime configuration

`CARDIOSENTRY_DEVICE`, `CARDIOSENTRY_NUM_THREADS`, and the descriptor's runtime
settings control inference. Set `CARDIOSENTRY_ACTIVE_MODEL` explicitly on
filesystems without symlink support. Model weights are ignored by Git.
