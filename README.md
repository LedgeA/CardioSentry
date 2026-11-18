# CardioSentry

CardioSentry is a research web application for processing smartphone photographs
of printed 12-lead ECGs. It checks uploads, detects the paper, lets the user
adjust the crop, and runs an image classifier built on
[ConvNeXt V1 Tiny](https://github.com/facebookresearch/ConvNeXt). Results include
model scores, thresholds, timing, and optional Grad-CAM overlays.

## Application workflow

1. Upload or photograph an ECG sheet.
2. Confirm the detected paper corners and straighten the image.
3. Review the classifier scores and optional explanations.
4. Save or print the result for research review.

The frontend uses React and TypeScript. FastAPI serves the application and
runs PyTorch inference; PostgreSQL stores accounts and capture records.
Images are stored on the application server.

## Run locally

See the [application guide](web-application/README.md) for setup and the
[model guide](web-application/models/README.md) for checkpoint requirements.
The classifier checkpoint is supplied separately. Inference uses the real
model; scores are not simulated.

## Scope

CardioSentry is an experimental research application. Classifier scores,
visual explanations, and guidance are not a diagnosis.
