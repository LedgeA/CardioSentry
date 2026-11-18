# CardioSentry web application

The application accepts photographs of printed ECGs, checks the input, detects
paper corners, straightens the sheet, and runs an image classifier. It stores
scores, operating thresholds, model identity, and the exact inference image.
Grad-CAM overlays and a printable report support research review.

## Requirements

- Python 3.12 or newer, Node.js and npm.
- A local PostgreSQL server and an empty application database.
- Tesseract OCR with English language data, available on `PATH`.
- The matching classifier checkpoint, supplied separately.

## Run locally

Run these commands from `web-application/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
cp .env.example .env
mkdir -p models/weights
cd frontend
npm ci
npm run build
cd ..
```

Configure `CARDIOSENTRY_DATABASE_URL` in `.env` with your local PostgreSQL URL.
The database role needs permission to create test databases when running the
full test suite. Set `CARDIOSENTRY_LOCAL_ONLY=true` to bind to loopback.

Place `convnext_tiny_rfpn_clean12_aug_inference.pt` in `models/weights/`.
See the [model guide](models/README.md) for checkpoint validation. Checkpoints,
credentials, uploaded images, and installed dependencies are ignored by Git.

```bash
python -m backend.main
```

Open `http://127.0.0.1:8000`. An empty installation creates an initial
`admin` / `admin` account; change its password in Settings. For a phone on the
same trusted network, disable local-only mode and use the printed LAN URL.
Camera capture in a mobile browser requires HTTPS; [make_cert.sh](scripts/make_cert.sh)
provides local certificate setup.

## Processing

The OCR gate checks for distinct printed lead labels before an upload is stored.
Non-ECG and poor-quality uploads are rejected. Accepted images pass through
paper detection, a corner editor, perspective correction, and descriptor-driven
preprocessing. Predictions use twelve independent labels and the configured
operating thresholds. These are experimental model outputs, not a diagnosis.

Grad-CAM explains a stored prediction using its inference image. Maps show
model attention rather than evidence of a particular physiological change.
[Guidance rules](docs/clinical-guidance.md) describe the separate symptom and
finding-based urgency layer.

## Storage and privacy

PostgreSQL stores account credentials, sessions, capture metadata, prediction
runs, and explanation records. Each account can access only its own captures.
Images and optional attribution arrays live in `CARDIOSENTRY_DATA_DIR`.
Deleting a capture moves its files to a local trash directory; permanent erasure
must also cover the database rows, trash, exports, and backups.

Set the privacy controller and contact in `.env` before collecting another
person's ECG. Use synthetic images for demonstrations. Keep captures, database
backups, credentials, certificates, and checkpoints outside public source
control. The application performs inference and OCR locally on its server.

## Verification

```bash
cd frontend
npm run build
```

The application runs without the local test suite or fixture images. The
optional startup fixture check is skipped when its image is unavailable;
checkpoint validation and real inference still run normally. For a standalone
inference check with your own image, use
`python scripts/verify_model.py --fixture /path/to/rectified-sheet.jpg`.

## Linux deployment

The [setup script](deploy/setup.sh), [systemd unit](deploy/cardiosentry.service),
and [Caddy template](deploy/Caddyfile) provide deployment configuration.
Install PostgreSQL, Python's venv support, Node.js, npm, Tesseract with English
language data, OpenSSL, and rsync first. Stage the classifier checkpoint in
`/opt/cardiosentry/weights/`, then run the setup script as root. Configure the
privacy contact and HTTPS before accepting uploads.
