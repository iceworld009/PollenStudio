# Pollen Studio

A local web app for building object detection datasets and training TFLite models for a Limelight camera. Built by **Harambe Cartel, Team 20912**.

Pollen Studio supports three game elements: pollen, blue Nectar, and red Nectar. Upload camera photos, draw or correct boxes, review model predictions, and train from the images you have approved.

## Install and start

Download the Linux or Windows source package from Releases and extract it into a permanent folder. Setup downloads Python 3.10 and the application dependencies; internet access is required. No private model or dataset is bundled.

### Linux (x86-64)

Install `curl` and CA certificates using your package manager, then run:

```bash
bash Setup-Linux.sh
bash Start-Linux.sh
```

Open **http://localhost:8765**. Keep the terminal open while using Studio. The managed Python environment lives in the application folder; no system Python changes are needed.

### Windows 10/11 (x86-64)

1. Double-click **Setup-Windows.cmd**. Setup uses Ubuntu 22.04 under WSL2. Windows may request administrator permission, a restart, and an Ubuntu username/password during first-time installation. Run setup again after completing those steps.
2. Double-click **Start-Windows.cmd**.
3. Open **http://localhost:8765** in your Windows browser.

Install an up-to-date NVIDIA Windows driver before GPU training. The application and datasets run inside WSL2 at `~/PollenStudio`; keep that folder when updating. Setup copies source files from the extracted package without replacing datasets or models. Do not install a separate Linux NVIDIA display driver inside WSL.

Compute defaults to **Automatic**: Studio checks the current driver and runs a small TensorFlow GPU operation. A usable GPU is selected automatically; otherwise training uses CPU and explains when detected hardware is unavailable to TensorFlow. You can explicitly select CPU or GPU. Device checks refresh periodically and Automatic mode checks again when a training job starts.

An RTX 3050 is an intended training target, but training batch size depends on available VRAM. The Windows package uses the same Linux TensorFlow trainer through WSL2, rather than native Windows CUDA. A real Windows/RTX 3050 installation has not yet been validated by the maintainers.

### Development setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./start_studio.sh --host 127.0.0.1 --port 8765
```

You can upload and manually annotate photos immediately. Inference needs a compatible uploaded model; training dependencies are installed separately from the Training tab.

## Work with your team

```bash
./start_studio.sh --host 0.0.0.0 --port 8765
```

Teammates can open `http://YOUR_SERVER_IP:8765` from the same network or through Tailscale. Allow TCP port 8765 through the server firewall. Keep this service on a trusted network: it has a shared workspace, no user accounts, and no per-user permissions. GitHub hosts the source code; the app still runs on your own computer.

## Dataset workflow

1. In **Dataset**, upload JPG, PNG, or WebP photos from any connected device. Each upload supports up to **1,000 photos and 512 MB**. Exact duplicates are skipped.
2. In **Review**, choose an element and draw its bounding box. Select an existing box to change its class or resize it. Save reviewed photos with **Save & include**.
3. Use the **Photo library** to inspect the dataset, review quality warnings, or move unwanted photos to Trash.
4. Archive reviewed photos to keep them in training while removing them from the auto-annotation queue. Archived annotations remain editable.
5. Export a reviewed TFRecord dataset or train a model from **Training**.

Only included photos enter training. Pending, excluded, and trashed photos do not. Label every target in an included photo, including Nectar in older pollen photos. Unlabeled targets teach the detector to treat them as background.

Keep separate recording sessions for training and validation. Reserve a fresh recording for final testing rather than choosing a model using the same footage repeatedly.

## Upload an existing dataset

In **Dataset → Original training data**, choose a TensorFlow TFRecord ZIP from your device and click **Upload dataset**. Archives up to **10 GB** are transferred in 8 MB chunks directly into server storage. The browser shows progress and can cancel an upload. Keep the page open until upload finishes; import then runs as an Activity job.

Training and validation folders and the three supported class IDs are preserved, including archives with an enclosing directory. This uploader accepts TFRecord ZIPs, not COCO JSON, VOC XML, or YOLO labels. Convert those formats before uploading. Maximum extracted size is 40 GB, with 100,000 archive entries. Leave enough free space for the ZIP and imported photos. Successful imports remove the uploaded ZIP; failed imports retain it in `studio_data/dataset-uploads/` for diagnosis.

## Models and labels

Studio accepts SSD TFLite models with `TFLite_Detection_PostProcess` outputs. Raw YOLO output layouts and Coral-compiled models cannot run in the Studio CPU preview.

Class order is fixed:

| Annotation ID | TFRecord ID | Label |
| --- | --- | --- |
| 0 | 1 | `pollen` |
| 1 | 2 | `blue_nectar` |
| 2 | 3 | `red_nectar` |

For a three-class model, upload a labels file containing those three names, one per line in that order. Existing single-class pollen models are supported with a one-line `pollen` labels file. They cannot annotate Nectar.

Select the input normalization used by your model exporter. Models trained in Studio use **−1 to 1**. Renaming a model does not change its position in the date-ordered model list.

Select a saved model in **Review** and click **Download model ZIP** to export its CPU TFLite file, matching labels, and inference metadata. The download excludes photos, checkpoints, and local filesystem paths. Full training bundles remain available in Activity.

## Training and Limelight export

Use **Prepare local trainer** in the Training tab to install the separate TensorFlow environment, pretrained SSD MobileNet V2 weights, and compiler. Setup downloads third-party software and requires internet access and disk space. NVIDIA GPU training also requires a compatible driver and access to its device files; CPU training is supported.

The first three-class model must start from pretrained weights. A pollen-only checkpoint has an incompatible detection head. Later three-class runs can continue from their saved checkpoints.

A completed run includes:

- Quantized CPU TFLite model and matching labels.
- Coral-compiled TFLite model for devices with an Edge TPU.
- Training loss graph, logs, and validation precision/recall.
- A local TensorFlow checkpoint for continued training.

Use the **CPU TFLite file on Limelight 3A**. Use the Coral file only on hardware with a compatible accelerator. Check actual camera FPS and detections before deploying a model to the robot. Fixed-threshold validation metrics are not a guarantee of performance on a new field.

## Camera capture

`capture.py` uses FFmpeg to capture a camera stream. Install FFmpeg and supply the stream URL for your camera:

```bash
python3 capture.py --help
python3 capture.py --url http://CAMERA_IP:5800/ images --interval 1 --seconds 120
```

Check the command help for preview, single-photo, and video options. Captures are stored separately from reviewed training data.

## Storage and development

The public app has no AI Review queue or built-in experiment history. Auto-annotation uses only a model you upload or train locally. Legacy local review records remain readable for compatibility.

The interface uses plain HTML, CSS, and JavaScript. The server and dataset tools use Python; no frontend build is required.

- `studio/`: server, annotation storage, inference, export, and training code.
- `studio/static/`: browser interface.
- `tests/`: automated tests.
- `studio_data/`: local database, uploads, training environments, jobs, and exports.
- `models/`: optional local model files.
- `captures/`: camera recordings.

Back up `studio_data/` and any imported source photos before moving an installation. Local datasets, environments, captures, and model binaries are excluded from Git. Do not force-add them when publishing source.

```bash
.venv/bin/python -m unittest discover -s tests -q
```

## Isolated validation

On Linux, install Bubblewrap and run `bash packaging/test-container.sh` after setup. This runs the automated suite and HTTP integration checks in temporary Linux namespaces, with a read-only filesystem, hidden home directory, disposable data storage, and no host GPU devices. Network access is shared for local HTTP tests. This is a Bubblewrap container, not a Docker image.

The smoke test exercises a fresh workspace, CPU fallback, request authentication, upload limits, a multi-chunk dataset import, preserved classes and splits, annotation saves, cancellation, and training setup guards. It uses synthetic images and deletes temporary data. It does not install the full training runtime or benchmark model quality. NVIDIA GPU execution must also be tested on a host with a usable driver; Windows/WSL2 still needs platform testing.

For an optional end-to-end training smoke test after runtime setup, run:

```bash
PYTHONPATH=. .venv/bin/python packaging/training_smoke.py
```

This performs two synthetic training steps, exports a three-class quantized TFLite model, checks CPU inference and labels, and runs Coral compilation. Outputs are temporary and are deleted afterward. It tests functionality, not model accuracy.

## License

[MIT](LICENSE) — Copyright © 2026 Harambe Cartel (Team 20912). You may use, modify, and redistribute Pollen Studio; retain the copyright and license notice. Downloaded dependencies, pretrained weights, and datasets have their own licenses.
