Photo library cleanup: remove obsolete AI/manual review warnings, local contact-sheet notes, capture-specific metadata and hidden auto-trash policy. Filter legacy cached reports so stale review flags do not reappear. Quality scans only report reproducible quality, duplicate and split-overlap warnings; they do not change photos or annotations.

Model download: select any saved model in Review and click Download model ZIP. Includes the CPU TFLite file, matching labels and inference metadata, without photos, checkpoints or local filesystem paths. Verified by isolated tests and a browser download check.

Validation: 50 automated tests in a temporary Bubblewrap container (one optional private-model test skipped); isolated HTTP upload/import/annotation checks; desktop/mobile browser checks; and a two-step synthetic three-class GPU train/export/quantize/inference/Coral compile test on GTX 1650. Fixed TensorFlow decoder aliases and narrow-range calibration compatibility discovered by that smoke test. No benchmark-quality model was produced or distributed.

Automatic compute selection: live TensorFlow GPU verification, CPU fallback, and manual overrides. Includes isolated clean-install tests, an optional synthetic training/export smoke test, serialized job launches, and a fix for TensorFlow record-decoder compatibility. Windows/WSL2 still needs hardware validation.

Dataset upload: import TensorFlow TFRecord ZIPs up to 10 GB directly from Windows or another browser. Includes chunked transfer, progress, cancellation, retry handling, disk-space checks, and preserved train/validation splits. Removes remaining PLN_600 wording from the interface.

Setup reliability fix: resolve requirements.txt relative to the installer, validate the complete source folder before copying to WSL, and report missing files explicitly.

Windows setup fix: preserve Windows path separators, spaces, and shell quoting when passing commands to WSL. Normalize script line endings before running in Linux.

Pollen Studio is a local annotation and training workspace from Harambe Cartel, Team 20912.

Choose the package for your operating system:

- **Linux:** extract and run `bash Setup-Linux.sh`, then `bash Start-Linux.sh`.
- **Windows:** extract and double-click `Setup-Windows.cmd`, then `Start-Windows.cmd`. Training runs inside WSL2 Ubuntu 22.04. First-time installation may require administrator permission, an Ubuntu user account, and a Windows restart.

Open http://localhost:8765 after starting the server. Internet access is required for setup. NVIDIA training requires a compatible host driver; the training runtime is installed separately from the Training tab.

Both packages contain the same application source plus platform launchers. No models or datasets are included. Supports pollen, blue Nectar, and red Nectar, with photo uploads of up to 1,000 files / 512 MB.

Linux application tests and fresh-workspace checks were performed by the maintainers. Windows/WSL2 hardware validation is still in progress. GPU training and Coral compilation remain dependent on external runtime downloads and compatible hardware.

MIT licensed. Retain the Harambe Cartel (Team 20912) copyright and license notice when redistributing.
