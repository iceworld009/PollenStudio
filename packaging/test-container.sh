#!/usr/bin/env bash
# Linux namespace container: read-only host filesystem, hidden home and devices,
# disposable app storage. Network is shared for local HTTP integration tests.
set -euo pipefail
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command -v bwrap >/dev/null || { echo 'Install bubblewrap to run the isolated tests.' >&2; exit 1; }
[[ -x "$release/.venv/bin/python" ]] || { echo 'Run Setup-Linux.sh first.' >&2; exit 1; }
python_real=$(readlink -f "$release/.venv/bin/python")
python_prefix=$(dirname -- "$(dirname -- "$python_real")")
if [[ -d "$release/.python" ]]; then python_prefix="$release/.python"; fi
exec bwrap --unshare-all --share-net --ro-bind / / \
  --tmpfs /home --dir "$python_prefix" --ro-bind "$python_prefix" "$python_prefix" \
  --tmpfs /mnt --ro-bind "$release" /mnt/app --tmpfs /mnt/app/studio_data \
  --tmpfs /tmp --tmpfs /run --proc /proc --dev /dev --clearenv \
  --setenv PATH /usr/bin:/bin --setenv HOME /tmp --setenv PYTHONWARNINGS ignore \
  --setenv PYTHONPATH /mnt/app --chdir /mnt/app \
  /usr/bin/bash -c '.venv/bin/python -m unittest discover -s tests -q && .venv/bin/python packaging/container_smoke.py'
