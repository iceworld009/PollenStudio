#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'This release requires 64-bit x86 Linux. On Windows, run Setup-Windows.cmd.' >&2
  exit 1
fi
if ! command -v curl >/dev/null; then
  echo 'Install curl with your system package manager, then run this script again.' >&2
  exit 1
fi
# Keep the installer and managed Python inside the application directory.
export UV_INSTALL_DIR="$PWD/.tools"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_NO_MODIFY_PATH=1
if [[ ! -x .tools/uv ]]; then
  installer=$(mktemp)
  trap 'rm -f "$installer"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh -o "$installer"
  sh "$installer"
fi
.tools/uv python install 3.10
if [[ ! -x .venv/bin/python ]]; then
  .tools/uv venv --python 3.10 --seed .venv
fi
.tools/uv pip install --python .venv/bin/python -r requirements.txt
printf '\nPollen Studio is ready. Run ./Start-Linux.sh, then open http://localhost:8765\n'
printf 'For GPU training, install a compatible NVIDIA driver, then use Prepare local trainer in Studio.\n'
