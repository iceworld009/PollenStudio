#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then bash packaging/setup-linux.sh; fi
exec .venv/bin/python -u -m studio.server --host 0.0.0.0 --port 8765 "$@"
