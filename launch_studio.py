#!/usr/bin/env python3
"""Start Pollen Studio in the background and open its interface."""
from pathlib import Path
import os
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
port = 8765
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
    try:
        connection.connect(('1.1.1.1', 80))
        host = connection.getsockname()[0]
    except OSError:
        host = '127.0.0.1'

def running(url):
    try:
        with urllib.request.urlopen(url + '/api/state', timeout=2) as response:
            import json
            return 'photo_audit' in json.load(response)
    except (OSError, ValueError):
        return False

url = f'http://{host}:{port}'
if not running(url):
    if running(f'http://127.0.0.1:{port}'):
        url = f'http://127.0.0.1:{port}'
    else:
        logs = ROOT / 'studio_data'
        logs.mkdir(exist_ok=True)
        with (logs / 'web-interface.log').open('ab') as log:
            process = subprocess.Popen([str(ROOT / 'start_studio.sh'), '--host', host, '--port', str(port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(40):
            if running(url):
                break
            if process.poll() is not None:
                raise SystemExit(f'Start failed. See {logs / "web-interface.log"}')
            time.sleep(.25)
        else:
            raise SystemExit(f'Start timed out. See {logs / "web-interface.log"}')
print(f'Pollen Studio: {url}', flush=True)
print(f'Log: {ROOT / "studio_data/web-interface.log"}', flush=True)
if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
    subprocess.Popen(['xdg-open', url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
