"""Build source-only platform distributions from the committed project tree."""
from pathlib import Path
import subprocess
import zipfile
import hashlib

root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
destination = root / 'dist'
destination.mkdir(exist_ok=True)
checksums = []
for platform in ('Linux', 'Windows'):
    archive = destination / f'PollenStudio-{platform}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(filter(None, files)):
            file = root / name
            if not file.is_file() or name.startswith(('.git/', 'studio_data/', 'models/', 'captures/', '.venv/')):
                continue
            bundle.write(file, 'PollenStudio/' + name)
    checksums.append(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name)
(destination / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n')
print('Packages created in', destination)
