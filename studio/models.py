"""Persist uploaded model versions after isolated CPU compatibility checks."""
import hashlib
import io
import re
import zipfile
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from studio.core import DATA, DEFAULT_MODEL, ROOT

MAX_MODEL_BYTES = 64 * 1024 * 1024
NORMALIZATIONS = {'zero_one', 'minus_one_one', 'raw'}


def available_models():
    result = [{'id':'pln_600', 'name':'PLN_600 · original', 'path':str(DEFAULT_MODEL),
               'normalization':'zero_one', 'labels':['pollen'],
               'created_at':DEFAULT_MODEL.stat().st_mtime if DEFAULT_MODEL.exists() else 0}] if DEFAULT_MODEL.is_file() else []
    for file in sorted((DATA / 'models').glob('*/metadata.json')):
        metadata = json.loads(file.read_text())
        metadata['path'] = str(file.with_name('model.tflite'))
        # Model bytes keep their date when display names or metadata change.
        metadata.setdefault('created_at', file.with_name('model.tflite').stat().st_mtime)
        result.append(metadata)
    return sorted(result, key=lambda model: (-model['created_at'], model['id']))


def resolve_model(ident):
    for model in available_models():
        if model['id'] == ident:
            return model
    raise ValueError('Model not found. Refresh the model list and select again.')


def store_upload(filename, content, normalization='zero_one', labels=b''):
    if Path(filename).suffix.lower() != '.tflite':
        raise ValueError('Choose a .tflite file. For Limelight ZIPs, extract and upload the 8bit CPU model.')
    if not 8 <= len(content) <= MAX_MODEL_BYTES or content[4:8] != b'TFL3':
        raise ValueError('Invalid TFLite file or file exceeds the 64 MB limit.')
    if normalization not in NORMALIZATIONS:
        raise ValueError('Choose a supported input normalization.')
    try:
        names = [line.strip() for line in labels.decode('utf-8-sig').splitlines() if line.strip()] if labels else ['pollen']
    except UnicodeDecodeError as exc:
        raise ValueError('Labels must be a UTF-8 text file.') from exc
    if names not in (['pollen'], ['pollen','blue_nectar','red_nectar']):
        raise ValueError('Labels must be pollen, or pollen / blue_nectar / red_nectar in that order.')
    ident = uuid.uuid4().hex
    directory = DATA / 'models' / ident
    directory.mkdir(parents=True)
    try:
        model_file = directory / 'model.tflite'
        model_file.write_bytes(content)
        config_file = directory / 'inspect.json'
        report_file = directory / 'inspection.json'
        config_file.write_text(json.dumps({'kind':'inspect-model', 'model':str(model_file),
                                          'normalization':normalization, 'report':str(report_file)}))
        with (directory / 'inspection.log').open('w') as log:
            try:
                result = subprocess.run([sys.executable, '-u', '-m', 'studio.worker', str(config_file)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('Model compatibility check timed out. Try a smaller CPU-compatible model.') from exc
        if result.returncode != 0 or not report_file.is_file():
            detail = (directory / 'inspection.log').read_text(errors='replace').splitlines()
            detail = next((line for line in reversed(detail) if line.strip()), 'CPU compatibility check failed.')
            raise ValueError('Cannot use this model: '+detail[:500])
        inspection=json.loads(report_file.read_text())
        if inspection.get('num_classes',1) != len(names):
            raise ValueError('Labels file must match the number of model classes.')
        metadata = {'id':ident, 'name':Path(filename.replace('\\','/')).name, 'labels':names,
                    'created_at':time.time(),
                    'normalization':normalization, 'sha256':hashlib.sha256(content).hexdigest(),
                    'inspection':json.loads(report_file.read_text())}
        temporary = directory / 'metadata.tmp'
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(directory / 'metadata.json')
        return metadata
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def export_model(ident):
    """Package inference files without local paths, datasets or checkpoints."""
    model=resolve_model(ident)
    content=Path(model['path']).read_bytes()
    stem=re.sub(r'[^A-Za-z0-9_.-]+','_',model['name']).removesuffix('.tflite').strip('._')[:100] or 'model'
    info={key:model[key] for key in ('name','labels','normalization','inspection','recommended_confidence') if key in model}
    info['sha256']=hashlib.sha256(content).hexdigest()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(stem+'.tflite',content)
        archive.writestr(stem+'_labels.txt','\n'.join(model['labels'])+'\n')
        archive.writestr('model-info.json',json.dumps(info,indent=2))
    return stem+'.zip',buffer.getvalue()
