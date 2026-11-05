"""Local web app with optional LAN binding; UI needs only the standard library."""
import argparse
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import signal
import shutil
import subprocess
import sys
import time
import threading
import uuid
import zipfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from studio.core import ROOT, DATA, DEFAULT_MODEL, connect, images, import_folder, snapshot, validate_boxes, trash_image, trash_images, archive_images, archive_all_included, save_review
from studio.models import available_models, resolve_model, store_upload, MAX_MODEL_BYTES, export_model
from studio import training, training_runtime, photo_audit, dataset_upload

TOKEN = secrets.token_urlsafe(32)
PROCESSES = {}
JOB_LOCK = threading.RLock()
TEST_PREVIEW_ARTIFACTS = {'test_gallery.html', 'test_preview.jpg', 'test_examples.json'} | {f'test_example_{n}.jpg' for n in range(1, 7)}


def live_experiment():
    # Reserved for future managed training searches; ordinary jobs use jobs().
    return None


def jobs():
    with JOB_LOCK:
        return _jobs()


def _jobs():
    result = []
    for file in sorted((DATA / 'jobs').glob('*/job.json'), reverse=True):
        job = json.loads(file.read_text())
        if job['status'] == 'running':
            p = PROCESSES.get(job['id'])
            code = p.poll() if p else -1
            if code is not None:
                job['status'] = 'completed' if code == 0 else 'failed' if p else 'interrupted'
                job['exit_code'] = code
                file.write_text(json.dumps(job))
        log = file.with_name('log.txt')
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 16000))
                job['log'] = stream.read().decode(errors='replace')
        progress = file.with_name('progress.json')
        if progress.is_file():
            job['progress'] = json.loads(progress.read_text())
        result_file = file.parent / 'model/result.json'
        if job['kind'] == 'train' and result_file.is_file():
            job['result'] = json.loads(result_file.read_text())
        result.append(job)
    return sorted(result, key=lambda j: j.get('created', 0), reverse=True)


def busy():
    return any(j['status'] == 'running' for j in jobs())


def launch(config, directory=None):
    with JOB_LOCK:
        return _launch(config,directory)


def _launch(config, directory=None):
    if busy():
        raise ValueError('A job is already running. Wait or stop it first.')
    directory = directory or DATA / 'jobs' / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / 'config.json'
    config_file.write_text(json.dumps(config))
    job = {'id': directory.name, 'kind': config['kind'], 'status': 'running', 'output': str(directory), 'created': time.time()}
    with (directory / 'log.txt').open('w') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'studio.worker', str(config_file)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PROCESSES[job['id']] = p
    (directory / 'job.json').write_text(json.dumps(job))
    return job


def stop_job(ident):
    with JOB_LOCK:
        return _stop_job(ident)


def _stop_job(ident):
    p = PROCESSES.get(ident)
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        file = DATA / 'jobs' / ident / 'job.json'
        job = json.loads(file.read_text()); job['status'] = 'cancelled'
        file.write_text(json.dumps(job))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, value, status=200, content_type='application/json', download=None):
        body = json.dumps(value).encode() if content_type == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="'+download+'"')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(body)

    def send_file(self, file, content_type, download=False):
        with file.open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(file.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if download:
                filename = download if isinstance(download, str) else 'pollen-reviewed-dataset.zip'
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile, length=1024*1024)

    def valid_host(self):
        hosts = {self.connection.getsockname()[0] if hasattr(self, 'connection') else '127.0.0.1', '127.0.0.1', 'localhost', getattr(self.server, 'server_address', ('127.0.0.1',))[0]}
        return self.headers.get('Host') in {f'{host}:{self.server.server_port}' for host in hosts}

    def upload_photos(self, size):
        from PIL import Image
        import io
        if busy():
            raise ValueError('Wait for the running job before uploading photos.')
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data;') or not 0 < size <= 512 * 1024 * 1024:
            raise ValueError('Choose photos, up to 512 MB per upload.')
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError('Upload interrupted. Try again.')
        message = BytesParser(policy=policy.default).parsebytes(b'Content-Type: '+content_type.encode()+b'\r\nMIME-Version: 1.0\r\n\r\n'+raw)
        photos = []; split = 'train'
        for part in message.iter_parts():
            name = part.get_param('name', header='content-disposition')
            if part.is_multipart() or name not in ('photos', 'split'):
                raise ValueError('Invalid photo upload.')
            data = part.get_payload(decode=True)
            if name == 'split':
                split = data.decode()
                continue
            suffix = Path(part.get_filename() or '').suffix.lower()
            if suffix not in ('.jpg', '.jpeg', '.png', '.webp'):
                raise ValueError('Choose JPG, PNG or WebP photos.')
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ('JPEG','PNG','WEBP'):
                    raise ValueError('Unsupported image format.')
                image.verify()
            photos.append((suffix, data))
        if not photos or len(photos) > 1000 or split not in ('train','val'):
            raise ValueError('Choose 1–1000 photos and a valid split.')
        folder = DATA / 'uploads' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        try:
            for index, (suffix, data) in enumerate(photos):
                (folder / (str(index)+suffix)).write_bytes(data)
            count = import_folder(folder, split)
            with connect() as db:
                retained = {row[0] for row in db.execute('SELECT path FROM images WHERE session=?',(str(folder),))}
            for file in folder.iterdir():
                if str(file) not in retained:
                    file.unlink()
            return {'count':count, 'skipped':len(photos)-count}
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def upload_model(self, size):
        if busy():
            raise ValueError('Wait for the running job or stop it before uploading a model.')
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data;') or size <= 0 or size > MAX_MODEL_BYTES + 131072:
            raise ValueError('Upload a .tflite file of at most 64 MB using the model form.')
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError('Upload was interrupted. Try again.')
        message = BytesParser(policy=policy.default).parsebytes(b'Content-Type: '+content_type.encode()+b'\r\nMIME-Version: 1.0\r\n\r\n'+raw)
        if not message.is_multipart():
            raise ValueError('Invalid multipart upload.')
        parts = {}
        for part in message.iter_parts():
            name = part.get_param('name', header='content-disposition')
            if name not in ('model', 'labels', 'normalization') or name in parts or part.is_multipart():
                raise ValueError('Invalid model upload fields.')
            parts[name] = part
        if 'model' not in parts or not parts['model'].get_filename():
            raise ValueError('Choose a .tflite model file.')
        labels = parts['labels'].get_payload(decode=True) if 'labels' in parts else b''
        if len(labels) > 65536:
            raise ValueError('Labels file is too large.')
        normalization = parts['normalization'].get_payload(decode=True).decode() if 'normalization' in parts else 'zero_one'
        return store_upload(parts['model'].get_filename(), parts['model'].get_payload(decode=True), normalization, labels)

    def do_GET(self):
        if not self.valid_host():
            return self.send({'error': 'Unrecognized server address'}, 403)
        path = urlsplit(self.path).path
        try:
            if path == '/api/state':
                return self.send({'token': TOKEN, 'images': images(), 'trash': images(trashed=True), 'jobs': jobs(), 'model': str(DEFAULT_MODEL), 'models':available_models(), 'training':training_runtime.status(), 'training_runs':training.available_runs(), 'photo_audit':photo_audit.report_for_state(), 'dependencies': {k: importlib.util.find_spec(k) is not None for k in ('numpy','PIL','ai_edge_litert')}})
            if path.startswith('/api/model-export/'):
                filename,content=export_model(path.rsplit('/',1)[1])
                return self.send(content,content_type='application/zip',download=filename)
            if path.startswith('/api/training/'):
                parts = path.split('/')
                if len(parts) != 5 or len(parts[3]) != 32 or any(c not in '0123456789abcdef' for c in parts[3]) or parts[4] not in training.ARTIFACT_NAMES | TEST_PREVIEW_ARTIFACTS:
                    raise ValueError('Invalid training artifact')
                file = DATA/'jobs'/parts[3]/'model/artifacts'/parts[4]
                inline = file.name in TEST_PREVIEW_ARTIFACTS and file.suffix in {'.html', '.jpg', '.png'} and parse_qs(urlsplit(self.path).query).get('view') == ['1']
                return self.send_file(file, mimetypes.guess_type(file.name)[0] or 'application/octet-stream', download=False if inline else file.name)
            if path.startswith('/api/thumbnail/'):
                ident=path.rsplit('/',1)[1]
                with connect() as db:
                    row=db.execute('SELECT * FROM images WHERE id=?',(ident,)).fetchone()
                if not row:raise ValueError('Image not found')
                from studio.thumbnails import thumbnail
                file=thumbnail(dict(row),boxes='boxes=0' not in urlsplit(self.path).query)
                return self.send_file(file,'image/jpeg')
            if path.startswith('/api/image/'):
                ident = path.rsplit('/', 1)[1]
                with connect() as db:
                    row = db.execute('SELECT path FROM images WHERE id=?', (ident,)).fetchone()
                if not row:
                    raise ValueError('Image not found')
                file = Path(row['path'])
                return self.send_file(file, mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            if path.startswith('/api/export/'):
                ident = path.rsplit('/', 1)[1]
                if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
                    raise ValueError('Invalid export')
                filename = 'pollen-limelight-model.zip' if (DATA/'jobs'/ident/'model/result.json').is_file() else 'pollen-reviewed-dataset.zip'
                return self.send_file(DATA / 'exports' / (ident+'.zip'), 'application/zip', download=filename)
            files = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in files:
                return self.send({'error':'Not found'}, 404)
            file = ROOT / 'studio/static' / files[path]
            return self.send(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or 'text/plain')
        except (ValueError, OSError) as exc:
            self.send({'error':str(exc)}, 400)

    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-Studio-Token') != TOKEN:
            return self.send({'error':'Refresh the local Studio page.'}, 403)
        try:
            path = urlsplit(self.path).path
            size = int(self.headers.get('Content-Length', '0'))
            if path.startswith('/api/dataset-chunk/'):
                if size < 1 or size > dataset_upload.CHUNK_BYTES:
                    self.close_connection=True
                    raise ValueError('Chunk exceeds 8 MB')
                ident=path.rsplit('/',1)[1]
                offset=int(self.headers.get('X-Upload-Offset','-1'))
                return self.send(dataset_upload.append(ident,offset,self.rfile,size))
            if size < 0 or size > (512*1024*1024+131072 if path == '/api/upload-photos' else MAX_MODEL_BYTES+131072 if path == '/api/upload-model' else 2_000_000):
                self.close_connection = True
                raise ValueError('Request too large')
            if path == '/api/upload-photos':
                return self.send(self.upload_photos(size))
            if path == '/api/upload-model':
                return self.send({'model':self.upload_model(size)})
            body = json.loads(self.rfile.read(size))
            path = urlsplit(self.path).path
            if path == '/api/stop':
                stop_job(body['id']); return self.send({'ok':True})
            if busy():
                raise ValueError('Wait for the running job or stop it before changing the dataset.')
            if path == '/api/archive-all':
                return self.send({'ok':True,'count':archive_all_included()})
            if path == '/api/purge-excluded':
                return self.send(launch({'kind':'purge-excluded'}))
            if path in ('/api/archive-images','/api/unarchive-images'):
                archive_images(body['ids'],boxes=body.get('boxes'),restore=path=='/api/unarchive-images')
                return self.send({'ok':True})
            if path == '/api/audit-photos':
                return self.send(launch({'kind':'audit-photos'}))
            if path in ('/api/delete-images','/api/restore-images'):
                trash_images(body['ids'],restore=path=='/api/restore-images')
                return self.send({'ok':True})
            if path == '/api/setup-training':
                if training_runtime.status()['ready']:
                    raise ValueError('The local TensorFlow trainer is already ready.')
                return self.send(launch({'kind':'setup-training'}))
            if path in ('/api/delete-image', '/api/restore-image'):
                trash_image(body['id'], restore=path == '/api/restore-image')
                return self.send({'ok':True})
            if path == '/api/import':
                return self.send({'count':import_folder(body['folder'], body['split'])})
            if path == '/api/dataset-begin':
                return self.send(dataset_upload.begin(body.get('name'),body.get('size')))
            if path == '/api/dataset-cancel':
                dataset_upload.cancel(body['id']);return self.send({'ok':True})
            if path == '/api/dataset-finish':
                archive=dataset_upload.complete(body['id'])
                return self.send(launch({'kind':'import','archive':str(archive),'uploaded_dataset':True}))
            if path == '/api/import-original':
                archive = Path(body['archive']).expanduser().resolve()
                if not archive.is_file():
                    raise ValueError('Dataset ZIP not found')
                return self.send(launch({'kind':'import','archive':str(archive)}))
            if path == '/api/review':
                save_review(body['id'], body['boxes'], body['status'])
                return self.send({'ok':True})
            if path in ('/api/predict', '/api/evaluate'):
                rows = [r for r in images() if r['status'] == 'pending' and not r['archived'] and (not body.get('id') or r['id'] == body['id'])]
                if path == '/api/evaluate':
                    rows = [r for r in images() if r['split'] == 'val' and r['status'] == 'include']
                if not rows:
                    raise ValueError('No eligible images for this job.')
                confidence = float(body['confidence'])
                if not 0.01 <= confidence <= 1:
                    raise ValueError('Confidence must be between 0.01 and 1')
                normalization = 'zero_one'
                if body.get('model_id'):
                    registered = resolve_model(body['model_id'])
                    model = Path(registered['path'])
                    normalization = registered['normalization']
                else:
                    model = Path(body.get('model') or DEFAULT_MODEL).expanduser().resolve()
                if not model.is_file() or (not body.get('model_id') and model != DEFAULT_MODEL):
                    raise ValueError('Select a TFLite model from the model selector.')
                directory = DATA / 'jobs' / uuid.uuid4().hex
                kind = 'evaluate' if path == '/api/evaluate' else 'predict'
                return self.send(launch({'kind':kind,'model':str(model),'normalization':normalization,'confidence':confidence,'images':rows,
                                         'metrics':str(directory / 'metrics.json')}, directory))
            if path == '/api/export':
                export_format = body.get('format', 'yolo')
                if export_format not in ('yolo', 'tfrecord'):
                    raise ValueError('Choose YOLO or TFRecord export.')
                if not any(r['status'] == 'include' for r in images()):
                    raise ValueError('Include reviewed images first.')
                ident = uuid.uuid4().hex
                return self.send(launch({'kind':'export', 'format':export_format, 'id':ident}, DATA / 'jobs' / ident))
            if path == '/api/train':
                experiment = live_experiment()
                if experiment and experiment['active']:
                    raise ValueError('The GPU search is running. Let it finish before starting another training job.')
                if not training_runtime.status()['ready']:
                    raise ValueError('Prepare the local TensorFlow trainer before training.')
                options = training.parameters(body)
                if options['device'] == 'gpu' and not training_runtime.status()['gpus']:
                    raise ValueError('TensorFlow GPU is unavailable. Choose CPU.')
                rows = images()
                if not all(any(r['split']==s and r['status']=='include' for r in rows) for s in ('train','val')):
                    raise ValueError('Training needs reviewed training images and a separate validation set.')
                folder = DATA / 'jobs' / uuid.uuid4().hex
                return self.send(launch({'kind':'train','directory':str(folder),**options}, folder))
            self.send({'error':'Unknown action'}, 404)
        except (ValueError, KeyError, OSError) as exc:
            self.send({'error':str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='Local interface IP; use your Wi-Fi IP for LAN sharing')
    args = parser.parse_args()
    connect().close(); (DATA / 'jobs').mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Pollen Studio → http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for ident in list(PROCESSES): stop_job(ident)
        server.server_close()


if __name__ == '__main__':
    main()
