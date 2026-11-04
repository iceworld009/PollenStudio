"""Disk-backed, bounded uploads for large dataset ZIPs."""
import json
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from studio import core

MAX_BYTES = 10 * 1024**3
CHUNK_BYTES = 8 * 1024**2
LOCK = threading.Lock()


def paths(ident):
    if not isinstance(ident,str) or len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):
        raise ValueError('Invalid upload ID')
    folder=core.DATA/'dataset-uploads'
    return folder/(ident+'.json'), folder/(ident+'.zip')


def begin(name, size):
    if not isinstance(name,str) or not name.lower().endswith('.zip') or type(size) is not int or not 0<size<=MAX_BYTES:
        raise ValueError('Choose a dataset ZIP of at most 10 GB.')
    folder=core.DATA/'dataset-uploads';folder.mkdir(parents=True,exist_ok=True)
    with LOCK:
        reserved=sum(max(0,json.loads(f.read_text())['size']-f.with_suffix('.zip').stat().st_size) for f in folder.glob('*.json') if f.with_suffix('.zip').exists())
        if shutil.disk_usage(folder).free < size+reserved+1024**3:
            raise ValueError('Not enough server disk space for this upload.')
        ident=uuid.uuid4().hex;meta,file=paths(ident)
        file.touch();meta.write_text(json.dumps({'size':size,'name':Path(name).name,'state':'uploading'}))
    return {'id':ident,'offset':0,'chunk_size':CHUNK_BYTES}


def append(ident,offset,stream,length):
    if not 0<length<=CHUNK_BYTES:raise ValueError('Invalid chunk size')
    with LOCK:
        meta,file=paths(ident);info=json.loads(meta.read_text())
        if info['state']=='uploading' and offset+length==file.stat().st_size:
            retry=stream.read(length)
            with file.open('rb') as existing:
                existing.seek(offset)
                if len(retry)==length and existing.read(length)==retry:
                    return {'offset':offset+length}
            raise ValueError('Retry chunk does not match stored data')
        if info['state']!='uploading' or offset!=file.stat().st_size or offset+length>info['size']:
            raise ValueError('Upload offset mismatch; retry the upload.')
        try:
            with file.open('ab') as output:
                remaining=length
                while remaining:
                    data=stream.read(min(1024**2,remaining))
                    if not data:raise ValueError('Upload interrupted')
                    output.write(data);remaining-=len(data)
        except Exception:
            with file.open('r+b') as output:output.truncate(offset)
            raise
        return {'offset':file.stat().st_size}


def complete(ident):
    with LOCK:
        meta,file=paths(ident);info=json.loads(meta.read_text())
        if info['state']!='uploading' or file.stat().st_size!=info['size']:raise ValueError('Upload is incomplete')
        if not zipfile.is_zipfile(file):raise ValueError('The uploaded file is not a valid ZIP.')
        with zipfile.ZipFile(file) as z:
            entries=z.infolist()
            if len(entries)>100000 or sum(i.file_size for i in entries)>40*1024**3:
                raise ValueError('Archive exceeds 100,000 entries or 40 GB extracted size.')
            if not any(i.filename.endswith('.tfrecord') for i in entries):
                raise ValueError('This uploader imports TFRecord dataset ZIPs. Export your annotated dataset as TensorFlow TFRecord first.')
            if shutil.disk_usage(file.parent).free < sum(i.file_size for i in entries)+1024**3:
                raise ValueError('Not enough server disk space to import this dataset.')
        info['state']='importing';meta.write_text(json.dumps(info))
        return file


def cancel(ident):
    with LOCK:
        meta,file=paths(ident)
        if meta.exists() and json.loads(meta.read_text())['state']=='importing':raise ValueError('Dataset import has already started.')
        file.unlink(missing_ok=True);meta.unlink(missing_ok=True)
