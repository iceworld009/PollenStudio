"""Reproducible local TF 2.15 runtime, official SSD weights and Coral compiler."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import tarfile
import urllib.request

from studio.core import DATA, ROOT

RUNTIME = DATA / 'training-runtime'
PYTHON = RUNTIME / 'venv/bin/python'
MODELS_COMMIT = '40695ad8964145a90ae1abbcc740c086e7364eac'
CORAL_COMMIT = '5020de9386ff370dcc1f63291a2d0f98eeb98adb'
BACKBONE = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PRETRAINED = RUNTIME / BACKBONE
RESEARCH = RUNTIME / 'models/research'
COMPILER = RUNTIME / 'compiler'


def compiler_command():
    return [str(COMPILER/'ld-linux-x86-64.so.2'), '--library-path', str(COMPILER), str(COMPILER/'edgetpu_compiler')]


def environment(device='cpu'):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(RESEARCH), str(RESEARCH/'slim')])
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['TF_NUM_INTRAOP_THREADS'] = '4'
    env['TF_NUM_INTEROP_THREADS'] = '2'
    env['OMP_NUM_THREADS'] = '4'
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    env['PYTHONUNBUFFERED'] = '1'
    # CUDA wheels live inside the training venv, outside the system loader paths.
    # Resolve lib/lib64 aliases so libraries are not added twice on Fedora.
    packages = sorted({p.resolve() for p in (RUNTIME/'venv').glob('lib*/python*/site-packages/nvidia')})
    libraries = sorted({str(p.resolve()) for root in packages for p in root.glob('*/lib') if p.is_dir()})
    if libraries:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(libraries + ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    for root in packages:
        toolkit = root/'cuda_nvcc'
        if (toolkit/'bin/ptxas').is_file():
            env['PATH'] = str(toolkit/'bin') + os.pathsep + env.get('PATH','')
            if '--xla_gpu_cuda_data_dir' not in env.get('XLA_FLAGS',''):
                env['XLA_FLAGS'] = (env.get('XLA_FLAGS','')+' --xla_gpu_cuda_data_dir='+str(toolkit)).strip()
            break
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    return env


_DEVICE_CACHE = {}
_DEVICE_LOCK = threading.Lock()


def detect_devices(refresh=False):
    """Probe the current driver and TensorFlow, rather than trusting setup history."""
    key=(str(PYTHON),PYTHON.is_file())
    with _DEVICE_LOCK:
        cached=_DEVICE_CACHE.get(key)
        if not refresh and cached and time.monotonic()-cached[0]<30:
            return dict(cached[1])
        result={'gpus':[], 'gpu_names':[], 'detected_gpu_names':[], 'gpu_error':None}
        try:
            hardware=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=3)
            if hardware.returncode==0:result['detected_gpu_names']=[n.strip() for n in hardware.stdout.splitlines() if n.strip()]
        except (OSError,subprocess.TimeoutExpired):pass
        if PYTHON.is_file():
            code = """import json,tensorflow as tf
physical=tf.config.list_physical_devices('GPU')
for gpu in physical: tf.config.experimental.set_memory_growth(gpu,True)
tf.config.set_soft_device_placement(False)
usable=[]; names=[]
for i,gpu in enumerate(physical[:1]):
 try:
  with tf.device('/GPU:'+str(i)):
   value=tf.matmul(tf.ones((2,2)),tf.ones((2,2))).numpy()
  usable.append(gpu.name);names.append(tf.config.experimental.get_device_details(gpu).get('device_name','NVIDIA GPU'))
 except Exception: pass
print(json.dumps({'gpus':usable,'gpu_names':names}))
"""
            try:
                probe=subprocess.run([str(PYTHON),'-c',code],env=environment('auto'),capture_output=True,text=True,timeout=25)
                if probe.returncode==0:
                    detected=json.loads(probe.stdout.strip().splitlines()[-1]);result.update(detected)
                else:result['gpu_error']='TensorFlow GPU check failed. CPU will be used in Automatic mode.'
            except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
                result['gpu_error']='TensorFlow GPU check could not complete. CPU will be used in Automatic mode.'
        if not result['gpus'] and result['detected_gpu_names'] and not result['gpu_error']:
            result['gpu_error']='NVIDIA hardware detected; TensorFlow GPU access is not available yet.'
        _DEVICE_CACHE[key]=(time.monotonic(),dict(result))
        return result


def status():
    marker = RUNTIME / 'ready.json'
    ready = marker.is_file() and PYTHON.is_file() and (PRETRAINED/'pipeline.config').is_file() and (COMPILER/'edgetpu_compiler').is_file()
    info = json.loads(marker.read_text()) if ready else {}
    expected = hashlib.sha256((ROOT/'requirements-training.txt').read_bytes()).hexdigest()
    ready = ready and info.get('requirements_sha256') == expected
    devices=detect_devices()
    return {'ready':bool(ready), 'path':str(RUNTIME), 'tensorflow':info.get('tensorflow'), **devices,
            'recommended_device':'gpu' if devices['gpus'] else 'cpu'}


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return
    print('Downloading',url,flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.part')
    try:
        request = urllib.request.Request(url,headers={'User-Agent':'PollenStudio/1.0'})
        with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response,output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_selected(archive, destination, prefix):
    destination = Path(destination)
    with tarfile.open(archive) as source:
        for member in source:
            parts = Path(member.name).parts
            if not member.isfile() or not parts or '..' in parts:
                continue
            relative = Path(*parts[1:]) if prefix == 'models' else Path(*parts)
            if prefix == 'models' and not (str(relative).startswith('research/object_detection/') or str(relative).startswith('research/slim/')):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as content, target.open('wb') as output:
                shutil.copyfileobj(content,output)


def prepare_assets():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    if not (RESEARCH/'object_detection/model_lib_v2.py').is_file():
        archive = RUNTIME/'models-source.tar.gz'
        download(f'https://codeload.github.com/tensorflow/models/tar.gz/{MODELS_COMMIT}',archive)
        extract_selected(archive,RUNTIME/'models','models')
    if not (PRETRAINED/'checkpoint/ckpt-0.index').is_file():
        archive = RUNTIME/'pretrained.tar.gz'
        download(f'https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/20200711/{BACKBONE}.tar.gz',archive)
        extract_selected(archive,RUNTIME,'pretrained')
    files = ['LICENSES','NOTICE','edgetpu_compiler','ld-linux-x86-64.so.2','libc++.so.1','libc++abi.so.1',
             'libc.so.6','libdl.so.2','libm.so.6','libpthread.so.0','libresolv.so.2','librt.so.1']
    for name in files:
        download(f'https://raw.githubusercontent.com/google-coral/edgetpu/{CORAL_COMMIT}/compiler/x86_64/edgetpu_compiler_bin/{name}',COMPILER/name)
        if name not in ('LICENSES','NOTICE'):
            (COMPILER/name).chmod(0o755)
    subprocess.run(compiler_command()+['--version'],check=True)


def finalize():
    protos = sorted((RESEARCH/'object_detection/protos').glob('*.proto'))
    subprocess.run([str(PYTHON),'-m','grpc_tools.protoc',f'-I{RESEARCH}',f'--python_out={RESEARCH}',*map(str,protos)],check=True)
    report = RUNTIME/'environment.json'
    code = "import json,tensorflow as tf; from object_detection.builders import model_builder; from object_detection import export_tflite_graph_lib_tf2; from pathlib import Path; gpus=tf.config.list_physical_devices('GPU'); Path("+repr(str(report))+").write_text(json.dumps({'tensorflow':tf.__version__,'gpus':[g.name for g in gpus],'gpu_names':[tf.config.experimental.get_device_details(g).get('device_name',g.name) for g in gpus]})); print('TensorFlow training imports verified')"
    subprocess.run([str(PYTHON),'-c',code],env=environment('auto'),cwd=ROOT,check=True)
    info = json.loads(report.read_text())
    info['requirements_sha256'] = hashlib.sha256((ROOT/'requirements-training.txt').read_bytes()).hexdigest()
    info['models_commit'] = MODELS_COMMIT
    info['coral_commit'] = CORAL_COMMIT
    ready_file=RUNTIME/'ready.tmp'
    ready_file.write_text(json.dumps(info,indent=2))
    ready_file.replace(RUNTIME/'ready.json')
    print('Local TensorFlow trainer and Coral compiler are ready.',flush=True)


def setup():
    python = sys.executable if sys.version_info[:2] == (3, 10) else shutil.which('python3.10')
    if not python:
        raise ValueError('Python 3.10 is required for this isolated TensorFlow trainer. Install Python 3.10, then run setup again.')
    if not PYTHON.is_file():
        subprocess.run([python,'-m','venv',str(RUNTIME/'venv')],check=True)
    subprocess.run([str(PYTHON),'-m','pip','install','--upgrade','pip','setuptools==69.5.1','wheel'],check=True)
    subprocess.run([str(PYTHON),'-m','pip','install','-r',str(ROOT/'requirements-training.txt')],check=True)
    prepare_assets()
    finalize()
