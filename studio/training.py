"""Snapshot → train SSD → quantize → compile Coral → validate → downloadable bundle."""
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

from studio.core import DATA, ROOT, images
from studio import training_runtime as runtime

ARTIFACT_NAMES = {'limelight_neural_detector_8bit.tflite','limelight_neural_detector_coral.tflite',
                  'limelight_neural_detector_labels.txt','training_loss_graph.png','validation_metrics.json',
                  'training_log.txt','model-info.json'}


def available_runs():
    result = []
    for file in (DATA/'jobs').glob('*/model/result.json'):
        info = json.loads(file.read_text())
        if not info.get('test_run') and (file.parent/'checkpoint/checkpoint').is_file():
            result.append({'id':file.parent.parent.name,'name':info['name'],'steps':info['final_step']})
    return result


def parameters(body):
    steps, batch = int(body.get('steps',4000)), int(body.get('batch',2))
    rate = float(body.get('learning_rate',.0003))
    name = str(body.get('name','Pollen SSD')).strip()
    device = body.get('device','cpu')
    if not 1 <= steps <= 100000 or not 1 <= batch <= 8 or not math.isfinite(rate) or not 0.000001 <= rate <= .01:
        raise ValueError('Choose 1–100000 steps, batch 1–8, and learning rate 0.000001–0.01.')
    if not name or len(name)>80 or any(ord(c)<32 for c in name):
        raise ValueError('Choose a model name of 1–80 characters.')
    if device not in ('cpu','gpu'):
        raise ValueError('Choose CPU or an available TensorFlow GPU.')
    source = body.get('source','pretrained')
    previous = None
    geometry = {}
    if source != 'pretrained':
        if source not in {r['id'] for r in available_runs()}:
            raise ValueError('Saved training run not found.')
        previous = str(DATA/'jobs'/source/'model/checkpoint')
        info = json.loads((DATA/'jobs'/source/'model/result.json').read_text())
        recipe = info.get('recipe', {})
        if recipe or 'image_size' in info:
            geometry = {'image_size':int(info.get('image_size', recipe.get('image_size', 300))),
                        'small_anchors':bool(info.get('small_anchors', recipe.get('small_anchors', False))),
                        'max_detections':int(info.get('max_detections', 10))}
    return {'steps':steps,'batch':batch,'learning_rate':rate,'name':name,'device':device,'previous_checkpoint':previous,**geometry}


def stage(folder, message):
    print('\n'+message,flush=True)
    with (folder/'progress.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        progress_file=folder/'progress.json'
        prior=json.loads(progress_file.read_text()) if progress_file.is_file() else {}
        temporary=folder/'progress-stage.tmp'
        temporary.write_text(json.dumps(prior|{'stage':message,'updated':time.time()}))
        temporary.replace(progress_file)


def training_snapshot(folder):
    selected = [r for r in images() if r['status']=='include']
    if not all(any(r['split']==split for r in selected) for split in ('train','val')):
        raise ValueError('Training requires included training and validation images.')
    rows=[]
    for row in selected:
        path=folder/'images'/row['split']/(row['id']+Path(row['path']).suffix.lower())
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row['path'],path)
        rows.append(row|{'path':str(path),'original_path':row['path']})
    (folder/'review-manifest.json').write_text(json.dumps(rows,indent=2))
    return rows


def run(config):
    if not runtime.status()['ready']:
        raise ValueError('Prepare the local TensorFlow trainer first.')
    folder = Path(config['directory'])
    output = folder/'model'; output.mkdir(parents=True,exist_ok=True)
    # A separate observer supplies live charts and CPU validation without blocking training.
    with (folder/'progress-monitor.log').open('a') as observer_log:
        subprocess.Popen([sys.executable,'-u','-m','studio.training_progress',str(folder),'--parent',str(os.getpid())],
                         cwd=ROOT,stdout=observer_log,stderr=subprocess.STDOUT)
    stage(folder,'1/7 · Snapshot reviewed images and preserve train/validation split')
    rows=training_snapshot(folder/'dataset')
    from studio.tfrecords import export_records
    export_records(rows,folder/'records')
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='val']
    # Calibration is reproducible and uses training images only.
    calibration = random.Random(42).sample(train,min(100,len(train)))
    from studio.classes import CLASSES
    if config.get('previous_checkpoint'):
        prior = json.loads((Path(config['previous_checkpoint']).parent/'result.json').read_text())
        if prior.get('labels',['pollen']) != list(CLASSES):
            raise ValueError('For the new three-class model, select fresh pretrained training. Old pollen checkpoints have a different output head.')
    config['labels']=list(CLASSES)
    tf_config=config|{'output':str(output),'records':str(folder/'records'),
                      'calibration_images':[r['path'] for r in calibration]}
    config_file=folder/'tensorflow-config.json';config_file.write_text(json.dumps(tf_config,indent=2))
    def tensorflow(action):
        subprocess.run([str(runtime.PYTHON),'-u','-m','studio.tf_training',action,str(config_file)],
                       env=runtime.environment(config['device'] if action == 'train' else 'cpu'),cwd=ROOT,check=True)
    stage(folder,'2/7 · Train SSD MobileNet V2 on reviewed images')
    tensorflow('train')
    stage(folder,'3/7 · Export the trained TensorFlow checkpoint')
    tensorflow('export')
    stage(folder,'4/7 · Calibrate and convert to 8-bit TFLite')
    tensorflow('quantize')
    artifacts=output/'artifacts'
    from studio.inference import Detector
    detector=Detector(artifacts/'limelight_neural_detector_8bit.tflite','minus_one_one')
    stage(folder,'5/7 · Compile the quantized detector for Coral Edge TPU')
    subprocess.run(runtime.compiler_command()+['--out_dir',str(artifacts),'--show_operations',
                   str(artifacts/'limelight_neural_detector_8bit.tflite')],check=True)
    compiled=artifacts/'limelight_neural_detector_8bit_edgetpu.tflite'
    if not compiled.is_file():
        raise ValueError('The compiler did not produce a Coral model.')
    compiled.replace(artifacts/'limelight_neural_detector_coral.tflite')
    stage(folder,'6/7 · Evaluate the exported model on the validation split')
    from studio.evaluation import match
    counts=[0,0,0]
    for index,row in enumerate(validation):
        predictions=detector.predict(row['path'],.35)
        counts=[a+b for a,b in zip(counts,match(predictions,row['boxes']))]
        if index%25==0: print(f'Validation {index+1}/{len(validation)}',flush=True)
    tp,fp,fn=counts
    metrics={'images':len(validation),'confidence':.35,'iou':.5,'true_positives':tp,
             'false_positives':fp,'missed_objects':fn,'precision':tp/(tp+fp) if tp+fp else 0,
             'recall':tp/(tp+fn) if tp+fn else 0,'note':'Fixed-threshold validation metrics, not COCO mAP or an untouched test result.'}
    (artifacts/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Validation:',json.dumps(metrics),flush=True)
    tensorflow('plot')
    stage(folder,'7/7 · Package Limelight files and retain the training checkpoint')
    state=json.loads((output/'training-state.json').read_text())
    info={'name':config['name'],'job_id':folder.name,'steps':config['steps'],**state,
          'test_run':config.get('test_run',False),'normalization':'minus_one_one','labels':config['labels'],
          'architecture':f"SSD MobileNet V2 / {config.get('image_size',300)}x{config.get('image_size',300)}",
          'image_size':config.get('image_size',300),'small_anchors':config.get('small_anchors',False),
          'max_detections':config.get('max_detections',10),'train_images':len(train),'validation_images':len(validation),
          'calibration_images':len(calibration),'compiler':'Coral official compiler 14.1',
          'metrics':metrics,'artifacts':sorted(ARTIFACT_NAMES)}
    log=folder/'log.txt'
    if log.is_file(): shutil.copyfile(log,artifacts/'training_log.txt')
    else: (artifacts/'training_log.txt').write_text('See the training job log for details.\n')
    # The separately retained checkpoint enables subsequent local training runs.
    if not config.get('test_run'):
        from studio.models import store_upload
        registered=store_upload(config['name']+'.tflite',(artifacts/'limelight_neural_detector_8bit.tflite').read_bytes(),'minus_one_one',labels=('\n'.join(config['labels'])+'\n').encode())
        info['model_id']=registered['id']
    (artifacts/'model-info.json').write_text(json.dumps(info,indent=2))
    exports=DATA/'exports';exports.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(exports/(folder.name+'.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for file in artifacts.iterdir():
            if file.is_file():archive.write(file,file.name)
    result_file=output/'result.tmp'
    result_file.write_text(json.dumps(info,indent=2))
    result_file.replace(output/'result.json')
    stage(folder,'Complete · Limelight CPU and Coral models are ready to download')
    print('Training checkpoint retained at',output/'checkpoint',flush=True)
