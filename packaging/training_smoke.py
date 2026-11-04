"""Tiny synthetic three-class training/export check; requires a prepared runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image
from studio import training_runtime as runtime
from studio.inference import Detector
from studio.tfrecords import export_records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-dir',type=Path,default=runtime.RUNTIME)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    args=parser.parse_args()
    runtime.RUNTIME=args.runtime_dir.resolve()
    runtime.PYTHON=runtime.RUNTIME/'venv/bin/python'
    runtime.RESEARCH=runtime.RUNTIME/'models/research'
    runtime.COMPILER=runtime.RUNTIME/'compiler'
    if not runtime.PYTHON.is_file():raise SystemExit('Prepare the local trainer before running this check.')
    devices=runtime.detect_devices(refresh=True)
    device=('gpu' if devices['gpus'] else 'cpu') if args.device=='auto' else args.device
    if device=='gpu' and not devices['gpus']:raise SystemExit('No usable TensorFlow GPU detected.')
    with tempfile.TemporaryDirectory(prefix='pollen-training-check-') as tmp:
        folder=Path(tmp);rows=[]
        for i,split in enumerate(['train','train','val']):
            image=folder/(str(i)+'.png');Image.new('RGB',(320,320),(i*60,30,40)).save(image)
            rows.append({'id':str(i),'path':str(image),'split':split,'boxes':[{'x1':.2,'y1':.2,'x2':.6,'y2':.6,'class_id':i}]})
        export_records(rows,folder/'records');(folder/'model').mkdir()
        config={'output':str(folder/'model'),'records':str(folder/'records'),'labels':['pollen','blue_nectar','red_nectar'],'steps':2,'batch':2,'learning_rate':.0001,'device':device,'requested_device':args.device,'name':'Synthetic validation','image_size':320,'max_detections':20,'calibration_images':[r['path'] for r in rows if r['split']=='train']}
        config_path=folder/'config.json';config_path.write_text(json.dumps(config))
        for action in ['train','export','quantize']:
            code="from pathlib import Path; import json; from studio import training_runtime as r; r.PRETRAINED=Path("+repr(str(runtime.RUNTIME/runtime.BACKBONE))+"); from studio import tf_training; tf_training."+action+"(json.loads(Path("+repr(str(config_path))+").read_text()))"
            subprocess.run([str(runtime.PYTHON),'-c',code],env=runtime.environment(device if action=='train' else 'cpu'),check=True,timeout=600)
        artifacts=folder/'model/artifacts'
        model=artifacts/'limelight_neural_detector_8bit.tflite'
        detector=Detector(model,'minus_one_one')
        assert detector.info['num_classes']==3
        assert (artifacts/'limelight_neural_detector_labels.txt').read_text().splitlines()==config['labels']
        detector.predict(rows[0]['path'],.35)
        subprocess.run(runtime.compiler_command()+['--out_dir',str(artifacts),str(model)],check=True,timeout=180)
        assert (artifacts/'limelight_neural_detector_8bit_edgetpu.tflite').is_file()
        print('PASS: synthetic training, checkpoint export, uint8 TFLite conversion, three labels, CPU inference, Coral compilation. Training device:',device)


if __name__=='__main__':main()
