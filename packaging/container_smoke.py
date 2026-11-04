"""Exercise a clean server in disposable storage; run inside a container."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from PIL import Image
from studio.tfrecords import export_records

URL='http://127.0.0.1:18768'
token=''


def request(path,body=None,headers=None):
    if body is not None and not isinstance(body,bytes):
        body=json.dumps(body).encode();content_type='application/json'
    else:content_type='application/octet-stream'
    h={'X-Studio-Token':token,'Content-Type':content_type,**(headers or {})}
    req=urllib.request.Request(URL+'/api/'+path,data=body,headers=h)
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)


def main():
    global token
    process=subprocess.Popen([sys.executable,'-u','-m','studio.server','--host','127.0.0.1','--port','18768'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:state=request('state');break
            except OSError:time.sleep(.1)
        assert not state['models'] and not state['images'] and not state['training']['gpus']
        token=state['token']
        try:request('dataset-begin',{'name':'x.zip','size':1},headers={'X-Studio-Token':'wrong'})
        except urllib.error.HTTPError as error:assert error.code==403
        else:raise AssertionError('Missing authentication check')
        try:request('dataset-begin',{'name':'x.zip','size':10*1024**3+1})
        except urllib.error.HTTPError as error:assert error.code==400
        else:raise AssertionError('Missing upload size limit')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);rows=[]
            for i,split in enumerate(['train','val']):
                image=tmp/(split+'.png');Image.new('RGB',(32,24),(i*70,20,50)).save(image)
                rows.append({'id':split,'path':str(image),'split':split,'boxes':[{'x1':.1,'y1':.1,'x2':.8,'y2':.8,'class_id':i+1}]})
            export_records(rows,tmp/'records');archive=tmp/'dataset.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for file in (tmp/'records').rglob('*'):
                    if file.is_file():z.write(file,'dataset/'+str(file.relative_to(tmp/'records')))
                z.writestr('padding.txt',b'x'*(9*1024**2))
            upload=request('dataset-begin',{'name':'dataset.zip','size':archive.stat().st_size})
            offset=0
            with archive.open('rb') as stream:
                while chunk:=stream.read(upload['chunk_size']):
                    offset=request('dataset-chunk/'+upload['id'],chunk,{'X-Upload-Offset':str(offset)})['offset']
            request('dataset-finish',{'id':upload['id']})
            for _ in range(100):
                state=request('state')
                if len(state['images'])==2 and state['jobs'][0]['status']=='completed':break
                time.sleep(.1)
            assert len(state['images'])==2 and state['jobs'][0]['status']=='completed',state['jobs']
            assert {r['split'] for r in state['images']}=={'train','val'}
            assert {r['boxes'][0]['class_id'] for r in state['images']}=={1,2}
            row=state['images'][0]
            request('review',{'id':row['id'],'boxes':[{'x1':.2,'y1':.2,'x2':.7,'y2':.7,'class_id':0}],'status':'include'})
            row=next(r for r in request('state')['images'] if r['id']==row['id'])
            assert row['boxes'][0]['class_id']==0
            pending=request('dataset-begin',{'name':'cancel.zip','size':100})
            request('dataset-cancel',{'id':pending['id']})
            assert not (Path('studio_data/dataset-uploads')/(pending['id']+'.zip')).exists()
            try:request('train',{'device':'auto'})
            except urllib.error.HTTPError as error:assert error.code==400
            else:raise AssertionError('Unprepared training was allowed')
        print('PASS: empty workspace, CPU fallback, token checks, upload limit, multi-chunk ZIP import, train/val classes, annotation save, cancellation, unprepared-training guard.')
    finally:
        process.terminate();process.wait(timeout=5)


if __name__=='__main__':main()
