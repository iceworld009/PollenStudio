import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
import importlib.util
import os
from unittest.mock import patch
from types import SimpleNamespace

from studio import core, server, tfrecords, models, training, training_runtime, photo_audit, thumbnails
from studio.evaluation import match
from studio.training_progress import LogProgress

BOX = dict(x1=.1, y1=.2, x2=.5, y2=.7, class_id=0)

class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        self.patchers = [patch.object(m, 'DATA', self.data) for m in (core, server, tfrecords, models, training, photo_audit, thumbnails)]
        self.patchers.append(patch.object(photo_audit,'REPORT',self.data/'photo-audit.json'))
        for p in self.patchers: p.start()
        (self.data / 'jobs').mkdir()
        core.connect().close()

    def tearDown(self):
        for p in self.patchers: p.stop()
        self.temp.cleanup()

    def add(self, ident, status='pending', split='train', session='new capture'):
        path=self.data / (ident+'.jpg'); path.write_bytes(ident.encode())
        with core.connect() as db:
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',(ident,str(path),session,split,status,json.dumps([BOX])))

    def request(self,path,body,token=server.TOKEN):
        handler=server.Handler.__new__(server.Handler)
        encoded=json.dumps(body).encode()
        handler.headers={'Host':'127.0.0.1:8765','X-Studio-Token':token,'Content-Length':str(len(encoded))}
        handler.server=SimpleNamespace(server_port=8765)
        handler.path=path;handler.rfile=io.BytesIO(encoded)
        result=[]
        handler.send=lambda value,status=200,**kw:result.append((status,value))
        handler.do_POST()
        return result[0]

    def test_soft_photo_policy_preserves_near_duplicates_and_restore(self):
        self.add('blur','include');self.add('similar','include');self.add('both','pending')
        report={'items':{'blur':{'warnings':[{'code':'soft_image'}]},
                         'similar':{'warnings':[{'code':'near_duplicate'}]},
                         'both':{'warnings':[{'code':'soft_image'},{'code':'near_duplicate'}]}}}
        self.assertEqual(photo_audit.apply_review_policy(report),[])
        (self.data/'photo-review-policy.json').write_text(json.dumps({'soft_images':'trash'}))
        self.assertEqual(set(photo_audit.apply_review_policy(report)),{'blur','both'})
        self.assertEqual([r['id'] for r in core.images()],['similar'])
        self.assertEqual(photo_audit.apply_review_policy(report),[])
        core.trash_image('blur',restore=True)
        self.assertEqual(next(r for r in core.images() if r['id']=='blur')['status'],'include')

    def test_box_validation(self):
        self.assertEqual(core.validate_boxes([BOX]),[BOX])
        for extra in ({'x1':float('nan')},{'x1':.9},{'y2':1.1},{'class_id':3}):
            with self.assertRaises(ValueError): core.validate_boxes([BOX|extra])

    def test_three_class_roundtrip_and_matching(self):
        from studio.evaluation import match
        boxes = [BOX | {'class_id':i} for i in range(3)]
        self.assertEqual(core.validate_boxes(boxes), boxes)
        self.assertEqual(match([boxes[1]], [boxes[0]]), (0,1,1))
        image=self.data/'three.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\n'+b'\0'*8+struct.pack('>II',32,24))
        destination=self.data/'three-records'
        tfrecords.export_records([{'id':'three','path':str(image),'split':'train','boxes':boxes}],destination)
        record=(destination/'train/pollen.tfrecord').read_bytes()
        size=struct.unpack('<Q',record[:8])[0]
        decoded=tfrecords.example(record[12:12+size])
        self.assertEqual(decoded['image/object/class/label'],[1,2,3])
        self.assertEqual(decoded['image/object/class/text'],[b'pollen',b'blue_nectar',b'red_nectar'])
        archive=self.data/'three.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.write(destination/'train/pollen.tfrecord','train/pollen.tfrecord')
        self.assertEqual(tfrecords.import_archive(archive),1)
        self.assertEqual([b['class_id'] for b in core.images()[0]['boxes']],[0,1,2])

    def test_active_gpu_search_prevents_a_second_training_job(self):
        with patch.object(server, 'live_experiment', return_value={'active': True}):
            status, result = self.request('/api/train', {'device': 'gpu'})
        self.assertEqual(status, 400)
        self.assertIn('GPU search is running', result['error'])

    def test_ai_review_requires_approval_and_preserves_old_labels(self):
        self.add('proposal','pending');self.add('original','include',session='Original dataset / train')
        changes=[{'id':ident,'before_status':status,'before_boxes':[BOX],'boxes':[BOX|{'x2':.6}],
                  'decision':'correct','reason':'Tightened target box'} for ident,status in [('proposal','pending'),('original','include')]]
        core.apply_ai_reviews(changes,'test')
        self.assertTrue(all(r['status']=='ai_review' and r['ai_before_boxes']==[BOX] for r in core.images()))
        with self.assertRaises(ValueError):core.snapshot(self.data/'snapshot')
        core.save_review('original',[BOX|{'x2':.6}],'include')
        original=next(r for r in core.images() if r['id']=='original')
        self.assertEqual(original['ai_accepted'],1)
        self.assertEqual([r['id'] for r in core.snapshot(self.data/'snapshot')],['original'])
        core.save_review('original',[BOX],'include')
        self.assertEqual(next(r for r in core.images() if r['id']=='original')['boxes'],[BOX])
        with self.assertRaises(ValueError):core.apply_ai_reviews(changes,'stale')

    def test_original_label_corrections_reach_future_training_snapshots(self):
        self.add('original','include',session='Original dataset / train')
        self.add('original-val','include',split='val',session='Original dataset / val')
        revised=[BOX|{'x2':.6}]
        previous=training.training_snapshot(self.data/'previous')
        for ident in ('original','original-val'):
            status,result=self.request('/api/review',{'id':ident,'boxes':revised,'status':'include'})
            self.assertEqual(status,200,result)
        updated=training.training_snapshot(self.data/'next')
        self.assertTrue(all(r['boxes']==revised for r in updated))
        self.assertEqual({r['split'] for r in updated},{'train','val'})
        self.assertTrue(all(r['boxes']==[BOX] for r in previous))
        self.assertTrue(all(r['boxes']==[BOX] for r in json.loads((self.data/'previous/review-manifest.json').read_text())))

    def test_archived_ai_approval_updates_protected_sidecar(self):
        self.add('archived','include')
        core.archive_images(['archived'])
        revised=[BOX|{'x2':.6}]
        core.apply_ai_reviews([dict(id='archived',before_status='include',before_boxes=[BOX],boxes=revised,decision='correct',reason='Tighten boundary')],'test')
        self.assertTrue(core.images()[0]['archived'])
        core.save_review('archived',revised,'include')
        self.assertEqual(json.loads((self.data/'training-archive/archived.json').read_text())['boxes'],revised)
        core.save_review('archived',[BOX],'include')
        self.assertEqual(json.loads((self.data/'training-archive/archived.json').read_text())['boxes'],[BOX])

    def test_archived_corrections_update_backup_and_future_training(self):
        self.add('archived','include');self.add('validation','include',split='val')
        core.archive_images(['archived'])
        archived=next(r for r in core.images() if r['id']=='archived')
        path=archived['path']
        previous=training.training_snapshot(self.data/'previous')
        revised=[BOX|{'x2':.6}]
        status,result=self.request('/api/review',dict(id='archived',boxes=revised,status='include'))
        self.assertEqual(status,200,result)
        row=next(r for r in core.images() if r['id']=='archived')
        self.assertTrue(row['archived']);self.assertEqual(row['status'],'include')
        self.assertEqual(row['split'],'train');self.assertEqual(row['path'],path)
        self.assertEqual(row['boxes'],revised)
        self.assertEqual(json.loads((self.data/'training-archive/archived.json').read_text())['boxes'],revised)
        updated=training.training_snapshot(self.data/'next')
        self.assertEqual(next(r for r in updated if r['id']=='archived')['boxes'],revised)
        self.assertEqual(next(r for r in previous if r['id']=='archived')['boxes'],[BOX])

    def test_ai_trash_restore_returns_to_review(self):
        self.add('soft','pending')
        core.apply_ai_reviews([{'id':'soft','before_status':'pending','before_boxes':[BOX],'boxes':[],
                               'decision':'trash','reason':'Severe blur hides target edges'}],'test')
        self.assertEqual(core.images(),[])
        core.trash_image('soft',restore=True)
        self.assertEqual(core.images()[0]['status'],'ai_review')

    def test_snapshot_excludes_pending_and_skipped(self):
        self.add('included','include');self.add('pending');self.add('skipped','skip');self.add('validation','include','val')
        dest=self.data/'export';rows=core.snapshot(dest)
        self.assertEqual({r['id'] for r in rows},{'included','validation'})
        self.assertTrue((dest/'labels/val/validation.txt').exists())
        self.assertEqual(len(list((dest/'images').rglob('*.jpg'))),2)
        label=(dest/'labels/train/included.txt').read_text().split()
        self.assertEqual(label[0],'0'); self.assertAlmostEqual(float(label[1]),.3)

    def test_dedup_keeps_validation_split(self):
        folder=self.data/'session';folder.mkdir();(folder/'a.jpg').write_bytes(b'identical')
        self.assertEqual(core.import_folder(folder,'val'),1)
        second=self.data/'second';second.mkdir();(second/'b.jpg').write_bytes(b'identical')
        self.assertEqual(core.import_folder(second,'train'),0)
        self.assertEqual(core.images()[0]['split'],'val')

    def test_cannot_split_same_capture_session(self):
        folder=self.data/'session';folder.mkdir();(folder/'a.jpg').write_bytes(b'a')
        core.import_folder(folder,'train');(folder/'b.jpg').write_bytes(b'b')
        with self.assertRaises(ValueError): core.import_folder(folder,'val')
        self.assertEqual(len(core.images()),1)

    def test_review_api_allows_original_edits_and_requires_token(self):
        self.add('new');self.add('old','include',session='Original dataset / train')
        status,_=self.request('/api/review',{'id':'new','status':'include','boxes':[]})
        self.assertEqual(status,200)
        status,_=self.request('/api/review',{'id':'old','status':'skip','boxes':[]})
        self.assertEqual(status,200)
        self.assertEqual(next(r for r in core.images() if r['id']=='old')['status'],'skip')
        self.assertEqual(self.request('/api/review',{},token='invalid')[0],403)

    def test_tensorflow_training_requires_ready_runtime_and_both_splits(self):
        body={'name':'Pollen next','steps':'4000','batch':'2','learning_rate':'.0003','device':'cpu','source':'pretrained'}
        with patch.object(server.training_runtime,'status',return_value={'ready':False}):
            self.assertEqual(self.request('/api/train',body)[0],400)
        with patch.object(server.training_runtime,'status',return_value={'ready':True,'gpus':[]}):
            self.add('training','include')
            self.assertEqual(self.request('/api/train',body)[0],400)
            self.add('validation','include','val')
            with patch.object(server,'launch',return_value={'kind':'train'}) as launched:
                self.assertEqual(self.request('/api/train',body)[0],200)
                config=launched.call_args.args[0]
                self.assertEqual(config['steps'],4000)
                self.assertEqual(config['kind'],'train')
                self.assertIsNone(config['previous_checkpoint'])
                self.assertNotIn('checkpoint',config)
                self.assertEqual(self.request('/api/train',body|{'device':'gpu'})[0],400)

    def test_training_parameters_reject_invalid_and_untrusted_sources(self):
        self.assertEqual(training.parameters({})['steps'],4000)
        for invalid in [{'steps':0},{'batch':9},{'learning_rate':'NaN'},{'device':'invalid'},
                        {'name':''},{'source':'../../outside'}]:
            with self.assertRaises(ValueError):training.parameters(invalid)

    def test_training_snapshot_freezes_reviewed_data_and_excludes_trash(self):
        self.add('train','include');self.add('val','include','val');self.add('pending');self.add('removed','include')
        core.trash_image('removed')
        rows=training.training_snapshot(self.data/'training-snapshot')
        self.assertEqual({r['id'] for r in rows},{'train','val'})
        row=next(r for r in rows if r['id']=='train')
        self.assertNotEqual(row['path'],row['original_path'])
        Path(row['original_path']).write_bytes(b'changed later')
        self.assertEqual(Path(row['path']).read_bytes(),b'train')

    def test_saved_training_run_can_be_selected_for_continuation(self):
        ident='b'*32
        output=self.data/'jobs'/ident/'model'
        (output/'checkpoint').mkdir(parents=True)
        (output/'checkpoint/checkpoint').write_text('model_checkpoint_path: "ckpt-1"')
        (output/'result.json').write_text(json.dumps({'name':'Pollen v1','final_step':4000}))
        options=training.parameters({'source':ident,'steps':1000})
        self.assertEqual(options['previous_checkpoint'],str(output/'checkpoint'))

    def test_continuation_preserves_search_model_geometry_across_generations(self):
        ident='c'*32
        output=self.data/'jobs'/ident/'model'
        (output/'checkpoint').mkdir(parents=True)
        (output/'checkpoint/checkpoint').write_text('model_checkpoint_path: "ckpt-1"')
        for geometry in ({'recipe':{'image_size':512,'small_anchors':True},'max_detections':20},
                         {'image_size':512,'small_anchors':True,'max_detections':20}):
            (output/'result.json').write_text(json.dumps({'name':'Search winner','final_step':10000,**geometry}))
            options=training.parameters({'source':ident,'steps':1000})
            self.assertEqual(options['image_size'],512)
            self.assertTrue(options['small_anchors'])
            self.assertEqual(options['max_detections'],20)

    def test_training_environment_finds_venv_cuda_and_preserves_cpu_mode(self):
        nvidia=self.data/'venv/lib/python3.10/site-packages/nvidia'
        (nvidia/'cudnn/lib').mkdir(parents=True)
        (nvidia/'cuda_nvcc/bin').mkdir(parents=True)
        (nvidia/'cuda_nvcc/bin/ptxas').touch()
        (self.data/'venv/lib64').symlink_to('lib',target_is_directory=True)
        with patch.object(training_runtime,'RUNTIME',self.data), patch.dict(os.environ,{
            'LD_LIBRARY_PATH':'/existing/lib','PATH':'/usr/bin','XLA_FLAGS':'--existing_flag=1','CUDA_VISIBLE_DEVICES':'0'}):
            gpu=training_runtime.environment('gpu')
            self.assertEqual(gpu['CUDA_VISIBLE_DEVICES'],'0')
            self.assertIn(str(nvidia/'cudnn/lib'),gpu['LD_LIBRARY_PATH'])
            self.assertEqual(gpu['LD_LIBRARY_PATH'].count(str(nvidia/'cudnn/lib')),1)
            self.assertTrue(gpu['LD_LIBRARY_PATH'].endswith('/existing/lib'))
            self.assertIn(str(nvidia/'cuda_nvcc/bin'),gpu['PATH'])
            self.assertIn('--existing_flag=1',gpu['XLA_FLAGS'])
            self.assertIn('--xla_gpu_cuda_data_dir='+str(nvidia/'cuda_nvcc'),gpu['XLA_FLAGS'])
            self.assertEqual(training_runtime.environment('cpu')['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')

    def test_live_training_log_counter_and_loss_deduplicate_lines(self):
        progress=LogProgress(12000)
        progress.feed('Training SSD; steps 0 → 12000; batch 2\nINFO:tensorflow:Step 100 per-step time 0.05s\n')
        progress.feed("INFO:tensorflow:Step 100 per-step time 0.05s\n{'Loss/total_loss': 0.25}\n{'Loss/total_loss': 0.25}\n")
        result=progress.report()
        self.assertEqual(result['completed_steps'],100)
        self.assertEqual(result['total_steps'],12000)
        self.assertEqual(result['train_history'],[(100,.25)])
        self.assertEqual(result['eta_seconds'],595)

    def test_live_training_progress_handles_continuation_and_partial_lines(self):
        progress=LogProgress(1000)
        progress.feed('Training SSD; steps 4000 → 5000; batch 2\nStep 4250 per-step time 0.')
        self.assertEqual(progress.report()['completed_steps'],0)
        progress.feed("1s\n{'Loss/total_loss': 3e-2}\n")
        result=progress.report()
        self.assertEqual(result['completed_steps'],250)
        self.assertEqual(result['global_step'],4250)
        self.assertEqual(result['percent'],25)
        self.assertEqual(result['train_loss'],.03)

    def test_live_training_progress_keeps_chart_bounded_and_latest_sample(self):
        progress=LogProgress(20000)
        progress.feed(''.join(f"Step {i} per-step time 0.1s\n'Loss/total_loss': 0.2\n" for i in range(1,5001)))
        result=progress.report()
        self.assertLessEqual(len(result['train_history']),1201)
        self.assertEqual(result['train_history'][-1],(5000,.2))
        self.assertEqual(result['train_history'][0],(1,.2))

    def test_delete_excludes_from_dataset_and_restore_preserves_labels(self):
        self.add('keep', 'include')
        self.add('remove', 'include', session='Original dataset / train')
        self.assertEqual(self.request('/api/delete-image', {'id':'remove'})[0],200)
        self.assertEqual([r['id'] for r in core.images()], ['keep'])
        self.assertEqual(core.images(trashed=True)[0]['id'], 'remove')
        self.assertTrue((self.data/'remove.jpg').exists())
        self.assertEqual([r['id'] for r in core.snapshot(self.data/'snapshot')], ['keep'])
        self.assertEqual(self.request('/api/delete-image', {'id':'remove'})[0],200)
        self.assertEqual(self.request('/api/restore-image', {'id':'remove'})[0],200)
        restored=next(r for r in core.images() if r['id']=='remove')
        self.assertEqual(restored['status'],'include')
        self.assertEqual(restored['boxes'],[BOX])

    def test_trash_survives_import_and_cannot_be_reviewed(self):
        folder=self.data/'session';folder.mkdir();(folder/'a.jpg').write_bytes(b'photo')
        core.import_folder(folder)
        ident=core.images()[0]['id']
        core.trash_image(ident)
        self.assertEqual(core.import_folder(folder),0)
        self.assertEqual(core.images(),[])
        self.assertEqual(self.request('/api/review',{'id':ident,'status':'include','boxes':[]})[0],400)
        core.trash_image(ident,restore=True)
        self.assertEqual(core.images()[0]['status'],'pending')

    def test_trash_respects_jobs_and_requires_valid_image(self):
        self.add('photo')
        with patch.object(server,'busy',return_value=True):
            self.assertEqual(self.request('/api/delete-image',{'id':'photo'})[0],400)
        self.assertEqual(core.images()[0]['status'],'pending')
        self.assertEqual(self.request('/api/delete-image',{'id':'missing'})[0],400)
        self.assertEqual(self.request('/api/restore-image',{'id':'photo'})[0],400)

    def test_bulk_photo_deletion_is_atomic_and_restorable(self):
        self.add('original','include',session='Original dataset / train')
        self.add('new','pending')
        self.assertEqual(self.request('/api/delete-images',{'ids':['original','missing']})[0],400)
        self.assertEqual(len(core.images()),2)
        self.assertEqual(self.request('/api/delete-images',{'ids':['original','new']})[0],200)
        self.assertEqual(core.images(),[])
        self.assertEqual(self.request('/api/restore-images',{'ids':['original','new']})[0],200)
        restored={r['id']:r['status'] for r in core.images()}
        self.assertEqual(restored,{'original':'include','new':'pending'})

    def test_training_archive_backs_up_labels_and_stays_in_dataset(self):
        self.add('reviewed','include')
        original_path=self.data/'reviewed.jpg'
        self.assertEqual(self.request('/api/archive-images',{'ids':['reviewed']})[0],200)
        row=core.images()[0]
        self.assertTrue(row['archived'])
        self.assertEqual(row['status'],'include')
        self.assertEqual(row['boxes'],[BOX])
        self.assertNotEqual(row['path'],str(original_path))
        original_path.unlink()
        self.assertEqual(Path(row['path']).read_bytes(),b'reviewed')
        self.assertEqual(core.snapshot(self.data/'archive-snapshot')[0]['id'],'reviewed')
        backup=json.loads((self.data/'training-archive/reviewed.json').read_text())
        self.assertEqual(backup['boxes'],[BOX])
        self.assertEqual(self.request('/api/review',{'id':'reviewed','status':'pending','boxes':[]})[0],400)
        self.assertEqual(self.request('/api/unarchive-images',{'ids':['reviewed']})[0],200)
        restored=core.images()[0]
        self.assertFalse(restored['archived'])
        self.assertEqual(restored['status'],'pending')
        self.assertEqual(restored['boxes'],[BOX])

    def test_bulk_archive_requires_review_and_can_include_skipped_correct_photos(self):
        self.add('correct','skip');self.add('pending')
        self.assertEqual(self.request('/api/archive-images',{'ids':['correct','pending']})[0],400)
        self.assertFalse(any(r['archived'] for r in core.images()))
        self.assertEqual(self.request('/api/archive-images',{'ids':['correct']})[0],200)
        correct=next(r for r in core.images() if r['id']=='correct')
        self.assertEqual(correct['status'],'include')
        self.assertEqual(self.request('/api/archive-images',{'ids':['pending'],'boxes':[BOX]})[0],200)
        self.assertTrue(all(r['archived'] for r in core.images()))

    def test_predict_never_selects_archived_photos_even_with_pending_status(self):
        self.add('archived','include');self.add('new')
        core.archive_images(['archived'])
        with core.connect() as db:db.execute('UPDATE images SET status="pending" WHERE id="archived"')
        with patch.object(server,'launch',return_value={'kind':'predict'}) as launched:
            self.assertEqual(self.request('/api/predict',{'confidence':.35})[0],200)
            self.assertEqual([r['id'] for r in launched.call_args.args[0]['images']],['new'])

    def test_archive_all_keeps_only_reviewed_included_new_photos(self):
        self.add('keep','include');self.add('excluded','skip');self.add('unreviewed')
        self.add('original','include',session='Original dataset / train')
        self.assertEqual(self.request('/api/archive-all',{})[0],200)
        rows={r['id']:r for r in core.images()}
        self.assertTrue(rows['keep']['archived'])
        for ident in ['excluded','unreviewed','original']:self.assertFalse(rows[ident]['archived'])
        self.assertEqual(rows['excluded']['status'],'skip')
        self.assertEqual(core.archive_all_included(),0)

    def test_permanent_purge_removes_only_excluded_files_and_exact_library_copies(self):
        folder=self.data/'captures';folder.mkdir()
        for name in ['trashed','excluded','included']:(folder/(name+'.jpg')).write_bytes(name.encode())
        core.import_folder(folder)
        rows={Path(r['path']).stem:r for r in core.images()}
        core.trash_image(rows['trashed']['id'])
        with core.connect() as db:db.execute('UPDATE images SET status="skip" WHERE id=?',(rows['excluded']['id'],))
        duplicate=self.data/'original/train/duplicate.jpg';duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(b'trashed')
        report=core.purge_excluded(scan_roots=[folder,self.data/'original'])
        self.assertEqual(report['trashed'],1)
        self.assertEqual(report['excluded'],1)
        self.assertFalse((folder/'trashed.jpg').exists())
        self.assertFalse((folder/'excluded.jpg').exists())
        self.assertFalse(duplicate.exists())
        self.assertTrue((folder/'included.jpg').exists())
        self.assertEqual([r['id'] for r in core.images()],[rows['included']['id']])
        self.assertEqual(core.images(trashed=True),[])
        (folder/'trashed.jpg').write_bytes(b'trashed')
        self.assertEqual(core.import_folder(folder),0)
        self.assertEqual(len(core.images()),1)

    @unittest.skipUnless(importlib.util.find_spec('PIL') and importlib.util.find_spec('numpy'),'Photo dependencies not installed')
    def test_photo_audit_warns_without_excluding_images(self):
        from PIL import Image
        import numpy as np
        for ident in ['black','copy','negative']:
            self.add(ident,'include')
        Image.new('RGB',(320,240)).save(self.data/'black.jpg')
        (self.data/'copy.jpg').write_bytes((self.data/'black.jpg').read_bytes())
        rng=np.random.default_rng(1)
        Image.fromarray(rng.integers(30,220,(240,320,3),dtype=np.uint8)).save(self.data/'negative.jpg')
        with core.connect() as db:db.execute('UPDATE images SET boxes="[]" WHERE id="negative"')
        report=photo_audit.analyze(core.images())
        self.assertTrue(any(w['code']=='very_dark' for w in report['items']['black']['warnings']))
        self.assertTrue(any(w['code']=='duplicate' for e in report['items'].values() for w in e['warnings']))
        self.assertEqual(report['items']['negative']['warnings'],[])
        self.assertTrue(all(r['status']=='include' for r in core.images()))

    @unittest.skipUnless(importlib.util.find_spec('PIL'),'Photo dependencies not installed')
    def test_photo_thumbnail_is_small_cached_and_label_sensitive(self):
        from PIL import Image
        self.add('photo','include')
        Image.new('RGB',(640,480),'gray').save(self.data/'photo.jpg')
        row=core.images()[0]
        first=thumbnails.thumbnail(row)
        self.assertEqual(Image.open(first).size,(320,240))
        self.assertEqual(thumbnails.thumbnail(row),first)
        self.assertNotEqual(thumbnails.thumbnail(row,boxes=False),first)

    def test_model_upload_rejects_invalid_files_and_labels(self):
        for filename, data, normalization, labels in [
            ('model.pt',b'1234TFL3','zero_one',b''),
            ('model.tflite',b'not tflite','zero_one',b''),
            ('model.tflite',b'1234TFL3','unknown',b''),
            ('model.tflite',b'1234TFL3','zero_one',b'other\npollen'),
        ]:
            with self.assertRaises(ValueError):
                models.store_upload(filename,data,normalization,labels)
        self.assertEqual(len(models.available_models()),1)
        with self.assertRaises(ValueError): models.resolve_model('../../outside')

    def test_upload_endpoint_requires_multipart_and_token(self):
        self.assertEqual(self.request('/api/upload-model',{})[0],400)
        self.assertEqual(self.request('/api/upload-model',{},token='invalid')[0],403)

    @unittest.skipUnless(importlib.util.find_spec('ai_edge_litert') and importlib.util.find_spec('numpy'), 'TFLite runtime not installed')
    def test_real_model_upload_and_compatibility_checks(self):
        model=models.store_upload('../my-pollen.tflite',core.DEFAULT_MODEL.read_bytes(),labels=b'pollen\n')
        self.assertEqual(model['name'],'my-pollen.tflite')
        self.assertEqual(model['inspection']['input_shape'],[1,300,300,3])
        selected=models.resolve_model(model['id'])
        self.assertTrue(Path(selected['path']).is_file())
        self.add('validation', 'include', 'val')
        with patch.object(server,'launch',return_value={'status':'running'}) as launched:
            status,_=self.request('/api/evaluate',{'model_id':model['id'],'confidence':.35})
            self.assertEqual(status,200)
            config=launched.call_args.args[0]
            self.assertEqual(config['model'],selected['path'])
            self.assertEqual(config['normalization'],'zero_one')
        with self.assertRaisesRegex(ValueError,'normalization'):
            models.store_upload('wrong-range.tflite',core.DEFAULT_MODEL.read_bytes(),'raw')
        coral=core.DEFAULT_MODEL.with_name('limelight_neural_detector_coral.tflite')
        with self.assertRaisesRegex(ValueError,'Coral'):
            models.store_upload('coral.tflite',coral.read_bytes())
        with self.assertRaises(ValueError):
            models.store_upload('broken.tflite',b'1234TFL3'+b'\0'*32)
        self.assertEqual(len(models.available_models()),2)

    def test_tf_example_roundtrip(self):
        features={'image/encoded':('bytes',[b'abc']), 'labels':('int',[1,128,300]), 'boxes':('float',[.1,.8]),'empty':('float',[])}
        decoded=tfrecords.example(tfrecords.encode_example(features))
        self.assertEqual(decoded['labels'],[1,128,300]);self.assertEqual(decoded['image/encoded'],[b'abc'])
        self.assertAlmostEqual(decoded['boxes'][0],.1,places=6);self.assertEqual(decoded['empty'],[])
        self.assertEqual(tfrecords.crc32c(b'123456789'),0xe3069283)

    def test_record_export_can_be_reimported(self):
        # Minimal PNG header is sufficient for dimension parsing in this codec test.
        image=self.data/'small.png';image.write_bytes(b'\x89PNG\r\n\x1a\n'+b'\0'*8+struct.pack('>II',32,24))
        rows=[{'id':'roundtrip','path':str(image),'split':'train','boxes':[BOX]}]
        destination=self.data/'records';tfrecords.export_records(rows,destination)
        record=(destination/'train/pollen.tfrecord').read_bytes()
        size=struct.unpack('<Q',record[:8])[0]
        self.assertEqual(record[8:12],tfrecords.masked_crc(record[:8]))
        self.assertEqual(record[-4:],tfrecords.masked_crc(record[12:12+size]))
        decoded=tfrecords.example(record[12:12+size])
        self.assertEqual(decoded['image/object/class/label'],[1]);self.assertEqual(decoded['image/width'],[32])
        archive=self.data/'roundtrip.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.write(destination/'train/pollen.tfrecord','train/pollen.tfrecord')
        self.assertEqual(tfrecords.import_archive(archive),1)
        self.assertEqual(core.images()[0]['boxes'][0]['class_id'],0)
        self.assertEqual(Path(core.images()[0]['path']).suffix,'.png')

    def test_matching_counts_duplicate_and_missed_detections(self):
        second=BOX|{'x1':.7,'x2':.9}
        self.assertEqual(match([BOX,BOX],[BOX,second]),(1,1,1))
        self.assertEqual(match([],[]),(0,0,0))
        self.assertEqual(match([BOX],[]),(0,1,0))

if __name__=='__main__': unittest.main()
