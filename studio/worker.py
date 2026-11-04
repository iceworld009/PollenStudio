"""Runs expensive work in a cancellable process, separate from the UI."""
import json
import sys
from pathlib import Path
from studio.core import connect, images, snapshot, DATA


def run(config):
    kind = config['kind']
    if kind == 'inspect-model':
        from studio.inference import Detector
        detector = Detector(config['model'], config['normalization'])
        Path(config['report']).write_text(json.dumps(detector.info))
        return
    if kind == 'import':
        from studio.tfrecords import import_archive
        print('Imported', import_archive(config['archive']), 'original images', flush=True)
        if config.get('uploaded_dataset'):
            archive=Path(config['archive']);archive.unlink(missing_ok=True);archive.with_suffix('.json').unlink(missing_ok=True)
        return
    if kind in ('predict', 'evaluate'):
        from studio.inference import Detector
        detector = Detector(config['model'], config.get('normalization','zero_one'))
        def predict(path):
            return detector.predict(path, config['confidence'])
        totals = [0, 0, 0]
        for i, row in enumerate(config['images']):
            boxes = predict(row['path'])
            if kind == 'predict':
                with connect() as db:
                    db.execute('UPDATE images SET boxes=? WHERE id=? AND status="pending" AND NOT EXISTS (SELECT 1 FROM training_archive WHERE image_id=images.id)', (json.dumps(boxes), row['id']))
            else:
                from studio.evaluation import match
                totals = [a+b for a,b in zip(totals, match(boxes,row['boxes']))]
            print(f"{i+1}/{len(config['images'])} · {Path(row['path']).name}: {len(boxes)} detections", flush=True)
        if kind == 'evaluate':
            tp, fp, fn = totals
            metrics = {'model':config['model'], 'images':len(config['images']), 'confidence':config['confidence'], 'iou':0.5,
                       'true_positives':tp, 'false_positives':fp, 'missed_objects':fn,
                       'precision':tp/(tp+fp) if tp+fp else 0, 'recall':tp/(tp+fn) if tp+fn else 0}
            Path(config['metrics']).write_text(json.dumps(metrics, indent=2))
            print('Validation result (fixed threshold, not mAP):', json.dumps(metrics, indent=2), flush=True)
        return
    if kind == 'export':
        import zipfile
        directory = DATA / 'exports' / config['id']
        if config['format'] == 'yolo':
            snapshot(directory)
        else:
            from studio.tfrecords import export_records
            rows = [r for r in images() if r['status'] == 'include']
            export_records(rows, directory)
            (directory / 'review-manifest.json').write_text(json.dumps(rows, indent=2))
        # JPEG and TFRecords contain compressed images; storing avoids wasteful recompression.
        with zipfile.ZipFile(directory.with_suffix('.zip'), 'w', zipfile.ZIP_STORED) as archive:
            for file in directory.rglob('*'):
                if file.is_file(): archive.write(file, file.relative_to(directory))
        print('Export ready:', directory.with_suffix('.zip'), flush=True)
        return
    if kind == 'purge-excluded':
        from studio.core import purge_excluded
        report=purge_excluded()
        print('Permanently deleted',report['trashed'],'trashed and',report['excluded'],'excluded photos;',report['files'],'library files removed.',flush=True)
        return
    if kind == 'audit-photos':
        from studio.photo_audit import run
        run()
        return
    if kind == 'setup-training':
        from studio.training_runtime import setup
        setup()
        return
    if kind == 'train':
        from studio.training import run
        run(config)
        return
    raise ValueError('Unknown job type')


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except ModuleNotFoundError as exc:
        print(f'Missing dependency: {exc.name}. Install requirements.txt for PLN inference, or use Prepare trainer in the training tab for TensorFlow training.', flush=True)
        sys.exit(1)
