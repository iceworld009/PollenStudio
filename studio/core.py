"""Persistent local annotation store and immutable dataset snapshots."""
import hashlib
import json
import math
import shutil
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'studio_data'
DEFAULT_MODEL = ROOT / 'models/PLN_600/limelight_neural_detector_8bit.tflite'


def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / 'studio.db')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, path TEXT UNIQUE, session TEXT, split TEXT, status TEXT DEFAULT "pending", boxes TEXT DEFAULT "[]")')
    db.execute('CREATE TABLE IF NOT EXISTS trash (image_id TEXT PRIMARY KEY, previous_status TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS training_archive (image_id TEXT PRIMARY KEY, original_path TEXT NOT NULL, archived_at REAL NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS purged_images (id TEXT PRIMARY KEY, purged_at REAL NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS ai_reviews (image_id TEXT PRIMARY KEY, batch TEXT NOT NULL, before_boxes TEXT NOT NULL, before_status TEXT NOT NULL, reason TEXT NOT NULL, decision TEXT NOT NULL, accepted INTEGER DEFAULT 0)')
    return db


def images(trashed=False):
    with connect() as db:
        query = ('SELECT images.*, CASE WHEN training_archive.image_id IS NULL THEN 0 ELSE 1 END AS archived, '
                 'COALESCE(training_archive.original_path, images.path) AS source_path, '
                 'ai_reviews.reason AS ai_reason, ai_reviews.decision AS ai_decision, ai_reviews.accepted AS ai_accepted, ai_reviews.before_boxes AS ai_before_boxes '
                 'FROM images LEFT JOIN training_archive ON images.id=training_archive.image_id '
                 'LEFT JOIN ai_reviews ON images.id=ai_reviews.image_id WHERE images.status '
                 + ('=' if trashed else '!=') + ' "deleted" ORDER BY images.session, images.path')
        return [dict(r) | {'boxes': json.loads(r['boxes']), 'ai_before_boxes':json.loads(r['ai_before_boxes']) if r['ai_before_boxes'] else None} for r in db.execute(query)]


def apply_ai_reviews(changes, batch):
    """Apply reviewed proposals only if the user's saved annotations still match."""
    with connect() as db:
        for change in changes:
            row=db.execute('SELECT * FROM images WHERE id=?',(change['id'],)).fetchone()
            if not row or row['status']!=change['before_status'] or json.loads(row['boxes'])!=change['before_boxes']:
                raise ValueError('Photo changed during review; refresh before applying proposals: '+change['id'])
            if change['decision'] not in ('keep','correct','trash','exclude'):
                raise ValueError('Invalid AI review decision')
            validate_boxes(change['boxes'])
        for change in changes:
            ident=change['id'];decision=change['decision']
            db.execute('INSERT OR REPLACE INTO ai_reviews VALUES (?,?,?,?,?,?,0)',
                       (ident,batch,json.dumps(change['before_boxes']),change['before_status'],change['reason'],decision))
            status='deleted' if decision=='trash' else 'skip' if decision=='exclude' else 'ai_review'
            if status=='deleted':
                # Restore returns AI-processed photos to the dedicated review queue.
                db.execute('INSERT OR REPLACE INTO trash VALUES (?,?)',(ident,'ai_review'))
            db.execute('UPDATE images SET boxes=?,status=? WHERE id=?',(json.dumps(change['boxes']),status,ident))


def save_review(ident, boxes, status):
    if status not in ('pending','skip','include'):raise ValueError('Invalid review status')
    boxes=validate_boxes(boxes)
    with connect() as db:
        row=db.execute('SELECT * FROM images WHERE id=?',(ident,)).fetchone()
        if not row:raise ValueError('Image not found')
        archived=db.execute('SELECT * FROM training_archive WHERE image_id=?',(ident,)).fetchone()
        ai=db.execute('SELECT * FROM ai_reviews WHERE image_id=? AND accepted=0',(ident,)).fetchone()
        reviewing_ai=row['status']=='ai_review' and ai is not None
        if row['status']=='deleted':raise ValueError('Restore this photo from Trash before editing it.')
        if archived and status=='pending':raise ValueError('Save archived corrections as included, or return the photo to annotation review first.')
        if reviewing_ai and status=='pending':raise ValueError('Approve or exclude this AI proposal.')
        db.execute('UPDATE images SET boxes=?,status=? WHERE id=?',(json.dumps(boxes),status,ident))
        if reviewing_ai:db.execute('UPDATE ai_reviews SET accepted=1 WHERE image_id=?',(ident,))
        if archived:
            sidecar=DATA/'training-archive'/(ident+'.json')
            saved=json.loads(sidecar.read_text()) if sidecar.exists() else {'id':ident,'source_path':archived['original_path'],'image':row['path'],'split':row['split']}
            saved['boxes']=boxes
            temporary=sidecar.with_suffix('.tmp');temporary.write_text(json.dumps(saved,indent=2));temporary.replace(sidecar)


def archive_images(ids, boxes=None, restore=False):
    """Retain reviewed examples and saved labels outside the annotation queue."""
    if not isinstance(ids,list) or not ids or len(ids)>1000 or any(not isinstance(i,str) for i in ids):
        raise ValueError('Select 1–1000 valid photos.')
    ids=list(dict.fromkeys(ids))
    if boxes is not None and len(ids)!=1:
        raise ValueError('Save annotations for one photo at a time.')
    with connect() as db:
        # Validate the whole selection before copying or changing any photo.
        rows=[]
        for ident in ids:
            row=db.execute('SELECT * FROM images WHERE id=?',(ident,)).fetchone()
            if not row or row['status']=='deleted':
                raise ValueError('Select active photos; restore photos from Trash first.')
            archived=db.execute('SELECT * FROM training_archive WHERE image_id=?',(ident,)).fetchone()
            if restore and not archived:
                raise ValueError('This photo is not in the training archive.')
            if not restore:
                if row['session'].startswith('Original dataset /'):
                    raise ValueError('Original photos are already retained for training.')
                if archived:continue
                if boxes is None and row['status'] not in ('include','skip'):
                    raise ValueError('Review and save annotations before archiving selected photos.')
                saved=validate_boxes(boxes if boxes is not None else json.loads(row['boxes']))
                if not Path(row['path']).is_file():raise ValueError('Source image is missing.')
            rows.append((row,archived,saved if not restore else None))
        for row,archived,saved in rows:
            if restore:
                db.execute('DELETE FROM training_archive WHERE image_id=?',(row['id'],))
                db.execute('UPDATE images SET status="pending" WHERE id=?',(row['id'],))
            else:
                folder=DATA/'training-archive';folder.mkdir(parents=True,exist_ok=True)
                destination=folder/(row['id']+Path(row['path']).suffix.lower())
                if Path(row['path']).resolve()!=destination.resolve():
                    shutil.copy2(row['path'],destination)
                (folder/(row['id']+'.json')).write_text(json.dumps({'id':row['id'],'source_path':row['path'],
                    'image':str(destination),'boxes':saved,'split':row['split']},indent=2))
                db.execute('INSERT INTO training_archive VALUES (?,?,?)',(row['id'],row['path'],time.time()))
                db.execute('UPDATE images SET path=?,boxes=?,status="include" WHERE id=?',
                           (str(destination),json.dumps(saved),row['id']))


def archive_all_included():
    ids=[r['id'] for r in images() if r['status']=='include' and not r['archived']
         and not r['session'].startswith('Original dataset /')]
    for offset in range(0,len(ids),1000):
        archive_images(ids[offset:offset+1000])
    return len(ids)


def purge_excluded(scan_roots=None):
    """Permanently remove current Trash/skipped records and their library image files."""
    with connect() as db:
        rows=[dict(r) for r in db.execute('SELECT * FROM images WHERE status IN ("deleted","skip")')]
        active=[dict(r) for r in db.execute('SELECT id,path FROM images WHERE status NOT IN ("deleted","skip")')]
        archives={r['image_id']:r['original_path'] for r in db.execute('SELECT * FROM training_archive')}
    target_ids={r['id'] for r in rows}
    protected={Path(r['path']).resolve() for r in active}
    protected.update(Path(path).resolve() for ident,path in archives.items() if ident not in target_ids)
    paths=set()
    for row in rows:
        paths.add(Path(row['path']))
        if row['id'] in archives:paths.add(Path(archives[row['id']]))
        paths.add(DATA/'training-archive'/(row['id']+'.json'))
        paths.update((DATA/'thumbnails').glob(row['id']+'-*'))
    # Imports deduplicate file bytes; remove redundant capture/extraction copies too.
    roots=scan_roots if scan_roots is not None else [ROOT/'captures',DATA/'original',DATA/'training-archive']
    for root in roots:
        for file in Path(root).rglob('*'):
            if file.is_file() and file.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                if file.resolve() not in protected and hashlib.sha256(file.read_bytes()).hexdigest()[:24] in target_ids:
                    paths.add(file)
    paths={p for p in paths if p.resolve() not in protected and p.is_file()}
    report={'photos':len(rows),'trashed':sum(r['status']=='deleted' for r in rows),
            'excluded':sum(r['status']=='skip' for r in rows),'files':len(paths),
            'image_ids':sorted(target_ids),'removed_paths':sorted(str(p) for p in paths),'time':time.time()}
    directory=DATA/'purge-reports';directory.mkdir(parents=True,exist_ok=True)
    report_file=directory/(str(time.time_ns())+'.json')
    report_file.write_text(json.dumps(report|{'status':'started'},indent=2))
    for file in paths:file.unlink(missing_ok=True)
    with connect() as db:
        for ident in target_ids:
            db.execute('INSERT OR IGNORE INTO purged_images VALUES (?,?)',(ident,time.time()))
            db.execute('DELETE FROM training_archive WHERE image_id=?',(ident,))
            db.execute('DELETE FROM trash WHERE image_id=?',(ident,))
            db.execute('DELETE FROM images WHERE id=? AND status IN ("deleted","skip")',(ident,))
    report_file.write_text(json.dumps(report|{'status':'complete'},indent=2))
    return report


def trash_image(ident, restore=False):
    trash_images([ident],restore)


def trash_images(ids, restore=False):
    if not isinstance(ids,list) or not ids or len(ids)>1000 or any(not isinstance(i,str) for i in ids):
        raise ValueError('Select 1–1000 valid photos.')
    ids=list(dict.fromkeys(ids))
    with connect() as db:
        for ident in ids:
            row = db.execute('SELECT status FROM images WHERE id=?', (ident,)).fetchone()
            if not row:
                raise ValueError('Image not found')
            if restore:
                prior = db.execute('SELECT previous_status FROM trash WHERE image_id=?', (ident,)).fetchone()
                if row['status'] != 'deleted' or not prior:
                    raise ValueError('This photo is not in Trash.')
                db.execute('UPDATE images SET status=? WHERE id=?', (prior['previous_status'], ident))
                db.execute('DELETE FROM trash WHERE image_id=?', (ident,))
            elif row['status'] != 'deleted':
                db.execute('INSERT INTO trash(image_id,previous_status) VALUES (?,?)', (ident, row['status']))
                db.execute('UPDATE images SET status="deleted" WHERE id=?', (ident,))


def import_folder(folder, split='train'):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir() or split not in ('train', 'val'):
        raise ValueError('Choose an existing image folder and train or val split.')
    count = 0
    with connect() as db:
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp') or not p.is_file():
                continue
            p = p.resolve()
            session = str(p.parent)
            prior = db.execute('SELECT split FROM images WHERE session=? LIMIT 1', (session,)).fetchone()
            if prior and prior['split'] != split:
                raise ValueError('This capture session already belongs to the other split. Use a separate recording session.')
            ident = hashlib.sha256(p.read_bytes()).hexdigest()[:24]
            if db.execute('SELECT 1 FROM purged_images WHERE id=?',(ident,)).fetchone():
                continue
            duplicate = db.execute('SELECT split FROM images WHERE id=?', (ident,)).fetchone()
            if duplicate:
                continue
            count += db.execute('INSERT OR IGNORE INTO images(id,path,session,split) VALUES (?,?,?,?)', (ident, str(p), session, split)).rowcount
    return count


def validate_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise ValueError('Invalid boxes.')
    result = []
    for b in boxes:
        coords = [float(b[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise ValueError('Boxes must have positive area and remain inside the image.')
        if type(b.get('class_id', 0)) is not int or b.get('class_id', 0) not in (0, 1, 2):
            raise ValueError('Choose pollen (0), blue Nectar (1), or red Nectar (2).')
        result.append(dict(zip(('x1','y1','x2','y2'), coords)) | {'class_id': b.get('class_id', 0)})
    return result


def snapshot(destination):
    """Only explicitly reviewed, included images; never pseudo-labels."""
    destination = Path(destination)
    rows = [r for r in images() if r['status'] == 'include']
    if not rows:
        raise ValueError('Include at least one reviewed image first.')
    manifest = []
    for r in rows:
        image_dir = destination / 'images' / r['split']
        label_dir = destination / 'labels' / r['split']
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / (r['id'] + Path(r['path']).suffix.lower())
        shutil.copyfile(r['path'], target)
        lines = []
        for b in validate_boxes(r['boxes']):
            lines.append(f"{b.get('class_id',0)} {(b['x1']+b['x2'])/2:.8f} {(b['y1']+b['y2'])/2:.8f} {b['x2']-b['x1']:.8f} {b['y2']-b['y1']:.8f}")
        (label_dir / (r['id'] + '.txt')).write_text('\n'.join(lines))
        manifest.append(r)
    (destination / 'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: pollen\n  1: blue_nectar\n  2: red_nectar\n')
    (destination / 'review-manifest.json').write_text(json.dumps(manifest, indent=2))
    return rows
