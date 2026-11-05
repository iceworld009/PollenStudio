"""Conservative photo-quality and redundancy warnings; never modifies labels or inclusion."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from studio.core import DATA, images, validate_boxes

REPORT = DATA/'photo-audit.json'
AUTOMATIC_CODES = {'low_resolution','low_information','soft_image','very_dark','very_bright','tiny_target','unreadable','split_overlap','duplicate','label_conflict','near_duplicate'}


def feature(row):
    import numpy as np
    from PIL import Image,ImageOps
    with Image.open(row['path']) as image:
        image=ImageOps.exif_transpose(image).convert('RGB')
        width,height=image.size
        if width*height>50_000_000:raise ValueError('Image exceeds 50 megapixels.')
        pixel_hash=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        small=np.asarray(image.resize((64,64)),dtype=np.float32)
        gray=np.asarray(image.convert('L').resize((320,240)),dtype=np.float32)
        lap=-4*gray[1:-1,1:-1]+gray[2:,1:-1]+gray[:-2,1:-1]+gray[1:-1,2:]+gray[1:-1,:-2]
        tiny=np.asarray(image.convert('L').resize((32,32)),dtype=np.float32)
        k=np.arange(8)[:,None];n=np.arange(32)[None,:]
        basis=np.cos(np.pi*(2*n+1)*k/64)
        spectrum=basis@tiny@basis.T
        bits=spectrum.flatten()[1:]>np.median(spectrum.flatten()[1:])
        phash=sum(int(bit)<<i for i,bit in enumerate(bits))
    metrics={'width':width,'height':height,'brightness':round(float(gray.mean()),2),
             'contrast':round(float(gray.std()),2),'sharpness':round(float(lap.var()),2),
             'dark_fraction':round(float((gray<12).mean()),4),'bright_fraction':round(float((gray>245).mean()),4)}
    return metrics,small,pixel_hash,phash


def box_similarity(first,second):
    from studio.evaluation import iou
    if len(first)!=len(second):return False
    unmatched=set(range(len(second)))
    for box in first:
        candidates=[i for i in unmatched if box.get('class_id',0)==second[i].get('class_id',0)]
        best=max(candidates,key=lambda i:iou(box,second[i]),default=None)
        if best is None or iou(box,second[best])<.85:return False
        unmatched.remove(best)
    return True


def analyze(rows):
    import numpy as np
    result={};features={}
    def warn(ident,code,text,level='review',related=None):
        entry={'code':code,'message':text,'level':level}
        if related:entry['related_id']=related
        if not any(w['code']==code and w.get('related_id')==related for w in result[ident]['warnings']):
            result[ident]['warnings'].append(entry)
    for index,row in enumerate(rows):
        ident=row['id'];result[ident]={'warnings':[],'metrics':{}}
        try:
            metrics,small,pixels,phash=feature(row)
            result[ident]['metrics']=metrics
            features[ident]=(small,pixels,phash)
            validate_boxes(row['boxes'])
            if metrics['width']<160 or metrics['height']<120:
                warn(ident,'low_resolution','Very low resolution. Check whether pollen remains identifiable at the model input size.')
            if metrics['contrast']<5 and metrics['sharpness']<12:
                warn(ident,'low_information','Almost uniform image. Consider excluding if the camera view is obstructed or unusable.','exclude_candidate')
            elif metrics['sharpness']<12:
                warn(ident,'soft_image','Low image sharpness. Inspect the target before deciding whether to exclude it.')
            if metrics['dark_fraction']>.92:
                warn(ident,'very_dark','Image is mostly near-black. Check that targets and scene details are visible.','exclude_candidate')
            if metrics['bright_fraction']>.92:
                warn(ident,'very_bright','Image is mostly overexposed. Check that target detail is recoverable.','exclude_candidate')
            if any(min((b['x2']-b['x1'])*300,(b['y2']-b['y1'])*300)<3 for b in row['boxes']):
                warn(ident,'tiny_target','A labeled target is under 3 pixels wide or high at 300×300. Check label quality; useful distant targets should be retained.')
        except (OSError,ValueError,KeyError) as error:
            features.pop(ident,None)
            warn(ident,'unreadable','Image or annotation cannot be used: '+str(error),'exclude_candidate')
        if index%100==0:print(f'Quality scan {index+1}/{len(rows)}',flush=True)
    # Sharpest representative is kept as the suggested comparison; no images are removed.
    ordered=sorted((r for r in rows if r['id'] in features),key=lambda r:result[r['id']]['metrics']['sharpness'],reverse=True)
    representatives=[]
    for row in ordered:
        ident=row['id'];small,pixels,phash=features[ident]
        for other in representatives:
            oid=other['id'];osmall,opixels,ohash=features[oid]
            exact=pixels==opixels
            if not exact:
                if (phash^ohash).bit_count()>4:continue
                a=result[ident]['metrics'];b=result[oid]['metrics']
                if abs(a['width']/a['height']-b['width']/b['height'])>.02:continue
                difference=float(np.abs(small-osmall).mean())
                if difference>3.0 or not box_similarity(row['boxes'],other['boxes']):continue
            if row['split']!=other['split']:
                warn(ident,'split_overlap','Very similar photo exists in the other split. Review recording-session separation before trusting validation.','review',oid)
                warn(oid,'split_overlap','Very similar photo exists in the other split. Review recording-session separation before trusting validation.','review',ident)
                continue
            if exact:
                same_labels=box_similarity(row['boxes'],other['boxes'])
                warn(ident,'duplicate' if same_labels else 'label_conflict',
                     'Identical decoded image. Consider retaining one copy.' if same_labels else 'Identical image has different annotations. Inspect both label sets.',
                     'exclude_candidate' if same_labels else 'review',oid)
            else:
                warn(ident,'near_duplicate','Very similar scene and target boxes. Consider keeping the sharper comparison photo if this adds no useful variation.','review',oid)
            break
        else:
            representatives.append(row)
    summary=Counter(w['code'] for entry in result.values() for w in entry['warnings'])
    return {'generated':time.time(),'scanned':len(rows),'items':result,'counts':dict(summary),
            'flagged':sum(bool(entry['warnings']) for entry in result.values()),
            'note':'Suggestions for review only. Difficult scenes and empty negative images can be valuable. No images were excluded automatically.'}


def run():
    rows=images()
    report=analyze(rows)
    temporary=REPORT.with_suffix('.tmp');temporary.write_text(json.dumps(report));temporary.replace(REPORT)
    print('Photo audit complete:',json.dumps({k:v for k,v in report.items() if k!='items'}),flush=True)
    return report


def report_for_state(rows=None):
    if not REPORT.is_file():return {'scanned':0,'items':{},'flagged':0,'counts':{},'pending':len(rows or [])}
    report=json.loads(REPORT.read_text())
    active=rows if rows is not None else images()
    ids={r['id'] for r in active+images(True)}
    report['items']={ident:entry for ident,entry in report['items'].items() if ident in ids}
    # Old local sidecars and manual reviews are not part of a reproducible scan.
    report['items']={ident:{'metrics':entry.get('metrics',{}),
        'warnings':[warning for warning in entry.get('warnings',[])
                    if warning.get('code') in AUTOMATIC_CODES
                    and (not warning.get('related_id') or warning['related_id'] in ids)]}
        for ident,entry in report['items'].items()}
    for key in ('visually_reviewed','dataset_notes','automatically_trashed_soft_images'):
        report.pop(key,None)
    report['note']='Automatic quality and duplicate checks only. Review warnings before excluding photos.'
    report['scanned']=len(report['items'])
    report['flagged']=sum(bool(entry['warnings']) for entry in report['items'].values())
    report['pending']=len(ids-set(report['items']))
    report['counts']=dict(Counter(w['code'] for entry in report['items'].values() for w in entry['warnings']))
    return report


if __name__=='__main__':run()
