"""Fixed-threshold detection precision/recall; not COCO mAP."""
def iou(a, b):
    intersection = max(0, min(a['x2'],b['x2'])-max(a['x1'],b['x1'])) * max(0, min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
    area = lambda r: (r['x2']-r['x1'])*(r['y2']-r['y1'])
    union = area(a)+area(b)-intersection
    return intersection/union if union else 0


def match(predictions, truth, threshold=.5):
    unmatched = set(range(len(truth))); true_positive = 0
    for box in sorted(predictions, key=lambda r:r.get('confidence',1), reverse=True):
        candidates = [i for i in unmatched if truth[i].get('class_id',0)==box.get('class_id',0)]
        best = max(candidates, key=lambda i:iou(box,truth[i]), default=None)
        if best is not None and iou(box,truth[best]) >= threshold:
            unmatched.remove(best); true_positive += 1
    return true_positive, len(predictions)-true_positive, len(unmatched)
