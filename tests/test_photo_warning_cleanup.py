import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from studio import photo_audit as audit


class WarningCleanupTests(unittest.TestCase):
    def test_duplicate_labels_respect_element_class(self):
        box={'x1':.1,'y1':.1,'x2':.5,'y2':.5,'class_id':0}
        self.assertFalse(audit.box_similarity([box],[box|{'class_id':1}]))
        self.assertTrue(audit.box_similarity([box],[box]))

    def test_old_cached_reviews_do_not_reappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'report.json'
            report.write_text(json.dumps({'items':{'photo':{'metrics':{'width':320},'visual_reviewed':True,'visual_note':'Old review','provenance':{'source':'old capture'},'warnings':[{'code':'ai_correction','message':'AI check'},{'code':'missed_ball','message':'Manual check'},{'code':'soft_image','message':'Low sharpness'}]}},'visually_reviewed':1,'dataset_notes':['Legacy capture']}))
            with patch.object(audit,'REPORT',report),patch.object(audit,'images',return_value=[{'id':'photo'}]):result=audit.report_for_state()
            self.assertEqual(result['counts'],{'soft_image':1})
            self.assertEqual(set(result['items']['photo']),{'metrics','warnings'})
            self.assertNotIn('visually_reviewed',result);self.assertNotIn('dataset_notes',result)
    def test_scan_ignores_local_manual_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);image=folder/'image.png';Image.new('RGB',(320,240),(100,100,100)).save(image)
            (folder/'photo-audit-review.json').write_text(json.dumps({'photo':{'warnings':[{'code':'ai_annotation','message':'Old AI review'}]}}))
            with patch.object(audit,'DATA',folder):result=audit.analyze([{'id':'photo','path':str(image),'boxes':[],'split':'train'}])
            self.assertFalse(any(w['code'].startswith('ai_') for w in result['items']['photo']['warnings']))
