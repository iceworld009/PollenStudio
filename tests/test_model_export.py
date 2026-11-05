import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from studio import models


class ModelExportTests(unittest.TestCase):
    def test_export_preserves_bytes_and_labels_without_private_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.tflite';path.write_bytes(b'1234TFL3example')
            model={'name':'FieldStrike.tflite','labels':['pollen','blue_nectar','red_nectar'],'normalization':'minus_one_one','path':str(path),'private':'not for export'}
            with patch.object(models,'resolve_model',return_value=model):name,content=models.export_model('example')
            self.assertEqual(name,'FieldStrike.zip')
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                self.assertEqual(z.read('FieldStrike.tflite'),path.read_bytes())
                self.assertEqual(z.read('FieldStrike_labels.txt').decode().splitlines(),model['labels'])
                info=json.loads(z.read('model-info.json'))
                self.assertNotIn('path',info);self.assertNotIn('private',info)
                self.assertEqual(len(z.namelist()),3)
    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):models.export_model('../../missing')
