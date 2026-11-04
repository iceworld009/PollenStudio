import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from studio import core, dataset_upload as upload


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.patch=patch.object(core,'DATA',Path(self.temp.name));self.patch.start()
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()
    def archive(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('dataset/train/pollen.tfrecord',b'')
        return data.getvalue()
    def test_chunk_retries_and_completion(self):
        data=self.archive();u=upload.begin('dataset.zip',len(data));ident=u['id']
        first=data[:20];rest=data[20:]
        self.assertEqual(upload.append(ident,0,io.BytesIO(first),20)['offset'],20)
        self.assertEqual(upload.append(ident,0,io.BytesIO(first),20)['offset'],20)
        upload.append(ident,20,io.BytesIO(rest),len(rest))
        self.assertEqual(upload.complete(ident).read_bytes(),data)
        with self.assertRaises(ValueError):upload.cancel(ident)
    def test_interrupted_chunk_rolls_back(self):
        u=upload.begin('data.zip',20)
        with self.assertRaises(ValueError):upload.append(u['id'],0,io.BytesIO(b'short'),20)
        self.assertEqual(upload.paths(u['id'])[1].stat().st_size,0)
        upload.cancel(u['id']);self.assertFalse(upload.paths(u['id'])[1].exists())
    def test_limits_paths_and_invalid_archives(self):
        with self.assertRaises(ValueError):upload.begin('x.zip',upload.MAX_BYTES+1)
        with self.assertRaises(ValueError):upload.paths('../outside')
        u=upload.begin('x.zip',4);upload.append(u['id'],0,io.BytesIO(b'nope'),4)
        with self.assertRaises(ValueError):upload.complete(u['id'])
    def test_reject_wrong_offset(self):
        u=upload.begin('data.zip',20)
        with self.assertRaises(ValueError):upload.append(u['id'],1,io.BytesIO(b'hi'),2)
