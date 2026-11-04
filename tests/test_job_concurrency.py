from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from studio import server


class JobConcurrencyTests(unittest.TestCase):
    def test_simultaneous_launches_create_only_one_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Mock();worker.poll.return_value=None
            with patch.object(server,'DATA',Path(tmp)),patch.object(server,'PROCESSES',{}),patch.object(server.subprocess,'Popen',return_value=worker) as start:
                def submit(_):
                    try:return server.launch({'kind':'export'})
                    except ValueError:return None
                with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(submit,range(2)))
                self.assertEqual(sum(r is not None for r in results),1)
                self.assertEqual(start.call_count,1)
