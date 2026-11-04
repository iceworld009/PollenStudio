import json
import subprocess
import unittest
from unittest.mock import patch
from studio import training, training_runtime as runtime


class DeviceTests(unittest.TestCase):
    def setUp(self):
        runtime._DEVICE_CACHE.clear()
    def tearDown(self):
        runtime._DEVICE_CACHE.clear()
    def test_automatic_prefers_usable_gpu(self):
        with patch.object(runtime,'status',return_value={'gpus':['GPU:0']}):
            result=training.parameters({})
        self.assertEqual(result['device'],'gpu');self.assertEqual(result['requested_device'],'auto')
    def test_automatic_falls_back_to_cpu(self):
        with patch.object(runtime,'status',return_value={'gpus':[]}):
            self.assertEqual(training.parameters({})['device'],'cpu')
    def test_cpu_override_is_respected(self):
        with patch.object(runtime,'status',side_effect=AssertionError('CPU override should not probe')):
            self.assertEqual(training.parameters({'device':'cpu'})['device'],'cpu')
    def test_driver_without_tensorflow_does_not_claim_usable_gpu(self):
        hardware=subprocess.CompletedProcess([],0,'NVIDIA RTX 3050\n','')
        failure=subprocess.CompletedProcess([],1,'','CUDA unavailable')
        with patch.object(runtime.PYTHON.__class__,'is_file',return_value=True),patch.object(runtime.subprocess,'run',side_effect=[hardware,failure]):
            result=runtime.detect_devices(refresh=True)
        self.assertEqual(result['gpus'],[]);self.assertEqual(result['detected_gpu_names'],['NVIDIA RTX 3050']);self.assertTrue(result['gpu_error'])
    def test_current_gpu_probe_and_cache(self):
        hardware=subprocess.CompletedProcess([],0,'NVIDIA RTX 3050\n','')
        success=subprocess.CompletedProcess([],0,json.dumps({'gpus':['GPU:0'],'gpu_names':['RTX 3050']}),'')
        with patch.object(runtime.PYTHON.__class__,'is_file',return_value=True),patch.object(runtime.subprocess,'run',side_effect=[hardware,success]) as run:
            first=runtime.detect_devices(refresh=True);second=runtime.detect_devices()
            self.assertEqual(first,second);self.assertEqual(run.call_count,2)
        self.assertEqual(first['gpus'],['GPU:0'])
    def test_gpu_probe_timeout_falls_back(self):
        with patch.object(runtime.PYTHON.__class__,'is_file',return_value=True),patch.object(runtime.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',25)):
            result=runtime.detect_devices(refresh=True)
        self.assertEqual(result['gpus'],[]);self.assertTrue(result['gpu_error'])
