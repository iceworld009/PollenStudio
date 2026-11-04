"""CPU inference for single-class SSD TFLite Detection_PostProcess models."""
import numpy as np
from PIL import Image, ImageOps


class Detector:
    def __init__(self, path, normalization='zero_one'):
        from pathlib import Path
        from ai_edge_litert import schema_py_generated as schema
        from ai_edge_litert.interpreter import Interpreter
        from flatbuffers import flexbuffers
        if normalization not in ('zero_one', 'minus_one_one', 'raw'):
            raise ValueError('Unsupported input normalization.')
        self.normalization = normalization
        data = Path(path).read_bytes()
        if data[4:8] != b'TFL3':
            raise ValueError('Not a TFLite model.')
        model = schema.Model.GetRootAsModel(data, 0)
        if model.SubgraphsLength() != 1:
            raise ValueError('Only single-graph SSD detectors are supported.')
        graph = model.Subgraphs(0)
        output_indices = None
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            code = model.OperatorCodes(op.OpcodeIndex()).CustomCode()
            if code and b'edgetpu' in code.lower():
                raise ValueError('Coral-compiled models need Edge TPU hardware. Upload the non-Coral 8bit .tflite file.')
            if code == b'TFLite_Detection_PostProcess':
                options = flexbuffers.Loads(bytes(op.CustomOptionsAsNumpy()))
                self.num_classes = options.get('num_classes')
                if self.num_classes not in (1,3):
                    raise ValueError('This workspace requires a single-class pollen detector.')
                output_indices = list(map(int, op.OutputsAsNumpy()))
        if output_indices is None or len(output_indices) != 4:
            raise ValueError('Unsupported output format. Upload an SSD detector with TFLite Detection_PostProcess outputs; raw YOLO TFLite exports are not supported.')
        self.interpreter = Interpreter(model_path=str(path), num_threads=2)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise ValueError('Expected one RGB image input.')
        self.input = inputs[0]
        shape = self.input['shape'].tolist()
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or not all(16 <= d <= 2048 for d in shape[1:3]):
            raise ValueError('Expected a fixed [1, height, width, 3] RGB input, at most 2048 pixels per side.')
        dtype = self.input['dtype']
        if dtype not in (np.uint8, np.int8, np.float32):
            raise ValueError('Only uint8, int8, and float32 inputs are supported.')
        if np.issubdtype(dtype, np.integer):
            scale, zero = self.input['quantization']
            if scale <= 0:
                raise ValueError('Input quantization scale is missing.')
            limits = np.iinfo(dtype)
            low, high = {'zero_one':(0,1), 'minus_one_one':(-1,1), 'raw':(0,255)}[normalization]
            if low / scale + zero < limits.min - 2 or high / scale + zero > limits.max + 2:
                raise ValueError('Selected input normalization does not fit this model’s quantization. Choose the range used during export.')
        by_index = {d['index']:d for d in self.interpreter.get_output_details()}
        if len(by_index) != 4 or any(i not in by_index for i in output_indices):
            raise ValueError('Expected all four SSD detection outputs at the model boundary.')
        self.slots = {i:by_index[index] for i,index in enumerate(output_indices)}
        # A real invocation catches unsupported CPU operations and resolves dynamic output shapes.
        self.interpreter.set_tensor(self.input['index'], np.zeros(shape,dtype=dtype))
        self.interpreter.invoke()
        values = [self.interpreter.get_tensor(self.slots[i]['index']) for i in range(4)]
        if values[0].ndim != 3 or values[0].shape[0] != 1 or values[0].shape[2] != 4:
            raise ValueError('Unsupported detection box shape.')
        n = values[0].shape[1]
        if values[1].shape != (1,n) or values[2].shape != (1,n) or values[3].size != 1:
            raise ValueError('Unsupported SSD classes, scores, or count shapes.')
        self.info = {'input_shape':shape, 'input_dtype':np.dtype(dtype).name,
                     'num_classes':self.num_classes, 'normalization':normalization, 'max_detections':int(n)}
        print('Model:', self.info, flush=True)

    def predict(self, path, threshold):
        h, w = self.input['shape'][1:3]
        with Image.open(path) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize((int(w), int(h)), Image.Resampling.BILINEAR), dtype=np.float32)
        if self.normalization == 'zero_one':
            pixels = pixels / 255.0
        elif self.normalization == 'minus_one_one':
            pixels = (pixels - 127.5) / 127.5
        dtype = self.input['dtype']
        if np.issubdtype(dtype, np.integer):
            scale, zero = self.input['quantization']
            info = np.iinfo(dtype)
            pixels = np.clip(np.rint(pixels / scale + zero), info.min, info.max).astype(dtype)
        else:
            pixels = pixels.astype(dtype)
        self.interpreter.set_tensor(self.input['index'], pixels[None])
        self.interpreter.invoke()
        def output(slot):
            d = self.slots[slot]
            a = self.interpreter.get_tensor(d['index']).astype(float)
            scale, zero = d['quantization']
            return (a - zero) * scale if scale else a
        boxes, classes, scores, count = [output(i) for i in range(4)]
        if not np.isfinite(count.flat[0]):
            raise ValueError('Invalid detection count returned by model.')
        result = []
        for box, cls, score in zip(boxes[0][:max(0,int(count.flat[0]))], classes[0], scores[0]):
            if not np.isfinite(score) or score < threshold:
                continue
            if not np.isfinite(cls) or cls != int(cls) or not 0 <= int(cls) < self.num_classes:
                raise ValueError(f'Unexpected class ID {cls}; expected pollen class 0.')
            if not np.all(np.isfinite(box)):
                raise ValueError('Invalid box coordinates returned by model.')
            y1, x1, y2, x2 = np.clip(box, 0, 1).tolist()
            if x2 > x1 and y2 > y1:
                result.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, class_id=int(cls), confidence=float(score)))
        return result
