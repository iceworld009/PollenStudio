"""TensorFlow-only subprocess stages for the local SSD training pipeline."""
import json
import math
from pathlib import Path
import shutil
import sys


def configure(config):
    import tensorflow as tf
    from google.protobuf import text_format
    from object_detection.protos import pipeline_pb2, train_pb2
    from studio.training_runtime import PRETRAINED
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse((PRETRAINED/'pipeline.config').read_text(),pipeline)
    pipeline.model.ssd.num_classes = len(config.get('labels', ['pollen']))
    pipeline.model.ssd.freeze_batchnorm = True
    size = config.get('image_size',300)
    pipeline.model.ssd.image_resizer.fixed_shape_resizer.height = size
    pipeline.model.ssd.image_resizer.fixed_shape_resizer.width = size
    if config.get('small_anchors'):
        anchors = pipeline.model.ssd.anchor_generator.ssd_anchor_generator
        anchors.min_scale = .03
        anchors.max_scale = .9
        anchors.reduce_boxes_in_lowest_layer = False
    maximum = config.get('max_detections',10)
    pipeline.model.ssd.post_processing.batch_non_max_suppression.max_total_detections = maximum
    pipeline.model.ssd.post_processing.batch_non_max_suppression.max_detections_per_class = maximum
    pipeline.model.ssd.post_processing.batch_non_max_suppression.score_threshold = 0.0
    train = pipeline.train_config
    train.batch_size = config['batch']
    train.use_bfloat16 = False
    train.sync_replicas = False
    train.replicas_to_aggregate = 1
    train.fine_tune_checkpoint = str(PRETRAINED/'checkpoint/ckpt-0')
    train.fine_tune_checkpoint_type = 'detection'
    train.fine_tune_checkpoint_version = train_pb2.V2
    train.load_all_detection_checkpoint_vars = False
    train.optimizer.Clear()
    train.optimizer.adam_optimizer.learning_rate.constant_learning_rate.learning_rate = config['learning_rate']
    train.gradient_clipping_by_norm = 10
    pipeline.train_input_reader.label_map_path = str(Path(config['records'])/'label_map.pbtxt')
    pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [str(Path(config['records'])/'train/pollen.tfrecord')]
    pipeline.train_input_reader.num_readers = 1
    pipeline.train_input_reader.num_parallel_batches = 2
    pipeline.train_input_reader.num_parallel_map_calls = 2
    pipeline.train_input_reader.prefetch_size = 2
    pipeline.train_input_reader.shuffle_buffer_size = 64
    for reader in pipeline.eval_input_reader:
        reader.label_map_path = pipeline.train_input_reader.label_map_path
        reader.tf_record_input_reader.input_path[:] = [str(Path(config['records'])/'valid/pollen.tfrecord')]
        reader.num_readers = 1
    checkpoint_dir = Path(config['output'])/'checkpoint'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    start = 0
    if config.get('previous_checkpoint'):
        source = tf.train.latest_checkpoint(config['previous_checkpoint'])
        if not source:
            raise ValueError('Previous training checkpoint is missing.')
        start = int(tf.train.load_variable(source,'step/.ATTRIBUTES/VARIABLE_VALUE'))
        for file in Path(source).parent.glob(Path(source).name+'.*'):
            shutil.copy2(file,checkpoint_dir/file.name)
        tf.compat.v1.train.update_checkpoint_state(str(checkpoint_dir),Path(source).name)
        print('Continuing from saved pollen checkpoint at step',start,flush=True)
    target = start + config['steps']
    train.num_steps = target
    pipeline_file = Path(config['output'])/'pipeline.config'
    pipeline_file.write_text(text_format.MessageToString(pipeline))
    return pipeline_file, checkpoint_dir, start, target


def train(config):
    import tensorflow as tf
    from absl import logging
    from object_detection import model_lib_v2
    logging.set_verbosity(logging.INFO)
    tf.get_logger().setLevel('INFO')
    if config['device'] == 'gpu' and not tf.config.list_physical_devices('GPU'):
        raise ValueError('TensorFlow cannot see a CUDA GPU. Choose CPU or install the compatible CUDA libraries in the training environment.')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    pipeline, checkpoint, start, target = configure(config)
    size = config.get('image_size', 300)
    print(f'Training SSD MobileNet V2, {size}x{size}, pollen; steps {start} → {target}; batch {config["batch"]}',flush=True)
    model_lib_v2.LOG_EVERY = 10
    target_device = '/GPU:0' if config['device'] == 'gpu' else '/CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(target_device)
    print('Training device:', target_device, flush=True)
    with strategy.scope():
        model_lib_v2.train_loop(str(pipeline),str(checkpoint),train_steps=target,use_tpu=False,
                               save_final_config=False,checkpoint_every_n=math.gcd(config['steps'],100),
                               checkpoint_max_to_keep=3,record_summaries=False,
                               num_steps_per_iteration=math.gcd(config['steps'],10))
    latest = tf.train.latest_checkpoint(str(checkpoint))
    if not latest:
        raise ValueError('Training did not produce a checkpoint.')
    actual = int(tf.train.load_variable(latest,'step/.ATTRIBUTES/VARIABLE_VALUE'))
    if actual != target:
        raise ValueError(f'Checkpoint step {actual} does not match requested step {target}.')
    state = {'start_step':start,'final_step':actual,'checkpoint':latest,'device':config['device']}
    if config['device'] == 'gpu':
        state['gpu_name'] = tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0]).get('device_name','NVIDIA GPU')
        state['gpu_peak_memory_bytes'] = tf.config.experimental.get_memory_info('GPU:0')['peak']
        print('GPU training memory peak:', state['gpu_peak_memory_bytes'], 'bytes', flush=True)
    (Path(config['output'])/'training-state.json').write_text(json.dumps(state))
    print('Saved final training checkpoint at step',actual,flush=True)


def export(config):
    import tensorflow as tf
    from google.protobuf import text_format
    from object_detection.protos import pipeline_pb2
    from object_detection import export_tflite_graph_lib_tf2
    output = Path(config['output'])
    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse((output/'pipeline.config').read_text(),pipeline)
    export_tflite_graph_lib_tf2.export_tflite_model(pipeline,str(output/'checkpoint'),str(output/'exported'),config.get('max_detections',10),False)
    print('Exported TensorFlow SavedModel.',flush=True)


def quantize(config):
    import tensorflow as tf
    import numpy as np
    from PIL import Image, ImageOps
    output = Path(config['output'])
    converter = tf.lite.TFLiteConverter.from_saved_model(str(output/'exported/saved_model'))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    paths = config['calibration_images']
    size = config.get('image_size', 300)
    def representative():
        for path in paths:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
                data = np.asarray(rgb,dtype=np.float32)/127.5-1.0
            yield [data[None]]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = True
    converter.inference_input_type = tf.uint8
    model = converter.convert()
    artifacts = output/'artifacts';artifacts.mkdir(exist_ok=True)
    (artifacts/'limelight_neural_detector_8bit.tflite').write_bytes(model)
    (artifacts/'limelight_neural_detector_labels.txt').write_text('\n'.join(config.get('labels',['pollen']))+'\n')
    print('Quantized uint8 SSD TFLite:',len(model),'bytes',flush=True)


def validation_loss(config):
    import tensorflow as tf
    from google.protobuf import text_format
    from object_detection.protos import pipeline_pb2
    from object_detection.builders import model_builder
    from object_detection import inputs, model_lib, model_lib_v2
    pipeline=pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse(Path(config['pipeline']).read_text(),pipeline)
    pipeline.eval_config.batch_size=1
    reader=pipeline.eval_input_reader[0]
    reader.num_epochs=1;reader.shuffle=False;reader.num_readers=1
    reader.num_parallel_batches=1;reader.num_prefetch_batches=1
    model=model_builder.build(pipeline.model,is_training=False)
    tf.keras.backend.set_learning_phase(False)
    dataset=inputs.eval_input(eval_config=pipeline.eval_config,eval_input_config=reader,
                              model_config=pipeline.model,model=model)
    step=tf.Variable(0,trainable=False,dtype=tf.int64)
    checkpoint=tf.train.latest_checkpoint(config['checkpoint'])
    restore=tf.train.Checkpoint(model=model,step=step).restore(checkpoint).expect_partial()
    @tf.function
    def loss(features,labels):
        labels=model_lib.unstack_batch(labels,unpad_groundtruth_tensors=True)
        losses,_=model_lib_v2._compute_losses_and_predictions_dicts(
            model,features,labels,training_step=None,
            add_regularization_loss=pipeline.train_config.add_regularization_loss)
        return losses['Loss/total_loss']
    values=[float(loss(features,labels).numpy()) for features,labels in dataset]
    restore.assert_existing_objects_matched()
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Validation produced no finite loss values.')
    result={'step':int(step.numpy()),'loss':sum(values)/len(values),'images':len(values)}
    report=Path(config['report']);temporary=report.with_suffix('.tmp')
    temporary.write_text(json.dumps(result));temporary.replace(report)
    print('Validation loss:',result,flush=True)


def plot(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import tensorflow as tf
    output = Path(config['output'])
    accumulator = EventAccumulator(str(output/'checkpoint/train'),size_guidance={'tensors':0,'scalars':0})
    accumulator.Reload()
    values = []
    tag='Loss/total_loss'
    if tag in accumulator.Tags().get('tensors',[]):
        values = [(e.step,float(tf.make_ndarray(e.tensor_proto))) for e in accumulator.Tensors(tag)]
    elif tag in accumulator.Tags().get('scalars',[]):
        values = [(e.step,e.value) for e in accumulator.Scalars(tag)]
    fig, ax = plt.subplots(figsize=(8,4))
    if values:
        ax.plot(*zip(*values),color='#70ad28',linewidth=1.2,label='Train')
    validation_file=output.parent/'validation-history.json'
    if validation_file.is_file():
        validation=json.loads(validation_file.read_text())
        if validation: ax.plot([p['step'] for p in validation],[p['loss'] for p in validation],color='#b2a700',linewidth=1.8,label='Val')
    if values: ax.legend()
    if not values:
        ax.text(.5,.5,'No scalar loss samples recorded in this short run.',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Training step',ylabel='Total training loss',title='Pollen · SSD MobileNet V2')
    ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/'artifacts/training_loss_graph.png',dpi=150)
    plt.close(fig)
    (output/'loss-history.json').write_text(json.dumps(values))


if __name__ == '__main__':
    import tensorflow as tf
    # TF-Slim 1.1 uses names moved out of this internal module in TF 2.15.
    from tensorflow.python.ops import control_flow_ops
    if not hasattr(control_flow_ops,'case'): control_flow_ops.case = tf.case
    if not hasattr(control_flow_ops,'cond'): control_flow_ops.cond = tf.cond
    config = json.loads(Path(sys.argv[2]).read_text())
    {'train':train,'export':export,'quantize':quantize,'plot':plot,'validation-loss':validation_loss}[sys.argv[1]](config)
