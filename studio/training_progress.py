"""Observe training logs and sample checkpoint validation without interrupting training."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from studio.core import ROOT
from studio import training_runtime as runtime

HEADER = re.compile(r'steps\s+(\d+)\s*→\s*(\d+)')
STEP = re.compile(r'\bStep\s+(\d+)\s+per-step time\s+([\d.eE+-]+)s')
LOSS = re.compile(r"['\"]Loss/total_loss['\"]:\s*([\d.eE+-]+)")


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class LogProgress:
    def __init__(self, steps):
        self.total = steps
        self.start = 0
        self.target = steps
        self.current = 0
        self.seconds_per_step = None
        self.points = {}
        self.buffer = ''

    def feed(self, text):
        self.buffer += text
        lines = self.buffer.split('\n')
        self.buffer = lines.pop()
        for line in lines:
            if match := HEADER.search(line):
                self.start, self.target = map(int, match.groups())
                self.current = max(self.current, self.start)
            if match := STEP.search(line):
                self.current = max(self.current, int(match[1]))
                seconds = float(match[2])
                if math.isfinite(seconds) and seconds > 0:
                    self.seconds_per_step = seconds
            if match := LOSS.search(line):
                value = float(match[1])
                if math.isfinite(value) and self.current > self.start:
                    self.points[self.current] = value

    def report(self):
        done = max(0, min(self.total, self.current-self.start))
        points = sorted(self.points.items())
        # Keep the JSON and SVG small for long runs, including first and latest samples.
        stride = max(1, math.ceil(len(points)/1200))
        chart = points[::stride]
        if points and chart[-1] != points[-1]: chart.append(points[-1])
        return {'completed_steps':done,'total_steps':self.total,'global_step':self.current,
                'start_step':self.start,'target_step':self.target,
                'percent':round(100*done/self.total,1) if self.total else 0,
                'train_loss':points[-1][1] if points else None,'train_history':chart,
                'seconds_per_step':self.seconds_per_step,
                'eta_seconds':round((self.total-done)*self.seconds_per_step) if self.seconds_per_step and done < self.total else None}


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return default


def checkpoint_snapshot(folder):
    """Pin a complete checkpoint so training may rotate old files safely."""
    source = folder/'model/checkpoint'
    state = source/'checkpoint'
    if not state.is_file(): return None
    match = re.search(r'model_checkpoint_path:\s*"([^"\n]+)"',state.read_text())
    if not match: return None
    name = Path(match[1]).name
    destination = folder/'live-validation/checkpoint'
    destination.mkdir(parents=True,exist_ok=True)
    files = list(source.glob(name+'.*'))
    if not any(p.suffix=='.index' for p in files): return None
    for file in destination.iterdir():
        if file.is_file(): file.unlink()
    for file in files: shutil.copy2(file,destination/file.name)
    (destination/'checkpoint').write_text(f'model_checkpoint_path: "{name}"\n')
    return destination


def _monitor(folder, parent=None):
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    parser = LogProgress(config['steps'])
    offset = 0
    evaluation = None
    evaluation_log = None
    last_requested = -1000
    last_attempt = 0
    history = []
    history_file = folder/'validation-history.json'
    if history_file.is_file(): history = json.loads(history_file.read_text())
    report_file = folder/'live-validation/result.json'
    try:
        while True:
            log = folder/'log.txt'
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(offset)
                    text = stream.read()
                    offset = stream.tell()
                parser.feed(text.decode(errors='replace'))
            training_state = folder/'model/training-state.json'
            if training_state.is_file():
                state = read_json(training_state)
                if state:
                    parser.start = state['start_step']
                    parser.current = state['final_step']
                    parser.target = parser.start+parser.total
            if evaluation is not None and evaluation.poll() is not None:
                if evaluation.returncode == 0 and report_file.is_file():
                    point = json.loads(report_file.read_text())
                    history = [p for p in history if p['step'] != point['step']]+[point]
                    history.sort(key=lambda p:p['step'])
                    atomic_json(history_file,history)
                evaluation_log.close(); evaluation_log = None; evaluation = None
            complete = (folder/'model/result.json').is_file()
            status = None
            job_file = folder/'job.json'
            if job_file.is_file(): status=read_json(job_file,{}).get('status')
            alive = True
            if parent:
                try: os.kill(parent,0)
                except ProcessLookupError: alive=False
            cancelled = status in ('cancelled','failed','interrupted')
            if cancelled: break
            ready = (folder/'model/pipeline.config').is_file() and (folder/'records/valid/pollen.tfrecord').is_file()
            latest_val = history[-1]['step'] if history else -1
            final_needed = parser.current >= parser.target and latest_val < parser.target
            eligible = parser.current >= 100 and parser.current-last_requested >= 1000
            if ready and evaluation is None and (eligible or final_needed) and time.time()-last_attempt > 30:
                try:
                    checkpoint = checkpoint_snapshot(folder)
                    if checkpoint:
                        last_requested=parser.current; last_attempt=time.time()
                        report_file.unlink(missing_ok=True)
                        eval_config = folder/'live-validation/config.json'
                        eval_config.write_text(json.dumps({'pipeline':str(folder/'model/pipeline.config'),
                                                          'checkpoint':str(checkpoint),'report':str(report_file)}))
                        evaluation_log=(folder/'validation-monitor.log').open('a')
                        evaluation=subprocess.Popen([str(runtime.PYTHON),'-u','-m','studio.tf_training','validation-loss',str(eval_config)],
                                                    env=runtime.environment('cpu'),cwd=ROOT,stdout=evaluation_log,stderr=subprocess.STDOUT)
                except OSError:
                    last_attempt=time.time()
            live=parser.report()
            live['validation_history']=[[p['step'],p['loss']] for p in history]
            live['validation_images']=history[-1]['images'] if history else None
            live['validation_running']=evaluation is not None
            live['updated']=time.time()
            atomic_json(folder/'training-live.json',live)
            # Existing servers already expose this file; no restart is needed for a running job.
            progress_file=folder/'progress.json'
            with (folder/'progress.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                stage=read_json(progress_file,{'stage':'Training'})
                stage.update(live)
                atomic_json(progress_file,stage)
            if (complete or not alive or status=='completed') and evaluation is None:
                # A finished run needs at most one final evaluation attempt.
                if not final_needed or last_requested >= parser.target: break
            time.sleep(2)
    finally:
        if evaluation is not None and evaluation.poll() is None:
            evaluation.terminate()
            try: evaluation.wait(timeout=5)
            except subprocess.TimeoutExpired: evaluation.kill(); evaluation.wait()
        if evaluation_log: evaluation_log.close()


def monitor(folder, parent=None):
    folder=Path(folder)
    with (folder/'observer.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        _monitor(folder,parent)


if __name__ == '__main__':
    arguments=argparse.ArgumentParser()
    arguments.add_argument('folder',type=Path)
    arguments.add_argument('--parent',type=int)
    args=arguments.parse_args()
    monitor(args.folder,args.parent)
