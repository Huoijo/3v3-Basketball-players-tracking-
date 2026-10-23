"""Embedded by build_kaggle_notebook.py; the generated notebook is standalone."""
import contextlib
import hashlib
import importlib
import importlib.util
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ENVIRONMENT / PATH: edit these only when using mounted offline inputs.
WORK_DIR = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd() / 'trackid3x3_work'
INPUT_DIR = Path('/kaggle/input')
DATA_INPUT_DIR = None     # e.g. Path('/kaggle/input/trackid3x3-tutorial')
MODEL_INPUT_DIR = None    # optional directory containing original YOLO/EasyOCR weights
WHEEL_DIR = None          # optional directory of compatible wheels for offline missing packages
ALLOW_DOWNLOADS = True   # False: no data/model/PyPI network requests
SAVE_VIDEOS = False      # optional visualization only; metrics/predictions are unchanged
PROBE_FRAMES = 12         # isolated real tracking format check; never used as baseline predictions
DATA_DIR = WORK_DIR / 'trackid3x3_tutorial_data'
OUTPUT_DIR = WORK_DIR / 'trackid3x3_runs'
RUN_ID = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%dT%H%M%S%f') + '_baseline_kaggle'
RUN_DIR = OUTPUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)
TEMP_DIR = RUN_DIR / 'tmp'
TEMP_DIR.mkdir()
OCR_MODEL_DIR = RUN_DIR / 'models' / 'easyocr'
OCR_MODEL_DIR.mkdir(parents=True)
IS_KAGGLE = INPUT_DIR.is_dir() and Path('/kaggle/working').is_dir()
ORIGINAL_NOTEBOOK_SHA256 = '33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00'
# Builder injects source-only notebook structure here, to capture exact executed code later.
# BUILD_LAYOUT


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(name, obj):
    (RUN_DIR / name).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


RUN_STATE = dict(status='RUNNING', platform='Kaggle' if IS_KAGGLE else 'other',
                 original_notebook_sha256=ORIGINAL_NOTEBOOK_SHA256,
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 seeds='unset, as in original notebook', stages=[])
save_json('run_manifest.json', RUN_STATE)
save_json('configuration.json', dict(WORK_DIR=str(WORK_DIR), INPUT_DIR=str(INPUT_DIR),
          DATA_INPUT_DIR=str(DATA_INPUT_DIR) if DATA_INPUT_DIR else None,
          MODEL_INPUT_DIR=str(MODEL_INPUT_DIR) if MODEL_INPUT_DIR else None,
          WHEEL_DIR=str(WHEEL_DIR) if WHEEL_DIR else None,
          DATA_DIR=str(DATA_DIR), OUTPUT_DIR=str(OUTPUT_DIR), RUN_DIR=str(RUN_DIR),
          TEMP_DIR=str(TEMP_DIR), ALLOW_DOWNLOADS=ALLOW_DOWNLOADS, SAVE_VIDEOS=SAVE_VIDEOS,
          PROBE_FRAMES=PROBE_FRAMES, seed_policy='unset; original behavior'))


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile
    def write(self, value):
        self.logfile.write(value)
        self.logfile.flush()
        return self.original.write(value)
    def flush(self):
        self.logfile.flush()
        self.original.flush()
    def __getattr__(self, key):
        return getattr(self.original, key)


@contextlib.contextmanager
def baseline_stage(number, name):
    start = time.perf_counter()
    entry = dict(stage=number, name=name, status='RUNNING')
    RUN_STATE['stages'].append(entry)
    save_json('run_manifest.json', RUN_STATE)
    with (RUN_DIR / 'stages.log').open('a') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f'[{number}] {name}')
            try:
                yield
            except BaseException:
                entry['status'] = 'FAILED'
                RUN_STATE['status'] = 'FAILED'
                (RUN_DIR / 'failure.log').write_text(traceback.format_exc())
                raise
            else:
                entry['status'] = 'COMPLETE'
            finally:
                entry['elapsed_seconds'] = time.perf_counter() - start
                save_json('run_manifest.json', RUN_STATE)
                print(f'[{number}] {entry["status"]} ({entry["elapsed_seconds"]:.2f}s)')


def choose_unique(root, filename):
    root = Path(root)
    matches = sorted(root.rglob(filename)) if root.is_dir() else []
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous {filename} in {root}: {matches}. Set a specific input directory.')
    return matches[0] if matches else None


def install_missing_dependencies():
    # DEPENDENCY: retain the Kaggle numeric and PyTorch/CUDA stack.
    from packaging.version import Version
    required = [('ultralytics', 'ultralytics', '8.3.0'), ('easyocr', 'easyocr', None),
                ('ndjson', 'ndjson', None), ('lap', 'lap', '0.5.12')]
    if ALLOW_DOWNLOADS:
        required.append(('gdown', 'gdown', None))
    if importlib.util.find_spec('cv2') is None:
        required.append(('opencv-python', 'cv2', None))
    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'IPython', 'packaging'):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f'Kaggle runtime is missing {name}; use the standard GPU image. Core stack is not auto-upgraded.')
    constraints = []
    protected = {}
    for dist in metadata.distributions():
        name = dist.metadata.get('Name', '').lower().replace('_', '-')
        if name in {'torch', 'torchvision', 'torchaudio', 'numpy', 'pandas', 'scipy',
                    'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'triton'} or name.startswith('nvidia-'):
            protected[name] = dist.version
            constraints.append(f'{name}=={dist.version}')
    constraint_path = RUN_DIR / 'protected-runtime-constraints.txt'
    constraint_path.write_text('\n'.join(sorted(set(constraints))) + '\n')
    missing = []
    for dist, module, minimum in required:
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = None
        if version is None or (minimum and Version(version) < Version(minimum)):
            missing.append(dist + (f'>={minimum}' if minimum else ''))
    save_json('dependency_plan.json', dict(install=missing, protected=protected))
    if missing:
        command = [sys.executable, '-m', 'pip', 'install', '--constraint', str(constraint_path)]
        if WHEEL_DIR is not None:
            command += ['--find-links', str(WHEEL_DIR)]
        if not ALLOW_DOWNLOADS:
            if WHEEL_DIR is None:
                raise RuntimeError(f'Missing packages {missing}. Enable Internet or add wheels and set WHEEL_DIR.')
            command += ['--no-index']
        command += missing
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (RUN_DIR / 'dependency_install.log').write_text(process.stdout)
        if process.returncode:
            raise RuntimeError('Dependency install failed under protected runtime constraints; see dependency_install.log. No automatic core-stack upgrade.')
        importlib.invalidate_caches()
    else:
        (RUN_DIR / 'dependency_install.log').write_text('No dependency installation needed.\n')
    for name, version in protected.items():
        if metadata.version(name) != version:
            raise RuntimeError(f'Protected package changed: {name}')
    # Import failures are recorded; do not speculatively replace installed packages.
    for _, module, _ in required:
        importlib.import_module(module)


def save_environment():
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'opencv-python', 'opencv-python-headless',
                 'easyocr', 'numpy', 'pandas', 'scipy', 'ndjson', 'gdown', 'lap'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               packages=packages, torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
               cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
               gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    save_json('environment.json', env)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (RUN_DIR / 'pip-freeze.txt').write_text(freeze.stdout + freeze.stderr)
    if freeze.returncode:
        raise RuntimeError('pip freeze failed; see pip-freeze.txt')
    print('torch version:', env['torch_version'])
    print('CUDA available:', env['cuda_available'])
    print('CUDA version:', env['cuda_version'])
    print('GPU name:', env['gpu_name'])
    print('Packages:', json.dumps(packages))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. On Kaggle enable a GPU accelerator, Restart Session, then Run All. CPU fallback is disabled.')
    torch.empty(1, device='cuda:0')  # verifies device 0 is usable


def resolve_data():
    filenames = ['IMG_0104_1.MOV', 'IMG_0104_1_tutorial_challenge.csv', 'Outdoor.ndjson', 'Outdoor.png']
    input_root = Path(DATA_INPUT_DIR) if DATA_INPUT_DIR is not None else INPUT_DIR
    found = {name: choose_unique(input_root, name) for name in filenames}
    if not all(found.values()):
        if DATA_INPUT_DIR is not None:
            raise FileNotFoundError(f'Incomplete explicit DATA_INPUT_DIR {input_root}: missing {[n for n,p in found.items() if p is None]}')
        # Downloaded cache is checked separately; never mix partial input datasets.
        found = {name: choose_unique(DATA_DIR, name) for name in filenames}
        if not all(found.values()):
            if not ALLOW_DOWNLOADS:
                raise FileNotFoundError('Add the original four tutorial files under /kaggle/input and set DATA_INPUT_DIR. Downloads are disabled.')
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'gdown',
                       'https://drive.google.com/drive/folders/1_yDMy8YVpxyqcKg2_eXN1xz_vg2FeCGu',
                       '--folder', '-O', str(DATA_DIR)]
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (RUN_DIR / 'data_download.log').write_text(process.stdout)
            if process.returncode:
                raise RuntimeError('Original gdown folder download failed. See data_download.log; add the same original files as a Kaggle Dataset input.')
            found = {name: choose_unique(DATA_DIR, name) for name in filenames}
    if not all(found.values()):
        raise FileNotFoundError(f'Missing original tutorial inputs: {[n for n,p in found.items() if p is None]}')
    known = {'IMG_0104_1_tutorial_challenge.csv': 'a86beb68f5e8351eea3b65b126db2a167b4c2f2dc805715818951c6760f4f2a2',
             'Outdoor.ndjson': '84bedee6b8b499e84bb98dba99b8c36a11c3eb08c60e88c3e7fbc95979b1b9ca'}
    inventory = {}
    for name, path in found.items():
        digest = file_sha256(path)
        inventory[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
        if name in known and digest != known[name]:
            raise RuntimeError(f'{name} differs from the audited tutorial input; verify the dataset instead of silently substituting it.')
        print(f'{name}: {path}')
    save_json('inputs.json', inventory)
    import cv2
    cap = cv2.VideoCapture(str(found[filenames[0]]))
    ok, _ = cap.read()
    info = dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                frames=cap.get(cv2.CAP_PROP_FRAME_COUNT), fps=cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    save_json('video_properties.json', info)
    if not ok or min(info.values()) <= 0:
        raise RuntimeError('Original tutorial video cannot be decoded; inspect video_properties.json.')
    return tuple(str(found[name]) for name in filenames)


def prepare_models():
    import ultralytics
    from easyocr.config import detection_models, recognition_models
    model_root = Path(MODEL_INPUT_DIR) if MODEL_INPUT_DIR is not None else INPUT_DIR
    weights = choose_unique(model_root, 'yolo11x.pt')
    target = RUN_DIR / 'models' / 'yolo11x.pt'
    if weights is None:
        cached = WORK_DIR / 'yolo11x.pt'
        weights = cached if cached.is_file() else None
    if weights:
        shutil.copy2(weights, target)
    elif ALLOW_DOWNLOADS:
        # Same YOLO11x model; destination is a writable run-local path.
        from ultralytics import YOLO
        YOLO(str(target))
    else:
        raise FileNotFoundError('Offline run needs the original yolo11x.pt in MODEL_INPUT_DIR or /kaggle/input.')
    if not target.is_file():
        raise FileNotFoundError(f'YOLO11x weight download did not produce {target}')
    # Reader(["en"], gpu=True) defaults: craft detector, standard English recognizer.
    required = [detection_models['craft']['filename'], recognition_models['gen2']['english_g2']['filename']]
    for filename in required:
        source = choose_unique(model_root, filename)
        if source is None:
            source = choose_unique(Path.home() / '.EasyOCR' / 'model', filename)
        if source:
            shutil.copy2(source, OCR_MODEL_DIR / filename)
        elif not ALLOW_DOWNLOADS:
            raise FileNotFoundError(f'Offline EasyOCR needs {filename} in MODEL_INPUT_DIR; no alternate OCR model will be selected.')
    yaml_source = Path(ultralytics.__file__).resolve().parent / 'cfg' / 'trackers' / 'bytetrack.yaml'
    if not yaml_source.is_file():
        raise FileNotFoundError(f'Installed Ultralytics ByteTrack YAML missing: {yaml_source}')
    shutil.copy2(yaml_source, RUN_DIR / 'bytetrack.yaml')
    save_json('models_before_inference.json', dict(yolo_path=str(target), yolo_sha256=file_sha256(target),
              bytetrack_source=str(yaml_source), bytetrack_sha256=file_sha256(yaml_source),
              easyocr_default_files=required))
    return str(target)


def capture_notebook_source(history):
    # Hash a persisted source notebook, not a guessed pathname or an output-bearing export.
    # Raw executed code comes from this kernel; markdown/metadata come from the build layout.
    observed = {}
    for raw in history:
        for cell in NOTEBOOK_LAYOUT['cells']:
            marker = '# TrackID3x3 Kaggle cell: ' + cell['id']
            if cell['cell_type'] == 'code' and raw.startswith(marker + '\n'):
                observed[cell['id']] = raw.strip() + '\n'
    rebuilt = json.loads(json.dumps(NOTEBOOK_LAYOUT))
    for cell in rebuilt['cells']:
        if cell['cell_type'] == 'code':
            if cell['id'] not in observed:
                raise RuntimeError(f'No executed source for {cell["id"]}; use Restart Session → Run All.')
            cell['source'] = observed[cell['id']].splitlines(keepends=True)
    path = RUN_DIR / 'kaggle_source_snapshot.ipynb'
    path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=1) + '\n')
    save_json('source_hashes.json', dict(original_notebook_sha256=ORIGINAL_NOTEBOOK_SHA256,
              kaggle_notebook_sha256=file_sha256(path), kaggle_notebook_path=str(path),
              hash_basis='Executed raw code + build markdown/metadata; output-free source notebook. Markdown-only edits in the UI are not captured.'))
