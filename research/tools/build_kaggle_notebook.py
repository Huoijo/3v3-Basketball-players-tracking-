#!/usr/bin/env python3
"""Build a standalone Kaggle notebook from the unchanged tutorial source."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'TrackID3x3_Challenge_Tutorial.ipynb'
TARGET = ROOT / 'notebooks' / 'TrackID3x3_Challenge_Tutorial_kaggle.ipynb'
EXPECTED_ORIGINAL = '33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00'


def source(nb, index):
    return ''.join(nb['cells'][index]['source'])


def parser_source(nb):
    raw = source(nb, 9)
    tree = ast.parse(raw)
    functions = [ast.get_source_segment(raw, n) for n in tree.body if isinstance(n, ast.FunctionDef)]
    return '\n\n'.join(functions) + '\n'


def build():
    digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    if digest != EXPECTED_ORIGINAL:
        raise RuntimeError('Original notebook hash differs; review the baseline before rebuilding.')
    original = json.loads(ORIGINAL.read_text())
    cells = []
    def add(kind, ident, content, origin=None, category=None):
        metadata = {}
        if origin is not None:
            metadata['original_cell'] = origin
        if category:
            metadata['change_categories'] = category
        entry = dict(cell_type=kind, id=ident, metadata=metadata)
        if kind == 'code':
            content = f'# TrackID3x3 Kaggle cell: {ident}\n' + content.strip() + '\n'
            entry.update(execution_count=None, outputs=[])
        entry['source'] = content.splitlines(keepends=True)
        cells.append(entry)
    def md(ident, content):
        add('markdown', ident, content)
    def code(ident, content, origin=None, category=None):
        add('code', ident, content, origin, category)
    def stage(ident, number, name, content, origin=None, category=None):
        code(ident, f'with baseline_stage({number!r}, {name!r}):\n' + textwrap.indent(content.strip()+'\n', '    '), origin, category or ['OBSERVABILITY'])

    md('intro', '''# TrackID3x3 — Kaggle GPU compatibility baseline

Status before execution: **READY_FOR_KAGGLE**. No successful Kaggle run is claimed by this file.

Enable a GPU and Internet in Kaggle Settings, then **Restart Session → Run All**.
For offline use, attach the original four tutorial files, `yolo11x.pt`, default EasyOCR weights,
and any missing package wheels; set `ALLOW_DOWNLOADS=False` and the input directories in the first cell.
Do not change models, thresholds, roster, filters or interpolation to match historical scores.

Outputs: `/kaggle/working/trackid3x3_runs/<run-id>/`. Videos are optional (`SAVE_VIDEOS=False` by default).
The original remains `TrackID3x3_Challenge_Tutorial.ipynb`, SHA-256
`33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00`.

Original tutorial: Kazuhiro Yamada & Keisuke Fujii, Nagoya University; CC BY 4.0.
Source: https://github.com/open-starlab/TrackID3x3 . This derivative changes environment,
paths, dependency setup, visualization and observability only.

The source notebook hash and a source-only replay snapshot are saved at completion.
Runtime code history captures code edits; UI markdown edits require exporting the notebook separately.
''')
    code('setup', (ROOT/'notebooks/kaggle_bootstrap.py').read_text(), category=['ENVIRONMENT','PATH','DEPENDENCY','OBSERVABILITY'])
    imports = source(original,3)
    imports = '\n'.join(line for line in imports.splitlines() if not line.startswith('!'))
    imports = imports[:imports.index('# Check GPU availability')]
    stage('environment',1,'Environment', '''install_missing_dependencies()
''' + imports + '''
save_environment()
print('Run directory:', RUN_DIR)
''',3,['ENVIRONMENT','DEPENDENCY','OBSERVABILITY'])
    stage('data',2,'Data', '''video_path, gt_file, keypoint_file, court_image_path = resolve_data()
YOLO_WEIGHTS = prepare_models()
''',5,['PATH','DEPENDENCY','OBSERVABILITY'])
    probe = (ROOT/'notebooks/kaggle_label_probe.py').read_text().replace('# ORIGINAL_PARSER',parser_source(original))
    stage('label-probe','3-probe','YOLO / ByteTrack real-label compatibility',
          f'''probe_source = {probe!r}
probe_path = TEMP_DIR / 'label_probe.py'
probe_path.write_text(probe_source)
process = subprocess.run([sys.executable, str(probe_path), video_path, YOLO_WEIGHTS, str(RUN_DIR), str(PROBE_FRAMES)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(RUN_DIR / 'label_probe.log').write_text(process.stdout)
if process.returncode:
    raise RuntimeError('Real tracking/parser compatibility probe failed. Inspect label_probe.log and label_probe/label_format.json; no parser behavior was changed.')
probe_summary = json.loads((RUN_DIR / 'label_probe' / 'label_format.json').read_text())
print('Actual sample:', probe_summary['frames'], 'frames;', probe_summary['tracked_label_rows'], 'parsed tracked rows')
''',category=['LIBRARY-COMPATIBILITY','OBSERVABILITY'])
    track = source(original,7).replace('YOLO("yolo11x.pt")','YOLO(YOLO_WEIGHTS)').replace('tracker="bytetrack.yaml"','tracker=str(RUN_DIR / "bytetrack.yaml")').replace('save=True,','save=SAVE_VIDEOS,').replace('    verbose=False\n','    verbose=False,\n    project=str(RUN_DIR),\n    name="tracking",\n    exist_ok=False\n')
    track += '''\nif Path(out_dir).resolve() != (RUN_DIR / 'tracking').resolve():
    raise RuntimeError(f'Unexpected tracking save directory: {out_dir}')
print('Saved label files:', len(list((Path(out_dir) / 'labels').glob('*.txt'))))
save_json('tracking_arguments.json', vars(model.predictor.args))
'''
    stage('tracking',3,'Detection + Tracking',track,7,['PATH','VISUALIZATION','OBSERVABILITY'])
    mot = source(original,9)
    split = mot.index('# Load MOT data for visualization')
    conversion, track_viz = mot[:split], mot[split:]
    conversion += '''\nwith open(mot_txt_path) as handle:
    mot_rows = [line for line in handle if line.strip()]
if not mot_rows:
    raise RuntimeError('MOT conversion produced no rows; inspect tracking labels.')
print('MOT rows:', len(mot_rows))
'''
    stage('mot',4,'MOT Conversion',conversion,9,['OBSERVABILITY'])
    stage('homography',5,'Homography',source(original,11) + '''\nsave_json('homography.json', dict(H=H.tolist(), image_points_px=image_points.tolist(), court_points_cm=court_points.tolist()))
np.save(RUN_DIR / 'homography.npy', H)
''',11,['OBSERVABILITY'])
    stage('court-filtering',6,'Court Filtering',source(original,15) + '''\nprint('Filtered rows:', len(df))
''',15,['OBSERVABILITY'])
    raw17 = source(original,17)
    offsets = [raw17.index('# Team identification by jersey-color clustering'),
               raw17.index('# Build semantic identities'),raw17.index('# Build the final prediction table')]
    segments=[raw17[:offsets[0]], raw17[offsets[0]:offsets[1]], raw17[offsets[1]:offsets[2]], raw17[offsets[2]:]]
    segments[0] = segments[0].replace('    gpu=True\n','    gpu=True,\n    model_storage_directory=str(OCR_MODEL_DIR),\n    download_enabled=ALLOW_DOWNLOADS\n')
    gpu_guard = '''\nif str(reader.device) != 'cuda':
    raise RuntimeError(f'EasyOCR did not select CUDA: {reader.device}. CPU fallback is disabled.')
print('EasyOCR device:', reader.device)
'''
    segments[0] = segments[0].replace('MAX_SAMPLED_FRAMES_PER_TRACK = 40',
                                    gpu_guard + '\nMAX_SAMPLED_FRAMES_PER_TRACK = 40')
    for ident,number,name,part,categories in zip(['ocr','teams','identity','prediction'],[7,8,9,10],
             ['OCR','Team Assignment','Semantic Identity','Final Prediction'],segments,
             [['PATH','ENVIRONMENT','OBSERVABILITY'],['OBSERVABILITY'],['OBSERVABILITY'],['OBSERVABILITY']]):
        stage(ident,number,name,part,17,categories)
    code('evaluator-definitions',source(original,21),21)
    evaluate = source(original,23) + '\n\n' + source(original,24)
    evaluate += '''\n# OBSERVABILITY: persist the returned metrics; matching and in-memory precision are unchanged.
save_json('ti_hota.json', dict(tau_cm=tau, alphas=alphas.tolist(), metrics=metrics))
clean_diagnostics = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                     for key,value in diagnostics.items()}
save_json('location_diagnostics.json', clean_diagnostics)
historical = {'TI-HOTA': 0.6351, 'TI-DetA': 0.6362, 'TI-AssA': 0.6341}
save_json('historical_comparison.json', dict(reference='Unverified saved original-notebook output',
          historical=historical, fresh={k:metrics[k] for k in historical},
          fresh_minus_historical={k:metrics[k]-v for k,v in historical.items()},
          note='No exact-score equality requirement; seeds and algorithm remain unchanged.'))
print('Persisted evaluation:', RUN_DIR / 'ti_hota.json')
'''
    stage('evaluation',11,'Evaluation',evaluate,[23,24],['OBSERVABILITY'])
    md('optional-video-note','''## Optional visualization

When `SAVE_VIDEOS=True`, the following cell creates the original tracker video, rectified
court image, GT/pred minimaps, and combined video. It runs **after persisted evaluation**.
Only Colab display and destination paths are replaced. The original minimap time handling
and drawing parameters are preserved. No metric depends on this cell.
''')
    homography_viz=source(original,13).replace('from google.colab.patches import cv2_imshow','from IPython.display import display, Image\n\ndef cv2_imshow(image):\n    ok, encoded = cv2.imencode(".png", image)\n    if not ok:\n        raise RuntimeError("Could not encode court preview")\n    display(Image(data=encoded.tobytes()))').replace('"/content/"','str(RUN_DIR) + "/"')
    viz = track_viz + '\n\n' + homography_viz + '\n\n' + source(original,26) + '\n\n' + source(original,28) + '\n\n' + source(original,30)
    viz = viz.replace('"/content"','str(RUN_DIR)')
    # Original cell 30 reuses H for display height; restore calibration afterwards.
    viz = '''saved_homography = H.copy()
try:
''' + textwrap.indent(viz.strip()+'\n','    ') + '''finally:
    H = saved_homography
'''
    code('optional-videos','''if SAVE_VIDEOS:
    with baseline_stage('optional', 'Visualizations'):
''' + textwrap.indent(viz,'        ') + '''else:
    print('Optional videos skipped (SAVE_VIDEOS=False); predictions and metrics are already saved.')
''',[9,13,26,28,30],['PATH','VISUALIZATION','OBSERVABILITY'])
    code('finish','''with baseline_stage('finalize', 'Evidence'):
    capture_notebook_source(get_ipython().history_manager.input_hist_raw)
    save_json('model_hashes.json', {str(p.relative_to(RUN_DIR)): file_sha256(p)
              for p in sorted((RUN_DIR / 'models').rglob('*')) if p.is_file()})
    save_json('output_summary.json', dict(predictions_path=predictions_file, prediction_rows=len(predictions_df),
              prediction_frames=int(predictions_df.frame_id.nunique()), predictions_sha256=file_sha256(predictions_file),
              metric_precision='original in-memory predictions', metrics_path=str(RUN_DIR / 'ti_hota.json'),
              original_notebook_sha256=ORIGINAL_NOTEBOOK_SHA256))
    RUN_STATE['status'] = 'PASS' if IS_KAGGLE else 'COMPLETED_ON_OTHER_CUDA_HOST'
    print('Metric pipeline completed. Outputs:', RUN_DIR)
save_json('artifact_hashes.json', {str(p.relative_to(RUN_DIR)): file_sha256(p)
          for p in sorted(RUN_DIR.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
''',category=['OBSERVABILITY'])
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                    language_info=dict(name='python'), accelerator='GPU'), nbformat=4,nbformat_minor=5)
    layout=copy.deepcopy(notebook)
    for cell in layout['cells']:
        if cell['cell_type']=='code':
            cell['source']=[]
    setup=next(c for c in notebook['cells'] if c['id']=='setup')
    text=''.join(setup['source']).replace('# BUILD_LAYOUT', 'NOTEBOOK_LAYOUT = '+repr(layout))
    setup['source']=text.splitlines(keepends=True)
    TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    digest=hashlib.sha256(TARGET.read_bytes()).hexdigest()
    (TARGET.with_suffix('.ipynb.sha256')).write_text(digest+'  '+TARGET.name+'\n')
    (ROOT/'notebooks/original_notebook.sha256').write_text(EXPECTED_ORIGINAL+'  '+ORIGINAL.name+'\n')
    print(TARGET.relative_to(ROOT))
    print('SHA-256:',digest)


if __name__=='__main__':
    build()
