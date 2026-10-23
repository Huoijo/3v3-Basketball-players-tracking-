#!/usr/bin/env python3
"""Inspect the unchanged tutorial and run its evaluator without model dependencies.

No model execution, methodology patching, seeding, or dependency installation.
Cell numbers are zero-based JSON indices. Run from the repository root.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import contextlib
import io
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'TrackID3x3_Challenge_Tutorial.ipynb'
GT_ID = '1ZIkheTLtqdqINd6gcRi1dZsJdaWgKhC0'
GT_NAME = 'IMG_0104_1_tutorial_challenge.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def evaluator_namespace(cells):
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    ns = dict(np=np, pd=pd, math=math, defaultdict=defaultdict,
              linear_sum_assignment=linear_sum_assignment)
    # Execute source definitions verbatim; exclude notebook top-level execution.
    for index in (21, 24):
        tree = ast.parse(''.join(cells[index]['source']))
        funcs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                           type_ignores=[])
        exec(compile(funcs, f'{NOTEBOOK.name}:cell{index}', 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='New run directory; existing directories are rejected')
    parser.add_argument('--download-gt', action='store_true',
                        help='Download only the 193 KB tutorial GT, not video or models')
    parser.add_argument('--gt', type=Path, help='Existing Challenge-format GT')
    parser.add_argument('--download-calibration', action='store_true',
                        help='Download Outdoor.ndjson and execute unchanged homography cell 11')
    parser.add_argument('--predictions', type=Path,
                        help='Existing five-column submission CSV; evaluate all rows')
    args = parser.parse_args()
    if args.predictions and not (args.gt or args.download_gt):
        parser.error('--predictions requires --gt or --download-gt')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    notebook = json.loads(NOTEBOOK.read_text())
    cells = notebook['cells']
    source_dir = out / 'source'
    source_dir.mkdir()
    index = []
    saved_outputs = []
    for i, cell in enumerate(cells):
        source = ''.join(cell['source'])
        path = source_dir / f'cell_{i:02d}.{ "py" if cell["cell_type"] == "code" else "md"}'
        path.write_text(source)
        entry = dict(cell=i, type=cell['cell_type'], source_file=str(path.relative_to(out)),
                     sha256=sha(path), execution_count=cell.get('execution_count'))
        if cell['cell_type'] == 'code':
            cleaned = '\n'.join(l for l in source.splitlines() if not l.lstrip().startswith('!'))
            try:
                ast.parse(cleaned)
                entry['syntax_without_shell_magics'] = 'PASS'
            except SyntaxError as exc:
                entry['syntax_without_shell_magics'] = str(exc)
        index.append(entry)
        saved_outputs.append(f'\n===== CELL {i}: HISTORICAL SAVED OUTPUT, NOT REPRODUCED =====\n')
        for item in cell.get('outputs', []):
            if 'text' in item:
                saved_outputs.append(''.join(item['text']))
            elif 'ename' in item:
                saved_outputs.append(f'{item["ename"]}: {item.get("evalue", "")}\n')
            elif 'text/plain' in item.get('data', {}):
                saved_outputs.append(''.join(item['data']['text/plain']))
    write_json(out / 'cell_index.json', index)
    (out / 'saved_notebook_outputs.txt').write_text(''.join(saved_outputs))
    versions = {}
    for package in ['numpy', 'pandas', 'scipy', 'torch', 'opencv-python', 'opencv-python-headless', 'ultralytics',
                    'easyocr', 'ndjson', 'gdown', 'nbclient', 'lap']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = dict(python=sys.version, executable=sys.executable,
                       platform=platform.platform(), machine=platform.machine(),
                       packages=versions)
    try:
        import torch
        environment['cuda_available'] = torch.cuda.is_available()
        environment['cuda_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        environment['cuda_available'] = None
        environment['cuda_device'] = None
    write_json(out / 'environment.json', environment)
    # Attempt the unchanged Python portion of cell 3. Do not run pip magics
    # against the user's global environment. Report exactly where it stops.
    prep = '\n'.join(l for l in ''.join(cells[3]['source']).splitlines()
                     if not l.lstrip().startswith('!'))
    try:
        exec(compile(prep, f'{NOTEBOOK.name}:cell3', 'exec'), {})
        preflight = 'PASS: cell 3 Python statements completed\n'
    except Exception:
        preflight = traceback.format_exc()
    (out / 'baseline_preflight.log').write_text(preflight)
    result = dict(baseline_status='BLOCKED', baseline_score=None,
                  reason='No detector/tracker/OCR execution in this audit runner; see preflight and reproduction recipe.',
                  seed_policy='Original notebook does not explicitly set seeds; unchanged.',
                  notebook_sha256=sha(NOTEBOOK), audit_script_sha256=sha(__file__),
                  command=sys.argv, inputs={})
    if args.download_calibration:
        try:
            import cv2
            import ndjson
            import numpy as np
            calibration = out / 'Outdoor.ndjson'
            url = 'https://drive.google.com/uc?export=download&id=1HsHq5mlKgzx3C9YDpD2GEpshvA3U3Zvd'
            with urllib.request.urlopen(url, timeout=30) as response:
                calibration.write_bytes(response.read(2_000_000))
            ns_h = dict(cv2=cv2, ndjson=ndjson, np=np, keypoint_file=str(calibration))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(''.join(cells[11]['source']),
                             f'{NOTEBOOK.name}:cell11', 'exec'), ns_h)
            (out / 'homography.log').write_text(output.getvalue())
            projected = cv2.perspectiveTransform(ns_h['image_points'].reshape(-1, 1, 2),
                                                ns_h['H']).reshape(-1, 2)
            residual = np.linalg.norm(projected - ns_h['court_points'], axis=1)
            assert np.isfinite(ns_h['H']).all() and float(residual.max()) < .01
            write_json(out / 'homography.json', dict(H=ns_h['H'].tolist(),
                       image_points_px=ns_h['image_points'].tolist(),
                       court_points_cm=ns_h['court_points'].tolist(),
                       calibration_fit_residual_cm=residual.tolist(),
                       note='Four calibration points fit exactly; not held-out localization accuracy'))
            result['inputs']['calibration'] = dict(path=str(calibration), sha256=sha(calibration))
            result['homography_status'] = 'PASS: unchanged cell 11 on real calibration'
        except Exception:
            (out / 'homography_error.log').write_text(traceback.format_exc())
            result['homography_status'] = 'FAIL'
    gt = args.gt.resolve() if args.gt else None
    if args.download_gt:
        gt = out / GT_NAME
        url = f'https://drive.google.com/uc?export=download&id={GT_ID}'
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(2_000_000)
                info = dict(url=url, status=response.status,
                            content_type=response.headers.get('Content-Type'), bytes=len(data))
            if not data.startswith(b'frame_id,court_x,court_y,team_id,jersey_number'):
                raise ValueError('Download did not contain expected GT CSV header')
            gt.write_bytes(data)
            write_json(out / 'download.json', info)
        except Exception:
            (out / 'download_error.log').write_text(traceback.format_exc())
            gt = None
            result['gt_download_status'] = 'FAIL'
    try:
        ns = evaluator_namespace(cells)
        obj = lambda identity='A4', xy=(0, 0): dict(position=list(xy), attributes=identity)
        frame = lambda gt, pred: dict(ground_truths=gt, predictions=pred)
        cases = {
            'perfect': ([frame([obj()], [obj()])], (1., 1., 1., 1., 0., 0.)),
            'wrong_identity': ([frame([obj()], [obj('B6')])], (0., 0., 0., 0., 1., 1.)),
            'missing_prediction': ([frame([obj()], [])], (0., 0., 0., 0., 0., 1.)),
            'extra_prediction': ([frame([obj()], [obj(), obj('B6')])],
                                 (math.sqrt(.5), .5, 1., 1., 1., 0.)),
            'empty_frame': ([frame([], [])], (0., 0., 0., 0., 0., 0.)),
        }
        keys = ['TI-HOTA', 'TI-DetA', 'TI-AssA', 'TI-TP', 'TI-FP', 'TI-FN']
        checks = []
        for name, (frames, expected) in cases.items():
            metrics, alphas = ns['compute_ti_hota'](frames, 100.)
            actual = [metrics[k] for k in keys]
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(actual, expected)), (name, actual)
            checks.append(dict(name=name, status='PASS', metrics=metrics))
        assert math.isclose(ns['loc_sim']([0, 0], [100, 0], 100.), .05, abs_tol=1e-12)
        # The evaluator drops tracker IDs: arbitrary tracker_id changes are invisible.
        frames = [frame([obj()], [dict(obj(), tracker_id=k)]) for k in (1, 99)]
        metrics, _ = ns['compute_ti_hota'](frames, 100.)
        assert metrics['TI-HOTA'] == 1.
        checks.append(dict(name='temporary_tracker_id_changes_invisible', status='PASS', metrics=metrics))
        write_json(out / 'synthetic_evaluator_checks.json',
                   dict(label='Synthetic contract checks; NOT baseline scores or oracle experiments',
                        tau_cm=100., alphas=alphas.tolist(), checks=checks,
                        distance_at_tau_similarity=.05))
        result['evaluator_smoke_status'] = 'PASS'
        if gt is not None:
            import pandas as pd
            result['inputs']['gt'] = dict(path=str(gt), sha256=sha(gt))
            gt_frames = ns['load_gt_frames'](gt)
            gt_df = pd.read_csv(gt)
            write_json(out / 'gt_inventory.json', dict(rows=len(gt_df), frames=len(gt_frames),
                       first_frame=min(gt_frames), last_frame=max(gt_frames),
                       columns=list(gt_df.columns), teams=sorted(gt_df.team_id.unique().tolist()),
                       identities=sorted({o['attributes'] for values in gt_frames.values() for o in values})))
            # A data-loading/missing-predictions smoke check, never a model prediction.
            subset = sorted(gt_frames)[:3]
            empty_predictions = pd.DataFrame(columns=gt_df.columns)
            tiny_gt = out / 'smoke_gt_first3.csv'
            gt_df[gt_df.frame_id.isin(subset)].to_csv(tiny_gt, index=False)
            frames = ns['load_video'](tiny_gt, empty_predictions)
            metrics, _ = ns['compute_ti_hota'](frames, 100.)
            assert metrics['TI-FN'] == sum(len(gt_frames[f]) for f in subset)
            write_json(out / 'real_gt_loader_smoke.json',
                       dict(label='Real GT first 3 frames with deliberately empty predictions; NOT baseline evaluation',
                            frame_ids=subset, tau_cm=100., metrics=metrics))
            if args.predictions:
                pred_path = args.predictions.resolve()
                pred = pd.read_csv(pred_path)
                if list(pred.columns) != list(gt_df.columns):
                    raise ValueError('Prediction CSV must contain exactly the five Challenge columns in order')
                frames = ns['load_video'](gt, pred)
                metrics, alphas = ns['compute_ti_hota'](frames, 100.)
                diagnostics = ns['compute_location_diagnostics'](gt, pred, 100.)
                diagnostics = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                               for k, v in diagnostics.items()}
                write_json(out / 'evaluation.json', dict(metrics=metrics, alphas=alphas.tolist(),
                           tau_cm=100., diagnostics=diagnostics, evaluated_frames=len(frames),
                           prediction_rows=len(pred), representation='CSV rounded values, not notebook in-memory floats'))
                result['inputs']['predictions'] = dict(path=str(pred_path), sha256=sha(pred_path))
                result['external_prediction_evaluation'] = 'PASS; model provenance must be verified separately'
    except Exception:
        (out / 'evaluator_error.log').write_text(traceback.format_exc())
        result['evaluator_smoke_status'] = 'FAIL'
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(out / 'run.json', result)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha(p) for p in files})
    print(json.dumps(result, indent=2))
    return 1 if (result.get('evaluator_smoke_status') != 'PASS'
                 or result.get('homography_status') == 'FAIL'
                 or result.get('gt_download_status') == 'FAIL') else 0


if __name__ == '__main__':
    raise SystemExit(main())
