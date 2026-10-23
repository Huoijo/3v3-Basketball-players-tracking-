#!/usr/bin/env python3
"""Local non-GPU validation; never claims a Kaggle inference result."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import textwrap
import traceback

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/TrackID3x3_Challenge_Tutorial_kaggle.ipynb'
ORIGINAL = ROOT / 'TrackID3x3_Challenge_Tutorial.ipynb'
EXPECTED = '33da120728e1ef5884ef001a8c30da3b0c183c165626b91c1075f1be7e9c1c00'
AUDIT_INPUTS = ROOT / 'research/runs/20261022-01_baseline-audit_tutorial_eval-smoke'
CALIBRATION = ROOT / 'research/runs/20261022-02_baseline-audit_tutorial_homography-smoke/Outdoor.ndjson'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(nodes):
    tree = copy.deepcopy(ast.Module(body=nodes, type_ignores=[]))
    # Stage context indentation changes docstring whitespace only.
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                value.value = inspect.cleandoc(value.value)
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='READY_FOR_KAGGLE', actual_kaggle_execution=False,
                  baseline_score=None, checks=[], original_sha256=sha(ORIGINAL),
                  kaggle_sha256=sha(NOTEBOOK), local_execution='CPU compatibility checks only')
    def check(name, function):
        try:
            detail = function()
            report['checks'].append(dict(name=name, status='PASS', detail=detail))
        except Exception:
            report['checks'].append(dict(name=name, status='FAIL', traceback=traceback.format_exc()))
            report['status']='BLOCKED'
            raise
    try:
        original = json.loads(ORIGINAL.read_text())
        nb = json.loads(NOTEBOOK.read_text())
        code = {c['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'}
        nodes = {key: ast.parse(value).body for key,value in code.items()}
        def body(key):
            return nodes[key][0].body  # ordinary baseline_stage context, no scope changes
        def orig(index):
            return ''.join(original['cells'][index]['source'])
        def schema():
            import nbformat
            nbformat.validate(nbformat.from_dict(nb))
            assert sha(ORIGINAL)==EXPECTED
            assert all(c.get('outputs',[])==[] and c.get('execution_count') is None for c in nb['cells'])
            assert not any('/content' in value or 'google.colab' in value for value in code.values())
            assert len(nodes)==16
            for value in code.values():compile(value, '<kaggle-cell>', 'exec')
            return dict(code_cells=len(nodes), nbformat='4.5', saved_outputs_cleared=True, original_unchanged=True)
        check('notebook_schema_and_python_syntax',schema)
        def preservation():
            # Full AST comparisons for every numerical/semantic baseline block.
            core = [(11,'homography'),(15,'court-filtering')]
            for index,ident in core:
                expected=ast.parse(orig(index)).body
                assert dump(expected)==dump(body(ident)[:len(expected)]),ident
            assert dump(ast.parse(orig(21)).body)==dump(nodes['evaluator-definitions'])
            eval_expected=ast.parse(orig(23)+'\n'+orig(24)).body
            assert dump(eval_expected)==dump(body('evaluation')[:len(eval_expected)])
            mot_source=orig(9).split('# Load MOT data for visualization')[0]
            mot_expected=ast.parse(mot_source).body
            assert dump(mot_expected)==dump(body('mot')[:len(mot_expected)])
            full17=[]
            for ident in ['ocr','teams','identity','prediction']:
                for node in body(ident):
                    if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and 'reader.device' in ast.unparse(node.test):
                        continue  # CUDA enforcement only
                    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value).startswith("print('EasyOCR device:'"):
                        continue
                    full17.append(node)
            # Reverse only documented Reader filesystem/download kwargs.
            for node in full17:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reader' for t in node.targets):
                    node.value.keywords=[k for k in node.value.keywords if k.arg not in {'model_storage_directory','download_enabled'}]
            assert dump(ast.parse(orig(17)).body)==dump(full17),'OCR/team/roster/merge/interpolation changed'
            # Reverse explicit storage/video-only changes to tracking call.
            track=body('tracking')
            for node in track:
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
                    call=node.value
                    if isinstance(call.func,ast.Name) and call.func.id=='YOLO':
                        call.args=[ast.Constant(value='yolo11x.pt')]
                    if isinstance(call.func,ast.Attribute) and call.func.attr=='track':
                        call.keywords=[k for k in call.keywords if k.arg not in {'project','name','exist_ok'}]
                        for kw in call.keywords:
                            if kw.arg=='tracker':kw.value=ast.Constant(value='bytetrack.yaml')
                            if kw.arg=='save':kw.value=ast.Constant(value=True)
            expected=ast.parse(orig(7)).body
            assert dump(expected)==dump(track[:len(expected)])
            return dict(unchanged=['YOLO11x inference parameters','ByteTrack configuration semantics',
                   'YOLO-to-MOT parser and conversion','homography','court filter','OCR thresholds',
                   'team assignment','roster','semantic merge','interpolation','TI-HOTA','location diagnostics'],
                   allowed_changes=['weight/YAML path','video save flag','Reader cache path/download policy','CUDA guard'])
        check('algorithm_ast_preservation',preservation)
        with tempfile.TemporaryDirectory(prefix='trackid3x3-kaggle-check-') as temp:
            temp=Path(temp)
            # Execute actual bootstrap in a temporary cwd; generated paths are isolated.
            import os
            previous=Path.cwd()
            try:
                os.chdir(temp)
                ns={}
                exec(compile(code['setup'],'kaggle:setup','exec'),ns)
            finally:
                os.chdir(previous)
            def source_capture():
                ns['capture_notebook_source'](list(code.values()))
                captured=ns['RUN_DIR']/'kaggle_source_snapshot.ipynb'
                assert captured.read_bytes()==NOTEBOOK.read_bytes()
                shutil.copyfile(captured,out/'kaggle_source_snapshot.ipynb')
                shutil.copyfile(ns['RUN_DIR']/'source_hashes.json',out/'source_hashes.json')
                return 'Reconstructed notebook is byte-identical to the shipped source artifact'
            check('executed_source_hash_round_trip',source_capture)
            def resolver():
                import cv2
                import numpy as np
                fixture=temp/'input with spaces'
                fixture.mkdir()
                shutil.copy2(AUDIT_INPUTS/'IMG_0104_1_tutorial_challenge.csv',fixture)
                shutil.copy2(CALIBRATION,fixture)
                # Clearly synthetic compatibility fixture, never used as baseline data.
                cv2.imwrite(str(fixture/'Outdoor.png'),np.zeros((48,64,3),np.uint8))
                writer=cv2.VideoWriter(str(fixture/'IMG_0104_1.MOV'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(64,48))
                assert writer.isOpened()
                for _ in range(3):writer.write(np.zeros((48,64,3),np.uint8))
                writer.release()
                ns.update(DATA_INPUT_DIR=fixture,ALLOW_DOWNLOADS=False)
                paths=ns['resolve_data']()
                assert len(paths)==4 and all(Path(p).is_file() for p in paths)
                ns['DATA_INPUT_DIR']=temp/'missing'
                try:ns['resolve_data']()
                except FileNotFoundError:pass
                else:raise AssertionError('Incomplete explicit input must fail')
                duplicate=fixture/'duplicate';duplicate.mkdir()
                shutil.copy2(CALIBRATION,duplicate/'Outdoor.ndjson')
                ns['DATA_INPUT_DIR']=fixture
                try:ns['resolve_data']()
                except RuntimeError:pass
                else:raise AssertionError('Duplicate input must fail')
                (duplicate/'Outdoor.ndjson').unlink()
                ns['DATA_INPUT_DIR']=fixture
                return dict(label='Synthetic video/PNG + real GT/calibration; filesystem test only',
                            offline_paths_with_spaces='PASS',missing_input='rejected',duplicate_input='rejected')
            check('offline_input_resolution',resolver)
            def homography():
                import cv2,ndjson,numpy as np
                ns.update(cv2=cv2,ndjson=ndjson,np=np,keypoint_file=str(CALIBRATION))
                exec(compile(code['homography'],'kaggle:homography','exec'),ns)
                projected=cv2.perspectiveTransform(ns['image_points'].reshape(-1,1,2),ns['H']).reshape(-1,2)
                residual=np.linalg.norm(projected-ns['court_points'],axis=1)
                assert residual.max()<.01
                shutil.copyfile(ns['RUN_DIR']/'homography.json',out/'homography.json')
                return dict(actual_tutorial_calibration=True,max_fit_residual_cm=float(residual.max()),
                            localization_accuracy_claimed=False)
            check('generated_homography_stage_real_calibration',homography)
            def mot_conversion():
                import cv2,glob,re,os
                from collections import defaultdict
                tracking=temp/'tracking';labels=tracking/'labels';labels.mkdir(parents=True)
                (labels/'IMG_0104_1_1.txt').write_text('0 0.5 0.5 0.2 0.4 0.9 2\n')
                ns.update(cv2=cv2,glob=glob,re=re,os=os,defaultdict=defaultdict,
                          out_dir=str(tracking),video_path=str(temp/'input with spaces'/'IMG_0104_1.MOV'))
                exec(compile(code['mot'],'kaggle:mot','exec'),ns)
                row=(tracking/'mot_results.txt').read_text().strip().split(',')
                assert row==['1','2','25.60','14.40','12.80','19.20','0.90','-1','-1','-1'],row
                return dict(label='Synthetic label fixture; NOT a real GPU label-format validation',row=row)
            check('generated_mot_conversion_fixture',mot_conversion)
            def probe_guard():
                # No Ultralytics inference: verify the gate rejects a known ambiguous line.
                raw=(ROOT/'notebooks/kaggle_label_probe.py').read_text()
                tree=ast.parse(raw)
                funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_result_label'],type_ignores=[])
                import math,numpy as np
                class Value:
                    def __init__(self,value):self.value=value
                    def item(self):return self.value
                class Box:
                    id=Value(2);conf=Value(.9);xywhn=np.array([.5,.5,.2,.4])
                env=dict(math=math,parse_yolo_tracking_line=ns['parse_yolo_tracking_line'])
                exec(compile(funcs,'label_probe_check','exec'),env)
                env['check_result_label']('0 .5 .5 .2 .4 .9 2'.split(),Box())
                Box.conf=Value(1.)
                try:env['check_result_label']('0 .5 .5 .2 .4 1 2'.split(),Box())
                except RuntimeError:pass
                else:raise AssertionError('Integer-looking confidence ambiguity must stop the compatibility gate')
                return 'Original parser retained; synthetic ambiguity is detected, not silently repaired'
            check('probe_guard_without_parser_change',probe_guard)
            def stage_failure():
                try:
                    with ns['baseline_stage']('fixture','Intentional failure'):
                        raise RuntimeError('intentional validation fixture')
                except RuntimeError:pass
                manifest=json.loads((ns['RUN_DIR']/'run_manifest.json').read_text())
                assert manifest['status']=='FAILED' and (ns['RUN_DIR']/'failure.log').is_file()
                return 'Runtime exceptions propagate and persist failure evidence'
            check('failure_persistence',stage_failure)
            def evaluator():
                import math,numpy as np,pandas as pd
                from scipy.optimize import linear_sum_assignment
                from collections import defaultdict
                env=dict(math=math,np=np,pd=pd,linear_sum_assignment=linear_sum_assignment,defaultdict=defaultdict)
                exec(compile(code['evaluator-definitions'],'kaggle:evaluator','exec'),env)
                obj=dict(position=[0.,0.],attributes='A4')
                metrics,_=env['compute_ti_hota']([dict(ground_truths=[obj],predictions=[obj])],100.)
                assert metrics['TI-HOTA']==1.
                metrics,_=env['compute_ti_hota']([dict(ground_truths=[obj],predictions=[])],100.)
                assert metrics['TI-FN']==1. and metrics['TI-HOTA']==0.
                return 'Generated evaluator perfect/missing synthetic contract checks pass; no model score'
            check('generated_evaluator_contract',evaluator)
    except Exception:
        report['status']='BLOCKED'
    finally:
        report['pending']=['Actual Kaggle GPU environment validation','Real YOLO/ByteTrack label probe',
                           'Full detection→OCR→semantic identity execution','Fresh predictions and TI-HOTA']
        report['tool_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/tools/build_kaggle_notebook.py',
                  ROOT/'research/tools/validate_kaggle_notebook.py',ROOT/'notebooks/kaggle_bootstrap.py',ROOT/'notebooks/kaggle_label_probe.py']}
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    return 0 if report['status']=='READY_FOR_KAGGLE' else 1


if __name__=='__main__':
    raise SystemExit(main())
