#!/usr/bin/env python3
"""Generic, preregistered OCR experiment. No GT or target identity in inference.

Stages: prepare, baseline, enhanced, aggregate. Diagnostic target inspection is
performed by a separate report script only after aggregate persists all tracks.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
VIDEO = OUT / 'source/IMG_0104_1.MOV'
CONFIG = json.loads((OUT / 'configuration.json').read_text())
STATE = json.loads((ROOT / 'research/experiments/exp01_t198_jersey_oracle/cached_mapping_state.json').read_text())
COLS = ['frame_id', 'id', 'x', 'y', 'width', 'height', 'conf']
EVIDENCE = ['tracker_id', 'frame_id', 'sampling', 'variant', 'ocr_text', 'candidate', 'confidence', 'in_roster_union', 'in_team_roster', 'accepted_baseline', 'accepted_new', 'quality', 'ocr_box', 'call_seconds']


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tracks():
    return pd.read_csv(ROOT / 'results/filtered_mot.txt', names=COLS)


def bbox(frame, row):
    x, y, w, h = [float(row[k]) for k in ['x', 'y', 'width', 'height']]
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(frame.shape[1], x + w)), int(min(frame.shape[0], y + h))
    return frame[y1:y2, x1:x2], max(0, x2-x1)*max(0, y2-y1)/(w*h)


def quality(crop, retained):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    area = gray.size
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    score = float(np.prod(np.clip([np.sqrt(area)/250, sharp/150, contrast/50], .05, 1))**(1/3)*retained)
    return dict(quality=score, area=area, sharpness=sharp, contrast=contrast, brightness=float(gray.mean()), retained_fraction=retained)


def decode(callback):
    cap = cv2.VideoCapture(str(VIDEO))
    metadata = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=cap.get(cv2.CAP_PROP_FPS), declared_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        callback(n, frame)
    cap.release()
    metadata['decoded_frames'] = n
    return metadata


def prepare():
    if (OUT / 'selected_frames.csv').exists():
        raise RuntimeError('Selection already exists; preserve this attempt')
    df = tracks()
    rows = []
    by_frame = {int(f): g.to_dict('records') for f, g in df.groupby('frame_id')}
    def measure(f, frame):
        for r in by_frame.get(f, []):
            crop, retained = bbox(frame, r)
            if not crop.size:
                raise RuntimeError(f'Unusable crop at frame {f}, track {r["id"]}')
            rows.append(dict(tracker_id=int(r['id']), frame_id=f, **quality(crop, retained)))
    meta = decode(measure)
    assert meta['decoded_frames'] == int(df.frame_id.max()) == 1118, meta
    assert len(rows) == len(df)
    q = pd.DataFrame(rows)
    q.to_csv(OUT / 'candidate_quality.csv', index=False)
    selected = []
    for t, g in q.groupby('tracker_id', sort=True):
        g = g.sort_values('frame_id')
        indices = np.unique(np.linspace(0, len(g)-1, min(CONFIG['baseline_samples'], len(g)), dtype=int))
        for r in g.iloc[indices].to_dict('records'):
            selected.append(dict(**r, sampling='baseline_uniform'))
        chosen = []
        for r in g.sort_values(['quality', 'frame_id'], ascending=[False, True]).to_dict('records'):
            if all(abs(r['frame_id']-f) >= CONFIG['minimum_frame_gap'] for f in chosen):
                chosen.append(r['frame_id'])
                selected.append(dict(**r, sampling='quality'))
            if len(chosen) >= CONFIG['quality_samples']:
                break
    sel = pd.DataFrame(selected).sort_values(['tracker_id', 'sampling', 'frame_id'])
    sel.to_csv(OUT / 'selected_frames.csv', index=False)
    keys = set(zip(sel.frame_id, sel.tracker_id))
    (OUT / 'crops').mkdir(exist_ok=True)
    def extract(f, frame):
        for r in by_frame.get(f, []):
            t = int(r['id'])
            if (f, t) in keys:
                crop, _ = bbox(frame, r)
                assert cv2.imwrite(str(OUT / 'crops' / f't{t}_f{f:04d}.png'), crop)
    decode(extract)
    save('visual_source.json', dict(path=str(VIDEO.relative_to(ROOT)), sha256=sha(VIDEO), size=VIDEO.stat().st_size, drive_file_id='1jK-3_OLWkBw8vk4rK27TB4NP2eeC3N-x', **meta, observations_reconstructed=len(rows), selected_unique_crops=len(keys), inference_source='Original MOV; no annotated-video pixels'))
    print('VISUAL_SOURCE_VERIFIED', meta, 'selected', len(sel), flush=True)


def variant_image(crop, variant):
    if variant == 'full_2x':
        return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = crop.shape[:2]
    a, b, c, d = CONFIG['torso_fractions']
    torso = crop[int(h*b):int(h*d), int(w*a):int(w*c)]
    if variant == 'torso_2x':
        return cv2.resize(torso, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(gray, 1.3, cv2.GaussianBlur(gray, (0, 0), 1), -.3, 0)


def ocr(stage):
    import easyocr
    import torch
    torch.set_num_threads(4)
    reader = easyocr.Reader(['en'], gpu=False, quantize=False, model_storage_directory=str(OUT / 'models'), verbose=True)
    save('environment.json', dict(device='CPU, 4 Torch threads; original Colab used CUDA/T4', packages={p:importlib.metadata.version(p) for p in ['easyocr','torch','torchvision','numpy','pandas','opencv-python-headless']}, model_hashes={p.name:sha(p) for p in (OUT/'models').glob('*.pth')}))
    sel = pd.read_csv(OUT/'selected_frames.csv')
    sampling = 'baseline_uniform' if stage == 'baseline' else 'quality'
    variants = ['full_2x'] if stage == 'baseline' else CONFIG['variants']
    dest = OUT / f'ocr_{stage}.csv'
    completed = set()
    if dest.exists():
        old = pd.read_csv(dest, keep_default_na=False)
        completed = set(zip(old.tracker_id, old.frame_id, old.variant))
    union = set().union(*[set(v) for v in STATE['team_rosters'].values()])
    calls = 0
    with dest.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE)
        if not completed:
            writer.writeheader()
        for r in sel[sel.sampling==sampling].to_dict('records'):
            t, frame = int(r['tracker_id']), int(r['frame_id'])
            team = STATE['cluster_to_team'][str(STATE['track_to_cluster'][str(t)])]
            allowed = STATE['team_rosters'][team]
            crop = cv2.imread(str(OUT/'crops'/f't{t}_f{frame:04d}.png'))
            for variant in variants:
                if (t, frame, variant) in completed:
                    continue
                started = time.perf_counter()
                results = reader.readtext(variant_image(crop, variant), allowlist='0123456789', detail=1, paragraph=False)
                elapsed = time.perf_counter()-started
                for box, text, conf in results or [(None, '', 0.)]:
                    digits = re.sub(r'\D', '', text)
                    candidate = int(digits) if digits else None
                    writer.writerow(dict(tracker_id=t, frame_id=frame, sampling=sampling, variant=variant, ocr_text=text, candidate=candidate, confidence=float(conf), in_roster_union=candidate in union, in_team_roster=candidate in allowed, accepted_baseline=bool(conf>=.1 and candidate in union), accepted_new=bool(conf>=.1 and candidate in allowed), quality=r['quality'], ocr_box=json.dumps(np.asarray(box).tolist()) if box is not None else '', call_seconds=elapsed))
                f.flush()
                calls += 1
                if calls % 20 == 0:
                    print(stage, 'calls_completed_this_session', calls, 'track', t, flush=True)
    print(stage, 'COMPLETE', flush=True)


def aggregate():
    if (OUT/'automatic_results.json').exists():
        raise RuntimeError('Automatic results already frozen; no tuning or overwrite')
    baseline = pd.read_csv(OUT/'ocr_baseline.csv', keep_default_na=False)
    new = pd.read_csv(OUT/'ocr_enhanced.csv', keep_default_na=False)
    pd.concat([baseline,new]).to_csv(OUT/'ocr_evidence.csv',index=False)
    selected = pd.read_csv(OUT/'selected_frames.csv')
    # Each selected input has an explicit OCR result or empty-call sentinel.
    for evidence, sampling, variants in [(baseline,'baseline_uniform',['full_2x']), (new,'quality',CONFIG['variants'])]:
        expected = {(int(r.tracker_id),int(r.frame_id),v) for r in selected[selected.sampling==sampling].itertuples() for v in variants}
        assert set(zip(evidence.tracker_id,evidence.frame_id,evidence.variant)) == expected
    outputs, scores, comparison, ablations = {}, [], [], []
    for t in sorted(tracks().id.unique()):
        t = int(t)
        b = baseline[(baseline.tracker_id==t)&(baseline.accepted_baseline==True)].copy()
        b['candidate'] = pd.to_numeric(b.candidate)
        bs = b.groupby('candidate').confidence.sum().to_dict()
        baseline_replayed = int(max(bs,key=bs.get)) if bs else None
        for variant in ['combined', *CONFIG['variants']]:
            g = new[(new.tracker_id==t)&(new.accepted_new==True)].copy()
            if variant != 'combined':
                g = g[g.variant==variant]
            g['candidate'] = pd.to_numeric(g.candidate)
            g['weight'] = g.confidence*(.5+.5*g.quality)
            votes = g.groupby(['frame_id','candidate']).weight.max().reset_index()
            ss = votes.groupby('candidate').weight.sum().to_dict()
            ordered = sorted(ss, key=lambda n:(-ss[n],n))
            best = int(ordered[0]) if ordered else None
            runner = int(ordered[1]) if len(ordered)>1 else None
            best_score = float(ss.get(best,0))
            runner_score = float(ss.get(runner,0))
            total = sum(ss.values())
            share = best_score/total if total else 0
            margin = (best_score-runner_score)/total if total else 0
            support = int((votes.candidate==best).sum()) if best is not None else 0
            accept = support>=2 and share>=.6 and margin>=.2
            result = dict(tracker_id=t, candidate_scores={str(int(k)):float(v) for k,v in ss.items()}, best_jersey=best, runner_up=runner, best_score=best_score, runner_score=runner_score, share=share, normalized_margin=margin, support_frames=support, accepted=accept, jersey=best if accept else None)
            ablations.append(dict(variant=variant, **result))
            if variant == 'combined':
                outputs[str(t)] = result
                scores.append(result)
                frozen = STATE['track_to_jersey'][str(t)]
                predicted = result['jersey']
                change = 'still_None' if frozen is None and predicted is None else 'None_to_recovered' if frozen is None else 'value_to_abstain' if predicted is None else 'agreement' if predicted==frozen else 'changed_value'
                comparison.append(dict(tracker_id=t, baseline_jersey=frozen, replay_baseline_jersey=baseline_replayed, baseline_replay_scores=json.dumps(bs), new_jersey=predicted, change=change, review_required=predicted!=frozen))
    pd.DataFrame(scores).to_csv(OUT/'track_jersey_scores.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'baseline_vs_exp1b_jerseys.csv',index=False)
    pd.DataFrame(ablations).to_csv(OUT/'variant_ablations.csv',index=False)
    save('automatic_results.json',dict(persisted_utc=pd.Timestamp.now(tz='UTC').isoformat(), configuration_sha256=sha(OUT/'configuration.json'), script_sha256=sha(__file__), tracks=outputs))
    print('ALL_TRACK_AUTOMATIC_RESULTS_PERSISTED',len(outputs),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','baseline','enhanced','aggregate'])
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='aggregate': aggregate()
    else: ocr(args.stage)
