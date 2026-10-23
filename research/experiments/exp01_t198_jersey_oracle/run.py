#!/usr/bin/env python3
"""One cached-state intervention; execute original downstream source verbatim.

No model inference, clustering, oracle other than T198 jersey, or tie repair.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
AUDIT=ROOT/'research/analysis/results_b11_b6_20261023'
GT=ROOT/'research/runs/20261022-01_baseline-audit_tutorial_eval-smoke/IMG_0104_1_tutorial_challenge.csv'
KEYS=['frame_id','team_id','jersey_number']
FIELDS=['frame_id','court_x','court_y','team_id','jersey_number']
METRICS=['TI-HOTA','TI-DetA','TI-AssA','TI-TP','TI-FP','TI-FN']


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):d.update(block)
    return d.hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')


def coverage(p):
    rows=[]
    for (team,jersey),g in p.groupby(['team_id','jersey_number']):
        rows.append(dict(identity=team+str(int(jersey)),first_frame=int(g.frame_id.min()),
                         last_frame=int(g.frame_id.max()),rows=len(g)))
    return rows


def compare_csv(a,b,prefix):
    joined=a.merge(b,on=KEYS,how='outer',suffixes=('_baseline','_exp1'),indicator=True)
    joined['coordinate_changed']=(joined._merge=='both')&((joined.court_x_baseline!=joined.court_x_exp1)|(joined.court_y_baseline!=joined.court_y_exp1))
    joined['change']=np.select([joined._merge=='left_only',joined._merge=='right_only',joined.coordinate_changed],
                              ['removed','added','coordinates_changed'],default='unchanged')
    joined['coordinate_displacement_cm']=np.hypot(joined.court_x_exp1-joined.court_x_baseline,joined.court_y_exp1-joined.court_y_baseline)
    changed=joined[joined.change!='unchanged'].copy()
    changed.to_csv(OUT/(prefix+'.csv'),index=False)
    summary=[]
    for (team,jersey),g in joined.groupby(['team_id','jersey_number']):
        summary.append(dict(identity=team+str(int(jersey)),unchanged=int((g.change=='unchanged').sum()),
                         added=int((g.change=='added').sum()),removed=int((g.change=='removed').sum()),
                         coordinates_changed=int(g.coordinate_changed.sum()),
                         max_common_coordinate_displacement_cm=float(g.coordinate_displacement_cm.max()) if g.coordinate_displacement_cm.notna().any() else None))
    save(prefix+'_summary.json',summary)
    return joined,summary


def main():
    manifest=dict(status='RUNNING',experiment='exp01_t198_jersey_oracle',started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                  only_intervention='track_to_jersey[198]: None -> 6',source_evaluator_unchanged=True,
                  models_or_clustering_rerun=False,tie_policy='Original global confidence sort/dedup; no repair or frozen-winner override')
    if (OUT/'experiment_manifest.json').exists():raise RuntimeError('Experiment already attempted; preserve it rather than overwrite')
    save('experiment_manifest.json',manifest)
    start=time.perf_counter()
    try:
        prereg=json.loads((OUT/'preregistration.json').read_text())
        for path,digest in prereg['inputs_sha256'].items():assert sha(ROOT/path)==digest,path
        assert prereg['status']=='PREREGISTERED_BEFORE_EXPERIMENT'
        frozen_paths=[p for p in (ROOT/'results').iterdir() if p.is_file()]+[ROOT/'TrackID3x3_Challenge_Tutorial.ipynb',GT,
             AUDIT/'source/executed_colab.ipynb',AUDIT/'source/recognition_stdout.txt',AUDIT/'homography_reconstructed.json']
        frozen_hashes={str(p.relative_to(ROOT)):sha(p) for p in frozen_paths}
        save('source_hashes.json',dict(inputs=frozen_hashes,script_sha256=sha(__file__),preregistration_sha256=sha(OUT/'preregistration.json')))
        versions={}
        for package in ['numpy','pandas','scipy','opencv-python-headless','ndjson']:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]=None
        save('environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions,
                                    device='CPU cached downstream replay; no GPU inference',seeds='No new seeds; no stochastic component rerun'))
        notebook=json.loads((AUDIT/'source/executed_colab.ipynb').read_text())
        code=''.join(notebook['cells'][18]['source'])
        original=json.loads((ROOT/'TrackID3x3_Challenge_Tutorial.ipynb').read_text())
        assert code==''.join(original['cells'][17]['source'])
        downstream=code[code.index('identity_df = df.copy()'):]
        evaluator=''.join(notebook['cells'][22]['source'])
        assert evaluator==''.join(original['cells'][21]['source'])
        diagnostic_cell=''.join(notebook['cells'][25]['source'])
        diagnostic_tree=ast.parse(diagnostic_cell)
        diagnostic='\n\n'.join(ast.get_source_segment(diagnostic_cell,n) for n in diagnostic_tree.body if isinstance(n,ast.FunctionDef))+'\n'
        (OUT/'source').mkdir()
        for name,text in [('downstream_original.py',downstream),('evaluator_original.py',evaluator),('location_diagnostic_original.py',diagnostic)]:
            (OUT/'source'/name).write_text(text)
        # Immutable original inputs, raw row order/index and cached retained support.
        cols=['frame_id','id','x','y','width','height','conf','class','visibility','empty']
        raw=pd.read_csv(ROOT/'results/mot_results.txt',names=cols)
        filtered=pd.read_csv(ROOT/'results/filtered_mot.txt',names=cols[:7])
        court=pd.read_csv(ROOT/'results/transformed_court.txt',names=['frame_id','id','court_x','court_y'])
        raw_keys=pd.MultiIndex.from_frame(raw[['frame_id','id']]);filter_keys=pd.MultiIndex.from_frame(filtered[['frame_id','id']])
        assert not filter_keys.has_duplicates
        df=raw.loc[raw_keys.isin(filter_keys)].copy()
        assert list(zip(df.frame_id,df.id))==list(zip(filtered.frame_id,filtered.id))
        assert np.array_equal(df[cols[:7]].to_numpy(),filtered.to_numpy())
        df['bot_cx']=df.x+df.width/2;df['bot_cy']=df.y+df.height
        court_lookup=court.set_index(['frame_id','id'])
        df[['court_x','court_y']]=court_lookup.loc[list(zip(df.frame_id,df.id)),['court_x','court_y']].to_numpy()
        stdout=(AUDIT/'source/recognition_stdout.txt').read_text()
        jerseys={int(a):int(b) for a,b in re.findall(r'^Track (\d+) -> jersey (\d+) ',stdout,re.M)}
        jerseys.update({int(t):None for t in re.findall(r'^Track (\d+): no valid jersey number recognized',stdout,re.M)})
        clusters={int(a):int(b) for a,b in re.findall(r'^Track (\d+) -> cluster (\d+)',stdout,re.M)}
        cluster_teams={int(a):b for a,b in re.findall(r'^Cluster (\d+) -> Team ([AB])',stdout,re.M)}
        logged_identities={int(a):b for a,b in re.findall(r'^Track (\d+) -> ([AB]\d+)$',stdout,re.M)}
        assert set(df.id)==set(jerseys)==set(clusters)
        parameters={}
        for node in ast.parse(code).body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in {'team_rosters','identity_order'}:parameters[target.id]=ast.literal_eval(node.value)
        H=np.array(json.loads((AUDIT/'homography_reconstructed.json').read_text())['H'])
        t198=df[df.id==198]
        conditions=dict(span=[int(t198.frame_id.min()),int(t198.frame_id.max())],observations=len(t198),
                        cached_court_retained=True,team=cluster_teams[clusters[198]],jersey=jerseys[198],
                        logged_semantic_identity=logged_identities.get(198))
        assert conditions==dict(span=[939,1118],observations=175,cached_court_retained=True,team='B',jersey=None,logged_semantic_identity=None),conditions
        save('t198_preconditions.json',conditions)
        frozen_state=dict(track_to_jersey=jerseys,track_to_cluster=clusters,cluster_to_team=cluster_teams,
                          team_rosters={k:sorted(v) for k,v in parameters['team_rosters'].items()},identity_order=parameters['identity_order'])
        save('cached_mapping_state.json',frozen_state)
        # The downstream source includes its original CSV write; out_dir is isolated per arm.
        def run_arm(name,mapping):
            directory=OUT/name;directory.mkdir()
            ns=dict(pd=pd,np=np,cv2=cv2,os=os,df=df.copy(deep=True),track_to_jersey=copy.deepcopy(mapping),
                    track_to_cluster=copy.deepcopy(clusters),cluster_to_team=copy.deepcopy(cluster_teams),
                    H=H.copy(),video_path='IMG_0104_1.MOV',out_dir=str(directory),**copy.deepcopy(parameters))
            with (directory/'downstream.log').open('w') as log,contextlib.redirect_stdout(log):
                exec(compile(downstream,'downstream_original.py','exec'),ns)
            ns['identity_df'].to_csv(directory/'selected_semantic_observations.csv',index=True)
            return ns
        print('[1] Frozen downstream replay',flush=True)
        baseline=run_arm('baseline_replay',jerseys)
        frozen=pd.read_csv(ROOT/'results/predictions.csv')
        replay=pd.read_csv(OUT/'baseline_replay/predictions.csv')
        shutil.copy2(OUT/'baseline_replay/predictions.csv',OUT/'predictions_baseline_replay.csv')
        assert len(replay)==5778
        cov={r['identity']:r for r in coverage(replay)}
        assert cov['B6']==dict(identity='B6',first_frame=1,last_frame=930,rows=930)
        assert cov['B11']==dict(identity='B11',first_frame=743,last_frame=1118,rows=376)
        assert set(map(tuple,frozen[KEYS].to_numpy()))==set(map(tuple,replay[KEYS].to_numpy()))
        merged=frozen.merge(replay,on=KEYS,suffixes=('_frozen','_replay'))
        errors=np.abs(merged[['court_x_frozen','court_y_frozen']].to_numpy()-merged[['court_x_replay','court_y_replay']].to_numpy())
        for jersey in [6,11]:
            mask=(merged.team_id=='B')&(merged.jersey_number==jersey)
            assert errors[mask].max()<=1e-5, 'Relevant B6/B11 replay is not reliable'
        replay_compare,replay_summary=compare_csv(frozen,replay,'replay_vs_frozen_diff')
        save('baseline_replay_verification.json',dict(status='PASS_FOR_INTERVENTION',rows=len(replay),coverage=coverage(replay),
                row_keys_equal=True,max_exported_coordinate_difference_cm=float(errors.max()),
                changed_common_rows=int(replay_compare.coordinate_changed.sum()),
                B6_coordinates_exact_at_csv_precision=bool((errors[(merged.team_id=='B')&(merged.jersey_number==6)]==0).all()),
                B11_coordinates_exact_at_csv_precision=bool((errors[(merged.team_id=='B')&(merged.jersey_number==11)]==0).all()),
                tie_policy='No artifact-constrained winner recovery was applied; original source executed as-is'))
        print('[2] Baseline B6/B11 replay verified; apply one mapping intervention',flush=True)
        intervention=copy.deepcopy(jerseys);intervention[198]=6
        assert {k for k in jerseys if jerseys[k]!=intervention[k]}=={198}
        save('intervention.json',dict(track=198,before=jerseys[198],after=intervention[198],team_before='B',team_after='B',other_mapping_keys_unchanged=True))
        exp=run_arm('exp1',intervention)
        shutil.copy2(OUT/'exp1/predictions.csv',OUT/'predictions_exp1.csv')
        predictions=pd.read_csv(OUT/'predictions_exp1.csv')
        selected_exp=exp['identity_df'];selected_base=baseline['identity_df']
        exp_cov={r['identity']:r for r in coverage(predictions)}
        tail=predictions[(predictions.team_id=='B')&(predictions.jersey_number==6)&predictions.frame_id.between(931,1118)].copy()
        observed_b6=selected_exp[(selected_exp.team_id=='B')&(selected_exp.jersey_number==6)]
        observed_frames=set(observed_b6.frame_id)
        tail['origin']=np.where(tail.frame_id.isin(observed_frames),'observed_T198','interpolated')
        t198_selected=selected_exp[selected_exp.id==198]
        hypotheses=dict(H1=bool(len(t198_selected) and (t198_selected.identity=='B6').all()),
                        H2=exp_cov.get('B6')==dict(identity='B6',first_frame=1,last_frame=1118,rows=1118),
                        H3=len(t198_selected)==175,H4=int((tail.origin=='interpolated').sum())==13,
                        H5=len(predictions)==5966 and len(predictions)-len(frozen)==188)
        save('hypothesis_results.json',hypotheses)
        save('coverage_before_after.json',dict(frozen=coverage(frozen),replay=coverage(replay),exp1=coverage(predictions),
               recovered_B6_tail_rows=len(tail),observed=int((tail.origin=='observed_T198').sum()),
               interpolated=int((tail.origin=='interpolated').sum()),interpolated_tail_frames=tail[tail.origin=='interpolated'].frame_id.tolist(),
               selected_T198_rows=len(t198_selected),new_B6_duplicate_key_count=int(observed_b6.duplicated(['frame_id','identity']).sum())))
        # Key/coordinate invariants; preserve and explain indirect sort-tie interactions.
        diff_frozen,summ_frozen=compare_csv(frozen,predictions,'csv_diff_vs_frozen')
        diff_replay,summ_replay=compare_csv(replay,predictions,'csv_diff_vs_replay')
        winners=selected_base[['frame_id','identity','id','conf','court_x','court_y']].merge(
            selected_exp[['frame_id','identity','id','conf','court_x','court_y']],on=['frame_id','identity'],suffixes=('_replay','_exp1'),how='inner')
        changed_winners=winners[winners.id_replay!=winners.id_exp1].copy()
        changed_winners['equal_confidence']=changed_winners.conf_replay==changed_winners.conf_exp1
        changed_winners.to_csv(OUT/'dedup_winner_changes.csv',index=False)
        assert changed_winners.equal_confidence.all(), 'Unexpected non-tie association change'
        frozen_positions=frozen.set_index(KEYS)
        interactions=[]
        for row in diff_replay[diff_replay.coordinate_changed & ~((diff_replay.team_id=='B')&(diff_replay.jersey_number==6))].itertuples(index=False):
            identity=row.team_id+str(row.jersey_number)
            matches=changed_winners[(changed_winners.frame_id==row.frame_id)&(changed_winners.identity==identity)]
            interactions.append(dict(frame_id=row.frame_id,identity=identity,
                           direct_tied_dedup_winner_change=bool(len(matches)),mechanism='Unchanged global sort receives 175 additional semantic rows; equal-confidence order changes'))
        pd.DataFrame(interactions).to_csv(OUT/'non_B6_interactions.csv',index=False)
        save('invariants.json',dict(vs_frozen=summ_frozen,vs_local_replay=summ_replay,
               all_non_B6_final_values_unchanged_vs_frozen=all(r['added']==r['removed']==r['coordinates_changed']==0 for r in summ_frozen if r['identity']!='B6'),
               all_non_B6_final_values_unchanged_vs_replay=all(r['added']==r['removed']==r['coordinates_changed']==0 for r in summ_replay if r['identity']!='B6'),
               changed_dedup_winners=len(changed_winners),all_changed_winners_equal_confidence=True,
               non_B6_changed_rows_directly_linked_to_ties=all(r['direct_tied_dedup_winner_change'] for r in interactions)))
        # Known semantic correspondence, not nearest-player relabeling.
        gt=pd.read_csv(GT);gt_b6=gt[(gt.team_id=='B')&(gt.jersey_number==6)]
        tail=tail.merge(gt_b6[['frame_id','court_x','court_y']],on='frame_id',suffixes=('_exp1','_gt'),validate='one_to_one')
        tail['error_cm']=np.hypot(tail.court_x_exp1-tail.court_x_gt,tail.court_y_exp1-tail.court_y_gt)
        tail['within_100cm']=tail.error_cm<=100
        tail.to_csv(OUT/'recovered_B6_rows.csv',index=False)
        tail.nlargest(15,'error_cm').to_csv(OUT/'largest_recovered_B6_errors.csv',index=False)
        quality=dict(rows=len(tail),GT_correspondences=len(tail),observed=int((tail.origin=='observed_T198').sum()),
                     interpolated=int((tail.origin=='interpolated').sum()),median_error_cm=float(tail.error_cm.median()),
                     max_error_cm=float(tail.error_cm.max()),within_100cm=int(tail.within_100cm.sum()),
                     over_100cm=int((tail.error_cm>100).sum()),over_200cm=int((tail.error_cm>200).sum()),
                     over_300cm=int((tail.error_cm>300).sum()),semantic_key_collisions=int(predictions.duplicated(KEYS).sum()),
                     by_origin={str(origin):dict(rows=len(g),median_error_cm=float(g.error_cm.median()),within100=int(g.within_100cm.sum())) for origin,g in tail.groupby('origin')})
        save('recovered_B6_quality.json',quality)
        evaluator_ns=dict(np=np,pd=pd,math=math,defaultdict=defaultdict,linear_sum_assignment=linear_sum_assignment)
        exec(compile(evaluator,'evaluator_original.py','exec'),evaluator_ns)
        exec(compile(diagnostic,'location_diagnostic_original.py','exec'),evaluator_ns)
        all_metrics={}
        for name,table in [('baseline',frozen),('baseline_replay',replay),('exp1',predictions)]:
            print('[3] Original evaluator:',name,'rows',len(table),flush=True)
            metric_start=time.perf_counter()
            frames=evaluator_ns['load_video'](str(GT),table)
            metrics,alphas=evaluator_ns['compute_ti_hota'](frames,100.)
            diagnostics=evaluator_ns['compute_location_diagnostics'](str(GT),table,100.)
            all_metrics[name]=metrics
            save(name+'_metrics.json',dict(metrics=metrics,alphas=alphas.tolist(),tau_cm=100.,prediction_rows=len(table),
                 frame_count=len(frames),elapsed_seconds=time.perf_counter()-metric_start,location_diagnostics=diagnostics,
                 precision='Five-decimal prediction CSV values, common precision for all three arms'))
            print(name,{k:metrics[k] for k in METRICS},flush=True)
        deltas={k:all_metrics['exp1'][k]-all_metrics['baseline'][k] for k in METRICS}
        replay_deltas={k:all_metrics['exp1'][k]-all_metrics['baseline_replay'][k] for k in METRICS}
        save('metric_delta.json',dict(exp1_minus_frozen_baseline=deltas,exp1_minus_local_replay=replay_deltas,
                  interpretation='Includes documented indirect global-sort tie interactions; not evidence that OCR can achieve oracle labels'))
        comparison=[dict(metric=k,baseline=all_metrics['baseline'][k],exp1=all_metrics['exp1'][k],delta=deltas[k],
                     local_replay=all_metrics['baseline_replay'][k],delta_vs_replay=replay_deltas[k]) for k in METRICS]
        pd.DataFrame(comparison).to_csv(OUT/'metric_comparison.csv',index=False)
        # Ensure inputs and cached maps stayed frozen, including original results/video.
        for path,digest in frozen_hashes.items():assert sha(ROOT/path)==digest,path
        assert jerseys[198] is None and cluster_teams[clusters[198]]=='B'
        status='CONFIRMED' if all(hypotheses.values()) else ('PARTIALLY_CONFIRMED' if hypotheses['H1'] else 'FALSIFIED')
        manifest.update(status=status,completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),hypotheses=hypotheses,
                        baseline_replay_valid_for_B6=True,all_frozen_input_hashes_unchanged=True,
                        other_mapping_keys_unchanged=True,non_B6_coordinate_interactions=interactions,
                        recovered_B6_quality=quality,metric_delta=deltas,elapsed_seconds=time.perf_counter()-start,
                        source_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'source').iterdir()},
                        caveat='Known confidence-tie order is unspecified. No sort/dedup modification was made to force other-row invariants.')
        save('experiment_manifest.json',manifest)
        print('STATUS:',status,flush=True)
    except BaseException:
        manifest.update(status='BLOCKED',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        save('experiment_manifest.json',manifest)
        (OUT/'failure.log').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
