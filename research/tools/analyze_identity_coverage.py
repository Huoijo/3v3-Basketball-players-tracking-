#!/usr/bin/env python3
"""Read-only causal audit of supplied results. No model inference or prediction edits.

Diagnostic GT correspondences are geometry evidence, not new semantic labels.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'research/analysis/results_b11_b6_20261023'
RAW_COLUMNS=['frame_id','tracker_id','x','y','width','height','conf','class','visibility','empty']
GT=ROOT/'research/runs/20261022-01_baseline-audit_tutorial_eval-smoke/IMG_0104_1_tutorial_challenge.csv'
CAL=ROOT/'research/runs/20261022-02_baseline-audit_tutorial_homography-smoke/Outdoor.ndjson'


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):d.update(block)
    return d.hexdigest()


def longest(frames):
    values=sorted(set(map(int,frames)))
    runs=[]
    for f in values:
        if runs and f==runs[-1][-1]+1:runs[-1].append(f)
        else:runs.append([f])
    return max(map(len,runs),default=0)


def intervals(frames):
    values=sorted(set(map(int,frames)));result=[]
    for f in values:
        if result and f==result[-1][1]+1:result[-1][1]=f
        else:result.append([f,f])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--extract-frames',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    results=ROOT/'results'
    results_hashes_before={p.name:sha(p) for p in results.iterdir() if p.is_file()}
    source=out/'source';source.mkdir(exist_ok=True)
    executed=source/'executed_colab.ipynb'
    if not executed.is_file():raise FileNotFoundError('Fetch the user-linked executed notebook into source/executed_colab.ipynb first')
    n=json.loads(executed.read_text());original=json.loads((ROOT/'TrackID3x3_Challenge_Tutorial.ipynb').read_text())
    snapshots={i:''.join(c.get('source',[])) for i,c in enumerate(n['cells'])}
    # Verify source equivalence against the audited local tutorial.
    cell_mapping={3:3,5:5,7:7,9:9,11:12,13:14,15:16,17:18,19:20,21:22,23:24,24:25,26:27,28:29,30:31}
    parity={str(a):dict(executed_cell=b,exact_source_equal=''.join(original['cells'][a]['source'])==snapshots[b]) for a,b in cell_mapping.items()}
    for i in [12,16,18,22,24,25,31]:(source/f'cell_{i:02d}.py').write_text(snapshots[i])
    stdout=''.join(''.join(o.get('text',[])) for o in n['cells'][18].get('outputs',[]))
    stdout=stdout[stdout.index('Jersey OCR results:'):]
    (source/'recognition_stdout.txt').write_text(stdout)
    ocr={}
    for match in re.finditer(r'^Track (\d+) -> jersey (\d+) \(([^\n]+)\)',stdout,re.M):
        scores={int(k):float(v) for k,v in re.findall(r'(\d+):([0-9.]+)',match[3])}
        ocr[int(match[1])]=dict(jersey=int(match[2]),scores=scores,accepted_scores_available=True)
    for match in re.finditer(r'^Track (\d+): no valid jersey number recognized',stdout,re.M):
        ocr[int(match[1])]=dict(jersey=None,scores={},accepted_scores_available=True)
    clusters={int(a):int(b) for a,b in re.findall(r'^Track (\d+) -> cluster (\d+)',stdout,re.M)}
    cluster_team={int(a):b for a,b in re.findall(r'^Cluster (\d+) -> Team ([AB])',stdout,re.M)}
    logged_identities={int(a):b for a,b in re.findall(r'^Track (\d+) -> ([AB]\d+)$',stdout,re.M)}
    raw=pd.read_csv(results/'mot_results.txt',header=None,names=RAW_COLUMNS)
    filtered=pd.read_csv(results/'filtered_mot.txt',header=None,names=RAW_COLUMNS[:7])
    court=pd.read_csv(results/'transformed_court.txt',header=None,names=['frame_id','tracker_id','court_x','court_y'])
    predictions=pd.read_csv(results/'predictions.csv');predictions['identity']=predictions.team_id+predictions.jersey_number.astype(str)
    gt=pd.read_csv(GT);gt['identity']=gt.team_id+gt.jersey_number.astype(str)
    for table in [raw,filtered,court]:
        assert not table.duplicated(['frame_id','tracker_id']).any()
    assert not predictions.duplicated(['frame_id','identity']).any()
    assert set(filtered.tracker_id)==set(ocr)==set(clusters)
    assert len(raw)==13890 and len(filtered)==len(court)==6489
    # Use the exact executed calibration cell with the same audited calibration input.
    ns=dict(cv2=cv2,np=np,keypoint_file=str(CAL))
    import ndjson
    ns['ndjson']=ndjson
    exec(compile(snapshots[12],'<executed-cell12-homography>','exec'),ns)
    H=ns['H']
    points=np.column_stack([raw.x+raw.width/2,raw.y+raw.height]).reshape(-1,1,2).astype(float)
    xy=cv2.perspectiveTransform(points,H).reshape(-1,2)
    raw['court_x']=xy[:,0];raw['court_y']=xy[:,1]
    raw['_raw_row']=np.arange(len(raw))
    (out/'homography_reconstructed.json').write_text(json.dumps(dict(H=H.tolist(),calibration_sha256=sha(CAL)),indent=2))
    # Replay only court gates for diagnostic decisions. Never execute original output writes.
    gate_stats=[];kept=[]
    for tid,g in raw.groupby('tracker_id',sort=True):
        inside=g[g.court_x.between(0,1505)&g.court_y.between(0,1105)]
        max_run=longest(inside.frame_id)
        total=g.frame_id.nunique()
        c1=int((g.court_y.between(-300,100)&g.court_x.between(0,1505)).sum())
        c2=int(((g.court_y>1002.5)&((g.court_x<502.5)|(g.court_x>1002.5))).sum())
        accepted=max_run>=10 and c1<=total/2 and c2<=total/2
        reasons=[]
        if max_run<10:reasons.append('inside_run_lt10')
        if c1>total/2:reasons.append('near_endline_majority')
        if c2>total/2:reasons.append('far_outside_paint_majority')
        gate_stats.append(dict(tracker_id=int(tid),court_track_gate_accepted=accepted,
                               court_gate_reason='accepted' if accepted else ';'.join(reasons),
                               longest_inside_run=max_run,c1=c1,c2=c2))
        if accepted:kept.append(tid)
    replay=raw[raw.tracker_id.isin(kept)&raw.court_x.between(-300,1805)&raw.court_y.between(-300,1405)]
    raw_keys=set(zip(replay.frame_id,replay.tracker_id));filtered_keys=set(zip(filtered.frame_id,filtered.tracker_id))
    assert raw_keys==filtered_keys,'Court filtering replay must match supplied filtered artifact exactly'
    f=filtered.merge(court,on=['frame_id','tracker_id'],how='left',validate='one_to_one',sort=False)
    projection_check=f.merge(raw[['frame_id','tracker_id','court_x','court_y']],on=['frame_id','tracker_id'],suffixes=('_file','_replay'))
    max_projection_error=float(np.abs(projection_check[['court_x_file','court_y_file']].to_numpy()-projection_check[['court_x_replay','court_y_replay']].to_numpy()).max())
    assert max_projection_error<1e-7
    f['_filtered_row']=np.arange(len(f))
    f['team_id']=f.tracker_id.map({tid:cluster_team[cl] for tid,cl in clusters.items()})
    f['jersey_number']=f.tracker_id.map({tid:v['jersey'] for tid,v in ocr.items()})
    roster={'A':{4,10,12},'B':{6,9,11}}
    f['jersey_available']=f.jersey_number.notna()
    f['roster_compatible']=[pd.notna(j) and int(j) in roster[t] for t,j in zip(f.team_id,f.jersey_number)]
    f['semantic_identity']=[t+str(int(j)) if ok else '' for t,j,ok in zip(f.team_id,f.jersey_number,f.roster_compatible)]
    semantic=f[f.roster_compatible].copy()
    # Same sort/dedup as source; exact final replay below checks observed consequences.
    selected=semantic.sort_values('conf',ascending=False).drop_duplicates(['frame_id','semantic_identity'],keep='first')
    # Equal-confidence sort ties are not a specified baseline policy. Recover
    # provenance from saved coordinates only among tied maximum-confidence rows.
    # This changes no supplied prediction and assigns no new player identity.
    tie_audit=[]
    prediction_lookup=predictions.set_index(['frame_id','identity'])
    selected_lookup=selected.set_index(['frame_id','semantic_identity'])
    replacements={}
    for (frame,identity),group in semantic.groupby(['frame_id','semantic_identity']):
        top=group[group.conf==group.conf.max()]
        if len(top)<2:continue
        point=prediction_lookup.loc[(frame,identity),['court_x','court_y']].to_numpy(dtype=float)
        matches=top[np.max(np.abs(top[['court_x','court_y']].to_numpy()-point),axis=1)<=5.01e-6]
        assert len(matches), 'Saved observed coordinate is inconsistent with all maximum-confidence tied rows'
        current=selected_lookup.loc[(frame,identity)]
        chosen=matches.iloc[0] if len(matches)==1 else current
        replacements[(frame,identity)]=int(chosen._filtered_row)
        tie_audit.append(dict(frame_id=int(frame),semantic_identity=identity,max_confidence=float(group.conf.max()),
                       local_sort_tracker_id=int(current.tracker_id),artifact_consistent_tracker_id=int(chosen.tracker_id),
                       tied_trackers=json.dumps(top.tracker_id.astype(int).tolist()),
                       matching_coordinate_candidates=len(matches),
                       local_choice_differs=int(current.tracker_id)!=int(chosen.tracker_id),
                       provenance='Saved prediction coordinates disambiguate equal-confidence observations; not independent tie replay'))
    if replacements:
        # Avoid namedtuple renaming of underscore-prefixed column names.
        chosen_rows=[replacements.get((row['frame_id'],row['semantic_identity']),int(row['_filtered_row']))
                     for _,row in selected.iterrows()]
        selected=semantic[semantic._filtered_row.isin(chosen_rows)].copy()
    pd.DataFrame(tie_audit).to_csv(out/'semantic_tie_audit.csv',index=False)
    selected_rows=set(selected._filtered_row)
    f['selected_after_dedup']=f._filtered_row.isin(selected_rows)
    f['drop_reason']=np.select([~f.jersey_available,~f.roster_compatible,~f.selected_after_dedup],
                              ['jersey_unavailable','roster_incompatible','semantic_duplicate'],default='survives_observed_semantic')
    f.to_csv(out/'filtered_observation_fates.csv',index=False)
    selected.to_csv(out/'selected_semantic_observations.csv',index=False)
    rebuilt=[];coverage=[]
    support_rows=[]
    for identity,g in selected.groupby('semantic_identity'):
        g=g.sort_values('frame_id').set_index('frame_id')
        support=set(g.index)
        interpolated=g.reindex(range(int(g.index.min()),int(g.index.max())+1))
        interpolated[['court_x','court_y','width','height']]=interpolated[['court_x','court_y','width','height']].interpolate(method='linear',limit_direction='both')
        for frame,row in interpolated.iterrows():
            rebuilt.append(dict(frame_id=frame,identity=identity,court_x=row.court_x,court_y=row.court_y))
            support_rows.append(dict(frame_id=frame,identity=identity,observed=frame in support,
                                      source_tracker_id=int(row.tracker_id) if frame in support else None))
    reconstructed=pd.DataFrame(rebuilt)
    compare=predictions.merge(reconstructed,on=['frame_id','identity'],how='outer',suffixes=('_file','_replay'),indicator=True)
    assert (compare._merge=='both').all()
    replay_error=float(np.abs(compare[['court_x_file','court_y_file']].to_numpy()-compare[['court_x_replay','court_y_replay']].to_numpy()).max())
    assert replay_error<=5.01e-6,'Interpolation replay differs from exported predictions'
    pd.DataFrame(support_rows).to_csv(out/'final_row_provenance_reconstruction.csv',index=False)
    for identity,g in predictions.groupby('identity'):
        missing=sorted(set(range(1,1119))-set(g.frame_id))
        coverage.append(dict(identity=identity,first_frame=int(g.frame_id.min()),last_frame=int(g.frame_id.max()),
                             final_rows=len(g),missing_frames=len(missing),missing_ranges=json.dumps(intervals(missing)),
                             surviving_observed_rows=int((selected.semantic_identity==identity).sum()),
                             interpolated_rows=int(sum(s['identity']==identity and not s['observed'] for s in support_rows))))
    pd.DataFrame(coverage).to_csv(out/'final_coverage.csv',index=False)
    # Global and team-assignment-conditional count accounting. No labels invented for court-rejected IDs.
    losses=[dict(scope='global',stage='raw_tracker',rows=len(raw),tracks=raw.tracker_id.nunique()),
            dict(scope='global',stage='after_court',rows=len(f),tracks=f.tracker_id.nunique()),
            dict(scope='global',stage='team_assigned',rows=int(f.team_id.notna().sum()),tracks=f[f.team_id.notna()].tracker_id.nunique()),
            dict(scope='global',stage='jersey_assigned',rows=int(f.jersey_available.sum()),tracks=f[f.jersey_available].tracker_id.nunique()),
            dict(scope='global',stage='roster_compatible',rows=len(semantic),tracks=semantic.tracker_id.nunique()),
            dict(scope='global',stage='semantic_dedup',rows=len(selected),tracks=selected.tracker_id.nunique()),
            dict(scope='global',stage='interpolated_final',rows=len(predictions),tracks=None)]
    for team in ['A','B']:
        z=f[f.team_id==team];s=semantic[semantic.team_id==team];d=selected[selected.team_id==team]
        for stage,count,tracks in [('after_court',len(z),z.tracker_id.nunique()),('team_assigned',len(z),z.tracker_id.nunique()),
                                   ('jersey_assigned',int(z.jersey_available.sum()),z[z.jersey_available].tracker_id.nunique()),
                                   ('roster_compatible',len(s),s.tracker_id.nunique()),('semantic_dedup',len(d),d.tracker_id.nunique()),
                                   ('interpolated_final',int((predictions.team_id==team).sum()),None)]:
            losses.append(dict(scope='inferred_team_'+team,stage=stage,rows=count,tracks=tracks))
    pd.DataFrame(losses).to_csv(out/'pipeline_loss_accounting.csv',index=False)
    # Geometry-only GT diagnostics: nearest and 100-cm-gated one-to-one Hungarian are distinct.
    trace=[];raw_nearest=[]
    fates={(int(row.frame_id),int(row.tracker_id)):row for row in f.itertuples(index=False)}
    for frame,gt_frame in gt.groupby('frame_id'):
        r=raw[raw.frame_id==frame].reset_index(drop=True)
        g=gt_frame.reset_index(drop=True)
        distances=np.linalg.norm(g[['court_x','court_y']].to_numpy()[:,None,:]-r[['court_x','court_y']].to_numpy()[None,:,:],axis=2)
        cost=distances.copy();cost[distances>100]=1e6
        rows,cols=linear_sum_assignment(cost)
        assigned={int(i):int(j) for i,j in zip(rows,cols) if distances[i,j]<=100}
        for j,row in r.iterrows():
            order=np.argsort(distances[:,j]);best=int(order[0]);second=int(order[1])
            raw_nearest.append(dict(frame_id=int(frame),tracker_id=int(row.tracker_id),nearest_gt_identity=g.iloc[best].identity,
                                    nearest_gt_distance_cm=float(distances[best,j]),nearest_gt_margin_cm=float(distances[second,j]-distances[best,j])))
        for i,grow in g.iterrows():
            j=int(np.argmin(distances[i]));row=r.iloc[j];tid=int(row.tracker_id);distance=float(distances[i,j])
            fate=fates.get((int(frame),tid))
            if distance>100:category='no_raw_within_100cm'
            elif fate is None:category='court_removed'
            elif not fate.jersey_available:category='jersey_unavailable'
            elif not fate.roster_compatible:category='roster_incompatible'
            elif fate.semantic_identity!=grow.identity:category='wrong_semantic_identity'
            elif not fate.selected_after_dedup:category='semantic_duplicate'
            else:category='correct_semantic_observation'
            match=assigned.get(int(i))
            trace.append(dict(frame_id=int(frame),gt_identity=grow.identity,nearest_tracker_id=tid,
                              nearest_distance_cm=distance,nearest_within100=distance<=100,
                              nearest_inferred_team=fate.team_id if fate else None,
                              nearest_semantic=fate.semantic_identity if fate else None,
                              nearest_stage_category=category,
                              hungarian_tracker_id=int(r.iloc[match].tracker_id) if match is not None else None,
                              hungarian_distance_cm=float(distances[i,match]) if match is not None else None))
    trace=pd.DataFrame(trace);trace.to_csv(out/'gt_observation_trace.csv',index=False)
    diagnostics=pd.DataFrame(raw_nearest)
    diagnostics.to_csv(out/'raw_observation_geometry_diagnostics.csv',index=False)
    budget=trace[((trace.gt_identity=='B11')&trace.frame_id.between(1,742))|((trace.gt_identity=='B6')&trace.frame_id.between(931,1118))]
    budget.groupby(['gt_identity','nearest_stage_category']).size().reset_index(name='frames').to_csv(out/'missing_window_geometry_accounting.csv',index=False)
    # All 37 tracker IDs; semantic availability is only reported where logs exist.
    lifecycle=[]
    for tid,g in raw.groupby('tracker_id',sort=True):
        g=g.sort_values('frame_id');z=f[f.tracker_id==tid];d=diagnostics[diagnostics.tracker_id==tid]
        freq=d.nearest_gt_identity.value_counts()
        info=ocr.get(int(tid));cl=clusters.get(int(tid));team=cluster_team.get(cl) if cl is not None else None
        row=dict(tracker_id=int(tid),first_frame=int(g.frame_id.min()),last_frame=int(g.frame_id.max()),
                 observed_frame_count=len(g),longest_continuous_run=longest(g.frame_id),
                 first_court_x=float(g.iloc[0].court_x),first_court_y=float(g.iloc[0].court_y),
                 last_court_x=float(g.iloc[-1].court_x),last_court_y=float(g.iloc[-1].court_y),
                 median_court_x=float(g.court_x.median()),median_court_y=float(g.court_y.median()),
                 filtered_observations=len(z),filtered_first_frame=int(z.frame_id.min()) if len(z) else None,
                 filtered_last_frame=int(z.frame_id.max()) if len(z) else None,
                 inferred_team=team,cluster_id=cl,jersey_candidate=info['jersey'] if info else None,
                 accepted_ocr_score_sums_rounded=json.dumps(info['scores']) if info else None,
                 ocr_sample_frames_scheduled=min(40,len(z)) if len(z) else 0,
                 actual_ocr_calls_recoverable=False,raw_rejected_ocr_available=False,
                 final_semantic_identity=logged_identities.get(int(tid)),
                 selected_observation_count=int(z.selected_after_dedup.sum()) if len(z) else 0,
                 contributes_observed_rows_to_final=bool(len(z) and z.selected_after_dedup.any()),
                 final_csv_has_tracker_ids=False,
                 semantic_drop_reasons=json.dumps(z.drop_reason.value_counts().to_dict()) if len(z) else 'not_reached',
                 nearest_gt_identity_diagnostic=freq.index[0],nearest_gt_diagnostic_counts=json.dumps(freq.to_dict()),
                 median_nearest_gt_distance_cm=float(d.nearest_gt_distance_cm.median()),
                 warning='Nearest GT is geometric evidence, not a physical track identity label')
        row.update(next(v for v in gate_stats if v['tracker_id']==tid));lifecycle.append(row)
    life=pd.DataFrame(lifecycle);life.to_csv(out/'tracker_lifecycle.csv',index=False)
    # Exact sample schedule, not invented OCR call-level outputs.
    samples=[]
    for tid,g in f.groupby('tracker_id'):
        g=g.sort_values('frame_id');idx=np.unique(np.linspace(0,len(g)-1,min(40,len(g)),dtype=int))
        for _,row in g.iloc[idx].iterrows():samples.append(dict(tracker_id=int(tid),frame_id=int(row.frame_id),
                             planned_by_source=True,actual_ocr_result_available=False))
    pd.DataFrame(samples).to_csv(out/'scheduled_ocr_samples.csv',index=False)
    windows=[]
    for name,lo,hi in [('B11_transition',700,780),('B6_transition',890,970)]:
        for tid,g in f[f.frame_id.between(lo,hi)].groupby('tracker_id'):
            for start,end in intervals(g.frame_id):
                z=g[g.frame_id.between(start,end)].sort_values('frame_id');first=z.iloc[0];last=z.iloc[-1]
                windows.append(dict(window=name,start_frame=start,end_frame=end,tracker_id=int(tid),
                          inferred_team=first.team_id,jersey_candidate=first.jersey_number,
                          semantic_identity=first.semantic_identity,accepted_ocr_scores=json.dumps(ocr[int(tid)]['scores']),
                          first_court_x=first.court_x,first_court_y=first.court_y,last_court_x=last.court_x,last_court_y=last.court_y,
                          selected_rows=int(z.selected_after_dedup.sum()),drop_reasons=json.dumps(z.drop_reason.value_counts().to_dict())))
    pd.DataFrame(windows).to_csv(out/'transition_windows.csv',index=False)
    # Neighbors are spatial candidates in the adjacent available semantic frame, never a merge decision.
    neighbors=[]
    for tid in [7,59,172,198]:
        track=f[f.tracker_id==tid].sort_values('frame_id')
        for direction,endpoint in [('before',track.iloc[0]),('after',track.iloc[-1])]:
            available=selected[selected.frame_id<endpoint.frame_id] if direction=='before' else selected[selected.frame_id>endpoint.frame_id]
            if available.empty:continue
            frame=available.frame_id.max() if direction=='before' else available.frame_id.min()
            options=available[available.frame_id==frame].copy()
            options['distance_cm']=np.hypot(options.court_x-endpoint.court_x,options.court_y-endpoint.court_y)
            for rank,(_,v) in enumerate(options.sort_values('distance_cm').head(3).iterrows(),1):
                neighbors.append(dict(candidate_tracker_id=tid,direction=direction,candidate_endpoint_frame=int(endpoint.frame_id),
                        neighbor_frame=int(frame),neighbor_tracker_id=int(v.tracker_id),neighbor_semantic=v.semantic_identity,
                        distance_cm=v.distance_cm,rank=rank,interpretation='Endpoint proximity only; not proof of physical continuity'))
    pd.DataFrame(neighbors).to_csv(out/'candidate_semantic_neighbors.csv',index=False)
    # Endpoint displacement for explicitly investigated track pairs; no new association is applied.
    links=[]
    for left,right in [(7,59),(59,74),(74,5),(5,157),(157,172),(172,179),(8,40),(40,65),(65,198)]:
        a=f[f.tracker_id==left].sort_values('frame_id').iloc[-1]
        b=f[f.tracker_id==right].sort_values('frame_id').iloc[0]
        links.append(dict(from_tracker_id=left,to_tracker_id=right,from_endpoint_frame=int(a.frame_id),to_start_frame=int(b.frame_id),
                          delta_frames=int(b.frame_id-a.frame_id),endpoint_distance_cm=float(np.hypot(a.court_x-b.court_x,a.court_y-b.court_y)),
                          note='Endpoints may overlap or span mixed tracks; descriptive geometry only'))
    pd.DataFrame(links).to_csv(out/'candidate_endpoint_links.csv',index=False)
    # Compact missing-window evidence runs; group by diagnostic category AND track, not each coordinate.
    stable=[]
    for identity,z in budget.groupby('gt_identity'):
        z=z.sort_values('frame_id');blocks=[]
        for row in z.itertuples(index=False):
            key=(row.nearest_tracker_id,row.nearest_stage_category)
            if blocks and blocks[-1]['key']==key and row.frame_id==blocks[-1]['end_frame']+1:
                blocks[-1]['end_frame']=row.frame_id;blocks[-1]['distances'].append(row.nearest_distance_cm)
            else:blocks.append(dict(key=key,start_frame=row.frame_id,end_frame=row.frame_id,distances=[row.nearest_distance_cm]))
        for b in blocks:stable.append(dict(gt_identity=identity,start_frame=b['start_frame'],end_frame=b['end_frame'],
                 nearest_tracker_id=b['key'][0],diagnostic_category=b['key'][1],median_distance_cm=float(np.median(b['distances']))))
    pd.DataFrame(stable).to_csv(out/'missing_window_stable_ranges.csv',index=False)
    video_path=results/'IMG_0104_1_with_minimap.mp4'
    cap=cv2.VideoCapture(str(video_path));video=dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)));cap.release()
    if args.extract_frames:extract_frames(raw,video_path,out)
    plot_coverage(predictions,life,selected,out)
    inputs=[*sorted(results.iterdir()),executed,ROOT/'TrackID3x3_Challenge_Tutorial.ipynb',GT,CAL]
    assert results_hashes_before=={p.name:sha(p) for p in results.iterdir() if p.is_file()}, 'Supplied results changed during analysis'
    manifest=dict(status='PARTIAL',scope='Causal diagnosis; baseline/result files read-only',
                  result_rows=len(predictions),expected_rows=1118*6,missing_rows=1118*6-len(predictions),
                  supplied_results_unchanged_during_analysis=True,
                  tracker_ids=len(life),filtered_tracker_ids=f.tracker_id.nunique(),
                  projection_max_absolute_error_cm=max_projection_error,final_replay_max_absolute_error_cm=replay_error,
                  equal_confidence_tie_groups=len(tie_audit),local_tie_choices_differ_from_export=sum(v['local_choice_differs'] for v in tie_audit),
                  tie_provenance='Disambiguated from unchanged saved predictions; not an independent numerical replay of tie ordering',
                  final_replay_row_keys_equal=True,source_cell_parity=parity,video=video,
                  geometry_diagnostic_gate_cm=100,geometry_matching_warning='GT geometric proximity, especially in crossings, is not identity ground truth for a tracker box',
                  remaining_limits=['No pre-tracker detector boxes or rejected detections','No raw OCR strings/rejections/confidences per sample',
                                    'Manual image evidence is sampled, not exhaustive frame relabeling'],
                  inputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in inputs if p.is_file()},
                  script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(__file__),command=sys.argv)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nFINAL COVERAGE\n'+pd.DataFrame(coverage).to_string(index=False))
    print('\nLOSSES\n'+pd.DataFrame(losses).to_string(index=False))
    print('\nMISSING-WINDOW GEOMETRY DIAGNOSTICS\n'+budget.groupby(['gt_identity','nearest_stage_category']).size().to_string())
    print('\nARTIFACT-CONSISTENT RECONSTRUCTION: max coordinate error vs rounded CSV',replay_error)
    print('Equal-confidence groups:',len(tie_audit),'; local sort differs at',sum(v['local_choice_differs'] for v in tie_audit),
          'rows. Those observation winners were inferred from saved coordinates, not independently reproduced.')


def extract_frames(raw,video_path,out):
    visual=out/'visual';visual.mkdir(exist_ok=True)
    # Diagnostic annotated copies only; supplied video is never written.
    requests={1:[7,5,8],280:[7,5],300:[7,5,47],301:[59,5],399:[59,5],408:[5,74],414:[5,74],
              417:[5,74],430:[5,74],700:[5],730:[5],741:[5],743:[157,84],780:[157],805:[157,84],
              820:[172,84],850:[172,84],853:[172,179],890:[65],930:[65],934:[],939:[198],950:[198],970:[198],1118:[198]}
    cap=cv2.VideoCapture(str(video_path));index=[]
    for frame,ids in requests.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode supplied video frame {frame}')
        annotated=im.copy();rows=raw[raw.frame_id==frame]
        for row in rows.itertuples(index=False):
            if row.tracker_id in [1,6,9,10,11,12,42,132,166,207,210]:continue
            x,y,w,h=map(int,[row.x,row.y,row.width,row.height])
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,255,255),3)
            label=f'T{row.tracker_id}';xy=(x,max(25,y-12))
            cv2.putText(annotated,label,xy,cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),7)
            cv2.putText(annotated,label,xy,cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3)
        cv2.putText(annotated,f'ANALYSIS ONLY: frame {frame}, raw MOT boxes/IDs',(30,annotated.shape[0]-35),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
        path=visual/f'frame_{frame:04d}.jpg';cv2.imwrite(str(path),cv2.resize(annotated,(1600,900)))
        for tid in ids:
            one=rows[rows.tracker_id==tid]
            if len(one)!=1:continue
            row=one.iloc[0];x,y,w,h=map(int,[row.x,row.y,row.width,row.height]);pad=35
            crop=im[max(0,y-pad):min(im.shape[0],y+h+pad),max(0,x-pad):min(im.shape[1],x+w+pad)]
            if crop.size:cv2.imwrite(str(visual/f'crop_f{frame:04d}_t{tid}.jpg'),crop)
        index.append(dict(frame=frame,seconds=(frame-1)/cap.get(cv2.CAP_PROP_FPS),path=str(path.relative_to(out)),
                          candidate_ids=ids,decoder_next_frame=cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cap.release();(visual/'index.json').write_text(json.dumps(index,indent=2))


def plot_coverage(predictions,life,selected,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,8),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    ax=axes[0]
    identities=['A4','A10','A12','B9','B6','B11']
    for i,identity in enumerate(identities):
        g=predictions[predictions.identity==identity]
        ax.broken_barh([(int(g.frame_id.min()),int(g.frame_id.max()-g.frame_id.min()+1))],(i-.3,.6),facecolors='#264653')
    ax.set(yticks=range(6),yticklabels=identities,xlim=(1,1118),xlabel='Video frame (one-based)',ylabel='Final semantic identity',title='Final prediction support: 930 missing player-frame rows')
    ax.grid(axis='x',alpha=.2)
    ax=axes[1];ids=[7,59,5,74,157,172,179,8,40,47,65,198]
    colors={'jersey_unavailable':'#9a6700','roster_incompatible':'#767676','semantic':'#264653'}
    for i,tid in enumerate(ids):
        row=life[life.tracker_id==tid].iloc[0]
        reason='jersey_unavailable' if pd.isna(row.jersey_candidate) else ('roster_incompatible' if tid==47 else 'semantic')
        ax.broken_barh([(row.first_frame,row.last_frame-row.first_frame+1)],(i-.3,.6),facecolors=colors[reason])
        ax.text(row.last_frame+5,i,str(row.final_semantic_identity) if pd.notna(row.final_semantic_identity) else 'no semantic ID',va='center',fontsize=8)
    ax.set(yticks=range(len(ids)),yticklabels=[f'T{v}' for v in ids],xlim=(1,1220),xlabel='Video frame (one-based)',ylabel='Temporary tracker ID',title='Relevant raw tracker spans and logged identity fate (span bars include internal gaps)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v,label=k.replace('_',' ')) for k,v in colors.items()],loc='lower center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False)
    for axis in axes:
        for boundary in [743,930]:axis.axvline(boundary,color='#555555',linestyle=':',alpha=.7)
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Source: supplied results + executed Colab stdout; frames 1–1118',fontsize=11)
    fig.savefig(out/'coverage_and_tracker_spans.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
