"""Post-inference gate and unchanged downstream/evaluator; no OCR tuning."""
import contextlib
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ORACLE=ROOT/'research/experiments/exp01_t198_jersey_oracle'
AUDIT=ROOT/'research/analysis/results_b11_b6_20261023'
GT=ROOT/'research/runs/20261022-01_baseline-audit_tutorial_eval-smoke/IMG_0104_1_tutorial_challenge.csv'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def finalize(decision,frozen):
    before=json.loads((OUT/'source_hashes.json').read_text())
    checks={p:sha(ROOT/p)==h for p,h in before['protected'].items()}
    assert all(checks.values()),'Protected files changed'
    config_ok=sha(OUT/'configuration.json')==before['configuration']
    prereg=json.loads((OUT/'preregistration.json').read_text())
    script_ok=sha(OUT/'run.py')==prereg['inference_script_sha256']
    assert config_ok and script_ok,'Preregistered inference changed'
    save('verification.json',dict(protected_files_unchanged=checks,configuration_unchanged=config_ok,inference_script_unchanged=script_ok,all_track_results_persisted_before_diagnostic=True))
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    manifest.update(status=decision['status'],completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),visual_source_verified=True,all_filtered_tracks=len(frozen['tracks']),automatic_target_jersey=decision['automatic_result']['jersey'],downstream_run=decision['downstream_run'])
    if 'downstream_failure' in decision:manifest['downstream_failure']=decision['downstream_failure']
    save('experiment_manifest.json',manifest)
    print(json.dumps(decision,indent=2))


def main():
    frozen=json.loads((OUT/'automatic_results.json').read_text())
    result=frozen['tracks']['198']
    e=pd.read_csv(OUT/'ocr_evidence.csv',keep_default_na=False)
    target=e[e.tracker_id==198].copy()
    target.to_csv(OUT/'target_ocr_evidence.csv',index=False)
    meaningful_six=target[(target.sampling=='quality')&(pd.to_numeric(target.candidate,errors='coerce')==6)&(target.confidence>=.1)]
    status='SUCCESS' if result['jersey']==6 else 'PARTIAL' if len(meaningful_six) and result['jersey'] is None else 'FAILURE'
    decision=dict(status=status,automatic_result=result,automatic_results_sha256=sha(OUT/'automatic_results.json'),accepted_six_readings=len(meaningful_six),six_frames=sorted(meaningful_six.frame_id.unique().tolist()),downstream_run=False)
    # Representative crops selected algorithmically, only inspected after frozen inference.
    sel=pd.read_csv(OUT/'selected_frames.csv')
    chosen=sel[(sel.tracker_id==198)&(sel.sampling=='quality')].sort_values(['quality','frame_id'],ascending=[False,True]).head(6)
    rep=OUT/'representative_crops';rep.mkdir(exist_ok=True)
    frames=set(chosen.frame_id.astype(int))|set(meaningful_six.frame_id.astype(int))
    for frame in sorted(frames):
        shutil.copy2(OUT/'crops'/f't198_f{frame:04d}.png',rep/f't198_f{frame:04d}.png')
    save('target_result.json',decision)
    if status=='SUCCESS':
        # All automatic mappings, including abstentions; no baseline or oracle fallback.
        mapping={int(k):v['jersey'] for k,v in frozen['tracks'].items()}
        state=json.loads((ORACLE/'cached_mapping_state.json').read_text())
        cols=['frame_id','id','x','y','width','height','conf','class','visibility','empty']
        raw=pd.read_csv(ROOT/'results/mot_results.txt',names=cols)
        filtered=pd.read_csv(ROOT/'results/filtered_mot.txt',names=cols[:7])
        court=pd.read_csv(ROOT/'results/transformed_court.txt',names=['frame_id','id','court_x','court_y'])
        keys=pd.MultiIndex.from_frame(filtered[['frame_id','id']])
        df=raw.loc[pd.MultiIndex.from_frame(raw[['frame_id','id']]).isin(keys)].copy()
        assert np.array_equal(df[cols[:7]].to_numpy(),filtered.to_numpy())
        df['bot_cx']=df.x+df.width/2;df['bot_cy']=df.y+df.height
        df[['court_x','court_y']]=court.set_index(['frame_id','id']).loc[list(zip(df.frame_id,df.id)),['court_x','court_y']].to_numpy()
        directory=OUT/'downstream';directory.mkdir(exist_ok=True)
        for name in ['downstream_original.py','evaluator_original.py']:
            shutil.copy2(ORACLE/'source'/name,OUT/'source'/name)
        ns=dict(pd=pd,np=np,cv2=cv2,os=os,df=df,track_to_jersey=mapping,
                track_to_cluster={int(k):v for k,v in state['track_to_cluster'].items()},
                cluster_to_team={int(k):v for k,v in state['cluster_to_team'].items()},
                team_rosters={k:set(v) for k,v in state['team_rosters'].items()},identity_order=state['identity_order'],
                H=np.array(json.loads((AUDIT/'homography_reconstructed.json').read_text())['H']),
                video_path='IMG_0104_1.MOV',out_dir=str(directory))
        try:
            with (directory/'run.log').open('w') as log,contextlib.redirect_stdout(log):
                exec(compile((OUT/'source/downstream_original.py').read_text(),'downstream_original.py','exec'),ns)
        except RuntimeError as exc:
            decision['downstream_failure']=str(exc)
            save('target_result.json',decision)
            finalize(decision,frozen)
            return
        ns['identity_df'].to_csv(directory/'selected_semantic_observations.csv',index=True)
        shutil.copy2(directory/'predictions.csv',OUT/'predictions_exp1b.csv')
        ens=dict(np=np,pd=pd,math=math,defaultdict=defaultdict,linear_sum_assignment=linear_sum_assignment)
        exec(compile((OUT/'source/evaluator_original.py').read_text(),'evaluator_original.py','exec'),ens)
        metrics={}
        coverage=[]
        for label,path in [('baseline',ROOT/'results/predictions.csv'),('exp01_oracle',ORACLE/'predictions_exp1.csv'),('exp01b',OUT/'predictions_exp1b.csv')]:
            p=pd.read_csv(path)
            values,alphas=ens['compute_ti_hota'](ens['load_video'](str(GT),p),100.)
            metrics[label]=values
            save(label+'_metrics.json',dict(metrics=values,alphas=alphas.tolist(),tau_cm=100,prediction_rows=len(p)))
            b6=p[(p.team_id=='B')&(p.jersey_number==6)]
            coverage.append(dict(arm=label,total_rows=len(p),B6_rows=len(b6),B6_first=int(b6.frame_id.min()),B6_last=int(b6.frame_id.max())))
        names=['TI-HOTA','TI-DetA','TI-AssA','TI-TP','TI-FP','TI-FN']
        pd.DataFrame([dict(metric=k,**{label:v[k] for label,v in metrics.items()}) for k in names]).to_csv(OUT/'metric_comparison.csv',index=False)
        save('coverage_comparison.json',coverage)
        keys=['frame_id','team_id','jersey_number']
        diff=[]
        new=pd.read_csv(OUT/'predictions_exp1b.csv')
        for label,path in [('baseline',ROOT/'results/predictions.csv'),('exp01_oracle',ORACLE/'predictions_exp1.csv')]:
            old=pd.read_csv(path)
            merged=old.merge(new,on=keys,how='outer',suffixes=('_old','_new'),indicator=True)
            merged['coordinate_changed']=(merged._merge=='both')&((merged.court_x_old!=merged.court_x_new)|(merged.court_y_old!=merged.court_y_new))
            changed=merged[(merged._merge!='both')|merged.coordinate_changed]
            changed.to_csv(OUT/f'csv_diff_vs_{label}.csv',index=False)
            diff.append(dict(comparator=label,added=int((merged._merge=='right_only').sum()),removed=int((merged._merge=='left_only').sum()),common_coordinates_changed=int(merged.coordinate_changed.sum())))
        save('csv_diff_summary.json',diff)
        decision['downstream_run']=True
        save('target_result.json',decision)
    finalize(decision,frozen)


if __name__=='__main__':main()
