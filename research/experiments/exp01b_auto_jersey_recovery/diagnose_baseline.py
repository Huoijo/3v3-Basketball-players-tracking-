"""Reporting only: inspect target after original recognition replay, before enhancement."""
import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent
e=pd.read_csv(OUT/'ocr_baseline.csv',keep_default_na=False)
g=e[e.tracker_id==198].copy()
assert g.frame_id.nunique()==40, 'Baseline target replay must finish before diagnosis'
accepted=g[g.accepted_baseline==True].copy()
accepted['candidate']=pd.to_numeric(accepted.candidate)
scores=accepted.groupby('candidate').confidence.sum().to_dict()
result=dict(team='B', scheduled_frames=40, calls=g.frame_id.nunique(),
            original_logged_scores={},original_logged_jersey=None,
            replay_scores=scores,replay_jersey=int(max(scores,key=scores.get)) if scores else None,
            raw_detections=int((g.ocr_text!='').sum()),empty_calls=int((g.ocr_text=='').sum()),
            weak_sixes=g[(pd.to_numeric(g.candidate,errors='coerce')==6)&(g.confidence<.1)].to_dict('records'),
            limitation='Original raw OCR evidence was not saved. This is a fresh CPU replay of original sampling/crop/readtext/union roster/confidence/vote, not a recovery of historical rejected reads.')
g.to_csv(OUT/'baseline_target_raw.csv',index=False)
(OUT/'baseline_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
