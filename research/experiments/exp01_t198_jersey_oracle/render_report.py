#!/usr/bin/env python3
"""Render persisted Exp1 diagnostics; does not execute an experiment."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
rows=pd.read_csv(out/'recovered_B6_rows.csv')
fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
for origin,label,color,marker in [('observed_T198','Observed T198 (175 rows)','#264653','o'),
                                 ('interpolated','Original interpolation (13 rows)','#9a6700','x')]:
    group=rows[rows.origin==origin]
    ax.scatter(group.frame_id,group.error_cm,s=22,label=label,color=color,marker=marker)
ax.axhline(100,color='#555555',linestyle='--',label='100 cm diagnostic boundary')
ax.set(xlabel='Video frame (one-based)',ylabel='Court-position error against same-identity B6 GT (cm)',
       title='Exp1 recovered B6 tail: 188 rows, median error 19.91 cm',xlim=(928,1121),ylim=(0,400))
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,loc='upper left')
fig.text(.5,-.035,'Source: predictions_exp1.csv and original tutorial GT · frames 931–1118 · 172/188 within 100 cm',ha='center',fontsize=9)
fig.savefig(out/'recovered_B6_errors.png',dpi=170,bbox_inches='tight')
plt.close(fig)
