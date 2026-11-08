#!/usr/bin/env python3
"""Draw the repository overview using the released 100k evaluation records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[1]
C = {'ink':'#18263B','muted':'#58677D','border':'#DCE4EE','blue':'#2563EB','green':'#087F70','purple':'#8B5CF6','orange':'#D97706','gray':'#536175'}

def record(h, mode, p=.1, schedule='constant', delta=0.):
    name=f'h{h}_{mode}_p{p:.2f}_{schedule}_d{delta:.2f}_s42'
    rows=[json.loads(l) for l in (ROOT/'results/80m'/name/'eval.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['step']==100000 and rows[0]['tokens_seen']==1638400000
    return rows[0]

def box(ax,x,y,w,h,text,color='#F1F5FB',edge=None,size=12,weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.012',facecolor=color,edgecolor=edge or C['border'],lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=C['ink'],weight=weight,linespacing=1.45)

def arrow(ax,start,end,color=None,style='-'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.6,color=color or C['muted'],linestyle=style))

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':C['muted'],'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.edgecolor':C['border'],'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(16,9.2),facecolor='#F7F9FC')
    fig.text(.045,.94,'Loop Transformer',fontsize=30,weight='bold',color=C['ink'])
    fig.text(.045,.895,'One shared layer stack. Dropout during training. Measurements at every loop.',fontsize=15,color=C['muted'])
    a=fig.add_axes([.035,.12,.48,.72]);a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
    a.text(.025,.96,'A   Reuse the same six layers',fontsize=17,weight='bold',color=C['ink'])
    a.text(.025,.905,'Weights stay shared; training masks are sampled again.',fontsize=11.5,color=C['muted'])
    box(a,.03,.72,.15,.105,'Tokens',size=12)
    box(a,.24,.72,.18,.105,'Loop 1',color='#E9F0FF',edge='#A9C4FF',weight='bold')
    box(a,.49,.72,.18,.105,'Loop 2',color='#E9F0FF',edge='#A9C4FF',weight='bold')
    box(a,.76,.72,.18,.105,'Loop H',color='#E9F0FF',edge='#A9C4FF',weight='bold')
    arrow(a,(.185,.772),(.23,.772));arrow(a,(.425,.772),(.48,.772));arrow(a,(.675,.772),(.75,.772));a.text(.705,.815,'...',ha='center',color=C['muted'])
    a.plot([.33,.33,.85,.85],[.71,.67,.67,.71],color=C['blue'],lw=1.2)
    a.text(.59,.625,'Same weights in every loop',ha='center',color=C['blue'],fontsize=12)
    box(a,.03,.22,.91,.34,'',color='white')
    a.text(.065,.505,'Inside each shared Transformer layer',fontsize=13,weight='bold',color=C['ink'])
    a.text(.065,.45,'Attention branch',fontsize=11,color=C['muted'])
    box(a,.065,.355,.20,.065,'Norm + attention',size=10.5)
    box(a,.31,.355,.145,.065,'Output norm',size=10.3)
    box(a,.50,.355,.16,.065,'Dropout',color='#FFF1DC',edge='#E8B768',size=11,weight='bold')
    box(a,.705,.355,.195,.065,'+ residual',size=11)
    arrow(a,(.27,.387),(.303,.387));arrow(a,(.46,.387),(.493,.387));arrow(a,(.665,.387),(.698,.387))
    a.text(.065,.285,'Then the FFN branch uses the same order.',fontsize=11.5,color=C['muted'])
    a.text(.065,.245,'Enable dropout on attention, FFN, or both.',fontsize=11.5,color=C['muted'])
    box(a,.03,.035,.44,.115,'TRAINING\nNew dropout masks each call',color='#FFF6E8',edge='#E8C98D',size=11)
    box(a,.50,.035,.44,.115,'EVALUATION\nDropout off; full dense model',color='#EAF6F2',edge='#A6D6C9',size=11)
    ax=fig.add_axes([.60,.565,.355,.24],facecolor='white')
    fig.text(.60,.83,'B   Prediction quality',fontsize=17,weight='bold',color=C['ink'])
    for mode,label,color,marker in [('none','Baseline',C['gray'],'o'),('ffn','FFN 10%',C['blue'],'s'),('attention','Attention 10%',C['purple'],'^'),('both','Both 10%',C['orange'],'D')]:
        ys=[record(h,mode,0 if mode=='none' else .1)['perplexity'] for h in [1,2,4,8]]
        ax.plot([1,2,4,8],ys,label=label,color=color,marker=marker,lw=2,ms=5)
    ax.set_xticks([1,2,4,8]);ax.set_xlim(.7,8.3);ax.set_ylim(35.9,39.7);ax.set_ylabel('Validation PPL  (lower is better)',fontsize=10.5);ax.set_xlabel('Loops',fontsize=10.5,labelpad=3);ax.grid(axis='y',color='#E9EDF3');ax.legend(ncol=2,frameon=False,fontsize=9.5,loc='upper right');ax.tick_params(labelsize=10)
    ax=fig.add_axes([.60,.17,.355,.24],facecolor='white')
    fig.text(.60,.46,'C   FFN outputs, layer by layer',fontsize=17,weight='bold',color=C['ink'])
    for mode,p,sch,delta,label,color in [('none',0,'constant',0,'Baseline',C['gray']),('ffn',.1,'constant',0,'FFN: constant 10%',C['blue']),('ffn',.1,'decreasing',.1,'FFN: 20% to 0%',C['green'])]:
        acts=record(8,mode,p,sch,delta)['diagnostics']['activations'];acts=sorted([x for x in acts if x['branch']=='ffn'],key=lambda x:(x['loop'],x['layer']))
        ys=[next(t['near_zero_ratio']*100 for t in row['threshold_sensitivity'] if t['threshold']==.01) for row in acts];ax.plot(range(1,49),ys,label=label,color=color,lw=1.7,marker='o',ms=2.5)
    for k in range(8):
        if k%2==0:ax.axvspan(k*6+.5,k*6+6.5,color='#EEF3FA',zorder=0)
    ax.set_xlim(.5,48.5);ax.set_xticks(np.arange(3.5,49,6),range(1,9));ax.set_ylim(0,11.2);ax.set_xlabel('Loop (six layer measurements per loop)',fontsize=10.5);ax.set_ylabel('Near-zero FFN outputs (%)',fontsize=10.5);ax.grid(axis='y',color='#E9EDF3');ax.legend(frameon=False,fontsize=9.5,loc='upper right');ax.tick_params(labelsize=10)
    fig.text(.045,.053,'RECORDED RESULTS',fontsize=9.5,color=C['blue'],weight='bold')
    fig.text(.18,.053,'80M parameters  |  100,000 steps  |  1.6384B training tokens  |  seed 42',fontsize=10.5,color=C['muted'])
    fig.text(.045,.023,'Near-zero threshold: 0.01. Diagnostics use 512 fixed tokens with dropout off. Curves show measurements, not skipped computation.',fontsize=9.5,color=C['muted'])
    out=ROOT/'assets';out.mkdir(exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'overview.{ext}',dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved assets/overview.png, .svg, and .pdf')

if __name__=='__main__':main()
