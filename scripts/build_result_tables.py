#!/usr/bin/env python3
"""Build public tables from the released scientific records."""
from pathlib import Path
import json
import statistics
ROOT=Path(__file__).resolve().parents[1]

def branch_mean(diag,branch):
    return statistics.mean(next(x['near_zero_ratio'] for x in row['threshold_sensitivity'] if x['threshold']==.01)
                           for row in diag['activations'] if row['branch']==branch)

def main():
    rows=[]
    for d in sorted((ROOT/'results/80m').iterdir()):
        cfg=json.loads((d/'config.json').read_text())['arguments']
        ev=[json.loads(x) for x in (d/'eval.jsonl').read_text().splitlines()]
        assert len(ev)==1 and ev[0]['step']==100000 and ev[0]['tokens_seen']==1638400000
        assert ev[0]['chunks']==128 and ev[0]['tokens']==65536
        rows.append({'name':d.name,**cfg,'evaluation':ev[0]})
    assert len(rows)==52
    labels={'none':'Baseline','ffn':'FFN','attention':'Attention','both':'Both'}
    text=['# Recorded results','',
      'The small-model tables use the same 100,000-step checkpoint, 1.6384B training tokens, seed 42, and fixed 128 validation chunks for every row. The repository includes the underlying evaluation record for all 52 settings. A historically longer run contributes its saved 100k record, not its later endpoint.','',
      'Near-zero percentages are averages across all layer uses in the named branch, with threshold 0.01. Measurements use the same 512 fixed tokens with dropout off. See [measurement definitions](METHOD.md).','',
      '## Fixed-rate dropout: 80M model','',
      'Bold loss and PPL values are the best within a loop count. Small differences come from one seed and have no cross-seed uncertainty estimate.','',
      '| Loops | Placement | Dropout p | Loss | PPL | FFN near-zero (%) | Attention near-zero (%) |',
      '| --- | --- | --- | --- | --- | --- | --- |']
    fixed=[r for r in rows if r['schedule']=='constant'];best={h:min(r['evaluation']['loss'] for r in fixed if r['loops']==h) for h in [1,2,4,8]}
    for r in sorted(fixed,key=lambda r:(r['loops'],r['p'],['none','ffn','attention','both'].index(r['mode']))):
        v=r['evaluation'];diag=v['diagnostics'];loss=f"{v['loss']:.6f}";ppl=f"{v['perplexity']:.6f}"
        if v['loss']==best[r['loops']]:loss=f'**{loss}**';ppl=f'**{ppl}**'
        text.append(f"| {r['loops']} | {labels[r['mode']]} | {r['p']:.2f} | {loss} | {ppl} | {branch_mean(diag,'ffn')*100:.3f} | {branch_mean(diag,'attention')*100:.3f} |")
    text += ['', '## Dropout across eight loops: 80M model','',
      'Every schedule below has mean dropout 0.10. The endpoints give the rates in Loop 1 and Loop 8. The three constant settings above serve as controls.','',
      '| Placement | Schedule | First / last loop rate | Loss | PPL | FFN near-zero (%) | Attention near-zero (%) |',
      '| --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted([r for r in rows if r['schedule']!='constant'],key=lambda r:(r['mode'],r['schedule'],r['delta'])):
        v=r['evaluation'];diag=v['diagnostics'];first,last=(r['p']-r['delta'],r['p']+r['delta'])
        if r['schedule']=='decreasing':first,last=last,first
        text.append(f"| {labels[r['mode']]} | {r['schedule'].capitalize()} | {first:.2f} / {last:.2f} | {v['loss']:.6f} | {v['perplexity']:.6f} | {branch_mean(diag,'ffn')*100:.3f} | {branch_mean(diag,'attention')*100:.3f} |")
    text += ['', '## What the layer plots show','',
      'With eight loops, FFN dropout decreasing from 20% to zero raises the first-loop mean near-zero fraction to 7.57%, while the last-loop value is 1.26%. The baseline changes from 1.25% to 1.03%. The effect is concentrated in early uses of the shared layers.','',
      '![FFN and attention outputs under different loop schedules](../figures/sparsedrop_layers/layer_sparsity_h8_ffn_schedule.png)','',
      'Applying the decreasing schedule to both branches creates a larger first-loop FFN near-zero fraction, 9.58%, but worse PPL: 36.614 versus 36.091 for FFN-only dropout. Counting more small outputs would choose the worse of these two models.','',
      '![Layer measurements and prediction quality](../figures/sparsedrop_layers/layer_sparsity_h8_quality.png)','',
      'Loop similarity is measured against Loop 1 of the same model. The decreasing FFN schedule keeps loop outputs closer to the first loop for longer, then changes more strongly near the end. This describes the learned computation; it does not by itself prove error correction.','',
      '![Each loop compared with the first loop](../figures/sparsedrop_additional/loop_diversity_reference_loop1_h8.png)','',
      'Output size and the share carried by the largest values help explain the near-zero counts. A branch can have more near-zero outputs because its whole output becomes smaller, even when its largest values account for a smaller share.','',
      '![Layer output magnitude](../figures/sparsedrop_additional/layer_output_rms_h8.png)','',
      '![Share of squared output in the largest ten percent of values](../figures/sparsedrop_additional/layer_top10_output_share_h8.png)','',
      '## Completed 400M / 20B results','',
      'Snapshot: November 7, 2026. One of 18 runs has completed. Only finished runs appear below. Each uses 20B training tokens and the complete 2,048-chunk validation set. These scores use a different validation scope from the small-model tables.','',
      '| Loops | Placement | Dropout p | Training tokens B | Loss | PPL | FFN near-zero (%) | Attention near-zero (%) | Mean update cosine |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    completed=json.loads((ROOT/'results/400m_completed.json').read_text())
    for r in completed['runs']:
        v=r['final_full_validation'];g=v['diagnostics'];assert r['training_tokens_B']==20 and r['steps']==1220704
        assert v['chunks']==2048 and v['tokens']==1048576
        text.append(f"| {r['loops']} | {labels[r['mode']]} | {r['p']:.2f} | 20.0000 | {v['loss']:.6f} | {v['perplexity']:.6f} | {branch_mean(g,'ffn')*100:.4f} | {branch_mean(g,'attention')*100:.4f} | {g['cross_loop']['mean_update_cosine']:.6f} |")
    text += ['', '## Limits of the comparison','',
      'The 80M study uses a single seed and an early stop on a longer learning-rate schedule. Validation also guided the choice of settings. More loops use more compute at the same token budget. The layer sample has 512 tokens, and near-zero outputs do not establish safe pruning. The 400M study is still incomplete; its first result does not yet compare dropout placements.','']
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(text))
    print('Wrote docs/RESULTS.md from 52 fixed-checkpoint records and completed 400M records.')
if __name__=='__main__':main()
