#!/usr/bin/env python3
"""Plot actual partial training records; no extrapolation or fabricated error bars."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aggregate_sparsedrop import load_run, pairing_errors


def records(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    rows = []
    series = []
    for name, label, color in [
        ('h4_none_p0.00_constant_d0.00_s42', 'Baseline (p=0)', '#2563eb'),
        ('h4_both_p0.10_constant_d0.00_s42', 'Attention + FFN (p=0.10)', '#e66f22'),
    ]:
        directory = args.results / name
        train = [x for x in records(directory / 'train.jsonl') if x.get('event') == 'train']
        validation = [x for x in records(directory / 'eval.jsonl') if x.get('event') == 'validation_fixed']
        if not train:
            continue
        series.append((label, train, validation, directory))
        x = [r['step'] for r in train]
        axes[0, 0].plot(x, [r['loss'] for r in train], label=label, color=color, lw=1.4)
        axes[0, 1].plot([r['step'] for r in validation], [r['loss'] for r in validation], color=color, marker='o', lw=1.5, label=label)
        axes[1, 0].plot(x, [r['grad_norm_preclip'] for r in train], color=color, lw=1.1)
        axes[1, 1].plot(x, [r['tokens_per_second']/1000 for r in train], color=color, lw=1.1)
    if not series:
        raise SystemExit('No actual training records found')
    for ax, title, ylabel in zip(axes.flat,
        ['Training loss (logged windows)', 'Fixed validation loss', 'Gradient norm before clipping', 'Training throughput (excludes eval/save)'],
        ['Cross entropy', 'Cross entropy', 'Global L2 norm', 'Thousand tokens / second']):
        ax.set(title=title, xlabel='Optimizer step', ylabel=ylabel)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('SparseDrop-Loop | H=4, seed=42 | early, incomplete runs', fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)
    if len(series) == 2:
        baseline = {r['step']: r for r in series[0][2]}
        treatment = {r['step']: r for r in series[1][2]}
        common = sorted(set(baseline) & set(treatment))
        if common:
            step = common[-1]
            try:
                base_run = load_run(series[0][3])
                treatment_run = load_run(series[1][3])
                base_run['evaluation'] = baseline[step]
                treatment_run['evaluation'] = treatment[step]
                errors = pairing_errors(base_run, treatment_run)
            except (OSError, ValueError, TypeError) as exc:
                errors = [str(exc)]
            if errors:
                payload = {'status': 'pairing_rejected', 'reasons': errors}
                args.output.with_suffix('.json').write_text(json.dumps(payload, indent=2)+'\n')
                print(json.dumps(payload, indent=2))
                return
            for label, data in [('baseline', baseline), ('both_p0.10', treatment)]:
                record = data[step]
                diag = record['diagnostics']
                branches = {}
                for branch in ('attention', 'ffn'):
                    acts = [a for a in diag['activations'] if a['branch'] == branch]
                    branches[branch] = {
                        'mean_near_zero_eps0.01': sum(next(t['near_zero_ratio'] for t in a['threshold_sensitivity'] if t['threshold'] == .01) for a in acts)/len(acts),
                        'mean_top10pct_energy': sum(a['top_10pct_energy'] for a in acts)/len(acts),
                        'mean_residual_rms': sum(a['residual_rms'] for a in acts)/len(acts),
                    }
                rows.append({'condition': label, 'step': step, 'loss': record['loss'], 'perplexity': record['perplexity'],
                             'mean_update_cosine': diag['cross_loop']['mean_update_cosine'],
                             'mean_linear_cka': diag['cross_loop']['mean_linear_cka'], 'activations_equal_weight_over_layers_loops': branches})
            payload = {'status': 'partial_single_seed', 'common_step': step, 'tokens_seen_per_run': baseline[step]['tokens_seen'],
                       'delta_loss_baseline_minus_dropout': baseline[step]['loss']-treatment[step]['loss'], 'rows': rows}
            args.output.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n')
            print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
