<h1 align="center">Loop Transformer</h1>

<p align="center"><strong>Where does dropout help a language model that reuses its layers?</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg" alt="PyTorch 2.4 or newer">
  <a href="docs/EXPERIMENTS.md"><img src="https://img.shields.io/badge/Experiments-52%20%2B%2018-0F766E.svg" alt="52 small-model settings and 18 larger-model settings"></a>
  <a href="docs/RESULTS.md"><img src="https://img.shields.io/badge/Results-Data%20included-6366F1.svg" alt="Recorded evaluation data included"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-475569.svg" alt="BibTeX citation"></a>
</p>

![Shared-layer model, validation perplexity, and layer-by-layer FFN measurements](assets/overview.png)

**SparseLoop** studies dropout in a dense language model that runs the same six Transformer layers several times. We apply dropout to the attention output, the feed-forward network (FFN) output, or both. We then measure prediction quality and inspect what each layer does in each loop.

The figure combines the model design with recorded **80M-model results at 100,000 steps / 1.6384B tokens**. Dropout is off during evaluation. A near-zero output does not mean a weight was removed or a calculation was skipped.

## What is included

- Shared-layer models, fixed-depth training, data preparation, evaluation, and plotting code.
- **52 small-model settings:** 1, 2, 4, or 8 loops; three dropout placements; several rates and loop schedules.
- **18 larger-model settings:** approximately 400M parameters, 2 to 12 loops, and 20B training tokens per run.
- Exact token accounting, checkpoint recovery, Slurm arrays, and tests for training and measurements.
- Saved 100k evaluation records for all 52 small-model settings, including every loop, layer, and output branch.
- The original LoopLM training, inference, arithmetic, CAPO, and MANO experiment code. See [code guide](docs/CODE_GUIDE.md).

## Main observations so far

1. **More loops help prediction in the small-model baseline.** Perplexity falls from 38.907 with one loop to 36.557 with eight loops at the same token budget.
2. **Where dropout is applied matters.** At four loops, FFN dropout at 10% lowers perplexity from 37.020 to 36.634. Attention-only dropout at the same rate gives 37.163.
3. **Early and late loops behave differently.** With eight loops and FFN dropout decreasing from 20% to zero, near-zero FFN outputs are much more common in the first loop than in the last.
4. **More near-zero outputs do not guarantee better prediction.** Their meaning depends on output size, loop position, and the validation result.

These are single-seed observations from an early training checkpoint, with small quality differences. They are not evidence of a general speedup or of weights that can safely be pruned. [Full results and measurement definitions](docs/RESULTS.md).

## Installation

Use Python 3.10 or newer. Install a PyTorch build that supports your GPU before running CUDA training; the experiments used PyTorch 2.4.0 and BF16 on H100/H200.

```bash
git clone https://github.com/Hik289/loop-transformer.git
cd loop-transformer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,plots,dev]"
```

The SparseLoop training path runs locally and does not need an LLM service or an API key.

## Quick start

First run a tiny CPU check. It uses synthetic tokens, verifies an exact 257-token budget, and removes its temporary files when finished.

```bash
python scripts/smoke_test.py
python -m pytest tests/test_sparsedrop_model.py tests/test_sparsedrop_training.py -q
```

Recreate the figures from the evaluation records already in this repository:

```bash
python scripts/plot_overview.py
python scripts/plot_sparsedrop_layer_profiles.py
python scripts/plot_sparsedrop_additional_profiles.py
```

## Run the experiments

Prepare FineWeb-Edu tokens once. The full training array contains 20B + 1 tokens: the extra token supplies the last next-token label. The training array alone occupies about 40GB before NumPy headers; downloads and temporary shards need additional space.

```bash
source cluster/env.sh
python scripts/prepare_fineweb_sparsedrop.py \
  --output-dir data/fineweb_edu20b --workers 8
```

Train one small-model condition from the repository root:

```bash
python scripts/train_sparsedrop.py \
  --config configs/sparsedrop_long/003_h4_ffn_p0.10_constant_d0.00_s42.json
```

Train one 400M condition:

```bash
python scripts/train_sparsedrop.py \
  --config configs/sparsedrop_400m20b/000_m404m_h2_ffn_p0.10_s42.json
```

The first command stops at **100,000 steps / 1.6384B tokens**. The second runs to **20B tokens / 1,220,704 steps**. Both use final-loop next-token loss. See [settings and budgets](docs/EXPERIMENTS.md) before launching a full grid.

### Slurm

Each array task requests one GPU. Supply your own partition and account. Submit from the repository root after creating the log directory:

```bash
mkdir -p logs
# The 18 larger-model runs split across two GPU types.
sbatch --partition=YOUR_H100_PARTITION --account=YOUR_ACCOUNT \
  --array=1,3-10%2 cluster/train_array.slurm
sbatch --partition=YOUR_H200_PARTITION --account=YOUR_ACCOUNT \
  --array=0,2,11-17%2 \
  cluster/train_array.slurm

# Alternatively, launch the 52 small-model settings.
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --export=ALL,SPARSEDROP_STUDY=80m --array=0-51%2 \
  cluster/train_array.slurm
```

The first two commands cover the 18 larger-model settings once each; the 80M command launches a separate study. Some high-loop profiles need more than 80GB of GPU memory. The array limit applies to each submission; your cluster may enforce a separate per-user limit. The wrapper saves before the seven-day limit and requeues the same task. [Cluster setup](docs/CLUSTER.md).

## Results and files

| Folder | Contents |
| --- | --- |
| `src/model/` | Shared-layer loop, attention, FFN, dropout, and optional MoE code |
| `scripts/` | Training, preparation, evaluation, diagnostics, and plotting |
| `configs/sparsedrop_long/` | 52 seed-42 conditions for the 80M comparison |
| `configs/sparsedrop_400m20b/` | 18 seed-42 conditions for the 400M study |
| `cluster/` | Portable launch wrapper and earlier cluster templates |
| `tests/` | Model, measurement, budget, and recovery checks |
| `results/` | Selected scientific records; no datasets or model weights |
| `docs/` | Methods, settings, results, and code guide |
| `assets/` | Main figure in PNG, SVG, and PDF |

As of **November 7, 2026**, one of the 18 larger-model runs has finished: 2-loop attention dropout, with full-validation loss **2.907877** and perplexity **18.317870** after 20B tokens. The remaining larger-model results are not presented as completed results. Their full-validation numbers should not be compared directly with the small study's 128-chunk scores.

## Citation

This citation identifies the code repository. No paper identifier is claimed for this release.

```bibtex
@software{looptransformer2026,
  title = {Loop Transformer: Dropout in Shared-Weight Language Models},
  year = {2026},
  url = {https://github.com/Hik289/loop-transformer},
  version = {0.1.0}
}
```

## Acknowledgments

Built on [rkstgr/LoopLM](https://github.com/rkstgr/LoopLM), a third-party reproduction of the shared-layer design in [Ouro](https://arxiv.org/abs/2510.25741). The experiments use [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and the [SmolLM2 tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). See [upstream provenance and licensing](THIRD_PARTY.md).
