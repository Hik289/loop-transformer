# Experiment settings

## Two separate studies

| Setting | Small-model comparison | Larger-model study |
| --- | --- | --- |
| Total parameters | 80,236,033 | 404,279,297 |
| Physical Transformer layers | 6 | 6 |
| Hidden width | 768 | 2048 |
| Attention heads | 12 | 32 |
| FFN width | 2048 | 5504 |
| Vocabulary size | 49,152 | 49,152 |
| Loops | 1, 2, 4, 8 | 2, 4, 6, 8, 10, 12 |
| Settings | 52 | 18 |
| Training tokens per comparison | 1.6384B | 20B |
| Optimizer steps | 100,000 | 1,220,704 |
| Seed | 42 | 42 |
| Main validation scope | Same 128 chunks | All 2,048 complete chunks |

Both models are dense, with tied input embeddings and output weights. Shared loops do not increase the parameter count. Both use BF16 computation, FP32 parameters and AdamW states, sequence length 512, and 32 sequences per optimizer update. There are 16,384 prediction tokens per full update. Loop count does not multiply the reported token count.

The optimizer uses learning rate 0.0003, betas (0.9, 0.95), weight decay 0.1 on matrix weights, and gradient clipping at 1.0. The learning rate warms up for 2,000 updates, then follows the original 20B-token cosine horizon to 0.00003. The small-model comparison stops early on that same horizon; it is not a separately tuned 100k schedule.

## Small-model grid

For each of four loop counts, there is one baseline and nine fixed-rate conditions: three placements (FFN, attention, both) times three rates (0.05, 0.10, 0.20). This gives 40 settings. At eight loops, another 12 settings combine the three placements with increasing/decreasing schedules and delta=0.05/0.10. Total: **52**, each with seed 42.

The released configurations all stop at 100k steps. The comparison records come from each run's actual 100k fixed-subset evaluation. This also allows a historically longer run to enter the table at exactly the same 100k checkpoint, without using its later endpoint.

## Larger-model grid

Six loop counts times three dropout placements gives **18 settings**. Every run uses fixed p=0.10 and seed 42. The final optimizer update contains 2,048 prediction tokens, so the total is exactly 20B.

| Run indices (zero-based) | Micro batch | Accumulation | Activation checkpointing |
| --- | --- | --- | --- |
| 0, 2 | 32 | 1 | Off |
| 1, 3 | 8 | 4 | On |
| 4-8, 11-17 | 16 | 2 | Off |
| 9, 10 | 8 | 4 | Off |

These are the current execution profiles. Indices 0 and 2 originally used 8 x 4 with checkpointing on; continuation was configured at steps 367,983 and 896,260. A fresh run uses its published profile from step zero and will not reproduce those historical mask sequences bit for bit. The effective batch and scientific budgets stay unchanged.

The two manifests list the exact index, run name, and configuration file. All configurations use repository-relative paths. Run commands from the repository root or override `--data-dir` and `--output-dir` explicitly.

## Data and outputs

FineWeb-Edu preparation uses pinned dataset and tokenizer revisions. A content hash assigns complete documents to train, validation, or test before tokenization, so identical text does not cross splits. Tokens are packed into uint16 arrays. The preparation script records source hashes, counts, and completion metadata.

Training creates `config.json`, `train.jsonl`, `eval.jsonl`, `best.pt`, and `last.pt`. At completion it writes `summary.json` and `sparsity_final.json`. Diagnostics must cover H x 6 x 2 layer/branch records. The 400M Slurm wrapper verifies weights-only exports before removing optimizer state from a completed run. Full checkpoints remain available while training is unfinished.

The repository includes selected evaluation records, not the training corpus or checkpoint weights. Regenerating the full studies requires preparing the data and running the configurations.
