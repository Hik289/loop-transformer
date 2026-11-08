# Running on Slurm

Use `cluster/train_array.slurm` for a new installation. It requests one GPU per task and allows two tasks in the array at once. Pass your site's partition, account, and any required QOS on the `sbatch` command line. The supplied 96G CPU-memory request is separate from GPU memory.

1. Clone the repository into a filesystem available on the compute nodes.
2. Create the environment and prepare the data using the README commands.
3. Run `mkdir -p logs` before submitting. Slurm opens log files before the script starts.
4. Submit from the repository root. The wrapper uses `SLURM_SUBMIT_DIR`, or an explicitly supplied `SPARSEDROP_ROOT`.

For a selected 400M run, use `--array=INDEX%1`. The default is all 18 indices with a concurrency limit of two. For the 80M grid, set `SPARSEDROP_STUDY=80m` and use indices 0-51.

Each 400M task checks the configuration, source hashes, prepared data, exact-token CUDA behavior, and a small save/resume run of the actual model shape before production training. The small-model wrapper also runs the CUDA token-budget check. A failed check stops the task.

The wrapper requests seven days and tells the trainer to save before that limit. On the expected pause exit code, it requeues the same array element. A user cancellation is not requeued. Your site must allow requeue for this behavior to work. If you shorten the time limit, preserve the advance signal so the trainer can save safely.

## Choosing a memory profile

Published 400M profiles were checked on 80GB H100 and 141GB-class H200 devices. Larger loop counts keep more activations when checkpointing is off. The H12, micro-batch-16 profile reached about 107GB of allocated memory in a short H200 test; it is not suitable for an 80GB card.

For another GPU, first reduce micro batch and increase accumulation so their product remains 32, or turn activation checkpointing on. Update the explicit profile policy and configuration together, then rebuild the public source checksums with `python scripts/refresh_source_hashes.py`. Changing an execution profile can change the exact dropout masks. Use a separate output directory for a new experiment.

## Earlier templates and helpers

Other `.slurm` files are retained to document the earlier experiments, including the original LoopLM tasks. Their site-specific account, partition, QOS, and directory settings have been removed. Review their resource requests and task ranges before using them. Original task scripts that use `$PROJECT` and `$SCRATCH` require you to set those variables.

Status, synchronization, and cross-cluster gate helpers are included for completeness. Set their remote root and SSH aliases for your installation. Do not use old job IDs: job registries and gate states are runtime files and are not released. The `.example.json` files show the format. The standard training path does not require a cross-cluster gate.
