# Upstream code

The model and supporting experiments build on [rkstgr/LoopLM](https://github.com/rkstgr/LoopLM), a third-party reproduction of Ouro-style looped language models. The starting revision was `8d79e31a871e23291a1afb21bf87d33116e8eb26`. This is not the official Ouro implementation.

The shared-layer model, inference tools, original training entry points, arithmetic and reasoning tasks, and their tests are retained. SparseDrop adds residual-output dropout, fixed-depth training, exact token budgets, recovery checks, and per-loop measurements.

The upstream repository did not provide a license at the recorded revision. Original contributions in this repository are covered by the [MIT License](LICENSE). That license does not apply to inherited upstream code or grant rights that the upstream authors have not provided. Dataset and tokenizer rights remain with their original providers.

References: [Ouro paper](https://arxiv.org/abs/2510.25741), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [SmolLM2 tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
