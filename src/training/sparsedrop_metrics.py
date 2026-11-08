"""Bounded, deterministic eval diagnostics for the SparseDrop-Loop protocol.

Only the first ``max_tokens`` valid positions of an ordered validation stream
are sampled. This is a diagnostic subset, not the validation-loss population.
Residual activations are summarized immediately; full GPU activations are never
retained. Callbacks must run with the model in eval mode and gradients disabled.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor


def token_cosine(x: Tensor, y: Tensor, epsilon: float = 1e-12) -> float:
    """Mean per-token cosine, rather than a cosine of flattened batches."""
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] == 0:
        raise ValueError("Cosine inputs must have equal, nonempty [tokens, hidden] shapes")
    x, y = x.double(), y.double()
    denominator = x.norm(dim=-1) * y.norm(dim=-1) + epsilon
    return ((x * y).sum(dim=-1) / denominator).mean().item()


def _centered_gram(x: Tensor) -> Tensor:
    x = x.double()
    x = x - x.mean(dim=0, keepdim=True)
    # The token-Gram identity is exactly equivalent to the feature-Gram formula,
    # and avoids three hidden_size**2 matrices for a small token sample.
    return x @ x.T


def linear_cka(x: Tensor, y: Tensor, epsilon: float = 1e-12) -> float:
    """Centered linear CKA on aligned [tokens, hidden] matrices."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA requires at least two aligned token observations")
    gx, gy = _centered_gram(x), _centered_gram(y)
    return ((gx * gy).sum() / (gx.norm() * gy.norm() + epsilon)).item()


def _off_diagonal_mean(matrix: list[list[float]]) -> float | None:
    values = [matrix[t][s] for t in range(len(matrix)) for s in range(t)]
    return sum(values) / len(values) if values else None


class SparseDropMetrics:
    """Collect the same bounded token sample at every loop and physical layer.

    ``begin_batch`` and ``end_batch`` bracket a model forward. Connect
    ``record_state`` to the model's state_collector (indices 0..H) and
    ``record_activation`` to activation_collector (loops 1..H, layers 0..M-1).
    Thresholds are fixed *before* comparing treatments; they are never fitted to
    a treatment's own activation scale. Observed RMS values allow the p=0
    baseline's empirical scale to be documented without changing the thresholds.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        thresholds: Iterable[float] = (0.001, 0.01, 0.1),
        topk_fractions: Iterable[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
        epsilon: float = 1e-12,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.thresholds = tuple(sorted(set(float(x) for x in thresholds)))
        self.topk_fractions = tuple(sorted(set(float(x) for x in topk_fractions) | {0.1}))
        if self.max_tokens <= 0 or epsilon <= 0:
            raise ValueError("max_tokens and numerical epsilon must be positive")
        if not self.thresholds or any(x < 0 or not math.isfinite(x) for x in self.thresholds):
            raise ValueError("Activation thresholds must be finite and nonnegative")
        if any(not 0 < x <= 1 for x in self.topk_fractions):
            raise ValueError("Top-k fractions must lie in (0, 1]")
        self.epsilon = float(epsilon)
        self.sampled_tokens = 0
        self._position_offset = 0
        self._sample_positions: list[int] = []
        self._state_chunks: dict[int, list[Tensor]] = defaultdict(list)
        self._activation_stats: dict[tuple[int, int, str], dict] = {}
        self._batch_open = False
        self._batch_indices: Tensor | None = None
        self._batch_mask: Tensor | None = None
        self._batch_shape: tuple[int, ...] | None = None
        self._batch_states: set[int] = set()
        self._batch_activations: set[tuple[int, int, str]] = set()
        self._expected_states: set[int] | None = None
        self._expected_activations: set[tuple[int, int, str]] | None = None

    @property
    def full(self) -> bool:
        return self.sampled_tokens >= self.max_tokens

    def begin_batch(self, valid_mask: Tensor | None = None, *, training: bool = False) -> None:
        if training:
            raise ValueError("SparseDrop diagnostics require model.eval(); training dropout is not inference sparsity")
        if self._batch_open:
            raise RuntimeError("end_batch must close the previous forward")
        self._batch_open = True
        self._batch_indices = None
        self._batch_shape = None
        self._batch_mask = None if valid_mask is None else valid_mask.detach().bool().cpu()
        self._batch_states = set()
        self._batch_activations = set()

    def _sample(self, tensor: Tensor) -> Tensor:
        if not self._batch_open:
            raise RuntimeError("begin_batch must be called before metric callbacks")
        if tensor.requires_grad:
            raise ValueError("Metrics callbacks must run under torch.no_grad() or torch.inference_mode()")
        if tensor.ndim < 2:
            raise ValueError("Expected tensor shape [...tokens, hidden]")
        shape = tuple(tensor.shape[:-1])
        if self._batch_indices is None:
            self._batch_shape = shape
            total = math.prod(shape)
            if self._batch_mask is not None:
                if tuple(self._batch_mask.shape) != shape:
                    raise ValueError("valid_mask shape must match activation token dimensions")
                indices = self._batch_mask.reshape(-1).nonzero().flatten()
            else:
                indices = torch.arange(total)
            self._batch_indices = indices[: max(0, self.max_tokens - self.sampled_tokens)]
        elif shape != self._batch_shape:
            raise ValueError("All callbacks in a forward must use the same token dimensions")
        indices = self._batch_indices.to(tensor.device)
        # Slice on the source device *before* copying; never retain the full
        # batch or its computation graph, even when the GPU batch is large.
        return tensor.detach().reshape(-1, tensor.shape[-1]).index_select(0, indices).float().cpu()

    def record_state(self, loop_index: int, tensor: Tensor) -> None:
        if loop_index < 0 or loop_index in self._batch_states:
            raise ValueError("State indices must be nonnegative and unique within a forward")
        sample = self._sample(tensor)
        self._batch_states.add(loop_index)
        if sample.numel():
            self._state_chunks[loop_index].append(sample)

    def record_activation(self, loop_index: int, layer_index: int, branch: str, tensor: Tensor) -> None:
        if loop_index < 1 or layer_index < 0 or branch not in {"attention", "ffn"}:
            raise ValueError("Expected loop >=1, layer >=0, and attention/ffn branch")
        key = (loop_index, layer_index, branch)
        if key in self._batch_activations:
            raise ValueError("Repeated residual callback; gradient-checkpoint recomputation must not be collected")
        sample = self._sample(tensor)
        self._batch_activations.add(key)
        if not sample.numel():
            return
        n, d = sample.shape
        energy = sample.double().square()
        sorted_energy = energy.sort(dim=-1, descending=True).values.cumsum(dim=-1)
        denominator = energy.sum(dim=-1) + self.epsilon
        stats = self._activation_stats.setdefault(key, {
            "tokens": 0, "hidden_size": d, "sum_squares": 0.0,
            "near_zero_counts": [0] * len(self.thresholds),
            "topk_energy_sums": [0.0] * len(self.topk_fractions),
        })
        if stats["hidden_size"] != d:
            raise ValueError("Residual hidden size changed between batches")
        stats["tokens"] += n
        stats["sum_squares"] += energy.sum().item()
        for idx, threshold in enumerate(self.thresholds):
            stats["near_zero_counts"][idx] += (sample.abs() <= threshold).sum().item()
        for idx, fraction in enumerate(self.topk_fractions):
            k = math.ceil(d * fraction)
            stats["topk_energy_sums"][idx] += (sorted_energy[:, k - 1] / denominator).sum().item()

    def end_batch(self) -> None:
        if not self._batch_open:
            raise RuntimeError("No diagnostic batch is open")
        if self._batch_indices is None:
            raise RuntimeError("No state or activation callback was received")
        expected = set(range(max(self._batch_states, default=0) + 1))
        if self._batch_states != expected or len(self._batch_states) < 2:
            raise ValueError("Each forward must record embedding h0 and every completed loop h1..hH")
        if self._expected_states is not None and self._batch_states != self._expected_states:
            raise ValueError("Loop depth changed across diagnostic batches")
        if self._expected_activations is not None and self._batch_activations != self._expected_activations:
            raise ValueError("Residual callback coverage changed across diagnostic batches")
        self._expected_states = self._batch_states.copy()
        self._expected_activations = self._batch_activations.copy()
        self._sample_positions.extend((self._batch_indices + self._position_offset).tolist())
        self.sampled_tokens += self._batch_indices.numel()
        self._position_offset += math.prod(self._batch_shape)
        self._batch_mask = None
        self._batch_indices = None
        self._batch_open = False

    def compute(self) -> dict:
        if self._batch_open:
            raise RuntimeError("Close the diagnostic batch before computing metrics")
        protocol = {
            "version": "sparsedrop_eval_v1",
            "model_mode": "eval",
            "sample_selection": "first_valid_tokens_in_fixed_validation_order",
            "max_tokens": self.max_tokens,
            "sampled_tokens": self.sampled_tokens,
            "sample_position_sha256": hashlib.sha256(
                ",".join(map(str, self._sample_positions)).encode("ascii")
            ).hexdigest(),
            "activation_location": "post_sublayer_RMSNorm_before_dropout_and_residual_addition",
            "activation_thresholds": list(self.thresholds),
            "threshold_policy": "fixed_a_priori_from_unit_scale_RMSNorm_baseline; never_refit_per_condition",
            "baseline_scale_reference": "record p=0 residual_rms alongside fixed thresholds; learned RMSNorm gains may change scale",
            "topk_fractions": list(self.topk_fractions),
            "numerical_epsilon": self.epsilon,
            "cosine_reduction": "mean_of_per_token_cosines",
            "cka_centering": "token_axis; centered_linear_CKA; bounded diagnostic subset",
        }
        if not self.sampled_tokens:
            return {"protocol": protocol, "status": "no_valid_tokens", "cross_loop": None, "activations": []}
        states = {idx: torch.cat(chunks, dim=0) for idx, chunks in sorted(self._state_chunks.items())}
        h = len(states) - 1
        if set(states) != set(range(h + 1)):
            raise ValueError("Missing state samples; expected h0..hH")
        updates = {idx: states[idx] - states[idx - 1] for idx in range(1, h + 1)}
        loop_summary = [{
            "loop": idx,
            "state_rms": states[idx].double().square().mean().sqrt().item(),
            "update_mean_l2": updates[idx].double().norm(dim=-1).mean().item(),
            "update_rms": updates[idx].double().square().mean().sqrt().item(),
        } for idx in range(1, h + 1)]
        cross_loop = {"status": "not_applicable_H1" if h == 1 else "ok", "loop_indices": list(range(1, h + 1)),
                      "state_cosine": None, "update_cosine": None, "linear_cka": None,
                      "mean_state_cosine": None, "mean_update_cosine": None, "mean_linear_cka": None}
        if h > 1:
            state_cosine = [[token_cosine(states[t], states[s], self.epsilon) for s in range(1, h + 1)] for t in range(1, h + 1)]
            update_cosine = [[token_cosine(updates[t], updates[s], self.epsilon) for s in range(1, h + 1)] for t in range(1, h + 1)]
            cross_loop.update(state_cosine=state_cosine, update_cosine=update_cosine,
                              mean_state_cosine=_off_diagonal_mean(state_cosine),
                              mean_update_cosine=_off_diagonal_mean(update_cosine))
            if self.sampled_tokens >= 2:
                grams = [_centered_gram(states[t]) for t in range(1, h + 1)]
                cka = [[((gx * gy).sum() / (gx.norm() * gy.norm() + self.epsilon)).item() for gy in grams] for gx in grams]
                cross_loop.update(linear_cka=cka, mean_linear_cka=_off_diagonal_mean(cka))
            else:
                cross_loop["cka_status"] = "not_applicable_fewer_than_two_tokens"
        activations = []
        for (loop, layer, branch), stats in sorted(self._activation_stats.items()):
            n, d = stats["tokens"], stats["hidden_size"]
            sparsity = [{"threshold": threshold, "near_zero_ratio": count / (n * d),
                         "active_ratio": 1.0 - count / (n * d)}
                        for threshold, count in zip(self.thresholds, stats["near_zero_counts"])]
            curve = [{"fraction": fraction, "k": math.ceil(d * fraction), "energy": energy / n}
                     for fraction, energy in zip(self.topk_fractions, stats["topk_energy_sums"])]
            activations.append({"loop": loop, "layer": layer, "branch": branch,
                                "tokens": n, "hidden_size": d,
                                "residual_rms": math.sqrt(stats["sum_squares"] / (n * d)),
                                "threshold_sensitivity": sparsity,
                                "top_10pct_energy": next(point["energy"] for point in curve if point["fraction"] == 0.1),
                                "topk_energy_curve": curve})
        return {"protocol": protocol, "status": "ok", "loop_states": loop_summary,
                "cross_loop": cross_loop, "activations": activations}
