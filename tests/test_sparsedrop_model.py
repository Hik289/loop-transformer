"""Research invariants for dense, fixed-depth SparseDrop recurrence."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.model.transformer import TransformerBlock


def tiny_config(**kwargs):
    return LoopLMConfig(vocab_size=47, hidden_size=32, num_layers=2,
                        num_heads=4, intermediate_size=64, max_seq_len=16,
                        max_recurrent_steps=3, **kwargs)


@pytest.mark.parametrize("schedule,expected", [
    ("constant", [0.1, 0.1, 0.1]),
    ("increasing", [0.05, 0.1, 0.15]),
    ("decreasing", [0.15, 0.1, 0.05]),
])
def test_schedule_mean_and_endpoints(schedule, expected):
    config = tiny_config(residual_dropout_p=0.1, residual_dropout_schedule=schedule,
                         residual_dropout_delta=0 if schedule == "constant" else 0.05)
    actual = [config.dropout_probability(t, 3) for t in range(1, 4)]
    assert actual == pytest.approx(expected)
    assert sum(actual) / len(actual) == pytest.approx(0.1)
    if schedule != "constant":
        assert sum(config.dropout_probability(t, 8) for t in range(1, 9)) / 8 == pytest.approx(0.1)


@pytest.mark.parametrize("kwargs", [
    {"residual_dropout_mode": "unknown"},
    {"residual_dropout_p": -0.1},
    {"residual_dropout_p": 1.0},
    {"residual_dropout_p": float("nan")},
    {"residual_dropout_schedule": "invalid"},
    {"residual_dropout_delta": -0.01},
    {"residual_dropout_delta": 0.1},
    {"residual_dropout_p": 0.05, "residual_dropout_schedule": "increasing", "residual_dropout_delta": 0.1},
    {"residual_dropout_p": 0.9, "residual_dropout_schedule": "decreasing", "residual_dropout_delta": 0.1},
])
def test_invalid_schedules_rejected(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)


def test_h1_and_loop_index_validation():
    config = tiny_config(residual_dropout_p=0.1, residual_dropout_schedule="increasing",
                         residual_dropout_delta=0.05)
    for index, count in [(1, 1), (0, 3), (4, 3), (1, 0)]:
        with pytest.raises(ValueError):
            config.dropout_probability(index, count)
    with pytest.raises(ValueError):
        replace(config, max_recurrent_steps=1)


@pytest.mark.parametrize("mode", ["attention", "ffn", "both"])
def test_p_zero_matches_unmodified_block_and_model(mode):
    torch.manual_seed(11)
    baseline = LoopLM(tiny_config()).train()
    candidate = LoopLM(tiny_config(residual_dropout_mode=mode, residual_dropout_p=0)).train()
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.randint(0, 47, (2, 8))
    assert set(candidate.state_dict()) == set(baseline.state_dict())
    torch.testing.assert_close(candidate(tokens).logits[-1], baseline(tokens).logits[-1], rtol=0, atol=0)
    # Independently reproduce the upstream block, including both sandwich norms.
    h = candidate.embed(tokens)
    cos, sin = candidate.rope.get_cos_sin(8, tokens.device)
    block = candidate.layers[0]
    original = h + block.post_attn_norm(block.attn(block.pre_attn_norm(h), cos, sin))
    original = original + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(original)))
    torch.testing.assert_close(block(h, cos, sin), original, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["attention", "ffn", "both"])
def test_eval_is_deterministic_dense_identity(mode):
    baseline = LoopLM(tiny_config()).eval()
    dropped = LoopLM(tiny_config(residual_dropout_mode=mode, residual_dropout_p=0.2)).eval()
    dropped.load_state_dict(baseline.state_dict())
    tokens = torch.randint(0, 47, (2, 8))
    with torch.no_grad():
        expected = baseline(tokens, final_only=True).logits[0]
        for _ in range(2):
            torch.testing.assert_close(dropped(tokens, final_only=True).logits[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode,branches", [("attention", 1), ("ffn", 1), ("both", 2)])
def test_independent_masks_correct_location_and_mode(mode, branches):
    torch.manual_seed(12)
    model = LoopLM(tiny_config(residual_dropout_mode=mode, residual_dropout_p=0.2)).train()
    tokens = torch.randint(0, 47, (8, 16))
    real_dropout = F.dropout
    observations = []
    normalized_outputs = []
    hooks = []
    for block in model.layers:
        norms = []
        if mode in {"attention", "both"}:
            norms.append(block.post_attn_norm)
        if mode in {"ffn", "both"}:
            norms.append(block.post_ffn_norm)
        for norm in norms:
            hooks.append(norm.register_forward_hook(
                lambda module, inputs, output: normalized_outputs.append(output.detach().clone())))

    def recording_dropout(x, p, training):
        result = real_dropout(x, p=p, training=training)
        observations.append((x.detach().clone(), result.detach().clone()))
        return result

    with patch("src.model.transformer.F.dropout", side_effect=recording_dropout):
        model(tokens, final_only=True)
        model(tokens, final_only=True)
    for hook in hooks:
        hook.remove()
    assert len(observations) == 2 * 3 * 2 * branches
    masks = []
    for (before, after), normalized in zip(observations, normalized_outputs):
        # Hooks independently verify that dropout consumes the actual
        # post-RMSNorm branch output, retaining 1/(1-p) survivor scaling.
        torch.testing.assert_close(before, normalized, rtol=0, atol=0)
        zero_mask = after == 0
        masks.append(zero_mask)
        assert abs(zero_mask.float().mean().item() - 0.2) < 0.03
        torch.testing.assert_close(after[~zero_mask], before[~zero_mask] / 0.8)
    for left, right in zip(masks, masks[1:]):
        assert not torch.equal(left, right)


def test_final_only_logits_and_collectors_match_full_forward():
    model = LoopLM(tiny_config()).eval()
    tokens = torch.randint(0, 47, (2, 8))
    states, activations, head_calls = {}, [], []
    hook = model.lm_head.register_forward_hook(lambda *args: head_calls.append(1))
    with torch.no_grad():
        out = model(tokens, final_only=True, collect_hidden_states=True,
                    activation_collector=lambda *args: activations.append(args),
                    state_collector=lambda index, h: states.__setitem__(index, h.clone()))
    hook.remove()
    assert len(head_calls) == 1
    assert len(out.logits) == 1 and out.exit_lambdas == [] and out.q_values is None
    assert len(out.hidden_states) == 4
    assert list(states) == [0, 1, 2, 3]
    torch.testing.assert_close(states[0], model.embed(tokens))
    for index, hidden in enumerate(out.hidden_states):
        torch.testing.assert_close(hidden, states[index])
    assert [(t, layer, branch) for t, layer, branch, _ in activations] == [
        (t, layer, branch) for t in (1, 2, 3) for layer in (0, 1) for branch in ("attention", "ffn")
    ]
    torch.testing.assert_close(out.logits[0], model(tokens).logits[-1], rtol=0, atol=0)


def test_future_tokens_cannot_change_past_predictions():
    model = LoopLM(tiny_config(residual_dropout_mode="both", residual_dropout_p=0.2)).eval()
    tokens = torch.randint(0, 47, (2, 12))
    changed = tokens.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 47
    with torch.no_grad():
        a = model(tokens, final_only=True).logits[0]
        b = model(changed, final_only=True).logits[0]
    torch.testing.assert_close(a[:, :6], b[:, :6], rtol=0, atol=0)


def test_checkpoint_preserves_loop_schedule_rng_and_gradients():
    config = tiny_config(residual_dropout_mode="both", residual_dropout_p=0.15,
                         residual_dropout_schedule="increasing", residual_dropout_delta=0.1)
    normal = LoopLM(config).train()
    checked = LoopLM(replace(config)).train()
    checked.load_state_dict(normal.state_dict())
    checked.gradient_checkpointing = True
    tokens = torch.randint(0, 47, (2, 8))
    targets = torch.randint(0, 47, (2, 8))
    results = []
    for model in (normal, checked):
        torch.manual_seed(7654)
        out = model(tokens, final_only=True).logits[0]
        loss = F.cross_entropy(out.reshape(-1, 47), targets.reshape(-1))
        loss.backward()
        results.append(out.detach())
    torch.testing.assert_close(*results, rtol=0, atol=0)
    for (name, p1), (_, p2) in zip(normal.named_parameters(), checked.named_parameters()):
        if name.startswith("exit_gate."):
            assert p1.grad is None and p2.grad is None
        else:
            assert p1.grad is not None, name
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-5, atol=1e-6)


def test_parameter_estimate_is_actual_dense_count():
    config = tiny_config()
    assert config.num_parameters() == sum(p.numel() for p in LoopLM(config).parameters())


def test_checkpoint_activation_collection_rejected():
    model = LoopLM(tiny_config()).train()
    model.gradient_checkpointing = True
    with pytest.raises(ValueError, match="checkpointed training"):
        model(torch.zeros(1, 2, dtype=torch.long), activation_collector=lambda *args: None)
