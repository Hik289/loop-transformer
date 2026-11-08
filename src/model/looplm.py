from typing import Callable, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from src.model.config import LoopLMConfig
from src.model.rope import RotaryEmbedding
from src.model.transformer import ActivationCollector, RMSNorm, TransformerBlock, PostNormTransformerBlock


StateCollector = Callable[[int, Tensor], None]


class LoopLMOutput(NamedTuple):
    # One tensor per recurrent step
    logits: list[Tensor]       # each (B, S, vocab_size)
    exit_lambdas: list[Tensor] # each (B, S) — raw sigmoid gate outputs λ_t
    q_values: list[Tensor] | None = None  # each (B, S, 2) — Q(halt), Q(continue)
    aux_losses: list[Tensor] | None = None  # auxiliary losses (e.g. MoE load balance)
    hidden_states: list[Tensor] | None = None  # optional unnormalized h0, h1, ..., hH


def compute_exit_distribution(exit_lambdas: list[Tensor]) -> Tensor:
    """Compute p(t|x) from raw gate outputs λ_t.

    For t < T_max:  p(t|x) = λ_t · S_{t-1}   where S_0 = 1
    For t = T_max:  p(T_max|x) = S_{T_max-1}

    Args:
        exit_lambdas: list of T tensors, each (B, S)

    Returns:
        exit_probs: (T, B, S) — sums to 1 over the T dimension
    """
    T = len(exit_lambdas)
    probs = []
    survival = torch.ones_like(exit_lambdas[0])  # S_0 = 1

    for t, lam in enumerate(exit_lambdas):
        if t < T - 1:
            p_t = lam * survival
            survival = survival * (1.0 - lam)
        else:
            # Last step absorbs remaining probability mass
            p_t = survival
        probs.append(p_t)

    return torch.stack(probs, dim=0)  # (T, B, S)


class LoopLM(nn.Module):
    """Looped Language Model with shared-weight recurrence.

    A stack of `num_layers` TransformerBlocks is applied `num_steps` times,
    reusing the same weights each pass. An LM head and an exit gate are applied
    after every recurrent step.

    Supports ablation modes:
        - H/L split: partition layers into slow (H) and fast (L) modules
        - Q-learning ACT: replace exit gate with Q-head for halt/continue
        - MoE recurrence: route to expert TransformerBlocks per step
    """

    def __init__(
        self,
        config: LoopLMConfig,
        use_postnorm: bool = False,
        use_lecun_init: bool = False,
        use_q_act: bool = False,
        use_moe_recurrence: bool = False,
        num_expert_layers: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()
        config.validate_sparsedrop()
        if (use_postnorm or use_moe_recurrence) and config.residual_dropout_mode != "none":
            raise ValueError("SparseDrop supports the standard dense sandwich-normalized LoopLM only")
        self.config = config
        self.use_q_act = use_q_act
        self.use_moe_recurrence = use_moe_recurrence

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        if use_moe_recurrence:
            from src.model.moe_layer import MoELayer
            self.moe_layer = MoELayer(
                config,
                num_experts=num_expert_layers,
                top_k=moe_top_k,
                use_postnorm=use_postnorm,
            )
            # Still keep regular layers for non-MoE parts (e.g. H/L split H-module)
            self.layers = nn.ModuleList()
        else:
            block_cls = PostNormTransformerBlock if use_postnorm else TransformerBlock
            self.layers = nn.ModuleList(
                [block_cls(config) for _ in range(config.num_layers)]
            )

        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Exit gate (standard Bernoulli)
        self.exit_gate = nn.Linear(config.hidden_size, 1, bias=True)

        # Q-learning ACT head (optional, replaces exit gate for halting decisions)
        if use_q_act:
            self.q_head = nn.Linear(config.hidden_size, 2, bias=True)

        # Tie LM head weights to the embedding matrix (standard practice)
        self.lm_head.weight = self.embed.weight

        self.rope = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_heads,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        self.gradient_checkpointing = False

        if use_lecun_init:
            self._init_weights_lecun()
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        nn.init.zeros_(self.exit_gate.bias)
        nn.init.normal_(self.exit_gate.weight, mean=0.0, std=std)
        if self.use_q_act:
            nn.init.zeros_(self.q_head.bias)
            nn.init.normal_(self.q_head.weight, mean=0.0, std=std)
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)
        if self.use_moe_recurrence:
            for p in self.moe_layer.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights_lecun(self) -> None:
        """Truncated LeCun Normal init (HRM-style): std = 1/sqrt(fan_in), truncated at 2σ."""
        import math
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                fan_in = module.embedding_dim
                std = 1.0 / math.sqrt(fan_in)
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def _apply_layers(
        self,
        h: Tensor,
        layers: list[nn.Module],
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        loop_index: int = 1,
        total_loops: int | None = None,
        activation_collector: ActivationCollector | None = None,
    ) -> Tensor:
        """Apply a list of transformer layers to hidden state."""
        use_ckpt = self.training and self.gradient_checkpointing
        for layer_index, layer in enumerate(layers):
            # Indices are immutable per-call arguments, including on checkpoint
            # recomputation. RNG preservation reuses each forward dropout mask.
            kwargs = {}
            if isinstance(layer, TransformerBlock):
                kwargs = dict(loop_index=loop_index, total_loops=total_loops,
                              layer_index=layer_index, activation_collector=activation_collector)
            if use_ckpt:
                h = checkpoint(layer, h, cos, sin, attn_mask, use_reentrant=False, **kwargs)
            else:
                h = layer(h, cos, sin, attn_mask, **kwargs)
        return h

    def _read_out(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute logits, exit lambda, and optionally Q-values from hidden state."""
        logits = self.lm_head(self.final_norm(h))
        lam = torch.sigmoid(self.exit_gate(h)).squeeze(-1)
        q_vals = self.q_head(h) if self.use_q_act else None
        return logits, lam, q_vals

    def forward(
        self,
        input_ids: Tensor,
        num_steps: int | None = None,
        attention_mask: Tensor | None = None,
        detach_between_steps: bool = False,
        # H/L split params
        use_hl_split: bool = False,
        n_h_layers: int = 2,
        t_inner: int = 2,
        t_outer: int = 2,
        final_only: bool = False,
        collect_hidden_states: bool = False,
        activation_collector: ActivationCollector | None = None,
        state_collector: StateCollector | None = None,
    ) -> LoopLMOutput:
        """
        Args:
            input_ids:      (B, S) token indices
            num_steps:      number of recurrent steps; defaults to config.max_recurrent_steps
            attention_mask: optional (B, 1, S, S) additive float mask
            detach_between_steps: detach hidden state between steps
            use_hl_split:   enable H/L module split
            n_h_layers:     number of H (slow) layers (first N layers)
            t_inner:        L inner loop steps
            t_outer:        H outer loop steps
            final_only:    compute only the final LM logits, skipping unused gates
            collect_hidden_states: return h0..hH (before final readout RMSNorm)
            activation_collector: callback(loop_index, layer_index, branch, tensor)
                for post-branch RMSNorm outputs before dropout/addition; loops
                are 1-based and physical layers are 0-based. Prefer in eval().
            state_collector: callback(loop_index, tensor), called on h0 then h1..hH;
                permits streaming a fixed token subset without retaining full states
        """
        if num_steps is None:
            num_steps = self.config.max_recurrent_steps
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.config.dropout_probability(1, num_steps)  # reject undefined H=1 schedules
        if self.training and self.gradient_checkpointing and activation_collector is not None:
            raise ValueError("activation_collector must not run during checkpointed training; use eval()")
        if use_hl_split and self.config.residual_dropout_mode != "none":
            raise ValueError("SparseDrop requires standard fixed-depth recurrence")
        if (self.use_moe_recurrence or use_hl_split) and (
            final_only or collect_hidden_states or activation_collector is not None or state_collector is not None
        ):
            raise ValueError("final-only output and collectors require standard dense recurrence")

        B, S = input_ids.shape
        cos, sin = self.rope.get_cos_sin(S, input_ids.device)
        h = self.embed(input_ids)

        if self.use_moe_recurrence:
            return self._forward_moe(h, num_steps, cos, sin, attention_mask, detach_between_steps)
        elif use_hl_split:
            return self._forward_hl_split(h, cos, sin, attention_mask, n_h_layers, t_inner, t_outer)
        else:
            return self._forward_standard(
                h, num_steps, cos, sin, attention_mask, detach_between_steps,
                final_only, collect_hidden_states, activation_collector, state_collector,
            )

    def _forward_standard(
        self,
        h: Tensor,
        num_steps: int,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        detach_between_steps: bool,
        final_only: bool = False,
        collect_hidden_states: bool = False,
        activation_collector: ActivationCollector | None = None,
        state_collector: StateCollector | None = None,
    ) -> LoopLMOutput:
        """Standard recurrent forward pass."""
        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []
        hidden_states = [h] if collect_hidden_states else None
        if state_collector is not None:
            state_collector(0, h)

        for step in range(1, num_steps + 1):
            if detach_between_steps:
                h = h.detach()
            h = self._apply_layers(h, list(self.layers), cos, sin, attn_mask,
                                   loop_index=step, total_loops=num_steps,
                                   activation_collector=activation_collector)
            if hidden_states is not None:
                hidden_states.append(h)
            if state_collector is not None:
                state_collector(step, h)
            if final_only:
                if step == num_steps:
                    logits_per_step.append(self.lm_head(self.final_norm(h)))
                continue
            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
            hidden_states=hidden_states,
        )

    def _forward_hl_split(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        n_h_layers: int,
        t_inner: int,
        t_outer: int,
    ) -> LoopLMOutput:
        """H/L split forward: L runs t_inner steps, H updates once, repeat t_outer times.

        H layers = self.layers[:n_h_layers]  (slow, global context)
        L layers = self.layers[n_h_layers:]  (fast, local refinement)

        Execution pattern per outer step:
            1. L layers run t_inner times (with detach between inner steps)
            2. H layers run once
            3. Read out logits/exit
        """
        h_layers = list(self.layers[:n_h_layers])
        l_layers = list(self.layers[n_h_layers:])

        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []

        for outer in range(t_outer):
            # L inner loop: fast refinement
            for inner in range(t_inner):
                if inner > 0 or outer > 0:
                    h = h.detach()
                h = self._apply_layers(h, l_layers, cos, sin, attn_mask)

            # H update: slow global context
            h = self._apply_layers(h, h_layers, cos, sin, attn_mask)

            # Read out after each H update
            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
        )

    def _forward_moe(
        self,
        h: Tensor,
        num_steps: int,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None,
        detach_between_steps: bool,
    ) -> LoopLMOutput:
        """MoE recurrence: route to expert layers at each step."""
        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        q_values: list[Tensor] = []
        aux_losses: list[Tensor] = []

        for _ in range(num_steps):
            if detach_between_steps:
                h = h.detach()
            h, lb_loss = self.moe_layer(h, cos, sin, attn_mask)
            aux_losses.append(lb_loss)

            logits, lam, q_vals = self._read_out(h)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)
            if q_vals is not None:
                q_values.append(q_vals)

        return LoopLMOutput(
            logits=logits_per_step,
            exit_lambdas=exit_lambdas,
            q_values=q_values if q_values else None,
            aux_losses=aux_losses,
        )
