import math
from dataclasses import dataclass


@dataclass
class LoopLMConfig:
    vocab_size: int = 49152
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    # SwiGLU intermediate size; paper uses ~2.67x hidden (rounded to multiple of 64)
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    max_recurrent_steps: int = 4
    rope_base: float = 10000.0
    dropout: float = 0.0
    beta_kl: float = 0.1
    # SparseDrop acts on normalized residual branches, never attention weights.
    # `dropout` above is retained for upstream config/checkpoint compatibility.
    residual_dropout_mode: str = "none"
    residual_dropout_p: float = 0.0
    residual_dropout_schedule: str = "constant"
    residual_dropout_delta: float = 0.0

    def __post_init__(self) -> None:
        self.validate_sparsedrop()

    def validate_sparsedrop(self) -> None:
        if self.residual_dropout_mode not in {"none", "attention", "ffn", "both"}:
            raise ValueError("residual_dropout_mode must be none, attention, ffn, or both")
        if self.residual_dropout_schedule not in {"constant", "increasing", "decreasing"}:
            raise ValueError("residual_dropout_schedule must be constant, increasing, or decreasing")
        p, delta = self.residual_dropout_p, self.residual_dropout_delta
        if not math.isfinite(p) or not 0 <= p < 1:
            raise ValueError("residual_dropout_p must be finite and in [0, 1)")
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("residual_dropout_delta must be finite and nonnegative")
        if self.residual_dropout_schedule == "constant":
            if delta != 0:
                raise ValueError("constant dropout requires residual_dropout_delta=0")
        elif not 0 <= p - delta <= p + delta < 1:
            raise ValueError("dropout schedule requires 0 <= p-delta <= p+delta < 1")
        if self.max_recurrent_steps < 1:
            raise ValueError("max_recurrent_steps must be positive")
        if self.residual_dropout_schedule != "constant" and self.max_recurrent_steps <= 1:
            raise ValueError("increasing/decreasing dropout requires more than one loop")

    def dropout_probability(self, loop_index: int, total_loops: int | None = None) -> float:
        """Drop probability for a 1-based loop index, with fixed mean across loops.

        The loop index is explicit so gradient checkpoint recomputation cannot
        accidentally read a mutable index left over from the last forward loop.
        """
        if total_loops is None:
            total_loops = self.max_recurrent_steps
        if total_loops < 1 or not 1 <= loop_index <= total_loops:
            raise ValueError("expected total_loops >= 1 and 1 <= loop_index <= total_loops")
        if self.residual_dropout_schedule == "constant":
            return self.residual_dropout_p
        if total_loops <= 1:
            raise ValueError("increasing/decreasing dropout requires more than one loop")
        offset = self.residual_dropout_delta * (2 * loop_index - total_loops - 1) / (total_loops - 1)
        if self.residual_dropout_schedule == "decreasing":
            offset = -offset
        return self.residual_dropout_p + offset

    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        # Each transformer layer:
        #   Attention: Q, K, V projections + output projection
        attn = 4 * self.hidden_size * self.hidden_size
        #   SwiGLU FFN: gate + up projections + down projection
        ffn = 3 * self.hidden_size * self.intermediate_size
        #   4 RMSNorm (sandwich: pre-attn, post-attn, pre-ffn, post-ffn) — each hidden_size params
        norms = 4 * self.hidden_size
        layer = attn + ffn + norms
        # LM head (tied with embedding by convention, so not counted separately)
        # Exit gate: hidden_size -> 1
        gate = self.hidden_size + 1
        return embed + self.num_layers * layer + gate + self.hidden_size  # final RMSNorm

    @classmethod
    def small(cls) -> "LoopLMConfig":
        """~100M parameter config for prototyping."""
        return cls(
            vocab_size=49152,
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            intermediate_size=2048,
            max_seq_len=2048,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )

    @classmethod
    def ouro_1_4b(cls) -> "LoopLMConfig":
        """1.4B parameter config (Table 2)."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            num_layers=24,
            num_heads=16,
            intermediate_size=5504,
            max_seq_len=4096,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )

    @classmethod
    def ouro_2_6b(cls) -> "LoopLMConfig":
        """2.6B parameter config — 48 layers, same hidden dim as 1.4B (Table 2)."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            num_layers=48,
            num_heads=16,
            intermediate_size=5504,
            max_seq_len=4096,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )
