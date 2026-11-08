# Method and measurements

## A small stack used several times

The model embeds the input tokens, applies a shared stack of six Transformer layers H times, and predicts the next token from the last loop. The same attention and FFN weights are used in every loop. Six layers and eight loops means 48 layer applications, not 48 separately parameterized layers.

SparseLoop training uses only the final loop's cross-entropy loss. It does not train an early-exit policy or use the original LoopLM auxiliary objective. Those separate upstream features remain available in the code.

## Where dropout is applied

Each attention or FFN branch follows this order:

```text
input -> RMSNorm -> attention or FFN -> RMSNorm -> dropout -> add residual
```

The three settings enable dropout on FFN outputs, attention outputs, or both. A new mask is sampled for each training call, including each reuse of a layer. Inverted dropout scales retained values by 1/(1-p). Evaluation turns dropout off and still executes the full dense model.

The dropout rate may be constant across loops, increase from early to late loops, or decrease. For H > 1, the increasing schedule is `p + delta * (2*t-H-1)/(H-1)`, where t starts at 1. The decreasing schedule subtracts the same term. Thus the mean rate remains p. The eight-loop study uses p=0.10 with delta=0.05 or 0.10.

## What the saved measurements mean

All layer measurements use the same 512 fixed validation tokens, with dropout off. They are recorded after a branch's output RMSNorm and before dropout and residual addition.

| Measurement | Meaning |
| --- | --- |
| Validation loss | Mean next-token cross-entropy; lower is better |
| Perplexity (PPL) | Exponential of loss; lower is better |
| Near-zero ratio | Fraction of branch output values with absolute value at most a fixed threshold |
| Output RMS | Typical size of the branch output values |
| Largest-10% share | Fraction of the squared output accounted for by the 10% largest absolute values, averaged over tokens |
| Update cosine | Similarity between the changes made by two loops, averaged over tokens |
| State cosine | Similarity between the outputs of two loops, averaged over tokens |
| Linear CKA | A centered comparison of token representations between loops |

The main near-zero threshold is 0.01; 0.001 and 0.1 are also saved. These values describe attention/FFN **branch outputs**, not FFN intermediate neurons or sparse weights. A small output may still matter to the prediction.

Loop diversity plots compare each loop with Loop 1 of the same model. Update cosine uses the change from a loop's input to its output. A negative value means opposing directions on average; it does not establish that one loop corrects another.

## What stays matched

Paired conditions use the same data, seed, tokenizer, batch, and learning-rate horizon. Data sampling has its own random generator, so changing dropout does not change the token order. Checkpoints save model, optimizer, sampler, and random states. An unchanged execution profile supports exact recovery within the tested environment.

Changing micro batch size can preserve 32 sequences per optimizer update while changing individual dropout masks and floating-point rounding. It is therefore not a claim of bitwise-identical training. Larger loop counts also use more computation at the same token budget; this study does not match FLOPs across loop counts.
