---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 8) - Decoder-Only Generation: Building GPT From Scratch and Fine-Tuning distilgpt2"
date: 2026-08-09 17:00:00 +0300
image: "/assets/img/nlp_week8_gpt.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    transformers,
    gpt,
    decoder-only,
    causal-attention,
    transfer-learning,
    language-modeling,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 8) - Decoder-Only Generation: Building GPT From Scratch and Fine-Tuning distilgpt2

This is Week 8 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Weeks 6 and 7 were both encoder work: bidirectional self-attention, `[CLS]` pooling, and models built to look at an entire sequence at once. Week 8 flips the direction. Every building block, `MultiHeadAttention`, `TransformerBlock`, the causal mask, already exists from Week 6, but until now the causal mask was one option among several. This week it stops being optional: a decoder-only model can never look forward, full stop, or it isn't a language model.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_8_GPT.ipynb`, supported by `utils.py`) is in the `week8-decoder/` folder.

---

## The Series at a Glance

| Phase                    | Weeks | Focus                           | Anchor Dataset |
| ------------------------ | ----- | ------------------------------- | -------------- |
| **Classical NLP**        | 1–2   | Vectorization, Tokenization     | IMDB           |
| **Deep Learning Bridge** | 3–5   | Embedding, RNNs, CNNs           | IMDB           |
| **Modern Transformers**  | 6–7   | Attention, BERT & SBERT         | IMDB           |
| **Modern Transformers**  | 8     | Decoder-Only Generation (GPT)   | WikiText-2     |
| **Modern Transformers**  | 9     | Encoder-Decoder Generation (T5) | OPUS Books     |

---

## Two Goals, One Notebook

This week's notebook has a deliberate split:

1. **Part A (pure PyTorch):** implement a tiny decoder-only Transformer, `TinyGPT`, and train it from random initialization on a self-contained toy corpus, to build intuition for the forward pass, causal masking, and next-token loss.
2. **Part B (Hugging Face):** fine-tune `distilgpt2`, a real pretrained decoder-only model, on a slice of WikiText-2, with Apple Metal Performance Shaders (MPS) , to see how differently a model that "already knows English" behaves compared to a model that is "learning English from 121 characters."

That contrast is this week's version of the theme Week 7 explored with BERT: architecture explains _how_ a model works, pretraining explains _why_ it works so much better in practice.

---

## Concept 1: A Decoder-Only Forward Pass, End to End

`TinyGPT` in `utils.py` follows the same high-level pattern as GPT:

$$
x_0 = \text{Embedding}(\text{input_ids}) + \text{PositionEmbedding}
$$

$$
x_l = \text{TransformerBlock}_l(x_{l-1}), \quad l = 1, \ldots, L
$$

$$
\text{logits} = \text{LMHead}(\text{LayerNorm}(x_L))
$$

Each `TransformerBlock` is pre-norm, exactly like Week 6: `x = x + Attention(LayerNorm(x))`, then `x = x + FeedForward(LayerNorm(x))`. What's new this week isn't the block, it's what the attention inside it is allowed to see.

### The Causal Mask Isn't Optional Anymore

`TinyGPT` builds one causal mask unconditionally on every forward pass:

```python
def build_causal_mask(query_len, key_len=None, device=None):
    key_len = key_len if key_len is not None else query_len
    diagonal_offset = key_len - query_len
    return torch.tril(
        torch.ones(query_len, key_len, dtype=torch.bool, device=device),
        diagonal=diagonal_offset,
    ).unsqueeze(0).unsqueeze(0)
```

The Position can attend to itself and everything before it, never after. The `diagonal_offset` term matters for a specific reason: in production autoregressive generation, a model doesn't recompute the whole sequence at every step, it caches previously computed keys and values and passes only the newest token forward, so `query_len=1` while `key_len` keeps growing. The offset is what makes that shape of mask still line up correctly. Worth being precise about what this notebook actually does with it: `evaluate_tiny_gpt` doesn't implement a KV cache. It re-slices the trailing `max_seq_len` tokens and runs a full forward pass at every generation step, so `query_len == key_len` throughout. The offset logic is real and it's why the causal mask _can_ support caching, it's just a code path this notebook doesn't exercise.

### Weight Tying

`TinyGPT`'s language-model head shares its weight matrix with the input embedding:

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.embedding.weight  # Weight Tying
```

The same matrix that maps token IDs to vectors also maps final hidden states back to vocabulary logits, one learned lookup table doing both jobs. For a 26-token character vocabulary this saves a comparatively small number of parameters; for a real subword vocabulary of 30–50K tokens the savings are substantial, since the embedding matrix is often one of the largest single components in a small model.

---

## Concept 2: Training TinyGPT From Nothing

The Part A dataset is intentionally tiny, three lines of text, 121 characters, 26-character vocabulary, so the whole loop finishes in seconds and every step is inspectable:

```
to be or not to be.
this is a tiny corpus for a tiny gpt.
we only want to demonstrate causal masking and next-token loss.
```

`TinyGPT` here has `d_model=128`, 2 layers, 4 heads, a 64-token context window, roughly **408K parameters**, small enough to train on CPU or MPS in under a minute for 1000 epochs:

```
epoch 1   | loss 81.4179
epoch 101 | loss 1.6524
epoch 201 | loss 1.1351
epoch 301 | loss 0.4998
epoch 401 | loss 0.2152
epoch 501 | loss 0.1668
epoch 601 | loss 0.0892
epoch 701 | loss 0.0377
epoch 801 | loss 0.0796
epoch 901 | loss 0.0388
```

Prompted with `"to be"`, the trained model generates:

```
to be or not to be.
this is a tiny corpus for a tiny gpt.
we only want to demonstrate causal masking and
```

That's not the model "understanding" anything, it's memorization, and that's expected: a 408K parameter model trained for 1000 epochs on 121 characters has more than enough capacity to reproduce its entire training set verbatim. The value of Part A isn't generation quality, it's confirming that the causal mask, the position embeddings, and the next-token loss are all wired together correctly for demonstration purposes only.

### Decoding With Temperature

`evaluate_tiny_gpt` samples one token at a time. Given logits $z_1, \ldots, z_V$ over the vocabulary and a temperature $T$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad x_{t+1} \sim \text{Categorical}(p)
$$

- $T = 1.0$: sample from the model's probabilities as-is.
- $T < 1.0$: sharpen the distribution, high-probability tokens get even more likely.
- $T \to 0$: converges to greedy decoding (always pick the argmax).
- $T > 1.0$: flatten the distribution, generation gets more random.

The generated text above used $T=1.0$, which is part of why it reproduces the training corpus so closely, the model's probabilities really are what peaked after memorizing 121 characters.

---

## Concept 3: What Pretraining Buys You, Measured in Parameters

Before touching the fine-tuning pipeline, it's worth putting the two models side by side:

```
TinyGPT (Part A, from scratch):
  d_model: 128, num_layers: 2, num_heads: 4, mlp_hidden_dim: 512
  vocab_size: 26 (character-level), max_seq_len: 64
  Total parameters: 408,320 (0.41M)
  Pretraining: none — random init, trained only on 121 characters

distilgpt2 (Part B, Hugging Face):
  d_model: 768, num_layers: 6, num_heads: 12
  vocab_size: 50,257 (GPT-2 byte-pair encoding)
  Total parameters: ~82,000,000 (82M)
  Pretraining: knowledge-distilled from GPT-2 (124M) on OpenWebText
```

distilgpt2 is roughly **200x** larger than `TinyGPT`, and unlike `TinyGPT`, it never starts from zero, it already encodes English syntax, common phrasing, and a huge amount of world knowledge before Part B ever runs a single fine-tuning step. That gap is the whole reason Part B's fine-tuning job can be so much lighter than Part A's training-from-scratch job and still produce far more fluent output.

---

## Concept 4: Fine-Tuning a Pretrained GPT on WikiText-2

Part B's pipeline has five stages: load WikiText-2 with the `datasets` SDK, tokenize with `distilgpt2`'s own byte-pair-encoding tokenizer, group tokenized text into fixed-length chunks, build a pretrained model plus `Trainer`, then train and generate.

**MPS-realistic defaults** keep the whole thing runnable without a dedicated GPU:

- `distilgpt2` instead of full GPT-2
- `max_seq_length=128` instead of a long context window
- a 5,000-example training slice and 500-example validation slice, not the full 36,718-row WikiText-2 train split
- `max_steps` instead of full epochs
- batch size 1, with gradient accumulation standing in for a larger effective batch

### Gradient Accumulation, Concretely

`per_device_train_batch_size=1` keeps memory usage low but makes each individual optimizer step noisy. `gradient_accumulation_steps=8` fixes that by summing gradients across 8 mini-batches before calling `optimizer.step()` once:

$$
\text{effective_batch_size} = \text{per_device_train_batch_size} \times \text{gradient_accumulation_steps} \times \text{num_devices}
$$

Here that's $1 \times 8 \times 1 = 8$ sequences per optimizer update, memory footprint close to batch size 1, optimization behavior closer to batch size 8.

### The Recipe

```python
training_args = TrainingArguments(
    max_steps=500,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=30,
    eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=100, save_total_limit=2,
    fp16=torch.cuda.is_available(),  # CPU/MPS stay full precision
)
```

`learning_rate=1e-5` is deliberately small, this is meant to nudge 82M pretrained parameters toward WikiText-2's style, not overwrite what they already encode. 500 optimizer steps at an effective batch size of 8 is 4,000 training examples seen, less than one pass over the 5,000-example slice.

### Generation After Fine-Tuning

Prompted with `"The meaning of life is"`:

```
The meaning of life is in the sense of a single individual who is "witted,
independent and free, or who does not possess the right to live, and who
does not have the right to live." The word was given in 1708 to the Roman
Catholic Church, which had no political influence over the Romans, and was
the subject of the Roman Catholic Church. The word was coined in 1710 to
describe what was to occur in Europe and the Americas. The Roman Catholic
Church did not become a religious organization until after
```

Worth being honest about this output rather than overselling it: it's grammatically coherent in a way `TinyGPT` could never manage from a 121-character corpus, but it's also meandering and repetitive ("the Roman Catholic Church" three times in five sentences), and the "facts" it states aren't reliable, they're the model pattern-matching WikiText-2's encyclopedic tone rather than retrieving anything true. That's consistent with the training recipe: 500 steps at `lr=1e-5` is a light nudge on top of a small (82M-parameter) base model, not remotely enough to fix distilgpt2's well-documented tendency to hallucinate specific-sounding dates and claims. The fluency is coming almost entirely from pretraining; the fine-tuning is steering style, not injecting reliable knowledge.

---

## TinyGPT vs. distilgpt2: What Architecture Buys You vs. What Pretraining Buys You

- **`TinyGPT`** proves the architecture works: causal masking, position embeddings, weight tying, and next-token loss all compose correctly, verified by watching a 408K-parameter model memorize 121 characters in under a thousand epochs.
- **distilgpt2** shows what 82M parameters and pretraining on a much larger corpus buys on top of that same architecture: fluent, grammatically sound multi-sentence generation after a training budget (500 steps, batch size 8, `lr=1e-5`) that would do essentially nothing for a model starting from random weights.
- **Weight tying** matters more as vocabulary grows; it's nearly invisible on a 26-token character vocabulary and substantial on distilgpt2's 50K-token BPE vocabulary.
- **The causal mask's diagonal offset** is crucial for KV-cache, although `evaluate_tiny_gpt` isn't using that capability.

---

## What We Learned

- Causal masking is a hard requirement for decoder-only generation: `TinyGPT` builds a fresh causal mask on every forward pass, with no code path that skips it.
- A small model with enough capacity and enough epochs will memorize a small corpus almost verbatim, that's expected behavior for Part A's toy setup, not evidence of "understanding," and the loss curve (81.4 → 0.04 over 1000 epochs) confirms straightforward overfitting to 121 characters.
- Temperature controls how peaked the sampling distribution is, not whether the model is right; `T=1.0` on an overfit model reproduces its training data closely simply because the model's own probabilities are that concentrated.
- Weight tying and gradient accumulation are both "free" wins for constrained compute: tying halves a meaningful chunk of parameter count on large vocabularies, and accumulation lets a batch-size-1 run behave like a larger batch without the memory cost.
- Pretraining, not architecture, explains almost all of the fluency gap between `TinyGPT`'s output and distilgpt2's output; distilgpt2's fine-tuning recipe here was intentionally light (500 steps, `lr=1e-5`) precisely because the heavy lifting was already done by pretraining, not this notebook's training loop.
- Fluent generation is not the same as accurate generation: distilgpt2's post-fine-tuning output reads smoothly but states specific "facts" with no grounding, worth flagging explicitly rather than letting fluency stand in for correctness.

---

## What's Next

Weeks 6 through 8 covered attention, encoder-only pretraining, and decoder-only generation. **Week 9** closes the architecture arc with encoder-decoder models: T5, trained on the OPUS Books dataset, where an encoder and a decoder work together instead of either one working alone.

> **Code:** This week's notebook is `Week_8_GPT.ipynb` in [`week8-decoder/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week8-decoder), supported by `utils.py`.
