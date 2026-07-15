---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 6) — Attention and Position: Letting Every Word Look Directly at Every Other Word"
date: 2026-07-26 17:00:00 +0300
image: "/assets/img/nlp_week6_transformer_attention.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    transformers,
    attention,
    rope,
    positional-encoding,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 6) — Attention and Position: Letting Every Word Look Directly at Every Other Word

This is Week 6 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 4 built four sequence models that read a review one word at a time. Week 5 flipped that around with a `TextCNN` that slides small fixed-width filters across the sentence instead of reading it in order. Both approaches share a limitation: an RNN's memory is a single running summary, and a CNN's filter only ever looks inside its own fixed window. Neither has a way to let a token look directly at any other token in the sentence and decide on the fly how much it matters. Week 6 introduces the mechanism built specifically for that: **attention**.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_6_transformer_attention.ipynb`, supported by `utils.py`) is in the `week6-transformer/` folder.

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

## Why Attention Is Worth a Whole Different Mechanism

An RNN's hidden state is a bottleneck: everything the model has read so far gets compressed into one fixed-size vector, updated token by token. A CNN's filter is a different kind of bottleneck: it can only ever see the `k` tokens inside its own window, no matter how many filters you stack side by side. Neither architecture gives a token a direct, unconstrained line of sight to any other token in the sentence.

**Self-attention** removes that constraint entirely. For every token, it computes a similarity score against every other token in the sequence, turns those scores into a probability distribution with softmax, and uses that distribution to build a weighted average of everyone else's content. No fixed window, no compressed running summary — just a direct, learned "how much should I care about you" score between every pair of tokens.

That flexibility comes at a cost though: attention on its own has no idea what order the tokens came in. A sentence and its scrambled anagram look identical to raw self-attention. That's the second big theme of this week's notebook — how do you give a permutation-invariant mechanism a sense of position?

---

## Step 1: Reusing Week 3's Artifacts, Again

Like Weeks 4 and 5, this notebook doesn't touch tokenization or vocabulary. It reloads the same train/val/test splits, the same byte-level BPE vocabulary and merge rules, and the same pretrained DAN embedding table from Week 3:

```python
train_ds = torch.load("../week3-embedding/datasets/train_ds.pt", weights_only=False)
val_ds = torch.load("../week3-embedding/datasets/val_ds.pt", weights_only=False)
test_ds = torch.load("../week3-embedding/datasets/test_ds.pt", weights_only=False)

bpe_vocab = load_json("../week3-embedding/datasets/bpe_vocab.json")
bpe_merges = load_json("../week3-embedding/datasets/bpe_merges.json")
vocab_config = load_json("../week3-embedding/models/vocab_config.json")
```

```
train: 40000 | val: 5000 | test: 5000
```

The same 384-dimensional DAN embedding matrix that fed the RNN family and the TextCNN feeds every attention demo this week too, so any difference in what these models learn comes from the architecture, not from a different starting point. For the cross-attention section, we also hand-write a tiny IMDb-style paired-text probe — a review snippet plus a question that attends back into it — encoded with the same BPE vocabulary so the token IDs line up with the same embedding space.

---

## Step 2: Self-Attention as a Learned Weighted Average

Scaled dot-product self-attention is:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

`Q`, `K`, and `V` all come from the same token sequence in self-attention. The core implementation in `utils.py` is `scaled_dot_product_attention`:

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if attention_mask is not None:
        weights = weights.masked_fill(~attention_mask, 0.0)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)

    output = torch.matmul(weights, value)
    return output, weights
```

Two details worth calling out:

- **Masking happens twice.** Masked positions are first filled with the dtype's minimum value before softmax, so they get essentially zero probability. Then the mask is applied a _second_ time after softmax, explicitly zeroing anything masked. This defensive second pass matters most for a fully-masked row (every key position padded) — without it, floating-point softmax over an all-`-inf`-ish row can produce tiny non-zero values instead of a clean zero. With it, a fully masked row of weights is guaranteed to sum to exactly zero rather than leak noise into the output.
- **A diagonal self-attention weight is not automatically `1`.** Scaling divides every score in a row by the same `sqrt(d)` factor, and softmax normalizes a token's self-score together with every other score in that row. A token only attends mostly to itself if its self-similarity is genuinely much larger than its similarity to everything else.

The first demo in the notebook runs this function directly — no learned `Q`/`K`/`V` projections yet, `Q = K = V =` the raw Week 3 DAN embeddings — specifically so the resulting heatmap shows the mechanics of scaling and masking without a trained projection muddying the picture. It's a useful sanity check, but it's raw embedding similarity, not a trained attention pattern.

---

## Step 3: Position Encoding — The Part Attention Is Missing

Self-attention is **permutation-equivariant**: reorder the input tokens and apply the same reordering to the output, and the computation doesn't change. Without extra information, token order is invisible to the model — which is exactly why negation scope, phrase structure, and long-range dependencies need some other signal injected. This week's notebook builds and contrasts two ways to do that.

### Traditional Sinusoidal Positional Encoding

The original Transformer adds a deterministic position vector to the token embedding once, right after the embedding layer:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right), \qquad
PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
$$

```python
def sinusoidal_position_encoding(
    seq_len: int,
    d_model: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal position encoding.")

    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

Early embedding dimensions oscillate quickly across positions; later ones oscillate much more slowly, since the `10000^{2i/d_model}` denominator grows with `i`. The result is a bank of waves at different frequencies — some capture fine local shifts, others capture broad position patterns. Because the whole table is deterministic, it can be precomputed once for a chosen max sequence length and reused across every batch. That's cheap, but it's also a hard ceiling: positions beyond that cached length simply don't exist unless the cache is rebuilt. And because this signal is injected only once, right at the input, it has to survive indirectly through every later block's residual stream — it can become diluted by the time it reaches the last layer.

### Rotary Positional Encoding (RoPE)

RoPE takes a completely different approach: instead of adding a position vector once at the input, it **rotates** pairs of dimensions inside the query and key vectors, inside every attention computation, at every layer. For frequency index `i` at position `p`:

$$
\theta_{p,i} = p \cdot 10000^{-2i / d}
$$

$$
\begin{bmatrix} x_{2i}^{\prime} \\ x_{2i+1}^{\prime} \end{bmatrix} =
\begin{bmatrix} \cos \theta_{p,i} & -\sin \theta_{p,i} \\ \sin \theta_{p,i} & \cos \theta_{p,i} \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

Rather than materialize an explicit 2×2 rotation matrix per pair, `utils.py` implements the equivalent element-wise identity:

$$
\mathrm{RoPE}(x) = x \odot \cos(\theta) + \mathrm{rotate\_half}(x) \odot \sin(\theta)
$$

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even to apply rotary embeddings.")

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[..., :seq_len, :].to(dtype=x.dtype, device=x.device)
    sin = sin[..., :seq_len, :].to(dtype=x.dtype, device=x.device)
    return (x * cos) + (rotate_half(x) * sin)
```

`rotate_half` swaps each even/odd coordinate pair and flips one sign, so that multiplying by `cos` and `sin` and adding the two together reproduces the 2×2 rotation exactly, one pair at a time, without ever building the matrix. Only `Q` and `K` get rotated. `V` is left untouched, since `V` is the content being aggregated, while position should only affect _which_ tokens attend to which through `QK^T`.

The key property this buys: the dot product between a rotated query and a rotated key depends only on their **relative offset**, not their absolute positions —

$$
\mathrm{RoPE}(q, p) \cdot \mathrm{RoPE}(k, m) = f(q, k, p - m)
$$

and as that offset grows, the rotations drift further out of alignment, which gives the model a soft, built-in bias toward attending more strongly to nearby tokens. That's a meaningfully different failure mode than sinusoidal PE's hard `max_seq_len` ceiling, which is part of why RoPE (and its descendants like NTK-aware scaling, YaRN, and LongRoPE) is the standard choice in most modern LLM architectures — though it's worth being precise that RoPE's better long-range behavior doesn't mean it "solves" long context on its own; it just gives the model better _tools_ for distance-aware attention, and whether the model exploits them well still comes down to training.

> **Note on RoPE Efficiency:** In our teaching implementation, `build_rope_cache` is called dynamically on every forward pass inside `MultiHeadAttention`. While functionally correct, this is computationally wasteful. In production models, RoPE caches are also precomputed once (like sinusoidal PE) into a buffer sized to `max_seq_len` and then sliced per batch, rather than being freshly generated per-block at every step.

---

## Step 4: Assembling a Multi-Head Attention Block

A single attention computation only lets the model learn one notion of "relevance" at a time. **Multi-head attention** splits `Q`, `K`, and `V` into several smaller subspaces, runs scaled dot-product attention independently in each, and concatenates the results back together — letting different heads specialize in different kinds of relationships (one might track negation scope, another local phrase structure, and so on).

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, use_rope=False, is_causal=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.is_causal = is_causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None, context=None, context_mask=None, return_attention=False):
        context = x if context is None else context
        key_mask = attention_mask if context_mask is None and context is x else context_mask

        query = _split_heads(self.q_proj(x), self.num_heads)
        key = _split_heads(self.k_proj(context), self.num_heads)
        value = _split_heads(self.v_proj(context), self.num_heads)

        if self.use_rope:
            q_cos, q_sin = build_rope_cache(query.size(-2), self.head_dim, device=query.device)
            k_cos, k_sin = build_rope_cache(key.size(-2), self.head_dim, device=key.device)
            query = apply_rope(query, q_cos, q_sin)
            key = apply_rope(key, k_cos, k_sin)

        causal_mask = build_causal_mask(query.size(-2), key.size(-2), device=query.device) if self.is_causal else None
        combined_mask = combine_attention_masks(padding_mask=key_mask, causal_mask=causal_mask)

        attended, attention_weights = scaled_dot_product_attention(
            query, key, value, attention_mask=combined_mask, dropout_p=self.dropout, training=self.training
        )
        output = self.out_proj(_merge_heads(attended))
        return (output, attention_weights) if return_attention else output
```

The same module handles both self-attention and cross-attention: when `context` isn't passed, it defaults to `x` and reuses `attention_mask` as the key-side padding mask; when a separate `context` and `context_mask` are passed, keys and values come from that other sequence entirely.

This wraps into a standard **pre-norm Transformer block** — LayerNorm before each sublayer, with the sublayer's output added back onto the residual stream:

- `x = x + self_attention(layer_norm(x))`
- `x = x + cross_attention(layer_norm(x), context)` _(optional)_
- `x = x + feed_forward(layer_norm(x))`

Pre-norm is the common choice in modern implementations because it tends to make deep Transformer stacks easier to optimize than the original post-norm design. The feed-forward sublayer uses `GELU` rather than `ReLU` — unlike ReLU's hard cutoff at zero, GELU lets mildly negative values survive with a small magnitude, which is one reason it's the default activation in most modern Transformers.

---

## Step 5: Cross-Attention on a Tiny Paired-Text Probe

IMDb is naturally single-sequence: one review, one label. Cross-attention needs two sequences with different roles — one that provides evidence, and one that asks what to focus on. So the notebook hand-writes three small paired records that keep the IMDb flavor while adding a query sequence, encoded with the same BPE vocabulary as everything else:

```python
{
    "title": "Borrowed Summer",
    "sentiment": "mixed",
    "context": "review: Borrowed Summer is not a disaster; the cast is charming, "
               "although the script repeats scenes and loses energy.",
    "query": "which phrase softens the negative review ?",
}
```

Running this through the reusable `TransformerBlock` in cross-attention mode (`enable_cross_attention=True`), the query sequence's self-attention is set to `is_causal=True` — like a decoder — while the cross-attention sublayer itself is _not_ causal: every query token can look across the full encoded review, and RoPE is deliberately left out of the cross-attention step, since mixing decoder-side query positions with encoder-side context positions doesn't have a clean interpretation here. Position is handled entirely inside the query's own self-attention path instead.

---

## Step 6: The IMDb Mini-Project — Absolute vs. RoPE

With attention, position encoding, and the block assembled, the notebook reuses the same components for two compact IMDb Transformer encoder classifiers, both initialized from the same Week 3 DAN embedding matrix so `d_model = 384`:

- **Model A** — traditional sinusoidal positional encoding, added once after the embedding layer
- **Model B** — rotary positional encoding, applied inside every attention layer

Same subset, same split, same hyperparameters, same embedding initialization — the only intentional difference is _where_ position information enters the model.

```
train / val / test sizes: 40000 5000 5000
max_seq_len from vocab config: 128
d_model from DAN embedding: 384
mlp_hidden_dim: 1536
```

Both models use 4 layers, 4 attention heads, and the note in the code is worth repeating here too: with limited compute and time, this is intentionally a _small_ Transformer, so raw accuracy isn't the point — the comparison between positional strategies is.

```
absolute | Epoch 01 train_acc=0.6622 val_acc=0.6848
absolute | Epoch 05 train_acc=0.7008 val_acc=0.6886
absolute | Epoch 09 train_acc=0.7212 val_acc=0.6938 (early stop)

rope     | Epoch 01 train_acc=0.6903 val_acc=0.6900
rope     | Epoch 05 train_acc=0.7275 val_acc=0.7082
rope     | Epoch 10 train_acc=0.7701 val_acc=0.7156
```

### Test Results

| Model                          | Test Loss  | Test Acc  |
| ------------------------------ | ---------- | --------- |
| Absolute (sinusoidal)          | 0.5570     | 70.7%     |
| RoPE                           | **0.5252** | **72.8%** |
| _(reference)_ TextCNN — Week 5 | 0.4743     | 76.7%     |

**RoPE outperformed absolute sinusoidal encoding in this setup** — both lower test loss and higher test accuracy. That doesn't mean RoPE is universally better; it means that for this small encoder classifier, trained from DAN embeddings on IMDb-length sequences, RoPE's relative-position information was easier for the model to exploit than absolute's additive signal. Neither Transformer variant caught up to Week 5's TextCNN, which lines up with something worth sitting with: IMDb sentiment is often carried by strong, self-contained local phrases (`"not bad"`, `"waste of time"`), and a small, undertrained attention model doesn't automatically have an edge over a model that's structurally built to detect exactly that kind of local cue.

---

## Step 7: The Hard-Review Probe

The same handwritten-probe idea from Weeks 4 and 5 shows up again — six short reviews targeting negation, contrast, faint praise, and a late sentiment reversal:

| Category        | Example                                                                   | Label | Absolute  | RoPE      |
| --------------- | ------------------------------------------------------------------------- | ----- | --------- | --------- |
| negation        | "The film is not good; the few funny moments cannot save the dull story." | 0     | ✅ (0.43) | ✅ (0.05) |
| negation        | "The film is not bad at all, with a surprisingly warm final act."         | 1     | ❌ (0.02) | ❌ (0.01) |
| contrast        | "The acting is excellent, but the movie becomes tedious and empty..."     | 0     | ❌ (0.56) | ❌ (0.99) |
| contrast        | "The opening is clumsy, but the story grows into a moving drama."         | 1     | ✅ (0.79) | ✅ (0.92) |
| faint_praise    | "It is beautifully shot... yet somehow lifeless, cold, and forgettable."  | 0     | ❌ (0.99) | ❌ (0.95) |
| sentiment_shift | "I expected to hate it after the first act, but... completely won over."  | 1     | ✅ (0.56) | ✅ (0.73) |

**Overall: both models land at exactly 3/6 (50%).** The failures are shared, not RoPE-specific or absolute-specific: both models get tripped up by `"not bad at all"` (a negation that flips to positive rather than confirming a negative) and by the excellent-acting-but-tedious-movie contrast sentence. That's a useful reminder that this is a limitation of a small, lightly-trained encoder classifier in general — not a referendum on rotary vs. sinusoidal position encoding specifically. A balanced 50/50 IMDb training set may simply not contain enough targeted examples of faint praise, contrastive reversal, or late sentiment shifts for either model to learn those patterns reliably.

---

## Position Encoding vs. Attention: What Each One Is Actually For

- **Attention** decides _who to listen to_. It's the mechanism that lets any token build a direct, learned, weighted summary of any other token, with no fixed window and no compressed running memory in the way.
- **Position encoding** decides _whether "who" even has a meaningful order to begin with_. Without it, attention is blind to sequence order entirely — negation scope, phrase structure, and relative distance all become invisible.
- **Sinusoidal PE** injects position additively, once, at the input — cheap and precomputable, but capped at a fixed max length and diluted across depth.
- **RoPE** injects position multiplicatively, inside every attention computation, at every layer — no hard length cap on the encoding itself, and a built-in bias toward nearby tokens, at the cost of being architecturally coupled to the attention mechanism rather than a simple add-on.
- **Cross-attention** is the same underlying mechanism as self-attention, just pointed at a second sequence instead of itself — the tool that lets a query decide which parts of a _different_ piece of context matter.

---

## What We Learned

- Self-attention removes the fixed-window and compressed-memory bottlenecks that limited the CNN and RNN family, at the cost of being completely blind to token order on its own.
- Sinusoidal positional encoding and RoPE solve the same problem — giving attention a sense of order — in structurally different ways: one additive and injected once, the other multiplicative and reapplied at every layer.
- In this small IMDb encoder classifier, RoPE (72.8% test accuracy) outperformed absolute sinusoidal encoding (70.7%), though neither caught up to Week 5's TextCNN (76.7%) — a reminder that a small, lightly-trained Transformer doesn't automatically beat an architecture that's structurally suited to the task's dominant signal.
- Cross-attention is self-attention pointed at a second sequence — same masking, scaling, and softmax machinery, just with `K` and `V` coming from somewhere other than the query's own tokens.
- The hard-review probe showed a limitation shared by both positional strategies (50% on both), not a RoPE-specific or absolute-specific weakness — a useful check against over-attributing a small model's mistakes to one architectural choice.

---

## What's Next

Attention and position encoding are the raw ingredients; **Week 7** assembles them into full pretrained architectures — `BERT` and `SBERT` — and looks at what changes when a Transformer is trained at scale on masked language modeling instead of from a small, task-specific setup like this week's demo.

> **Code:** This week's notebook is `Week_6_transformer_attention.ipynb` in [`week6-transformer/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week6-transformer), supported by `utils.py`.
