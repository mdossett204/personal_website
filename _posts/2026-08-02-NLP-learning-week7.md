---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 7) - Transfer Learning: BERT, Sentence-BERT, and Why Pre-Training Changes Everything"
date: 2026-08-02 17:00:00 +0300
image: "/assets/img/nlp_week7_bert_sbert.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    transformers,
    bert,
    sbert,
    transfer-learning,
    sentence-embeddings,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 7) - Transfer Learning: BERT, Sentence-BERT, and Why Pre-Training Changes Everything

This is Week 7 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 6 built `TransformerBlock`, `MultiHeadAttention`, and two positional encoding strategies from scratch, then trained a small IMDb classifier from random initialization. Everything in that model: every weight, every representation, started from nothing and learned only from the ~40,000 IMDb reviews we gave it. Week 7 asks the obvious next question: what happens if a model doesn't start from nothing? We stack our own blocks into a full encoder, then bring in Bidirectional Encoder Representations from Transformers (BERT), a model with the same architecture, except pre-trained on 3.3 billion words before it ever saw a single IMDb review and measure exactly how much that head start is worth.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_7_encoder.ipynb`, supported by `utils.py`) is in the `week7-BERT/` folder.

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

## What Is Depth Actually Doing?

Week 6 already stacked four `TransformerBlock`s: 4 layers, 4 attention heads for both its sinusoidal and RoPE classifiers. So the stacking itself isn't new this week. What's new is packaging that same pattern into a reusable `TinyBERT` encoder class, and asking a question Week 6 never measured directly: does stacking those blocks actually give you a deeper, more differentiated understanding of the input at each layer, or does depth alone buy you very little without training? Concept 1 of this week's notebook answers that directly, with a tool built for exactly this question: **Centered Kernel Alignment (CKA)**.

Once we've built the stacked encoder ourselves, the rest of the week swaps our own from-scratch weights for a checkpoint that's already been pre-trained at scale BERT and spends the remaining concepts on what that buys you: architecture inspection, fine-tuning, and finally Sentence-BERT (SBERT), a variant purpose-built for a task standard BERT is structurally bad at.

---

## Concept 1: Probing a Stacked Encoder with CKA

Stacking is literal: each block takes the previous block's output as its new input, so the encoder pipeline (with absolute position encoding) is just:

$$
x_0 = \text{Embedding}(\text{input_ids}) + \text{PositionalEncoding}
$$

$$
x_l = \text{TransformerBlock}_l(x_{l-1}), \quad l = 1, \ldots, L
$$

$$
h = \text{LayerNorm}(x_L)
$$

`TinyBERT` in `utils.py` does exactly this: four `TransformerBlock`s stacked, using absolute positions instead of RoPE. The interesting part isn't building it; it's looking _inside_ it.

### Measuring Layer Similarity with CKA

Linear CKA compares two representation matrices $X$ and $Y$ (each `[n_examples, d]`):

$$
\text{CKA}(X, Y) = \frac{\lVert X^T Y \rVert_F^2}{\lVert X^T X \rVert_F \cdot \lVert Y^T Y \rVert_F}
$$

It ranges from 0 (completely different representations) to 1 (identical linear transformation), a useful diagnostic for whether later layers are building on earlier ones or just repeating the same computation.

```python
tiny_bert = create_bert(
    vocab_size=vocab_size, d_model=embedding_dim, num_heads=4,
    mlp_hidden_dim=4 * embedding_dim, num_layers=4, pad_id=pad_id,
    use_rope=True, use_absolute_positions=False,
    pretrained_embedding_weight=embedding_weight.detach().cpu(),
).to(device)

layer_reps = []
with torch.no_grad():
    x = tiny_bert.dropout(tiny_bert.embedding(probe_ids))
    for block in tiny_bert.blocks:
        x = block(x=x, attention_mask=probe_masks)
        layer_reps.append(masked_mean_pool(x, probe_masks).detach().cpu())

cka_matrix = torch.zeros(4, 4)
for i in range(4):
    for j in range(4):
        cka_matrix[i, j] = linear_cka(layer_reps[i], layer_reps[j])
```

`TinyBERT` here is completely untrained random block weights, a frozen embedding table carried over from Week 3. The natural guess is that random weights should produce a noisy, unpredictable CKA matrix. That's not what happens:

```
CKA matrix (row i, col j = similarity between layer i and layer j):
tensor([[1.0000, 0.9840, 0.9700, 0.9520],
        [0.9840, 1.0000, 0.9880, 0.9730],
        [0.9700, 0.9880, 1.0000, 0.9880],
        [0.9520, 0.9730, 0.9880, 1.0000]])
```

Every pair of layers, including the first and the last, scores above 0.95. That's not noise, and it's not a bug: it's the residual connection. Every `TransformerBlock` computes `x = x + block_output`, and with random, untrained weights, each block's fresh contribution is small relative to the residual connection it's added to. The original embedding-derived signal dominates all the way through, by construction, regardless of training. The takeaway: a flat, uniformly-high CKA matrix in an untrained encoder isn't evidence of anything interesting happening (or not happening) in the blocks themselves, it's what skip connections guarantee _before_ any learning reshapes them.

### The Trained Comparison: Where the Staircase Actually Shows Up

The obvious follow-up: if training is what's supposed to break that flat pattern, does it actually? We ran the same CKA analysis on the `[CLS]` token across all 13 hidden states (embedding + 12 layers) of a real pre-trained `bert-base-uncased`, decoding our BPE-tokenized IMDb batch back to text so BERT's own tokenizer could process it:

```python
bert_inputs = bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    bert_outputs = bert_model(**bert_inputs)

trained_cka_matrix = torch.zeros((13, 13))
for i in range(13):
    for j in range(13):
        rep_i = bert_outputs.hidden_states[i][:, 0, :]
        rep_j = bert_outputs.hidden_states[j][:, 0, :]
        trained_cka_matrix[i, j] = linear_cka(rep_i, rep_j)
```

The result is the staircase pattern the untrained encoder didn't show: nearby layers stay highly similar to each other, but similarity drops off the further apart two layers are, and the embedding layer looks the least like the final layer of anything in the stack. After 3.3 billion words of pre-training, the blocks have learned to make transformations substantial enough to actually pull representations away from that shared residual connection, layer by layer, exactly what "the encoder is doing real work" should look like, and exactly what was invisible in the untrained version.

---

## Concept 2: Inside a Pre-Trained BERT

Every architectural piece BERT uses is something Week 6 already built:

- **Bidirectional self-attention**: the same `MultiHeadAttention` with `is_causal=False`.
- **`[CLS]` / `[SEP]`**: `[CLS]` accumulates a sequence-level representation; `[SEP]` separates paired segments.
- **Token type embeddings**: an extra learned 0/1 embedding marking which segment a token belongs to.
- **Learned absolute position embeddings**: the same design as `TinyTransformerClassifier`'s `use_absolute_positions=True` path, except BERT learns its position table instead of using the fixed sinusoidal formula.

```
Total parameters: 109,482,240 (109.5M)
Number of encoder layers: 12
Number of attention heads: 12
Hidden size (embedding_dim): 768
Intermediate FFN size: 3072
Vocab size: 30522
Max position embeddings: 512

TinyBert for comparison:
  embedding_dim: 384, num_layers: 4, num_heads: 4
  Total parameters: 7,325,184 (7.33M)
```

BERT-base is roughly 15x larger than our `TinyBERT`, before even accounting for the training data gap.

### Watching Masked Language Modeling in Action

BERT's main pre-training objective, Masked Language Modeling (MLM), masks 15% of input tokens and asks the model to reconstruct them from bidirectional context:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x})
$$

Instead of re-running the pre-training, we watch the resulting objective in action with a `fill-mask` pipeline:

```
Sentence: The film was absolutely [MASK], I could not stop watching.
  [0.191] beautiful       → the film was absolutely beautiful, i could not stop watching.
  [0.110] incredible      → the film was absolutely incredible, i could not stop watching.
  [0.095] stunning        → the film was absolutely stunning, i could not stop watching.

Sentence: The acting was [MASK] but the plot was predictable and dull.
  [0.594] good            → the acting was good but the plot was predictable and dull.
  [0.096] excellent       → the acting was excellent but the plot was predictable and dull.

Sentence: I would [MASK] this movie to anyone who enjoys slow-burn thrillers.
  [0.926] recommend       → i would recommend this movie to anyone who enjoys slow-burn thrillers.
```

Every top prediction is plausible, fluent, and grammatically consistent with both the left and right context, the direct payoff of bidirectional attention that a causal, left-to-right model doesn't get.

BERT's secondary objective, Next Sentence Prediction (NSP), teaches the `[CLS]` token to represent whether one segment plausibly follows another:

$$
\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid h_{[\text{CLS}]})
$$

Later work (RoBERTa, ALBERT) found dropping NSP often _helps_ improve performance, the randomly sampled negative pairs are trivially identifiable from topic mismatch alone, so NSP may add more noise than signal. It's still worth understanding, though, because it's the reason `[CLS]` and `token_type_ids` exist throughout the Hugging Face API.

### Early vs. Late Layers Quantified

Cosine similarity between the `[CLS]` vector at layer 1 and layer 12, for three IMDb-style sentences:

```
The cinematography was breathtaking and the performances were superb...     0.204
A tedious and poorly written film that wasted two hours of my life...       0.163
Somewhere between brilliant and baffling, this film defies easy categori... 0.199
```

All three sit around 0.16–0.20, low similarity between where a sentence starts and where it ends up after 12 layers of bidirectional attention. That's the same "real work" conclusion the CKA staircase pointed to, from a second, independent angle.

---

## Concept 3: Fine-Tuning BERT on IMDb

Fine-tuning takes those pre-trained weights and continues training on a small labeled dataset, at a low learning rate (`2e-5` here) so the adjustment steers existing knowledge rather than overwriting it. For classification: prepend `[CLS]`, run all 12 layers, take `[CLS]`'s final hidden state, feed it through a small linear head.

```python
bert_ft = BertForSequenceClassification.from_pretrained(bert_name, num_labels=2)

training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
trainer = Trainer(model=bert_ft, args=training_args,
                   train_dataset=tokenized_train, eval_dataset=tokenized_val,
                   compute_metrics=compute_metrics)
```

Fine-tuned on 5,000 IMDb examples and evaluated on 1,000 held-out examples, against the Week 6 `TinyTransformerClassifier` (RoPE variant), evaluated on the full IMDb test set it was originally validated on:

| Model                                                   | Accuracy  |
| ------------------------------------------------------- | --------- |
| Week 6 `TinyTransformerClassifier` (RoPE), from scratch | 72.8%     |
| BERT-base, fine-tuned                                   | **91.6%** |

A nearly 19-point accuracy gap, and BERT reached it in far fewer epochs on far less labeled data than Week 6's model saw during its own training. The gap is attributable to two compounding ones: roughly 15x the parameters (109.5M vs. 7.3M), _and_ 3.3 billion words of bidirectional pre-training before either model saw a single IMDb label. Neither factor alone accounts for the whole gap; fine-tuning is effectively steering knowledge the model already has, rather than building it from nothing.

One limitation worth flagging before moving on: BERT's `[CLS]` representation was pre-trained for NSP, a binary, cross-sentence task, not for open-ended semantic similarity. Naively comparing two independently-encoded `[CLS]` vectors for similarity doesn't work well. That gap is exactly what Concept 4 addresses.

---

## Concept 4: Sentence-BERT and the Problem It Solves

Suppose you want to find which of 10,000 sentences is most similar to a query. Standard BERT cross-attends between two sequences _inside_ the forward pass:

$$
\text{score}(q, c_i) = f_{\text{BERT}}([\text{CLS}], q, [\text{SEP}], c_i, [\text{SEP}])
$$

That means representations can't be pre-computed or cached, every (query, candidate) pair needs its own full forward pass. 10,000 candidates means 10,000 forward passes at query time; a million candidates makes this approach infeasible.

**SBERT** (Reimers & Gurevych, 2019) fixes this with a Siamese network: two copies of the same model, shared weights, each sentence encoded _independently_ into a fixed-size vector via mean pooling:

$$
u = \text{Pool}(\text{BERT}(s_A)), \qquad v = \text{Pool}(\text{BERT}(s_B)), \qquad \text{sim}(s_A, s_B) = \cos(u, v)
$$

Because each sentence is encoded independently, the entire candidate corpus can be pre-computed once; retrieval at query time is a single cosine similarity operation. The pooling step is exactly `masked_mean_pool` from `utils.py`, the same function the Week 3 DAN baseline and Week 6's classifier both use, just now pooling over 12 layers of bidirectional Transformer context instead of a single embedding lookup.

```python
siamese_bert = create_sbert(bert_model)   # wraps any BERT-shaped model in mean-pooling + cosine similarity

sim = siamese_bert(enc_a["input_ids"], enc_a["attention_mask"],
                    enc_b["input_ids"], enc_b["attention_mask"])
```

```
[0.806]  A: The film was a masterpiece of modern cinema.
         B: This movie is one of the greatest films ever made.

[0.555]  A: The film was a masterpiece of modern cinema.
         B: The food at the restaurant was disappointing and cold.

[0.779]  A: I loved the performances, especially the lead actor.
         B: The acting in this film was wonderful, particularly the main character.
```

Even without any SBERT-specific fine-tuning (this is plain `bert-base-uncased` wrapped in mean pooling), semantically related sentences score noticeably higher than unrelated ones: 0.806 and 0.779 for genuinely similar review pairs, versus 0.555 for a review paired with an unrelated restaurant complaint.

---

## Concept 5: Semantic Search: SBERT vs. the Week 3 DAN Baseline

A semantic search system separates cleanly into an offline indexing phase (encode every candidate once) and an online retrieval phase (encode the query, find nearest neighbors by cosine similarity). Because SBERT encodes independently, indexing is $O(n)$ up front. Retrieval against $n$ candidates is $O(n \cdot d)$ per query, where $d$ is the embedding dimensionality (384 for `all-MiniLM-L6-v2`), each cosine similarity comparison is a dot product over all $d$ dimensions, so comparing the query against every one of the $n$ candidates costs $n$ dot products of size $d$. (An approximate index like FAISS trades a small amount of accuracy for $O(\log n)$ retrieval instead.) That per-query cost is a direct structural payoff from Concept 4, SBERT's independent encoding is what makes this pre-computable in the first place.

We compared a proper fine-tuned SBERT model (`all-MiniLM-L6-v2`, 6 layers, fine-tuned on ~1B sentence pairs) against the Week 3 DAN baseline, mean-pooled, order-agnostic, non-contextual embeddings on the same retrieval task:

```
Query: 'Bad movie with terrible writing and poor effects.'
  SBERT top-3:
    [0.560] Cheap special effects and a lazy script make this a forgettable blockbuster.
    [0.481] Despite strong performances, the film struggles to find emotional resonance.
    [0.454] The plot holes are glaring and the pacing is painfully slow.
  DAN baseline top-3:
    [0.435] Cheap special effects and a lazy script make this a forgettable blockbuster.
    [0.301] A deeply moving story about grief, loss, and unexpected friendship.
    [0.245] Breathtaking visuals and a haunting score make this an unforgettable experience.
```

SBERT's similarity scores are both higher in magnitude and more correctly scored than DAN's. DAN still retrieves some reasonable neighbors, but with much weaker, muddier selections (one of the top 3 candidates is "Breathtaking visuals and a haunting score make this an unforgettable experience." for bad movie match).

### Quantifying the Gap: STS Spearman Correlation

We constructed a small Semantic Textual Similarity (STS) benchmark, 8 sentence pairs, each with an LLM-generated reference similarity score from 0 to 1, scored by both SBERT and DAN models and compared via Spearman rank correlation:

```
STS Spearman Correlation (higher = better)
  SBERT (all-MiniLM-L6-v2): 0.881
  DAN baseline:             -0.143
```

DAN's correlation is _negative_ not just weak, but pointed the wrong way. The full 8-pair table shows why, and it's important to look at all 8 rather than any one or two in isolation:

```
  ref   SBERT    DAN  A vs B
 0.95   0.585  0.067  'The acting was phenomenal.' vs 'The performances were outstanding.'
 0.90   0.804  0.234  'A brilliant comedy.' vs 'A hilarious and clever film.'
 0.85   0.547  0.166  'The film is too long.' vs 'The movie drags on unnecessarily.'
 0.80   0.389  0.336  'Poor special effects.' vs 'The CGI looked cheap and unconvincing.'
 0.75   0.403  0.135  'I loved the ending.' vs 'The conclusion was satisfying.'
 0.10   0.436  0.099  'Great soundtrack.' vs 'The music was mediocre.'
 0.05   0.389  0.216  'A horror film.' vs 'A romantic comedy set in Paris.'
 0.02   0.102  0.175  'The director is talented.' vs 'The restaurant has good pasta.'
```

On the lowest-similarity pairs, DAN's raw number does land closer to the reference score than SBERT's does. But look at the five genuinely _high_-similarity pairs above them: DAN predicts its single lowest score in the whole set (0.067) for the pair with the _highest_ reference similarity "the acting was phenomenal" and "the performances were outstanding" are close to paraphrases of each other, and DAN rates them less similar than it rates "a horror film" vs. "a romantic comedy set in Paris." That's a bigger miss in the opposite direction. Averaged over all 8 pairs, mean absolute error against the llm generated reference score is also higher for DAN (0.454) than for SBERT (0.285). DAN's output is compressed into a narrow low range (roughly 0.07–0.34) that incidentally overlaps the _low_ end of the reference scale while badly undershooting the high end. That compression, largely indifferent to actual meaning, is exactly what non-contextual mean-pooled embeddings tend to produce: they mostly pick up _topical_ overlap (both sentences mention film, or both mention music) rather than the semantic relationship a reference score is meant to capture. SBERT's own raw cosine values are much less compressed (0.102 → 0.804), and its _ordering_ still tracks the reference ranking far better than DAN's does, which is what both the Spearman correlation and the MAE comparison are independently confirming.

A quick 2D PCA projection of SBERT vs. DAN embeddings for eight clearly positive/negative review snippets shows the same story visually: SBERT's positive and negative clusters separate cleanly, while DAN's separation is fuzzier, consistent with everything the STS numbers already say quantitatively.

---

## BERT, SBERT, and DAN: What Each One Is Actually For

- **BERT** produces token-level, bidirectionally-contextual representations, refined progressively across 12 layers of self-attention and pre-trained at scale, but its `[CLS]` output wasn't built for comparing two independently-encoded sentences.
- **SBERT** takes the same BERT-shaped architecture and adds nothing structurally new, same mean pooling as the Week 3 DAN baseline, except it pools over deep bidirectional context instead of a single embedding lookup, which is the entire source of the gap in this week's results.
- **DAN** is fast and simple, and its correlation with the reference similarity scores here (-0.143, and a higher mean absolute error than SBERT) shows exactly where "fast and simple" stops being enough: topical overlap isn't the same thing as semantic similarity.
- **CKA** is a diagnostic, not a training signal, it tells you whether an encoder's layers are doing meaningfully different work, and this week showed that the answer depends entirely on whether the residual-dominated flatness of an untrained network has actually been broken by training.

---

## What We Learned

- Stacking `TransformerBlock`s into a full encoder is architecturally trivial; what depth actually _buys_ you is a separate, measurable question that CKA is built to answer.
- An untrained encoder's CKA matrix is flat and uniformly high (0.95+) specifically because of residual connections preserving a shared identity pathway, not because randomness happens to look similar.
- A real pre-trained BERT shows the opposite pattern, a staircase where nearby layers stay similar and distant ones diverge, confirming that training is what breaks the residual connection's dominance, not depth alone.
- Fine-tuned BERT-base (91.6% accuracy) beat the from-scratch Week 6 `TinyTransformerClassifier` (72.8%) by nearly 19 points, from two compounding factors: ~15x the parameters, and 3.3B words of pre-training before any labeled IMDb data.
- Standard BERT can't efficiently support similarity search, cross-attention requires a forward pass per pair. SBERT fixes this by pooling each sentence independently, enabling one-time indexing and O(n·d) or better retrieval.
- On a small STS benchmark with LLM-generated reference scores, SBERT correlated with the reference ranking at 0.881 Spearman and a lower mean absolute error; the non-contextual DAN baseline scored -0.143 Spearman with a higher error, not a marginal gap, a near-inverted ranking on the high-similarity pairs specifically.

---

## What's Next

Weeks 6 and 7 covered attention, position, and encoder-only pre-training. **Week 8** moves to the generative side: decoder-only, GPT-style autoregressive models, trained on WikiText-2, where the causal masking Week 6 already built becomes a mandatory parameter.

> **Code:** This week's notebook is `Week_7_encoder.ipynb` in [`week7-BERT/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week7-BERT), supported by `utils.py`.
