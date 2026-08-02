---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 9) - Encoder-Decoder Generation: Building T5 From Scratch and Fine-Tuning t5-small"
date: 2026-08-16 17:00:00 +0300
image: "/assets/img/nlp_week9_t5.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    transformers,
    t5,
    encoder-decoder,
    cross-attention,
    seq2seq,
    translation,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 9) - Encoder-Decoder Generation: Building T5 From Scratch and Fine-Tuning t5-small

This is Week 9, the final week, of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 8 was decoder-only: one stack, causal masking, and next-token prediction. Week 9 adds a second stack. `EncoderBlock` and `DecoderBlock` share the same `MultiHeadAttention` and `FeedForward` building blocks from Weeks 6–8, but this week they're wired together for the first time, with a third kind of attention, **cross-attention**, connecting them.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_9_T5.ipynb`, supported by `utils.py`) is in the `week9-t5/` folder.

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

## Three Parts

This week's notebook has a deliberate three-way split:

1. **Part A (concept):** contrast decoder-only (GPT) architecture against encoder-decoder (T5) architecture: bidirectional encoder self-attention, causal decoder self-attention, and the cross-attention bridge that connects them.
2. **Part B (pure PyTorch):** build a tiny encoder-decoder Transformer, `TinyT5`, from scratch and train it on a toy sequence-reversal task, to make the seq2seq mechanics: masking, teacher forcing, cross-attention, concrete.
3. **Part C (Hugging Face):** fine-tune `t5-small`, a real pretrained encoder-decoder model, on German-to-English translation, to see the same architecture doing genuinely useful work at scale.

Same theme as every week in this series: architecture explains _how_ a model works, and building it from scratch is the only way to be sure you actually understand the _how_ before leaning on a pretrained model for the _why-it-works-well_.

---

## Concept 1: What Changes When You Add an Encoder

A decoder-only model like `TinyGPT` from Week 8 has one stack and one rule: never look forward. An encoder-decoder model like T5 has two stacks with two different rules:

- **The Encoder** processes the entire source sequence with **bidirectional self-attention**: every source token can attend to every other source token, in both directions, building a context-rich representation of the whole input before generation ever starts.
- **The Decoder** generates the target sequence with **causal self-attention** (same rule as `TinyGPT`), but also gets a second attention mechanism, **cross-attention**, that lets each decoder position query the _encoder's_ output directly:

$$
\text{CrossAttention}(X_{\text{dec}}, H_{\text{enc}}) = \text{Softmax}\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}
$$

Queries come from the decoder; keys and values come from the encoder. That's the entire "translation bridge": at every generation step, the decoder decides which source tokens are relevant right now.

In `utils.py`, this isn't a separate mechanism. `MultiHeadAttention.forward` takes an optional `context` argument: when `context` is left as `None` it defaults to `x` (self-attention), and when `context` is explicitly passed as the encoder's output, it's cross-attention, same class, same weights structure, different tensor wired in as keys/values.

---

## Concept 2: Building TinyT5 From Scratch

`TinyT5` assembles two symmetric stacks:

```python
self.encoders = nn.ModuleList([EncoderBlock(...) for _ in range(num_layers)])
self.decoders = nn.ModuleList([DecoderBlock(...) for _ in range(num_layers)])
```

Each `EncoderBlock` is pre-norm self-attention + feed-forward, same pattern as every block in this series. Each `DecoderBlock` adds a second sub-layer in between: self-attention, then cross-attention (querying the encoder output), then feed-forward.

### Two Masks, Doing Different Jobs

```python
def make_src_mask(self, src_ids):
    return (src_ids != self.pad_id).view(src_ids.size(0), 1, 1, src_ids.size(1))

def make_tgt_mask(self, tgt_ids, offset=0):
    ...
    return pad_mask & causal_mask
```

The source mask only has to hide `<PAD>` tokens because the encoder is bidirectional, so there's no "future" to hide. The target mask does two jobs at once: it hides `<PAD>` tokens _and_ enforces causality via a lower-triangular mask, combined with a logical `AND`.

### Teacher Forcing, Shift-Right

Training feeds the decoder the _actual_ target sequence, shifted right by one position and prefixed with `<BOS>`:

$$
\text{Target}: [\text{Word}_1, \text{Word}_2, \text{Word}_3, \text{EOS}] \rightarrow \text{Decoder Input}: [\text{BOS}, \text{Word}_1, \text{Word}_2, \text{Word}_3]
$$

This lets the whole target sequence train in one parallel forward pass instead of token-by-token, while still forcing the model to predict position $t$ using only positions $<= t-1$.

### Weight Tying

Same trick as `TinyGPT` in Week 8:

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.embedding.weight  # Weight Tying optimization
```

For `TinyT5`'s config (`d_model=128`, 2 encoder layers, 2 decoder layers, `vocab_size=32`), the full model comes out to **537,472 parameters**, small enough to train the toy task in well under a minute on CPU or MPS.

---

## Concept 3: Training TinyT5: The Copying Shortcut and the Phase Change

The toy task is sequence reversal: map a random source sequence to its reverse, plus an `<EOS>` token. Simple enough to fully diagnose, hard enough that the model can't solve it by accident.

```
epoch 1    | loss 86.7831
epoch 101  | loss 3.4703
epoch 251  | loss 3.1373
epoch 501  | loss 2.9946
epoch 751  | loss 2.7046
epoch 851  | loss 2.3287
epoch 901  | loss 1.8203
epoch 951  | loss 0.6052
```

Notice the shape: a long plateau around ~3.0–3.3 for hundreds of epochs, then a sharp break starting around epoch 850, dropping the loss by more than 2 nats in the last 150 epochs. That plateau isn't the model failing to learn, it's the model exploiting a **copying shortcut**: causal self-attention in the decoder is a very simple linear mapping, so early on it's cheaper for the model to learn "copy the token I just saw, lagging by one step" than to learn true cross-attention alignment with the encoder. The phase change is what happens when continued training finally forces the cross-attention pathway to converge and the shortcut stops being competitive.

By epoch 1000, the held-out example matches exactly:

```
source:  [25, 9, 21, 31, 25, 6, 14, 4]
target:  [4, 14, 6, 25, 31, 21, 9, 25, 2]
pred:    [4, 14, 6, 25, 31, 21, 9, 25, 2]
```

Target is the reversed source plus `<EOS>` (`2`) and the prediction matches it token-for-token.

---

## Concept 4: KV-Caching

`MultiHeadAttention` and `TinyT5.decode` both support key-value caching for incremental generation: cache the encoder's static keys/values once for cross-attention, and concatenate only the newest token's keys/values into a growing cache for self-attention, dropping per-step generation cost from $O(t^2)$ to $O(t)$.

This notebook never actually turns `use_cache=True` on, because `train_tiny_T5` and `evaluate_tiny_T5` both run full parallel forward passes, the same simplification `TinyGPT` made in Week 8.

---

## Concept 5: Fine-Tuning t5-small on Real Translation Data

Part C moves from the from-scratch model to a pretrained one: `t5-small`, fine-tuned for German-to-English translation.

**Dataset note:** the original plan was `opus100` (en-de), but it's occasionally unavailable or flaky on the Hugging Face Hub, so this notebook uses `opus_books` (`de-en`) instead, a lightweight, reliably-available corpus of translated literature. `opus_books` only ships a single `train` split (46,320 rows), so we partition it ourselves: 90/10 into train/validation, then take a 5,000/500-example slice of that for a CPU/MPS-realistic run.

Preprocessing follows T5's text-to-text convention, a task prefix tells the model what to do:

```python
prefix = "translate German to English: "

def preprocess(batch):
    src_texts = [prefix + ex["de"] for ex in batch["translation"]]
    tgt_texts = [ex["en"] for ex in batch["translation"]]
    return tokenizer(src_texts, max_length=128, truncation=True, text_target=tgt_texts)
```

Training uses the same gradient-accumulation trick as Week 8: batch size 1, accumulated over 8 steps, to keep peak memory low on local hardware:

```python
training_args = TrainingArguments(
    max_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    warmup_steps=30,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    fp16=False,  # CPU/MPS stay full precision
    dataloader_pin_memory=False,  # no benefit on mps
)
```

---

## Generation After Fine-Tuning

```
DE: Ich habe diesen Film wirklich genossen, aber das Ende war enttäuschend.
EN predicted: I really enjoyed this film, but the end was a tumultuous affair.
```

Worth being precise about what this output actually shows: the first half is a correct, fluent translation. The second half isn't: "enttäuschend" means _disappointing_, not "a tumultuous affair." That's a real mistranslation, and a plausible explanation is sitting right there in the data: `opus_books` is literary text, and 1000 steps at an effective batch size of 8 (8,000 examples seen, well under two passes over the 5,000-example training slice) is a light nudge, not a deep retraining. It's enough to pull the model's phrasing toward a more literary register; it's not enough to fully correct where that pull produces something fluent-sounding but wrong. A good reminder from this series generally: fluency and correctness are different axes, and a model can move on one without moving much on the other.

### Beam Search, Briefly

Generation used beam search (`num_beams=3`) instead of greedy decoding. Where greedy always takes $\arg\max_w P(w   
 \mid y_{< t}, X)$ at each step, beam search keeps the top-$B$ highest cumulative-log-probability hypotheses alive at every step, expanding all $B \times |V|$ candidates and pruning back down to $B$ each time. For translation, where there's a near-correct target and greedy's early mistakes can lock out better completions downstream, beam search generally wins. It also has a real cost:it favors safe, generic phrasing over more surprising ones, which is part of why open-ended chat generation typically prefers sampling instead.

---

## What We Learned

- Cross-attention is not a new operation, it's the same `MultiHeadAttention` module used for self-attention, with the encoder's output wired in as keys and values instead of the decoder's own hidden states.
- Small seq2seq models can spend a long time exploiting a "copying shortcut" in causal self-attention before a sharp phase change forces true cross-attention alignment to emerge, the loss curve's plateau-then-cliff shape is diagnostic of that.
- Fine-tuning a pretrained encoder-decoder model for translation is dramatically cheaper than training one from scratch, but a light fine-tune (1000 steps on a 5,000-example slice) nudges style more than it guarantees correctness: this week's translation demo got half a sentence right and half a sentence fluently wrong.
- Beam search's advantage over greedy decoding is real for structured tasks like translation, precisely because it doesn't commit to an early token that forecloses a better full sequence.

---

## Closing the Arc

Weeks 7 through 9 built the same handful of primitives: multi-head attention, layer norm, feed-forward blocks, into three different architectures: BERT's encoder-only bidirectional representations (Week 7), GPT's decoder-only causal generation (Week 8), and now T5's encoder-decoder sequence-to-sequence generation (Week 9). Every week's from-scratch build existed to answer the same question in a different setting: does this architecture actually work the way the theory says it should, verified by writing it.

That's a wrap on the 9-week series: from TF-IDF vectorization in Week 1 to fine-tuning a pretrained encoder-decoder Transformer in Week 9. All nine notebooks and their supporting `utils.py` files are in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).
