---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 4) — When Order Enters the Picture: RNNs, GRUs, LSTMs, and Bidirectionality"
date: 2026-07-12 17:00:00 +0300
image: "/assets/img/nlp_week4_sequence_models.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    rnn,
    lstm,
    gru,
    bidirectional-lstm,
    sequence-models,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 4) — When Order Enters the Picture: RNNs, GRUs, LSTMs, and Bidirectionality

This is Week 4 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 3 closed the _similarity_ gap — token IDs became points in a space where distance meant something, via a Deep Averaging Network (DAN) trained on IMDB sentiment. But a DAN's core operation is mean pooling, and mean pooling has one deliberate blind spot: it throws sequence order away completely. Week 4 exists specifically to address that blind spot.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_4_sequence_models.ipynb`, supported by `utils.py`) is in the `week4-sequence-models/` folder.

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

## Why Order Matters

Take two sentences: `"This movie is not bad."` and `"This movie is bad."` They share almost every word. Last week's DAN turns a sentence into a vector by averaging the embeddings of its words together — and since these two sentences share almost all the same words, that average comes out nearly identical for both, aside from whatever small contribution `"not"` makes on its own. But the two sentences mean opposite things. Negation (`"not bad"`), contrast words like `"but"` (`"...but the ending was fantastic"`), and sentences that change direction partway through (`"I expected to hate this movie, but I loved it"`) all depend on _where_ a word sits relative to the rest of the sentence — not just whether that word shows up somewhere in it.

Sequence models fix this by reading one word at a time and carrying forward a running summary of what they've seen so far — including whether something earlier flipped the meaning of what comes next. This week builds four of these on top of last week's frozen DAN embedding: a basic RNN, a GRU, an LSTM, and a bidirectional LSTM. Then all four get tested against a small, handwritten set of sentences built specifically to require this kind of order-aware reasoning.

---

## Step 1: Reusing Week 3's Artifacts, Not Rebuilding Them

Nothing about tokenization or vocabulary changes this week, so the notebook starts by reloading everything Week 3 already built and saved: the train/val/test splits, the BPE vocabulary and merge rules, and the frozen DAN embedding table.

```python
train_ds = torch.load(train_ds_path, weights_only=False)
val_ds   = torch.load(val_ds_path,   weights_only=False)
test_ds  = torch.load(test_ds_path,  weights_only=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

vocab_config = load_json(imdb_vocab_config_path)
bpe_vocab = load_json(imdb_vocab_path)
```

One detail worth calling out: as of PyTorch 2.6, `torch.load` refuses by default to load anything beyond plain tensors — a safety measure, since a booby-trapped checkpoint file could otherwise run arbitrary code the moment you load it. `train_ds`, `val_ds`, and `test_ds` were saved as `Subset` objects rather than plain tensors, so they need the safety check turned off (`weights_only=False`) to load at all. That's fine here specifically because these files were created locally in Week 3 and never came from an outside source — it's not something to do by default with a checkpoint you didn't create yourself.

The embedding table comes back the same way it was saved — on its own, not bundled with the rest of the DAN:

```python
embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
embedding_layer.load_state_dict(torch.load("../week3-embedding/models/imdb_embedding_layer.pt", map_location=device))
embedding_layer.weight.requires_grad_(False)
```

Every sequence model this week starts from a copy of these same weights — but, unlike the `embedding_layer` above, each model's copy is allowed to keep changing during training rather than staying frozen. That's an intentional choice, not an oversight: it lets each model adjust the word representations toward whatever helps it specifically, instead of being stuck with representations tuned for a completely different kind of model.

---

## Step 2: A Basic RNN

The simplest sequence model keeps one running summary — its "hidden state" — and updates it after reading each word:

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_size, num_layers, dropout=0.2, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, nonlinearity="relu")
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, mask):
        lengths = mask.sum(dim=1).cpu()
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        output = hidden[-1]
        output = self.dropout(output)
        return self.linear(output)
```

Two details here matter more than they look:

- **`pack_padded_sequence` uses each review's real length, not the fixed `max_seq` size.** `lengths = mask.sum(dim=1)` counts how many real (non-padding) words each review actually has — using the same mask built back in Week 3. "Packing" tells PyTorch to skip the padding entirely when reading a review, instead of reading through a bunch of meaningless pad tokens and hoping they don't throw off the running summary. `enforce_sorted=False` just means the reviews in a batch don't need to be pre-sorted by length, which they aren't here.
- **This RNN uses ReLU instead of the more textbook `tanh`.** The classic RNN formula is usually written with tanh: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$. Tanh has a problem over long sentences, though: once its input gets large in either direction, its output curve flattens out almost completely, and in that flat zone barely any gradient gets through during training — which is part of why a plain tanh RNN can lose track of things over a long review. ReLU doesn't have that flattening problem on the positive side. But tanh also keeps its output squeezed between -1 and 1 no matter what, acting like a built-in safety limit — and ReLU has no such limit, so its output can keep growing without bound. So switching to ReLU trades away that built-in safety limit for better information flow during training. It's a real tradeoff, not a strictly free upgrade.

---

## Step 3: GRU and LSTM — Adding Gates

A basic RNN struggles over long reviews because each new hidden state is built directly from the previous one, over and over, and training has to send gradient signal backward through every single one of those steps. The more steps there are, the more that signal tends to shrink toward zero or blow up — the classic "vanishing/exploding gradient" problem, and the reason a plain RNN often forgets something it read near the start of a long review by the time it reaches the end.

**GRU** tackles this with two gates — think of a gate as a valve that controls how much of something gets through. The _reset gate_ controls how much of the old summary feeds into a proposed new update; the _update gate_ then decides how much of that proposed update actually replaces the old summary, versus how much of the old summary just stays as-is:

$$h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \bar{h}_t$$

Here $h_{t-1}$ is the old summary, $\bar h_t$ is the proposed update, and $z_t$ (the update gate) decides the mix between them. Because this is a blend rather than a full overwrite, there's a much more direct path for gradient signal to flow backward during training, instead of getting squeezed at every single step.

**LSTM** takes this a step further with a completely separate memory channel — the _cell state_ — dedicated purely to carrying information forward:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The _forget gate_ $f_t$ can stay close to 1, letting old information pass through mostly untouched instead of being overwritten every step — so information can survive across many words without constantly getting diluted. An _output gate_ then decides how much of that memory actually gets exposed as the model's working summary at each step.

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_size, num_layers, dropout=0.2, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, mask):
        lengths = mask.sum(dim=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed)
        output = self.dropout(hidden[-1])
        return self.linear(output)
```

The GRU model is nearly identical, just swapping `nn.LSTM` for `nn.GRU` and dropping the separate memory channel. One small difference from the RNN: `nn.GRU` and `nn.LSTM` don't let you choose ReLU vs. tanh — PyTorch always uses tanh (and sigmoid for the gates) inside them, so there's no explicit activation choice to make here.

---

## Step 4: Bidirectionality

Bidirectional models read a sentence twice — once forward, once backward — so every word's representation can draw on what comes both before _and_ after it. That's useful for something like `"Only the final scene is good"`, where the word `"good"` alone reads as positive, but the word `"Only"` earlier in the sentence flips the overall meaning — you need context from both sides to catch that.

```python
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_size, num_layers, dropout=0.2, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, mask):
        lengths = mask.sum(dim=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed)
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        output = torch.cat([last_forward, last_backward], dim=1)
        output = self.dropout(output)
        return self.linear(output)
```

Worth being precise about the indexing here, since it's easy to get backward (no pun intended): for a single-layer bidirectional LSTM, PyTorch stores the final states as `[forward, backward]`, so `hidden[-2]` is where the forward-reading pass ended up, and `hidden[-1]` is where the backward-reading pass ended up. Gluing the two together gives the classifier information from both directions at once. Reading both ways doesn't fix the vanishing-gradient issue by itself — each direction still has to send its own gradient signal backward through every step — but it does mean the model has seen both sides of a word like "Only" or "but" before it has to make a final call.

---

## Step 5: A Small, Targeted Test Set — Not a Formal Benchmark

Standard IMDB test accuracy doesn't tell you _why_ a model gets things right or wrong — a model can score well just by picking up on strong sentiment words like `"terrible"` or `"fantastic"`, without needing to understand order at all. So this week adds a small, handwritten set of test sentences aimed squarely at what a DAN can't do: **negation**, **contrast**, **sentiment shift**, and **scope**.

```python
benchmark_set = {
    "negation": [
        {"text": "This movie is not good.", "label": 0},
        {"text": "This movie is bad.", "label": 0},
        {"text": "This movie is not bad.", "label": 1},
        {"text": "I did not enjoy this film.", "label": 0},
        {"text": "The plot was not convincing.", "label": 0},
    ],
    "contrast": [
        {"text": "The acting was weak, but the ending was fantastic.", "label": 1},
        {"text": "The acting was fantastic, but the ending was weak.", "label": 0},
        {"text": "The first half drags, but the film becomes excellent.", "label": 1},
        {"text": "The first half is excellent, but the film becomes dull.", "label": 0},
    ],
    "sentiment_shift": [
        {"text": "I expected to hate this movie, but I loved it.", "label": 1},
        {"text": "I expected to love this movie, but I hated it.", "label": 0},
        {"text": "It begins like a disaster and ends beautifully.", "label": 1},
        {"text": "It begins beautifully and ends like a disaster.", "label": 0}
    ],
    "scope": [
        {"text": "Only the final scene is good.", "label": 0},
        {"text": "The final scene is good.", "label": 1},
        {"text": "Only the soundtrack is memorable.", "label": 0},
        {"text": "The soundtrack is memorable.", "label": 1},
        {"text": "Only the ending works.", "label": 0},
        {"text": "The ending works.", "label": 1},
    ],
}
```

This is a small, hand-built test set, not a proper benchmark — there aren't nearly enough examples per category to draw statistically solid conclusions from. But it's useful for spotting patterns in _how_ a model fails, which a single overall accuracy number would otherwise hide completely.

All four sequence models (RNN, GRU, LSTM, BiLSTM) run against this set. Each model gets one throwaway "warm-up" run first — the very first time any model runs on new input, there's some one-time setup cost behind the scenes (memory getting allocated, and on some hardware, code getting compiled for the first time) that has nothing to do with how fast the model actually runs afterward. Only the second run gets timed, so the speed comparison reflects steady, real-world performance rather than that one-time startup cost. Accuracy, speed, and parameter count all get collected into one comparison table, and every example each model gets wrong is logged by category for a closer look.

---

## What the Comparison Showed

**Test accuracy and challenge-set accuracy tell two different stories.** On the regular IMDB test set, GRU comes out on top (77.5% correct), followed by BiLSTM (76.2%), LSTM (74.8%), with RNN well behind the rest (71.2%). But on the small handwritten challenge set — the sentences built specifically to test negation, contrast, sentiment shift, and scope — the order changes: LSTM leads (63.2%), then BiLSTM (57.9%), GRU (52.6%), and RNN again in last place (47.4%). RNN is clearly the weakest model either way. But there's no single "best" model overall — the model that does best on everyday reviews (GRU) isn't the one that's best at untangling negation and contrast (LSTM). Which one you'd actually pick depends on what you need it to be good at.

Breaking the challenge set down by category shows why the two rankings disagree:

| Category (n)        | RNN | GRU | LSTM    | BiLSTM  |
| ------------------- | --- | --- | ------- | ------- |
| negation (5)        | 40% | 60% | **80%** | 40%     |
| contrast (4)        | 50% | 50% | **75%** | **75%** |
| sentiment shift (4) | 50% | 50% | 50%     | **75%** |
| scope (6)           | 50% | 50% | 50%     | 50%     |

- **LSTM is the clear winner on negation.** A sentence like _"not good"_ needs the model to remember, all the way to the end, that something earlier flipped the meaning. LSTM has a separate memory channel (the cell state) built specifically to carry a signal like that forward without it getting overwritten at every step. GRU doesn't have that separate channel — it mixes memory and output together — which may be why it falls behind here.
- **BiLSTM is best at sentiment shift, and ties LSTM on contrast — but is tied-worst on negation.** Sentences like _"I expected to hate this movie, but I loved it"_ need the model to connect two halves of a sentence to each other, which is exactly what reading in both directions is good for. But negation usually only needs the one word right before it — a much more local, short-range signal — and reading the sentence backward too doesn't help with that, and here it actually seems to hurt: BiLSTM ties RNN for the worst negation score.
- **RNN is weakest across the board.** It has no memory gates at all — every new word directly overwrites its one running summary of the sentence, with nothing held separately in reserve. It also uses a ReLU activation here instead of the more common tanh, which has no upper limit on how large its outputs can grow, giving it fewer built-in safeguards against noisy signals. For a task that depends on holding onto something steady across several words — a negation, a contrast — this is the model with the least built-in support for doing that.
- **No model beats a coin flip on scope — and looking at the actual predictions shows why.** All four models score exactly 50% on sentences like _"Only the final scene is good"_ vs. _"The final scene is good."_ Looking at what each model actually predicted: GRU and LSTM both behave like simple word-spotters here — they see a positive word like "good" or "memorable" and predict positive, ignoring the word "Only" in front of it entirely. BiLSTM does something slightly different but no more correct: it predicts negative for both the "Only..." version _and_ the plain version whenever the sentence is about a soundtrack or an ending — meaning it picked up a topic-based habit, not an understanding of what "only" does to a sentence's meaning. None of the four ever learned that "only" can flip the sentiment of an otherwise-positive sentence.

  A full scan of the 25,000-review training set explains why. Contrast patterns ("but," "however") show up in 18,657 reviews — 74.6%, the large majority of the dataset. Negation patterns like "not good" appear in 2,012 reviews — 8.0%, far less common but still thousands of real examples to learn from. Scope patterns like "only the... is good" appear in just 212 reviews — 0.85%. That's roughly a 90-to-1 gap between how often negation shows up and how often this specific scope pattern does. With that little exposure, there's no real chance for any of the four models to pick up on what "only" is doing here — the difference between architectures we saw on negation and contrast simply doesn't have anything to work with on scope, because the pattern is barely present in what the models were trained on in the first place.

The main takeaway: **different architectures are good at different things, not ranked on a single scale.** LSTM's extra memory channel helps specifically with negation. Reading both directions helps specifically with connecting two halves of a sentence. Neither trick helps with scope, because that's not a memory problem or a direction problem — it's that the models likely never saw enough examples of that particular pattern to learn it. A single accuracy number, from either the standard test set or the challenge set alone, would have hidden all of this. It only shows up once you break results down by category and look at what each model is actually predicting.

---

## What We Learned

- Packing sequences by their real length (using the padding mask) rather than running the model over padding is what lets `nn.RNN`/`nn.GRU`/`nn.LSTM` handle reviews of different lengths correctly in the same batch.
- GRU and LSTM both fight vanishing gradients the same basic way — letting old information add into the new state instead of being fully overwritten — but LSTM's separate memory channel gives it an extra edge specifically on negation, where a signal has to survive across the whole sentence.
- Reading in both directions (BiLSTM) helps most when a sentence's meaning depends on connecting two separated parts of it (contrast, sentiment shift) — but doesn't help, and here even slightly hurts, on negation's much shorter-range pattern.
- One overall accuracy number can rank models completely differently depending on which one you use: GRU wins on the standard test set, LSTM wins on the small targeted test set. Neither number alone tells the whole story.
- A model being _able_ to use an earlier word doesn't mean it _learns_ to use it — that depends on whether training gives it enough reason to. A full scan of the training set shows why negation and contrast worked (8.0% and 74.6% of reviews contain those patterns) while scope didn't (just 0.85%) — the "only + positive sentence" pattern isn't any harder for these architectures to represent, there simply wasn't enough of it for any of the four to learn from.

---

## What's Next

This week showed that different sequence models pick up on different kinds of order-related patterns — but none of them cracked scope, and it's an open question whether more training data, a bigger model, or a genuinely different approach is what's needed. **Week 5** turns to Convolutional Neural Networks (CNNs) applied to text — a different way of picking up on local structure, sliding small fixed-size filters across the sentence to spot short patterns wherever they show up, instead of reading through it one word at a time and carrying a running summary forward.

> **Code:** This week's notebook is `Week_4_sequence_models.ipynb` in [`week4-sequence-models/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week4-sequence-models), supported by `utils.py`.
