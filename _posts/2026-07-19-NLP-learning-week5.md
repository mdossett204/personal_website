---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 5) — Local Patterns Without Order: CNNs for Text"
date: 2026-07-19 17:00:00 +0300
image: "/assets/img/nlp_week5_cnn_text.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    cnn,
    text-classification,
    1d-convolution,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 5) — Local Patterns Without Order: CNNs for Text

This is Week 5 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 4 built four sequence models — RNN, GRU, LSTM, BiLSTM — that read a review one word at a time and carry a running summary forward, specifically to catch order-dependent patterns like negation and contrast that a plain averaging model can't see. Week 5 asks a different question: what if you don't read the whole sentence in order at all, and instead just slide a small window across it, looking for short local patterns wherever they happen to show up?

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_5_cnn_for_text.ipynb`, supported by `utils.py`) is in the `week5-CNN-for-text/` folder.

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

## Why Local Patterns Are Worth a Whole Different Architecture

A sequence model like last week's LSTM reads a review word by word, updating one running summary as it goes. That's powerful, but it's also doing more work than some patterns actually need. A phrase like `"not good"` or `"absolutely wonderful"` is a purely local signal — you don't need to have read the first half of the review to recognize it, and you don't need a summary that's been updated fifty times to still remember it clearly by the end.

A **1D convolution** is built for exactly this kind of pattern. Instead of reading the sentence in order and accumulating state, it slides a small fixed-width window across the token sequence and asks the same question at every position: _"does this specific short pattern show up here?"_ The same detector — the same learned weights — gets reused at every position in the sentence. That's parameter sharing, and it's the whole idea behind a `TextCNN`.

---

## Step 1: Reusing Week 3's Artifacts, Not Rebuilding Them

Like Week 4, this notebook doesn't touch tokenization or vocabulary — it reloads the same train/val/test splits, the same BPE vocabulary and merge rules, and the same pretrained DAN embedding table from Week 3:

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

The embedding layer is initialized from the same Week 3 DAN weights used by every sequence model last week — so all of these architectures start from the same learned embedding space, and any differences in what they end up learning come from the architecture itself, not from a different starting point. This week's embedding stays trainable, letting the CNN nudge the embedding space toward whatever helps it detect short local phrases.

---

## Step 2: 1D Convolution as a Sliding Window

For a sequence of length `n`, a filter of width `k` produces `n - k + 1` scores — one for every valid window position. Each of those scores is just: take the `k` embeddings inside the window, apply the filter's learned weights, and sum. If the filter has learned to fire on a positive local pattern, high scores mean "this window looks like that pattern."

This notebook uses three filter widths — `6`, `10`, and `16` tokens — with `64` filters at each width. That's `64` independent short-context detectors, `64` medium-context detectors, and `64` wider-context detectors, all scanning the same sequence in parallel. A useful analogy: it's how a reader's eye catches phrases like `"not good"` or `"too slow"` no matter where they appear on a page, without needing to have read the sentence in order to notice them.

---

## Step 3: Pooling — Turning a Feature Map Into One Number Per Filter

After the convolution and a ReLU, each filter has produced a feature map: one activation per window position. That's still not a sentence-level signal — it needs to be collapsed into a single number per filter before it can feed a classifier. **Global pooling** does that collapse:

- **Max pooling** keeps the single largest activation in the feature map — "this pattern fired somewhere, and here's how strongly."
- **Mean pooling** averages every activation in the feature map — "how active was this filter across the whole review, on average."

A quick toy example makes the contrast concrete. Take one hypothetical width-2 filter tuned to positive local patterns, scored across the bigrams of a sentence:

```
Sentence:
I expected very little but this movie was absolutely wonderful

Bigram windows and one filter's scores:
            I expected -> 0.0
         expected very -> 0.1
           very little -> 0.0
            little but -> 0.0
              but this -> 0.1
            this movie -> 0.2
             movie was -> 0.4
        was absolutely -> 1.8
  absolutely wonderful -> 3.2

Max pooling over the whole sentence keeps: 3.2
Mean pooling over the whole sentence keeps: 0.644
```

Max pooling reports one strong match (`3.2`) and ignores everything else in the sentence; mean pooling reports a modest average (`0.644`) that's dragged down by all the near-zero bigrams that came before it. The actual `TextCNN` in this notebook uses **max pooling only** — the goal is to catch the strongest local evidence for or against a sentiment, not to average it out.

---

## Step 4: The TextCNN Architecture

The full pipeline is:

`input ids → embedding → padding-aware masking logic → Conv1d + ReLU → global max pool → concatenate → dropout → linear layer → logit`

```python
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int,
        num_filters: int,
        kernel_sizes=(6, 10, 16),
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.kernel_sizes = tuple(kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in self.kernel_sizes]
        )
        self.linear = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def _pool_feature_map(self, feature_map: torch.Tensor, conv_mask: torch.Tensor) -> torch.Tensor:
        mask = conv_mask.unsqueeze(1)
        has_valid = mask.any(dim=-1, keepdim=True)
        mask = mask | ~has_valid
        masked_map = feature_map.masked_fill(~mask, -1e9)
        return masked_map.max(dim=-1).values

    def extract_feature_maps(self, input_ids: torch.Tensor, padding_mask: torch.Tensor):
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        conv_input = embedded.transpose(1, 2)

        feature_maps = {}
        pooled_outputs = []
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            conv_out = conv(conv_input)
            activated = torch.relu(conv_out)
            conv_mask = build_conv_mask(padding_mask, kernel_size)
            pooled = self._pool_feature_map(activated, conv_mask)
            feature_maps[kernel_size] = {
                "activations": activated,
                "mask": conv_mask,
                "pooled": pooled,
            }
            pooled_outputs.append(pooled)

        sentence_features = torch.cat(pooled_outputs, dim=1)
        sentence_features = self.dropout(sentence_features)
        return feature_maps, sentence_features

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        _, sentence_features = self.extract_feature_maps(input_ids, padding_mask)
        return self.linear(sentence_features)
```

```
TextCNN(
  (embedding): Embedding(590, 384, padding_idx=589)
  (dropout): Dropout(p=0.3, inplace=False)
  (convs): ModuleList(
    (0): Conv1d(384, 64, kernel_size=(6,), stride=(1,))
    (1): Conv1d(384, 64, kernel_size=(10,), stride=(1,))
    (2): Conv1d(384, 64, kernel_size=(16,), stride=(1,))
  )
  (linear): Linear(in_features=192, out_features=1, bias=True)
)
```

A couple of details are easy to miss but matter for correctness:

- **Padding is neutral, but it still needs to be masked out.** `padding_idx` makes the pad embedding exactly zero, so a window that overlaps padding never gets _contaminated_ by garbage values. But the convolution still produces a valid, non-zero activation for that window if the real tokens inside it are strongly activating — a window that's mostly padding can still register a high score just from the one or two real tokens it does contain. `build_conv_mask(padding_mask, kernel_size)` exists to discard those windows before pooling picks its winner, so the model never selects an activation propped up by an artificially narrow, mostly-empty window.
- **Very short reviews need a deliberate fallback.** If a review's real (non-padded) length is shorter than a kernel width — a five-token review against a 16-token kernel, for example — there's no window that's _fully_ real content, so the strict mask above would leave nothing valid to pool from. Rather than silently defaulting to a meaningless placeholder, `_pool_feature_map` detects that case and unmasks every window instead of only the fully-valid ones. Because padding contributes exactly zero to each window's score, this doesn't inject noise — it just means the model is scoring the short phrase at whichever offsets the sliding window happens to place it at (padding-heavy or not), and max pooling picks whichever alignment gave the strongest signal, rather than being blocked from making a prediction at all.

---

## Step 5: Training TextCNN

```python
criterion = nn.BCEWithLogitsLoss()
patience = 5
num_epochs = 30
lr = 5e-4
weight_decay = 1e-4

best_cnn_model = train_test_model(
    device=device,
    model=model,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    patience=patience,
    num_epochs=num_epochs,
    lr=lr,
    weight_decay=weight_decay,
    is_binary=True,
    save_path=save_path,
)
```

```
Epoch 01 | train_loss=0.6141 val_loss=0.5508 | train_acc=0.6558 val_acc=0.7224
Epoch 05 | train_loss=0.4374 val_loss=0.4823 | train_acc=0.7934 val_acc=0.7600
Epoch 09 | train_loss=0.2928 val_loss=0.4922 | train_acc=0.8756 val_acc=0.7754
Epoch 11 | train_loss=0.2147 val_loss=0.5193 | train_acc=0.9176 val_acc=0.7682
Early stopping at epoch 11
```

```
test_loss=0.4743 | test_acc=0.7672
```

TextCNN lands at **76.7% test accuracy** — right in the middle of last week's sequence models (RNN 71.2%, LSTM 74.8%, BiLSTM 76.2%, GRU 77.5%). On everyday reviews, sliding local filters do about as well as a model that reads the whole sentence in order — which makes sense, since most sentiment in a typical review really is carried by a handful of strong local phrases, not by long-range structure.

---

## Step 6: A Challenge Set Built for a Local-Pattern Model

The same principle from Week 4 applies here: standard test accuracy hides _why_ a model gets things right or wrong. This week's handwritten set targets the specific things a sliding-window, order-blind model should struggle or succeed with:

```python
challenge_set = {
    "negation_bigrams": [
        {"text": "This movie is not good.", "label": 0},
        {"text": "This movie is not bad.", "label": 1},
        {"text": "The ending is not terrible.", "label": 1},
        {"text": "The acting is not interesting.", "label": 0},
    ],
    "phrase_relocation": [
        {"text": "Absolutely wonderful acting saves this movie.", "label": 1},
        {"text": "This movie is saved by absolutely wonderful acting.", "label": 1},
        {"text": "Really disappointing writing ruins this movie.", "label": 0},
        {"text": "This movie is ruined by really disappointing writing.", "label": 0},
    ],
    "local_cue_with_noise": [
        {"text": "After a long setup with many side characters and plenty of dialogue, the ending is wonderful.", "label": 1},
        {"text": "After a long setup with many side characters and plenty of dialogue, the ending is awful.", "label": 0},
        {"text": "There are many scenes, many conversations, and many detours, but the final scene is brilliant.", "label": 1},
        {"text": "There are many scenes, many conversations, and many detours, but the final scene is dreadful.", "label": 0},
    ],
    "competing_local_cues": [
        {"text": "The opening is wonderful but the ending is awful.", "label": 0},
        {"text": "The opening is awful but the ending is wonderful.", "label": 1},
        {"text": "Parts are charming but the overall movie is disappointing.", "label": 0},
        {"text": "Parts are disappointing but the overall movie is charming.", "label": 1},
    ],
}
```

- `negation_bigrams`: short local flips such as `not good` versus `not bad`.
- `phrase_relocation`: the same key phrase moved to different positions in the sentence.
- `local_cue_with_noise`: one strong local cue surrounded by mostly neutral text.
- `competing_local_cues`: multiple local sentiment phrases pulling in different directions — the category where a max-pooling model's blind spot should show up most clearly, since it has no way to know which of two competing local phrases the sentence actually wants to weigh more heavily.

## What the Challenge Set Showed

| Category (n)             | TextCNN  |
| ------------------------ | -------- |
| negation_bigrams (4)     | 25%      |
| phrase_relocation (4)    | **100%** |
| local_cue_with_noise (4) | **100%** |
| competing_local_cues (4) | 25%      |
| **Overall (16)**         | 62.5%    |

The split is stark, and it lines up with what global max pooling can and can't do:

- **`phrase_relocation` and `local_cue_with_noise` are exactly where a sliding-window detector should shine, and TextCNN gets both perfect.** Moving `"absolutely wonderful acting"` from the front of a sentence to the back doesn't matter to a filter that gets reused at every position — the same detector fires wherever the phrase lands. Same story for a single strong cue buried in a longer, mostly neutral sentence: the filter only needs to catch that one window, and max pooling makes sure the rest of the sentence can't dilute it.
- **`competing_local_cues` is where the order-blindness costs the model the most.** `"The opening is wonderful but the ending is awful"` and `"The opening is awful but the ending is wonderful"` contain the exact same two local phrases — just swapped. Global max pooling has no mechanism for knowing that the _second_ clause should outweigh the first; it just reports "both of these fired somewhere," and the classifier is stuck weighing two contradictory signals without any positional context to break the tie. TextCNN gets 1 of 4 right here, which is close to what you'd expect from a model that's structurally unable to use word order to resolve the conflict.
- **`negation_bigrams` is a genuine miss, and it's a fair one.** `"This movie is not good"` and `"This movie is not bad"` differ by a single word, but that word flips the label. A width-6 filter absolutely can span `"is not good"` as one window — the information is there — but there's nothing in the architecture that treats `"not"` as special the way an LSTM's gates can choose to. The model got 1 of 4 right here too, and the one correct case (`"The acting is not interesting"`) looks more like it landed on the right answer by coincidence than by genuinely parsing the negation.

---

## Step 7: Reading a Trained Filter's Feature Map

With training done, the feature maps are far more interpretable than they'd be at random initialization. Running the probe sentence `"Parts are disappointing but the overall movie is charming."` through the model:

```
probe text: Parts are disappointing but the overall movie is charming.
word tokens: ['Parts', 'are', 'disappointing', 'but', 'the', 'overall', 'movie', 'is', 'charming', '.']
predicted positive probability: 0.204
kernel=6  | feature_map shape=(64, 123) | pooled shape=(64,)
kernel=10 | feature_map shape=(64, 119) | pooled shape=(64,)
kernel=16 | feature_map shape=(64, 113) | pooled shape=(64,)
```

A raw activation on its own isn't a class score — it's just "how strongly did this filter fire here." What actually drives the prediction is:

`filter contribution = pooled_activation × classifier_weight`

Decomposing the logit for this sentence:

```
bias contribution: 0.0787
sum of filter contributions: -1.4431
final logit from decomposition: -1.3644
probability from decomposition: 0.2035
```

The top negative contributors are almost all anchored on `"Parts are disappointing"` and `"disappointing but the overall"` — filters that clearly picked up the negative half of the sentence and, in this case, outweighed the positive filters anchored on `"movie is charming"`. The label for this sentence is `1` (positive) — the model gets it wrong here, and the decomposition shows exactly why: it's a genuine instance of the `competing_local_cues` pattern, where the negative-leaning filters simply won the vote.

---

## CNN vs. RNN

- CNNs are fast and parallelizable, since every window can be scored at once instead of one step at a time.
- CNNs are especially good when the label depends on local phrase evidence — strong, self-contained sentiment expressions the model can catch anywhere in the review.
- RNNs (and their gated variants) are more naturally aligned with information that has to accumulate or get resolved gradually across a long sequence.
- Gradient clipping matters more for RNNs, where recurrence makes optimization less stable; it's kept here mainly as a general safeguard, not because it's doing the same load-bearing work it does for a recurrent model.
- A CNN with global max pooling is **order-invariant at the sentence level** — it knows "this pattern exists somewhere," but not "where, relative to any other pattern." That's exactly why `competing_local_cues` and `negation_bigrams` are the two categories where it struggles, and exactly why an LSTM's dedicated memory channel had the edge on negation back in Week 4.

---

## What We Learned

- A 1D convolution turns "read the whole sentence in order" into "slide a small fixed-width detector across it and reuse the same weights everywhere" — a fundamentally different, and much cheaper, way to catch local sentiment patterns.
- Global max pooling is the right tool for "did this pattern fire anywhere," but it throws away _where_ each pattern fired relative to every other pattern — which is precisely the information negation and competing-cues sentences depend on.
- Padding needs careful handling twice over: once to keep contaminated windows out of pooling entirely, and again as a deliberate fallback for reviews shorter than the largest kernel, where the honest move is to stop excluding the padding-touching windows and just take the highest activation across all of them.
- Decomposing a prediction into `pooled_activation × classifier_weight` per filter turns a black-box logit into something you can actually trace back to specific spans of the sentence — useful both for debugging and for building trust in what the model is picking up on.
- TextCNN landed in the middle of last week's sequence models on standard test accuracy (76.7%, versus 71.2–77.5% for RNN/LSTM/BiLSTM/GRU) but showed a very different failure pattern on the challenge set — a reminder that architectures with similar headline accuracy can be strong and weak in completely different, non-overlapping places.

---

## What's Next

Both CNNs and RNNs have now shown a version of the same limitation: neither has a way to weigh one part of a sentence against another based on genuine relevance, but just position or local cue. **Week 6** turns to attention — a mechanism built specifically to let a model decide, for each word, which other words in the sentence actually matter to it.

> **Code:** This week's notebook is `Week_5_cnn_for_text.ipynb` in [`week5-CNN-for-text/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week5-CNN-for-text), supported by `utils.py`.
