---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 3) — From IDs to Meaning: Embedding in Vector Space"
date: 2026-07-05 17:00:00 +0300
image: "/assets/img/nlp_week3_embedding.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    embedding,
    word-vectors,
    glove,
    sbert,
    sentence-transformers,
    deep-averaging-network,
    rag,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 3) — From IDs to Meaning: Embedding in Vector Space

This is Week 3 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 2 ended with tokens — the string pieces a BPE tokenizer produces, like `love</w>` or `##ing`. Week 3 picks up from there: those tokens get mapped to stable integer IDs for the first time, and then those IDs become dense, learned vectors where proximity in space actually means something.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_3_embeddings.ipynb`, supported by `utils.py`) is in the `week3-embedding/` folder.

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

## Why Embedding?

Week 2 left us with a clean pipeline: text → tokens. BPE token strings like `love</w>` or `k` and `in` and `ds</w>` — but no integer IDs yet. A token string on its own has no place in a neural network; it needs a stable row index before anything can look it up. And even once it has one, that index carries no notion of meaning — ID `7` isn't mathematically "closer to" ID `3` in any sense a model can use.

**Embeddings** replace that arbitrary index with a dense vector — a point in a continuous space where distance and direction are supposed to carry meaning. The goal of this week is to make that claim concrete in four stages:

1. **Build the vocabulary and token IDs** — turn last week's BPE merges into a stable integer mapping, and raw text into fixed-length ID sequences ready for model input.
2. **Mechanics first** — what an embedding lookup actually is, with no training involved.
3. **Learn one from scratch** — train a small model on IMDB sentiment and see what its embedding space picks up.
4. **Compare against the industry standard** — put that from-scratch embedding next to GloVe and SBERT and see where each one wins and loses.

---

## Step 1: Turning BPE Merges into a Vocabulary, and Text into Fixed-Length Features

Week 2 produced `bpe_merges` — an ordered list of byte-pair merges learned from the IMDB corpus. That list, on its own, isn't a vocabulary yet. A merge like `("l", "l")` tells you _how_ to merge two symbols into `"ll"`, but it doesn't assign that resulting token a stable integer ID. Before any embedding lookup is possible, every distinct token the tokenizer can produce needs exactly one fixed row index — and every raw string needs to become a fixed-length sequence of those indices, padded or truncated to a consistent length so batches of different-length reviews can be stacked into a single rectangular batch tensor.

That's the job of `create_vocab_mapping`:

```python
def create_vocab_mapping(bpe_merges, pad_token="[PAD]"):
    vocab_id_mapping = {BYTE_ENCODER[i]: i for i in range(256)}
    current_id = 256
    for a, b in bpe_merges:
        token = a + b
        if token not in vocab_id_mapping:
            vocab_id_mapping[token] = current_id
            current_id += 1
    vocab_id_mapping[pad_token] = current_id
    return vocab_id_mapping
```

Two details here matter more than they look:

- **The first 256 IDs are reserved for raw bytes**, `0` through `255`, before a single merge is even considered. The keys are `BYTE_ENCODER[i]` — the GPT-2 printable Unicode characters introduced in Week 2 — not string representations of integers like `"0"`, `"1"`, etc. That distinction matters because the BPE tokenizer also uses `BYTE_ENCODER` when producing tokens, so the keys in this vocabulary map directly onto the token strings the tokenizer outputs. Because the BPE is byte-level, every possible input — including bytes that never got merged into anything — already has a guaranteed home in the vocabulary. There's no `[UNK]` fallback needed; the byte-level foundation is what makes that true.
- **IDs are assigned in merge order, not alphabetical or random order.** This has to be deterministic and reproducible, because the vocabulary built here gets saved to disk and reloaded in every later session. If the ID assignment ever shifted between runs, a model trained against one vocabulary mapping would silently misinterpret IDs produced by a different one — a quiet, hard-to-debug failure mode rather than a crash.

With a vocabulary in hand, `convert_text_to_token_ids` turns one string into the actual model-ready inputs:

```python
def convert_text_to_token_ids(text, bpe_merges, bpe_vocab, max_seq, pad_token="[PAD]"):
    tokenized_text = bpe_tokenizer(text, bpe_merges, byte_level=True)
    n = len(tokenized_text)
    padding_mask = [1] * n if n <= max_seq else [1] * max_seq
    if n < max_seq:
        tokenized_text.extend([pad_token for _ in range(max_seq - n)])
        padding_mask.extend([0 for _ in range(max_seq - n)])
    token_ids = [bpe_vocab[token] for token in tokenized_text[:max_seq]]
    return token_ids, padding_mask
```

It's worth being precise about what this function actually returns, because both outputs travel through every stage downstream and it's easy to conflate them.

**`token_ids`** is a flat list of integers of exactly length `max_seq`. Each integer is the row index in the embedding table that corresponds to a BPE token — the same IDs assigned in `create_vocab_mapping`. A short review of 40 tokens padded to `max_seq=128` would produce 40 real IDs followed by 88 copies of `pad_id`. These integers are what gets fed into `nn.Embedding` — the model never sees the token strings, only these indices.

**`padding_mask`** is a parallel list of the same length, containing only `0`s and `1`s. A `1` means "this position is real content — count it." A `0` means "this position is a pad token — ignore it." The mask is built at the same time as the token IDs because they're two views of the same thing: you know exactly which positions are padding as you're building the sequence, so there's no reason to recompute it later.

Concretely, for a review with 40 tokens at `max_seq=128`:

```
token_ids:    [23, 107, 4, ..., real_token_40, pad_id, pad_id, ..., pad_id]
padding_mask: [  1,   1, 1, ...,            1,      0,      0, ...,      0]
                <------- 40 real positions -------> <--- 88 padding positions --->
```

This distinction matters more than it might look. Without the mask, any mean-pooling operation would treat all 128 positions as equally real — averaging 40 tokens of actual sentiment signal with 88 zeros from padding. The model would still learn, but its sentence vectors would be consistently diluted by a ratio of `max_seq / actual_length`. The mask is what lets mean pooling produce a fair average over real tokens regardless of how long each review actually is.

Building this vocabulary and running every review in the corpus through `convert_text_to_token_ids` is the kind of one-time, computationally expensive step that shouldn't rerun on every notebook execution — which is exactly why it's commented out in the notebook itself:

```python
# bpe_merges = train_bpe(texts, vocab_size=500, min_freq=5, byte_level=True)
# save_json("datasets/bpe_merges.json", bpe_merges)

# bpe_vocab = create_vocab_mapping(bpe_merges, pad_token="[PAD]")
# save_json("datasets/bpe_vocab.json", bpe_vocab)
# vocab_config = {
#     "pad_token": pad_token,
#     "pad_id": bpe_vocab[pad_token],
#     "byte_vocab_size": 256,
#     "max_seq": max_seq,
#     "num_merges": len(bpe_merges),
#     "vocab_size": len(bpe_vocab)
# }
# save_json("models/vocab_config.json", vocab_config)

# tokenized_features = []
# padding_masks = []
# for i, text in enumerate(texts):
#     tokenized_text, padding_mask = convert_text_to_token_ids(text, bpe_merges, bpe_vocab, max_seq, pad_token)
#     tokenized_features.append(tokenized_text)
#     padding_masks.append(padding_mask)
# save_torch_dataset(tokenized_features, labels, padding_masks, "datasets/imdb_bpe_dataset.pt")
```

Run once, each of these gets cached to disk — the merges and vocabulary as JSON, the fully tokenized dataset as a single `.pt` tensor file. Every subsequent run just reloads the cached artifacts:

```python
bpe_merges = load_json("datasets/bpe_merges.json")
bpe_vocab = load_json("datasets/bpe_vocab.json")
input_ids, labels, padding_masks = load_torch_dataset("datasets/imdb_bpe_dataset.pt")
```

This matters for more than just speed. `vocab_config.json` is what later guarantees the DAN's embedding table is built with exactly the right `vocab_size` and `pad_id` — values derived once, frozen, and reused everywhere downstream, instead of being recomputed every time the notebook runs.

---

## Step 2: The Embedding Layer Is Just a Lookup Table

Before training anything, it's worth being precise about what `nn.Embedding` actually does, because it's easy to over-mystify. It's a matrix of shape `(vocab_size, embedding_dim)`. Given a token ID, you get back the row at that index. That's the entire operation.

```python
toy_vocab_size = 10
embedding_dim = 4
emb = torch.nn.Embedding(toy_vocab_size, embedding_dim)

toy_ids = torch.tensor([1, 3, 7, 3])
toy_vectors = emb(toy_ids)
```

Token ID `3` shows up twice in that input and returns the exact same vector both times — at this stage, there's no notion of context, no notion of similarity between IDs, and no awareness of order. The "meaning" only shows up after these vectors get trained against some objective. Which is exactly what Step 3 does.

---

## Step 3: From Tokens to a Sentence Vector — the Deep Averaging Network (DAN)

With the vocabulary and tokenized dataset from Step 1 in hand, every sentence already arrives as a fixed-length ID sequence plus a mask. What's new in this step is turning that sequence of token IDs into a single vector representation of the entire sentence.

The simplest way to do that — simpler than an RNN, simpler than attention — is to **average the token embeddings**. That's the entire idea behind a **Deep Averaging Network (DAN)**: embed every token, mean-pool across the sequence (ignoring `[PAD]` positions), then pass that pooled vector through a small feed-forward classifier.

```python
class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_dim, n_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, mask):
        embedding = self.embedding(input_ids)
        mask = mask.unsqueeze(-1).float()
        masked_embedding = embedding * mask
        mean_pool = masked_embedding.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        mean_pool = self.dropout(mean_pool)
        hidden = self.hidden(mean_pool)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        return self.linear(hidden)
```

A few details worth calling out:

- **The mask isn't optional.** Padding tokens get embedded just like real tokens, but they don't represent any actual content. Multiplying by the mask before summing — then dividing by the _mask's_ sum rather than the raw sequence length — is what keeps padding from diluting the average. `clamp(min=1)` is a small but necessary guard against dividing by zero on an all-padding edge case.
- **Mean pooling throws away order on purpose.** `"the movie was good, not bad"` and `"the movie was bad, not good"` average to the same vector. That's the central limitation of a DAN, and it's the reason this architecture is a stepping stone rather than a destination — Week 4's RNNs exist specifically to stop discarding sequence order.
- **`embedding_size=384`, `hidden_dim=768`** in this notebook's configuration — the hidden layer is sized at twice the embedding dimension, a common default rather than a tuned choice. `vocab_size` and `pad_id` come straight from `vocab_config.json`, the artifact cached in Step 1.

Training itself follows a standard supervised binary-classification loop: `BCEWithLogitsLoss`, early stopping on validation loss, a learning-rate scheduler that backs off when validation loss plateaus. None of that is embedding-specific — it's the same training loop that would sit underneath almost any classifier. What _is_ embedding-specific is what comes out the other end: once training finishes, `model.embedding.weight` is no longer an arbitrary lookup table. It's been pulled, by gradient descent, toward a geometry where sentiment-relevant similarity is encoded in distance.

### Saving the Model and the Embedding Layer Separately

Once training is done, the notebook saves two artifacts:

```python
# torch.save(best_model.state_dict(), "models/dan_imdb.pt")
# torch.save(best_model.embedding.state_dict(), "models/imdb_embedding_layer.pt")
```

These are saving two different things, and the distinction matters. `best_model.state_dict()` saves _every_ parameter in the full DAN — the embedding matrix, the hidden layer weights, and the output layer weights. That's the complete classifier, useful if you ever want to resume training or run inference on new reviews.

`best_model.embedding.state_dict()` saves _only_ the embedding matrix — a single tensor of shape `(vocab_size, embedding_size)`. That's all you need to turn token IDs into vectors. The hidden and output layers are irrelevant for that job, so saving them separately keeps things lightweight and explicit about intent.

Both cells are commented out for the same reason as the earlier caching steps: they're one-time operations that shouldn't silently overwrite artifacts already on disk on every notebook run.

### Saving the Dataset Split

One more artifact gets saved alongside the model weights — the train/val/test splits themselves:

```python
# torch.save(train_ds, "datasets/train_ds.pt")
# torch.save(val_ds, "datasets/val_ds.pt")
# torch.save(test_ds, "datasets/test_ds.pt")
```

This is easy to overlook but important. The IMDB dataset in this notebook is re-split from scratch (80% train / 10% val / 10% test) rather than using HuggingFace's original boundaries. Future weeks will reuse the embedding weights trained here — and if those weeks re-split the data independently, even with the same random seed, there is no guarantee the splits align across environments or library versions. Reviews that were in this week's test set could silently end up in a future training set, meaning the model would be evaluated on data it has indirectly already seen through the shared embedding weights.

`train_ds`, `val_ds`, and `test_ds` are PyTorch `Subset` objects — a TensorDataset plus the list of indices that define each split — and they're fully serializable. Future weeks reload them and wrap in new DataLoaders:

```python
train_ds = torch.load("datasets/train_ds.pt")
val_ds   = torch.load("datasets/val_ds.pt")
test_ds  = torch.load("datasets/test_ds.pt")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)
```

The DataLoader configuration (batch size, shuffle) is the only thing specified again, which is appropriate since that may vary week to week. Note that DataLoaders themselves can't be saved directly — they contain non-serializable objects like worker processes and iterators — so the pattern is always: save the dataset, recreate the loader.

### Loading Back Just the Embedding Layer for Sentence Vectors

When the comparison section runs later in the notebook, it doesn't reload the full classifier. It loads the full model's weights only long enough to extract the embedding layer:

```python
dan_model = DAN(vocab_size, embedding_size, pad_id, hidden_dim, 1, dropout)
state = torch.load("models/dan_imdb.pt", map_location="cpu")
dan_model.load_state_dict(state)
dan_embedding = dan_model.embedding
dan_embedding.weight.requires_grad_(False)
```

This is doing something specific. A fresh `DAN` is constructed with the same architecture as training — the weight shapes have to match or `load_state_dict` will raise an error. `torch.load` reads the saved `state_dict` off disk, and `load_state_dict` copies every tensor into the model's parameters in place. Then `dan_model.embedding` is pulled out as a standalone `nn.Embedding` object — this is just a Python attribute access, no copying involved, so `dan_embedding` points directly at the same embedding matrix that was just restored from disk.

`requires_grad_(False)` freezes the weights. Since we're only using `dan_embedding` to produce sentence vectors, not to train anything further, disabling gradients prevents PyTorch from tracking computation history through these tensors — which wastes memory and adds overhead with no benefit. From this point forward, `dan_embedding` is a pure lookup table backed by trained weights.

Alternatively, since we saved the embedding layer weights separately as imdb_embedding_layer.pt, we can skip reconstructing the full DAN entirely and load directly into a standalone nn.Embedding:

```python
dan_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
dan_embedding.load_state_dict(torch.load("models/imdb_embedding_layer.pt"))
dan_embedding.weight.requires_grad_(False)
```

This is cleaner when you only need the embedding layer and have no use for the classifier head — no need to define the full DAN architecture just to immediately discard the hidden and output layers.

---

## Step 4: How Does a Learned Embedding Compare to the Industry Standard?

Training your own embedding from scratch on a few thousand reviews is a useful exercise, but it's worth being honest about where it sits relative to embeddings designed for general use. This notebook compares three approaches side by side on the same handful of sentences:

- **GloVe** — unsupervised word vectors trained on global word co-occurrence statistics across a massive corpus. Captures broad semantic relationships, but every word maps to exactly one static vector regardless of the sentence it's in. `"bank"` means the same thing whether you're talking about a river or a loan.
- **SBERT (Sentence-BERT)** — an encoder-only Transformer using bidirectional attention to build sentence-level embeddings directly, rather than averaging word vectors after the fact. It's _contextual_: the same word contributes differently to the sentence vector depending on what surrounds it.
- **The trained DAN embedding** — learned end-to-end on IMDB sentiment, with no positional information and mean pooling baked in, exactly as described above.

To compare them fairly, we embed the same five sentences with all three, then compute pairwise cosine similarity and visualize it as a heatmap.

For the DAN, `bpe_batch_encode` first tokenizes the sentence list into a batch of `(input_ids, padding_mask)` tensors, then `get_dan_sent_embed` turns those into one sentence vector per sentence:

```python
def get_dan_sent_embed(embedding_layer: nn.Module,
                       input_ids: torch.tensor,
                       padding_mask: torch.tensor) -> torch.tensor:
    # input_ids: [B, S]
    # padding_mask: [B, S]
    with torch.no_grad():
        emb = embedding_layer(input_ids)               # [B, S, D]
        mask = padding_mask.unsqueeze(-1).float()      # [B, S, 1]
        masked = emb * mask
        sent_emb = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]
    return sent_emb
```

This is the same masked mean-pool from `DAN.forward`, pulled out as a standalone function and wrapped in `torch.no_grad()`. It's worth tracing the shapes explicitly:

- `embedding_layer(input_ids)` looks up each token ID in the trained matrix and returns a 3-D tensor `[B, S, D]` — batch size × sequence length × embedding dimension.
- `padding_mask.unsqueeze(-1)` adds a trailing dimension to turn the `[B, S]` mask into `[B, S, 1]`, so it broadcasts correctly across the `D` embedding dimensions when multiplied against `[B, S, D]`.
- `masked.sum(dim=1)` collapses the sequence dimension, summing only the real token vectors (padding positions have been zeroed out by the mask multiply).
- Dividing by `mask.sum(dim=1).clamp(min=1)` divides by the count of real tokens per sequence — not the fixed `max_seq` — giving a true average over content. `clamp(min=1)` guards against the degenerate case of an all-padding sequence causing a divide-by-zero.

The output is `[B, D]` — one vector per sentence, ready to be compared. `torch.no_grad()` is there for the same reason as `requires_grad_(False)` was set on the weights: we're producing representations, not training, so there's no need to build a computation graph.

```python
# DAN sentence embeddings
input_ids, padding_mask = bpe_batch_encode(sentences, bpe_merges, bpe_vocab, max_seq, pad_token)
dan_sent_emb = get_dan_sent_embed(dan_embedding, input_ids, padding_mask)
# GloVe sentence embeddings
glove = load_glove_txt(path="models/glove.6B.300d.txt", dim=300)
glove_sent_emb = torch.stack([get_glove_sent_embed(s, glove, 300) for s in sentences])
# SBERT sentence embeddings
sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert_sent_emb = get_sbert_embed(sbert, sentences)

plot_heatmap(cos_sim_matrix(dan_sent_emb), "DAN Cosine Similarity", sentences)
plot_heatmap(cos_sim_matrix(glove_sent_emb), "GloVe Cosine Similarity", sentences)
plot_heatmap(cos_sim_matrix(sbert_sent_emb), "SBERT Cosine Similarity", sentences)
```

**Why cosine similarity and not, say, Euclidean distance?** Cosine similarity measures the angle between two vectors and deliberately ignores their magnitude. Two sentence embeddings can have very different lengths — longer sentences tend to produce larger raw vectors before normalization — but if they point in the same semantic direction, cosine similarity treats them as close. For embedding comparison, direction is where the meaning lives; magnitude is mostly an artifact of how the vector was produced.

```python
def cos_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return x @ x.T
```

Normalizing every row to unit length before the matrix multiply is what turns a plain dot product into cosine similarity — it's a one-line implementation once you see it, but it's doing real conceptual work.

### What the comparison actually shows

The qualitative pattern across the three heatmaps lines up with what each method is built for, not by coincidence:

- The **trained DAN embedding** does reasonably well on movie-review-flavored similarity, because that's the exact signal it was optimized for — sentences with similar sentiment and similar review vocabulary end up close together.
- **GloVe** is noticeably weaker here. Averaging static, context-free word vectors tends to flatten sentence-level meaning — two sentences that share a few common words can look more similar than they really are, while sentences that _mean_ the same thing using different vocabulary can look less similar than they should.
- **SBERT** is the strongest and most general of the three, because it's purpose-built for sentence-level semantic similarity across arbitrary domains, not just movie reviews.

The honest takeaway isn't "SBERT wins" in some abstract sense — it's that **embedding quality is task- and domain-dependent**. A DAN trained on IMDB sentiment will outperform a general-purpose sentence embedder on IMDB-shaped sentiment tasks if you give it enough data and training time. It just doesn't generalize past that.

---

## Step 5: Real-World Embedding Use — Retrieval (a RAG-Style Demo)

The most common production use of sentence embeddings today isn't classification — it's **retrieval**. Given a query, find the most semantically relevant documents from a larger corpus. This is the embedding step that sits underneath Retrieval-Augmented Generation (RAG): embed the query, embed the documents, rank by cosine similarity, hand the top-_k_ matches to an LLM as context. RAG is a broader topic that goes well beyond NLP and embedding alone — this demo focuses only on the retrieval half to show embeddings doing real work, not as a full treatment of the RAG pipeline.

```python
def semantic_search(query, docs, doc_embs, query_embed_fn, k):
    q_emb = query_embed_fn(query)
    sims = F.cosine_similarity(doc_embs, q_emb.unsqueeze(0), dim=1)
    topk = torch.topk(sims, k=k)
    return [(docs[i], sims[i].item()) for i in topk.indices.tolist()]
```

To stress-test this a little, the notebook builds two query types against a mixed corpus of IMDB-style review snippets and completely unrelated documents (router resets, two-factor authentication, travel tips, personal finance):

```python
imdb_query = "I want a movie with strong acting and an emotional story."
non_imdb_query = "How can I make my online account more secure?"
```

This is where the gap between the three embedding approaches stops being abstract and starts mattering practically:

- On the **IMDB-shaped query**, all three embedders retrieve reasonable matches — the trained DAN embedding included, since it's operating squarely inside its training distribution.
- On the **out-of-domain query about account security**, the trained DAN embedding struggles to retrieve the right snippets. It was never trained to represent concepts like two-factor authentication or password strength — its embedding space simply doesn't have meaningful structure out there. GloVe does somewhat better since its training corpus wasn't domain-restricted, but it still suffers from collapsing distinct sentences into similar averaged vectors. SBERT handles both query types comparably well, because general-domain semantic similarity is precisely its training objective.

The practical lesson generalizes well beyond this toy example: **retrieval quality in a RAG pipeline is bottlenecked by embedding quality.** If the embedding model doesn't represent your domain well, similarity search will confidently return irrelevant context — and a downstream LLM fed irrelevant context tends to produce a confidently wrong answer, not an honestly uncertain one.

---

## What We Learned

- A vocabulary isn't free — BPE merges have to be turned into stable, deterministic integer IDs, and every text has to be padded or truncated into a fixed-length sequence because training the DAN requires a consistent input dimension across all examples in a batch. The embedding lookup itself can handle variable-length sequences; the fixed length is a batching constraint, not an embedding one.
- An embedding layer is a lookup table until something trains it — token ID similarity only becomes meaningful after gradient descent shapes the space.
- Mean pooling (the core of a DAN) is a cheap, effective way to go from token vectors to a sentence vector, but it throws away word order entirely — a real architectural ceiling, not a minor caveat.
- Cosine similarity is the right tool for comparing embeddings because it isolates _direction_ (meaning) from _magnitude_ (mostly an artifact of vector construction).
- A domain-trained embedding (our DAN) can match or beat a general-purpose one _inside_ its training domain, and falls off sharply _outside_ it.
- SBERT's advantage isn't "bigger model" in some vague sense — it's that bidirectional attention captures context before pooling, instead of pooling first and hoping context survives the average.
- Embedding quality is the bottleneck for retrieval-based systems like RAG: bad embeddings don't fail loudly, they fail by quietly returning the wrong context with high confidence.

---

## What's Next

This week closed the _similarity_ gap — token IDs are now points in a space where distance means something. But mean pooling still throws sequence order away completely: `"not bad, actually great"` and `"not great, actually bad"` would produce identical DAN embeddings. **Week 4** picks up exactly there, introducing Recurrent Neural Networks (RNNs) — the first architecture in this series where _position and order_ carry information, not just the bag of tokens present.

> **Code:** This week's notebook is `Week_3_embeddings.ipynb` in [`week3-embedding/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week3-embedding), supported by `utils.py`.
