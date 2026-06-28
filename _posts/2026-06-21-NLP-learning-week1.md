---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 1) — Turning Text into Numbers"
date: 2026-06-21 17:00:00 +0300
image: "/assets/img/nlp_week1_vectorization.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    text-classification,
    vectorization,
    bag-of-words,
    tf-idf,
    pytorch,
    scikit-learn,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 1) — Turning Text into Numbers

This post kicks off a **9-week hands-on series** taking you from classical NLP all the way to modern Large Language Models. Each week builds on the last, using the same datasets and a growing codebase so you can see exactly how the field evolved — and why each leap mattered.

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** Each post highlights the key ideas; the repo has the complete runnable notebooks.

---

## The Series at a Glance

| Phase                    | Weeks | Focus                           | Anchor Dataset |
| ------------------------ | ----- | ------------------------------- | -------------- |
| **Classical NLP**        | 1–2   | Vectorization, Tokenization     | IMDB           |
| **Deep Learning Bridge** | 3–5   | Embeddings, RNNs, CNNs          | IMDB           |
| **Modern Transformers**  | 6–7   | Attention, BERT & SBERT         | IMDB           |
| **Modern Transformers**  | 8     | Decoder-Only Generation (GPT)   | WikiText-2     |
| **Modern Transformers**  | 9     | Encoder-Decoder Generation (T5) | OPUS Books     |

Week 1 is the foundation everything else stands on: how do you turn raw text into something a model can actually learn from?

---

## Why Start Here?

Before embeddings, before attention, before GPT — there was a simpler question that nobody could skip: **how do you feed words into a math machine?**

Computers don't read. They compute over numbers. So the very first problem in NLP is representation: turning a sentence like _"This film was brilliant"_ into a vector a classifier can reason about.

Week 1 is about that translation layer — and it matters even now, because the limitations of these classical methods are precisely what motivated every architectural breakthrough that follows.

---

## The Dataset: IMDB Movie Reviews

The first 7 weeks anchor on the **IMDB Movie Reviews** dataset — 50,000 labeled reviews, split evenly between positive and negative. It's a clean binary classification problem that let us focus on the method rather than the task. Weeks 8 and 9 swap to generation-focused datasets (WikiText-2 and OPUS Books) as we move from classification into language modeling and translation.

```python
from datasets import load_dataset

imdb_ds = load_dataset("stanfordnlp/imdb")
```

The `utils.py` helper in the repo handles shuffling and splitting into train/val/test subsets cleanly:

```python
train_df, val_df, test_df = get_imdb_ds(
    seed=204,
    train_size=10000,
    val_size=200,
    test_size=250
)
```

We deliberately carve out the validation split from the original training data to keep the test set pure — an important discipline to prevent data leakage.

---

## Step 1: Bag of Words — The Simplest Possible Representation

Before TF-IDF, it helps to understand the simpler idea it improves on: **Bag of Words (BoW)**. BoW answers the representation question in the bluntest way possible — count how many times each word appears in a document, ignore all word order, and use that count vector as your feature. A review becomes a bag of word counts, with no notion of where those words appeared or how they related to each other.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=2000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])
```

The result is a **sparse matrix** — for each review, you get a vector of 2,000 dimensions where most values are zero. The word _"brilliant"_ might be column 843; _"terrible"_ might be column 1,201.

**What BoW gets right:** It's fast, interpretable, and captures strong signal for sentiment — a review full of _"terrible"_ and _"awful"_ is probably negative regardless of sentence structure.

**What BoW gets wrong:** All words are treated as equally important. The word _"the"_ appearing 50 times scores higher than _"cinematography"_ appearing 5 times, despite carrying almost no signal. Common words dominate the representation.

TF-IDF is the direct fix for this.

---

## Step 2: TF-IDF — Smarter Counting

The vectorization method we use in the notebooks for modeling is **TF-IDF (Term Frequency–Inverse Document Frequency)**. It answers a deceptively simple question: how important is a word to a specific document, relative to the rest of the corpus?

To understand it, you need to be precise about two distinct quantities:

**Term Frequency (TF):** How many times a term appears in a **single document**. If the word _"brilliant"_ appears 4 times in one review, its TF for that review is 4. This is purely local — it says nothing about other documents.

**Document Frequency (DF):** How many **documents in the entire corpus** contain the term. If _"the"_ appears in all 25,000 training reviews, its DF is 25,000. If _"cinematography"_ appears in only 300 reviews, its DF is 300.

**Inverse Document Frequency (IDF)** flips DF into a weight — words with high DF (ubiquitous, uninformative) get penalized; words with low DF (rare, distinctive) get boosted.

The final score combines both:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{1 + N}{1 + \text{DF}(t)} + 1$$

Where N is the total number of documents in the corpus.

**Intuition:** _"The"_ has a high TF in almost every document, but its DF is also enormous — so IDF crushes its weight to near zero. _"Cinematography"_ appearing 5 times in a single review is genuinely distinctive because most reviews never use it — high TF, low DF, high TF-IDF score.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=2000)
X_train_tfidf = tfidf.fit_transform(train_df["text"])
X_test_tfidf  = tfidf.transform(test_df["text"])
```

The result is a **sparse matrix** — for each review, a vector of 2,000 dimensions where most values are zero. Only the terms that actually appear in that review have non-zero weights.

---

## Step 3: From Sparse Matrix to PyTorch Tensor

Sklearn vectorizers produce **scipy sparse matrices** — efficient storage but incompatible with PyTorch's training loop. The conversion step is worth understanding:

```python
import torch

# Convert sparse → dense → tensor
X_train_tensor = torch.tensor(
    X_train_tfidf.toarray(), dtype=torch.float32
)
y_train_tensor = torch.tensor(
    train_df["label"].values, dtype=torch.long
)
```

For 10,000 reviews × 2,000 features, this dense tensor is ~80MB. That's manageable here, but it's exactly why we'll move to embedding layers in Week 3 — dense representations don't scale.

The repo's `utils.py` includes `save_sklearn_features()` and `load_sklearn_features()` to cache these so you don't recompute them across notebooks:

```python
save_sklearn_features(X_train, y_train, X_val, y_val, X_test, y_test,
                      base_filename='imdb_sklearn')
```

---

## Step 4: The Opaque Box vs. The Clear Box

With features ready, Week 1 trains two flavors of classifier on the same data — and the contrast is the point.

### The Opaque Box: Scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)
print(accuracy_score(y_test, lr.predict(X_test)))
```

One line to train. You get accuracy. You don't see how.

### The Clear Box: PyTorch Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()        # Reset gradients — critical to prevent accumulation from the previous epoch
    logits = model(X_batch)          # Forward pass
    loss = criterion(logits, y_batch) # Loss calculation
    loss.backward()                   # Backward pass
    optimizer.step()                  # Weight update
```

More verbose — but now you see exactly what's happening at each step. One detail worth calling out explicitly: `optimizer.zero_grad()` **must** be called before every backward pass. PyTorch accumulates gradients by default, so without this reset, gradients from the previous batch would compound into the current update, corrupting training.

Every deep learning framework you'll use — Hugging Face `Trainer`, PyTorch Lightning — is abstracting this exact loop. Understanding it here pays dividends in every subsequent week.

---

## Step 5: From Linear to Non-Linear — Adding a Hidden Layer (MLP)

Everything so far — sklearn's Logistic Regression and the PyTorch `LinearModel` — is doing the same thing mathematically: a single weighted sum of TF-IDF features, squashed through Sigmoid (or Softmax for multiple classes). No matter how you train it, a linear model can only draw a straight-line boundary between classes.

Week 1's multi-class notebook (`multiclass-classification-pytorch-vs-sklearn.ipynb`) switches datasets to **20 Newsgroups** — text grouped into 20 topics instead of 2 sentiment classes — and uses that switch to ask a different question: what happens if we let the model learn _non-linear_ combinations of words instead of just a weighted sum?

The answer is the simplest possible neural network: a **Multi-Layer Perceptron (MLP)**. Concretely, it inserts one hidden layer between the TF-IDF input and the output:

```python
class MLPModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, X):
        return self.layers(X)
```

Three pieces matter here, conceptually:

**The hidden layer itself.** `nn.Linear(feature_dim, hidden_dim)` projects the sparse TF-IDF vector into a smaller, learned space (here, 256 dimensions). On its own this is still just a linear transformation — nothing new yet.

**ReLU — where the non-linearity actually comes from.** `ReLU(x) = max(0, x)` zeroes out negative values and passes positive ones through unchanged. That's it — but stacking a linear layer, a non-linearity, and another linear layer lets the network approximate functions a single linear layer mathematically cannot, like combinations of features that only matter together. (TF-IDF itself is still order-blind going in, so the MLP can only combine the _features it's given_ — it can't recover word order that was never in the input.)

**Dropout — fighting overfitting, not adding capacity.** `nn.Dropout(0.5)` randomly zeroes half the hidden units on every training step. This forces the network to not rely on any single hidden neuron too heavily, which matters more as you add capacity — a plain linear model has no neurons to drop in the first place.

The output layer simply maps the 256 hidden units down to 20 raw scores — one per newsgroup topic — and `CrossEntropyLoss` (the multi-class generalization of the binary loss from Step 4) handles the Softmax and the loss calculation in one numerically stable step, the same way `BCEWithLogitsLoss` did for binary classification.

**The interesting part isn't that the MLP exists — it's whether the extra capacity actually helps.** On this dataset, the MLP, plain Logistic Regression, and Naive Bayes all land in a similar performance range. That's not a bug — it's the same TF-IDF ceiling from Step 2 showing up again: when the _input features_ are the bottleneck (a fixed bag of word counts with no order or semantics), adding a hidden layer gives the model more ways to combine those same limited features, but it can't manufacture information that was never in the input. The MLP is a genuine architectural step forward — it's also a preview of exactly the kind of "more parameters, same ceiling" pattern we'll keep probing as the series moves into embeddings (Week 3) and beyond.

---

## What the Results Tell Us (and What They Don't)

The goal of Week 1 is not to chase accuracy numbers — it's to understand the mechanics of each approach and where they break down. That said, classical TF-IDF classifiers perform respectably on IMDB, and comparing Logistic Regression, SVM, Tree models, and a simple PyTorch neural network gives you a concrete feel for the tradeoffs between the opaque-box and clear-box approaches.

**The concept is the takeaway, not the score.** A well-tuned linear model on TF-IDF features is a genuinely strong baseline — and building intuition for _why_ it works prepares you to reason clearly about why later architectures improve on it.

---

## What's Missing — and Why It Matters

These methods have hard ceilings:

**Order blindness:** _"The film was not boring"_ and _"The film was boring"_ produce nearly identical TF-IDF vectors. Sentiment that depends on negation, contrast, or sentence structure is invisible. RNNs (Week 4) introduce a structurally different idea — treating text as a sequence where order carries meaning — but whether that actually resolves negation in practice, on a dataset this size, is something we'll test with a small handwritten benchmark rather than assume going in.

**Vocabulary explosion:** Even 2,000 features is already a meaningful constraint, and real corpora have millions of unique tokens — you can't scale a sparse matrix indefinitely. Embeddings (Week 3) take a different path: instead of one dimension per vocabulary word, they map words into a fixed, dense, low-dimensional space. That sidesteps the scaling problem; it doesn't make the underlying vocabulary any smaller.

**No semantic understanding:** _"brilliant"_ and _"masterpiece"_ are completely unrelated in TF-IDF space, despite meaning similar things. There's no notion of word similarity. Embeddings introduce the geometry needed to represent that kind of similarity — whether it shows up as measurably better classification on our specific dataset and task is a separate question we'll check rather than declare upfront.

**No scalable phrase structure:** TF-IDF can capture bigrams and trigrams via `ngram_range=(1,2)`, but the vocabulary explodes combinatorially — every word pair becomes a new feature — and the model still can't generalize across positions or share what it learns about a phrase from one part of a document to another. CNNs (Week 5) introduce a different mechanism: sliding filters that can, in principle, detect a phrase pattern anywhere in the text with far fewer parameters than an n-gram vocabulary would need.

Each of these gaps points to a different architectural idea worth understanding on its own terms. That's not the same as saying each one cleanly closes the gap it's paired with — we're still working with the same modest dataset and a learning-focused scale, not a production pipeline. But the progression through the series isn't arbitrary: each new architecture is a direct, motivated response to a specific limitation you've seen firsthand in Week 1, and the later posts will check honestly whether it actually delivers, rather than assuming it does because the theory says it should.

---

## The Complete LLM Learning Stack

This series follows a deliberate progression:

**Phase 1 — Classical NLP (Weeks 1–2):** Establish baselines. Understand vectorization and tokenization from first principles.

**Phase 2 — Deep Learning Bridge (Weeks 3–5):** Move to dense representations. Introduce sequence modeling and spatial features.

**Phase 3 — Modern Transformers (Weeks 6–9):** Attention, BERT, GPT, T5. Fine-tuning pretrained models for real tasks.

Each phase inherits the previous one's datasets and artifacts, so the progression is concrete rather than theoretical.

---

## Conclusion

Week 1 establishes something easy to underestimate: a strong classical baseline and a clear mental model of what a machine learning pipeline actually does — vectorize, represent, train, evaluate.

The numbers look good. But the ceiling is visible. That tension is what drives the next eight weeks.

**Next up — Week 2: The Mechanics of Meaning (Tokenization).** We open the opaque box of how modern NLP systems actually split text, and implement WordPiece and Byte Pair Encoding(BPE) from scratch.

> **Code:** All notebooks for this week are in [`week1-classical-nlp/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week1-classical-nlp) in the repo. The three notebooks are `classical-ML-vectorization.ipynb`, `binary-classification-pytorch-vs-sklearn.ipynb`, and `multiclass-classification-pytorch-vs-sklearn.ipynb`, supported by `utils.py`.
