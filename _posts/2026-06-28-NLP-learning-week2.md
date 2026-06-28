---
layout: post
title: "Zero to Hero: NLP Classification & Generation (Week 2) — Before Meaning: How Text Becomes Tokens"
date: 2026-06-28 17:00:00 +0300
image: "/assets/img/nlp_week2_tokenization.png"
tags:
  [
    nlp,
    ai,
    machine-learning,
    tokenization,
    bpe,
    wordpiece,
    byte-level-bpe,
    huggingface,
    TechBlog,
  ]
mathjax: true
---

# Zero to Hero: NLP Classification & Generation (Week 2) — Before Meaning: How Text Becomes Tokens

This is Week 2 of the **9-week hands-on series** taking us from classical NLP to modern Large Language Models. Week 1 turned text into numbers using TF-IDF and trained classifiers on top of those numbers. Week 2 opens the box one level deeper: **how does text become "words" in the first place?**

> **Full code for all weeks lives in the [GitHub repo](https://github.com/mdossett204/AI-maths-foundations/blob/main/NLP-learning-notebooks/nlp_learning_readme.md).** This week's notebook (`Week_2_Tokenization.ipynb`, supported by `utils.py`) is in the `week2-tokenization/` folder.

---

## The Series at a Glance

| Phase                    | Weeks | Focus                           | Anchor Dataset |
| ------------------------ | ----- | ------------------------------- | -------------- |
| **Classical NLP**        | 1–2   | Vectorization, Tokenization     | IMDB           |
| **Deep Learning Bridge** | 3–5   | Embeddings, RNNs, CNNs          | IMDB           |
| **Modern Transformers**  | 6–7   | Attention, BERT & SBERT         | IMDB           |
| **Modern Transformers**  | 8     | Decoder-Only Generation (GPT)   | WikiText-2     |
| **Modern Transformers**  | 9     | Encoder-Decoder Generation (T5) | OPUS Books     |

---

## Why Tokenization?

`CountVectorizer` and `TfidfVectorizer` from Week 1 quietly did something important for us: they decided what counts as a "word." Every modern natural language processing (NLP) pipeline — from a simple bag-of-words model to GPT — has to answer that question explicitly before anything else can happen. **Tokenization** is the step that turns a raw string into the discrete units a model actually operates on.

It sounds like a solved problem until you hit messy real-world text: emojis, accented characters, Chinese, Japanese, Korean (CJK) scripts, URLs, contractions. The strategy you pick determines how big your vocabulary is, how long your sequences get, and whether your model can handle text it's never seen before.

---

## The Setup: IMDB + a Stress Test

We keep IMDB as the anchor corpus — same dataset as Week 1, now used to train and evaluate tokenizers instead of classifiers. To see where each tokenization strategy actually breaks, we add a small **stress test set** designed to be deliberately uncomfortable:

```python
stress_texts = [
    "I loved it 😂🔥",
    "café naïve résumé",
    "今天天气很好",
    "Check https://example.com #NLP",
    "price=$19.99, don't miss it!",
]
```

Five lines, five different failure modes: emoji, accented Latin characters, CJK script, URLs/hashtags, and punctuation-heavy text. Every tokenizer in this post gets run against the same IMDB sample _and_ this stress set, so the comparisons are concrete rather than theoretical.

---

## Step 1: Word-Level Tokenization

The simplest strategy: split on whitespace and punctuation.

```python
def word_tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
```

On a real IMDB review, this looks reasonable:

```
num_tokens: 195
tokens[:20]: ['We', 'love', 'movies', 'of', 'all', 'kinds', '.', 'This', 'movie', 'was', ...]
```

The stress test is where it starts to show its limits:

```
'今天天气很好'       -> ['今天天气很好']                          (1 token — the whole sentence)
'Check https://example.com #NLP' -> ['Check', 'https', ':', '/', '/', 'example', '.', 'com', '#', 'NLP']
```

The Chinese sentence collapses into a single, useless "word" because there's no whitespace to split on. The URL explodes into ten fragments because every punctuation character becomes its own token. **Word-level tokenization assumes whitespace-delimited, Latin-script text** — an assumption that quietly breaks the moment your input doesn't match that shape.

---

## Step 2: Character-Level Tokenization

The opposite extreme: split into individual characters.

```python
def character_tokenizer(text: str) -> List[str]:
    return list(text)
```

This fixes the out-of-vocabulary (OOV) problem completely — every input is just a sequence of characters, and the vocabulary is tiny (47 unique characters for our IMDB sample, versus 106 unique word tokens). The cost is sequence length: the same review goes from 195 word tokens to **764 character tokens**. On the stress set, the Chinese sentence correctly splits into its six characters instead of collapsing into one. But that robustness comes at a steep efficiency cost — four times the sequence length for the same text, which matters a lot once attention cost scales quadratically with sequence length (more on that later).

Word-level and character-level are two ends of the same tradeoff: vocabulary size versus sequence length. **Subword tokenization exists to sit in between.**

---

## Step 3: Subword Tokenization — WordPiece and Byte-Level Byte Pair Encoding (BPE) (Pretrained)

Before building anything from scratch, it's worth seeing what production tokenizers actually do, using Hugging Face's `AutoTokenizer`.

**BERT's WordPiece** keeps frequent words whole and splits rare ones into pieces marked with `##`:

```python
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tokens = bert_tokenizer.tokenize(sample_text)
# num_tokens: 199
# tokens[:20]: ['we', 'love', 'movies', 'of', 'all', 'kinds', '.', 'this', ...]
```

On the stress set, WordPiece's `[UNK]` token shows up exactly where you'd expect:

```
'I loved it 😂🔥'   -> ['i', 'loved', 'it', '[UNK]']        has_[UNK]=True
'café naïve résumé' -> ['cafe', 'naive', 'resume']          has_[UNK]=False
'今天天气很好'        -> ['[UNK]', '天', '天', '[UNK]', '[UNK]', '[UNK]']  has_[UNK]=True
```

Interesting detail: `café` becomes `cafe` and `résumé` becomes `resume` — BERT's uncased vocabulary normalizes away the accents, so no `[UNK]` is needed there. But emoji and most CJK characters genuinely aren't in BERT's vocabulary, so they fall back to `[UNK]` — information is lost, not just compressed.

**GPT-2's byte-level BPE** takes a structurally different approach — it never gives up on a character, because it doesn't operate on characters at all. It operates on UTF-8 _bytes_:

```python
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_tokens = gpt2_tokenizer.tokenize(sample_text)
# num_tokens: 184
# tokens[:20]: ['We', 'Ġlove', 'Ġmovies', 'Ġof', 'Ġall', 'Ġkinds', '.', 'ĠThis', ...]
```

(That `Ġ` isn't a typo — GPT-2 represents a leading space as part of the following token, using a special character standing in for byte 32.)

On the same stress set, **zero `[UNK]` tokens, ever**:

```
'I loved it 😂🔥'   -> ['I', 'Ġloved', 'Ġit', 'ĠðŁĺ', 'Ĥ', 'ðŁ', 'Ķ', '¥']
'今天天气很好'        -> ['ä»', 'Ĭ', 'å¤©', 'å¤©', 'æ°', 'Ķ', 'å¾', 'Ī', 'å¥', '½']
```

The emoji and Chinese characters get chopped into byte-level fragments, and decoding them back gives you the exact original text. **Byte-level BPE can represent anything you can encode in UTF-8, because that's literally all it sees.** This is why it's the default for modern decoder LLMs.

### Encoding Primer: American Standard Code for Information Interchange (ASCII) vs. Unicode vs. UTF-8

This is worth being precise about, because it's the whole reason byte-level tokenization works:

- **ASCII**: 128 symbols for basic English letters, digits, and punctuation. `A` = 65.
- **Unicode**: a universal standard assigning every character a unique code point — `U+0041` for `A`, `U+4E2D` for `中`, `U+1F92A` for `🤪`.
- **UTF-8**: the encoding that turns Unicode code points into actual bytes (1–4 bytes per character). ASCII stays 1 byte, so UTF-8 is backward-compatible with it.

```python
s = "Aé中🤪"
print([f"U+{ord(ch):04X}" for ch in s])
# ['U+0041', 'U+00E9', 'U+4E2D', 'U+1F92A']
print(list(s.encode("utf-8")))
# [65, 195, 169, 228, 184, 173, 240, 159, 164, 170]
```

`A` is one byte. `é` is two. `中` is three. `🤪` is four. A byte-level tokenizer works on this final list of integers (0–255) — there's no such thing as an "unknown byte," which is exactly why it never needs an `[UNK]` token.

---

## Special Tokens: Tokenization vs. Input Formatting

It's worth separating two ideas that get conflated: **tokenization** splits text into pieces; **input formatting** wraps those pieces with the control tokens a specific model expects.

```python
bert_ids_plain = bert_tokenizer.encode("Hello world!", add_special_tokens=False)
bert_ids_special = bert_tokenizer.encode("Hello world!", add_special_tokens=True)
# plain:   ['hello', 'world', '!']
# special: ['[CLS]', 'hello', 'world', '!', '[SEP]']
```

BERT wraps every input with `[CLS]` (classification token) / `[SEP]` (separator token). GPT-2, by contrast, has no dedicated `[CLS]`/`[SEP]` — `bos_token` and `eos_token` both default to the same `<|endoftext|>`, and there's no `pad_token` at all unless you set one yourself. Special tokens are not a property of "tokenization in general" — they're a property of how a specific architecture was trained to read its input.

---

## Quick Comparison (Corpus-Wide)

Running each tokenizer over the first 2,000 IMDB reviews:

| Method         | Avg. tokens | Median | Min | Max  |
| -------------- | ----------- | ------ | --- | ---- |
| Word           | 287.79      | 218.0  | 16  | 1600 |
| Character      | 1276.16     | 964.0  | 65  | 7382 |
| BERT WordPiece | 303.84      | 230.0  | 19  | 1681 |
| GPT-2 Byte-BPE | 290.47      | 221.0  | 16  | 1659 |

The qualitative tradeoffs line up exactly with what you'd predict:

| Method         | OOV risk        | Vocab size | Seq. length                       | Multilingual | Emoji/Symbols |
| -------------- | --------------- | ---------- | --------------------------------- | ------------ | ------------- |
| Word           | High            | Largest    | Comparable to subword             | Weak         | Weak          |
| Character      | Very low        | Smallest   | Longest (4x+)                     | Strong       | Strong        |
| WordPiece      | Lower than word | Medium     | Comparable to word                | Medium       | Medium        |
| Byte-level BPE | Very low        | Medium     | Comparable to word, often shorter | Strong       | Strong        |

Character-level is the real outlier here — 4x+ the sequence length of every other method, for almost no benefit on clean English text. Word, WordPiece, and byte-level BPE all land in a similar range; which one is actually shortest varies by document rather than following a strict ordering. Our own single-review numbers show this directly: GPT-2's byte-level BPE produced **184** tokens, _fewer_ than our plain word tokenizer's **195** — punctuation gets split into its own tokens under word-level splitting, while a well-trained subword vocabulary can pack a common multi-character span (a whole word, a frequent suffix) into a single token. "Word-level is shortest" isn't a safe claim; "character-level is the outlier" is.

---

## From Scratch: Building BPE

Pretrained tokenizers are a black box if you stop here. The rest of the notebook builds **BPE and WordPiece from scratch** on a 5,000-review IMDB subset, so the mechanics stop being theoretical.

### The BPE Algorithm

1. Pre-tokenize into words, split each word into characters plus an end-of-word marker: `"hello"` → `h e l l o </w>`.
2. Count adjacent symbol pairs across the whole corpus, weighted by word frequency.
3. Merge the most frequent pair everywhere it occurs (e.g. `l + l → ll`).
4. Repeat until the vocabulary hits the target size or no pair meets the minimum frequency.
5. To tokenize new text, apply the learned merges _in the order they were learned_.

```python
bpe_merges = train_bpe(imdb_corpus[:5000], vocab_size=1000, min_freq=2, byte_level=False)
bpe_tokens_scratch = bpe_tokenizer(sample_text, bpe_merges, byte_level=False)
# num_tokens: 281
# tokens[:20]: ['W', 'e</w>', 'love</w>', 'movies</w>', 'of</w>', 'all</w>',
#               'k', 'in', 'ds</w>', '.</w>', 'This</w>', 'movie</w>', 'was</w>',
#               'fil', 'm', 'ed</w>', 'in</w>', 'F', 'in', 'l']
```

281 tokens against BERT's 199 and GPT-2's 184 — worse than production tokenizers, as expected from a vocabulary trained on 5,000 reviews instead of billions of words, but the merges it does learn (`love</w>`, `movies</w>`, `all</w>`) are recognizably correct whole words. The mechanics check out.

### Byte-Level BPE (Scratch)

Byte-level BPE trains the exact same merge algorithm, just over UTF-8 bytes instead of characters, so any input is representable. The key design choice is _how_ each byte gets represented as a mergeable symbol. The natural-looking option — representing each byte as its decimal string (`108`, `101`, `111`...) — turns out to be the wrong one: merging those strings directly produces symbols like `"108111"`, which are visually meaningless and structurally ambiguous, since different byte pairs can concatenate into the same digit string.

The right approach is the trick GPT-2 actually uses: map each of the 256 byte values to its own dedicated, printable Unicode character, so merges stay legible and reversible.

```python
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~")+1))
          + list(range(ord("¡"), ord("¬")+1))
          + list(range(ord("®"), ord("ÿ")+1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

BYTE_ENCODER = bytes_to_unicode()
```

Printable ASCII bytes (33–126) map to themselves. Everything else — control characters, the space character, bytes that aren't valid standalone UTF-8 — gets remapped to a printable stand-in character above code point 256. That's exactly why GPT-2's tokenizer shows a space as `Ġ` instead of breaking.

Here's the sample text tokenized with byte-level BPE:

```python
bpe_merges_byte = train_bpe(imdb_corpus[:5000], vocab_size=1000, min_freq=2, byte_level=True)
bpe_tokens_byte = bpe_tokenizer(sample_text, bpe_merges_byte, byte_level=True)
# num_tokens: 281
# tokens[:20]: ['W', 'e</w>', 'love</w>', 'movies</w>', 'of</w>', 'all</w>',
#               'k', 'in', 'ds</w>', '.</w>', 'This</w>', 'movie</w>', 'was</w>',
#               'fil', 'm', 'ed</w>', 'in</w>', 'F', 'in', 'l']
```

Identical to the character-level result, token for token. That's not a coincidence — it's the correct outcome. Every byte in this English-language review falls in the printable ASCII range, where `BYTE_ENCODER` maps each byte to itself. Byte-level and character-level BPE are mathematically the same algorithm on pure ASCII text; the two approaches only diverge once you feed in something a single byte can't represent on its own — accented characters, CJK, emoji, exactly the cases the stress test is built for. The corpus-wide stats over 2,000 reviews bear this out: 469.73 average tokens for character-based scratch BPE versus 469.9 for byte-level — a fraction of a token apart, with the tiny gap coming from the handful of non-ASCII characters scattered through real IMDB text.

---

## From Scratch: Building WordPiece

### The WordPiece Algorithm

1. Word-tokenize the corpus and count word frequencies.
2. Initialize the vocabulary with `[UNK]` and every character seen — first characters of a word standalone, all others prefixed with `##`.
3. Score every adjacent pair: `score(a, b) = freq(a, b) / (freq(a) × freq(b))`.
4. Merge the highest-scoring pair, update the splits.
5. Repeat until the vocabulary hits the target size or no pair clears the minimum frequency.

```python
wp_vocab = train_wordpiece(imdb_corpus[:5000], vocab_size=1000, min_freq=2)
wp_tokens_scratch = word_piece_tokenizer(sample_text, wp_vocab)
# num_tokens: 627
# tokens[:20]: ['W', '##e', 'l', '##o', '##v', '##e', 'm', '##o', '##v', '##i',
#               '##e', '##s', 'o', '##f', 'a', '##l', '##l', 'k', '##i', '##n']
```

627 tokens — only marginally better than raw character-level tokenization on the same text (764 tokens) and far worse than scratch BPE's 281, despite an identical vocabulary budget and an identical training corpus. Why would the same budget, on the same data, produce such different results?

### Why Did WordPiece Perform Worse Here?

This isn't a bug — it's the merge _criterion_ itself, and it's worth understanding precisely. WordPiece's score, `freq(a,b) / (freq(a) × freq(b))`, is a mutual-information-style measure: it rewards pairs that show up together far more often than chance would predict, _relative to how rare each symbol already is_. On a small corpus, this systematically favors merging rare character pairs that happen to co-occur once or twice — the denominator is tiny, so the score spikes — over genuinely common English patterns like `ing</w>` or `the</w>`.

BPE has no such bias. It just merges whatever pair is most frequent, full stop, which means it converges almost immediately onto common affixes and whole short words regardless of corpus size. WordPiece's smarter-looking score is exactly what makes it slower to converge on a small corpus — it needs a much larger training set before the "rare-but-correlated" pairs stop outranking the genuinely common ones. BERT's production WordPiece vocabulary was trained on billions of words; ours was trained on 5,000 reviews. The gap in this notebook is the gap you'd expect.

---

## Compare Scratch vs. Pretrained

Side by side, first 30 tokens of the same review:

```
Pretrained BERT WordPiece:  ['we', 'love', 'movies', 'of', 'all', 'kinds', '.', 'this', ...]
Pretrained GPT-2 BPE:       ['We', 'Ġlove', 'Ġmovies', 'Ġof', 'Ġall', 'Ġkinds', '.', 'ĠThis', ...]
Scratch BPE:                ['W', 'e</w>', 'love</w>', 'movies</w>', 'of</w>', 'all</w>', 'k', 'in', ...]
Scratch BPE (byte-level):   ['W', 'e</w>', 'love</w>', 'movies</w>', 'of</w>', 'all</w>', 'k', 'in', ...]
Scratch WordPiece:          ['W', '##e', 'l', '##o', '##v', '##e', 'm', '##o', '##v', '##i', ...]
```

And across the full 2,000-review subset:

| Method             | Avg. tokens | Median | Min | Max  |
| ------------------ | ----------- | ------ | --- | ---- |
| BERT WordPiece     | 303.84      | 230.0  | 19  | 1681 |
| GPT-2 Byte-BPE     | 290.47      | 221.0  | 16  | 1659 |
| Scratch BPE        | 469.73      | 349.5  | 24  | 2829 |
| Scratch BPE (byte) | 469.9       | 349.5  | 24  | 2829 |
| Scratch WordPiece  | 1049.2      | 791.5  | 55  | 6120 |

Scratch BPE lands roughly 1.5x the sequence length of production tokenizers — a real but modest gap, explained entirely by vocabulary scale. Scratch WordPiece lands at over 3x — the convergence gap from the section above, compounding across an entire corpus instead of one review.

A second, smaller-scale check on a single budget sentence makes the same point even more concretely:

```
Text: "This is a short paragraph to compare token budgets across tokenizers.
       It includes contractions, punctuation, and a URL: https://example.com."

BERT WordPiece tokens:        36
GPT-2 Byte-BPE tokens:        31
Scratch BPE tokens:           61
Scratch BPE byte-level tokens: 61
Scratch WordPiece tokens:      122
```

Same text, four different token budgets. **Context length is measured in tokens, not characters or words** — this is exactly why that distinction matters in practice.

---

## Tying It Together: Tokenization → Embeddings

A modern large language model (LLM) pipeline, end to end:

1. Raw text
2. Tokenizer → token strings
3. Tokenizer → token IDs (integers)
4. Input formatting (special tokens, padding, attention masks)
5. Embedding lookup (token IDs → vectors)
6. Positional information added
7. Transformer layers

Each token ID is just a row index into an embedding matrix — which is exactly why vocabulary size matters architecturally, not just statistically. A bigger vocabulary means a bigger embedding table, more parameters, more memory.

```python
toy_vocab_size = 10
embedding_dim = 4
emb = torch.nn.Embedding(toy_vocab_size, embedding_dim)

toy_ids = torch.tensor([1, 3, 7, 3])
toy_vectors = emb(toy_ids)
# Token IDs: [1, 3, 7, 3]
# Embedding shape: torch.Size([4, 4])
```

Notice token ID `3` appears twice in the input and maps to the identical embedding vector both times — at this stage, embeddings are just a per-ID lookup table, with no notion of similarity between IDs and no awareness of sequence order. **Week 3** closes the similarity gap: token IDs become dense, _learned_ vectors where nearby points in the embedding space actually mean something. Order-awareness is a separate gap — Week 3's Deep Averaging Network (DAN) architecture pools those embeddings by averaging them, which is simple but still throws sequence order away entirely. That gap doesn't close until **Week 4**, where Recurrent Neural Networks (RNNs) treat text as a sequence where position and order carry meaning.

**Which tokenizer family shows up where in practice:** BPE-family methods (often byte-level) dominate modern decoder-only LLMs; WordPiece is the BERT-family standard. The common thread across both: byte-level handling is what makes a tokenizer robust to literally any input, which is exactly why it's become the default choice for general-purpose LLMs.

---

## What We Learned

- Word-level is simple but brittle — OOV-heavy, and falls apart on non-whitespace-delimited scripts.
- Character-level is robust but inefficient — 4x+ the sequence length of subword methods for the same text.
- Subword methods (WordPiece, BPE) are the practical middle ground production systems actually use.
- Byte-level BPE is the strongest default for messy, multilingual, real-world text — and on pure ASCII text, it's mathematically identical to character-level BPE; the two only diverge once non-ASCII bytes enter the picture.
- WordPiece's mutual-information merge score is theoretically elegant but needs a much larger corpus to converge well than raw-frequency BPE does.
- Special tokens belong to **input formatting**, not tokenization itself — and different architectures format that input very differently.

The scratch implementations aren't meant to match production tokenizers — vocab sizes of 1,000 trained on 5,000 reviews were never going to. The goal was to see the mechanics clearly enough that BERT's and GPT-2's tokenizer output stops looking like magic.

---

## What's Next

Tokenization gives us integers. Those integers, on their own, carry no notion of meaning or similarity — token ID 7 isn't "closer to" token ID 3 in any sense that matters. **Week 3** picks up exactly where the embedding demo above left off: turning token IDs into dense, _learned_ vectors where similarity is actually meaningful, then building a Deep Averaging Network on top of them as our first taste of representation learning beyond TF-IDF.

> **Code:** This week's notebook is `Week_2_Tokenization.ipynb` in [`week2-tokenization/`](https://github.com/mdossett204/AI-maths-foundations/tree/main/NLP-learning-notebooks/week2-tokenization), supported by `utils.py`.
