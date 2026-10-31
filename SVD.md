# SVD, and why LLMs are built on it

A tutorial in the same spirit as the main README: a tiny worked example first,
then the real thing. SVD (Singular Value Decomposition) is the single piece of
linear algebra that shows up most often in modern language models — in the
embeddings the models start from, in the way they are fine-tuned, and in how
they get compressed to fit on a phone.

```
$ uv run wordvec
word math> build 3

  co-occurrence table (19 × 19)
    = U · S · Vᵀ

  singular values (how much each pattern explains):

    pattern 1   ##########################...  48.2   "pets vs. baking"
    pattern 2   ##############..............   27.1   "action vs. object"
    pattern 3   ####........................    8.9   (noise)
    pattern 4   #...........................    2.3   (noise)
    ...
    pattern 19  ............................    0.0

  keep the top 2 → every word gets 2 numbers instead of 19.
```

The point of the tutorial: the three letters `U`, `S`, `Vᵀ` are the whole trick,
and every use of SVD in an LLM is a variation on *keep the big singular values,
throw the small ones away*.

## The one-sentence version

> **SVD rewrites any matrix as a sorted list of rank-1 ingredients, biggest first.**

That is it. Every row of the matrix, every column, every entry, is a sum of
simple pieces. SVD lists those pieces in order of how much of the matrix they
account for. You keep the first few and the rest is noise you can afford to
lose.

## A 2×2 you can check by hand

```
A = | 3  0 |
    | 4  5 |
```

SVD factors this as `A = U · S · Vᵀ`, three matrices multiplied together:

| Piece | Shape | What it is |
| --- | --- | --- |
| `U` | 2×2, orthonormal | the output directions, ranked |
| `S` | 2×2, diagonal | how strong each direction is (the *singular values*) |
| `Vᵀ` | 2×2, orthonormal | the input directions, ranked |

For this `A` you get `S ≈ diag(6.70, 2.24)`. The first singular value is three
times the second: most of what `A` does is one direction, and only a quarter of
it is the second. Zero the second one out and you have a rank-1 approximation
that is already close to `A`. That is lossy compression that preserves the
dominant behaviour of the matrix.

The jargon shortcut: `U` and `Vᵀ` are *rotations*, `S` is a *stretch*. Any
matrix is a rotation, then a stretch along axes, then another rotation. SVD
finds those axes.

## Why this is useful at all

Three facts fall out of that one factorization, and all of them matter later:

1. **Low-rank approximation is optimal.** Keep the top `k` singular values, zero
   the rest, and you get the best possible rank-`k` approximation to `A` in the
   least-squares sense. (Eckart–Young theorem.) This is the mathematical reason
   "keep the top patterns" is not a hack.
2. **Noise lives in the small singular values.** Random perturbations spread
   mass across *all* singular values roughly evenly, so the small ones are
   mostly noise. Cutting them off denoises the matrix.
3. **The rank tells you the true dimensionality.** If `S` has 50 non-zero
   entries and the rest are zero, the matrix only ever moves things around in a
   50-dimensional subspace, no matter how big it looks.

## Where this repo already uses it

The `build` command in this project is a working SVD demo already — look at
step 3. You have a 19×19 word-by-word co-occurrence table from 16 sentences,
you run SVD, and you keep the top 2 patterns. The result: `dog` and `puppy`
land in the same corner, `oven` and `flour` land in the opposite corner, with
no labels given.

That is Latent Semantic Analysis (LSA) in miniature. It is also where the
story of language models begins. In 1990 Deerwester et al. showed that running
SVD on a word-by-document matrix of newspaper articles produced "topic"
dimensions that behaved like meaning. In 2014 GloVe wrote this out explicitly:
their training objective is provably equivalent to a weighted SVD of the
log co-occurrence matrix.

So the first useful word embeddings — the kind this repo uses — are literally
SVD outputs. The 50 numbers per word in GloVe are the top 50 singular
directions of a giant co-occurrence table.

## From SVD to modern LLMs

Modern transformers do not train their embeddings with SVD. They train them
with gradient descent on next-token prediction. But SVD still shows up in at
least four places that matter:

### 1. Initialization and analysis

The weight matrices inside a transformer are initialized to be roughly
orthogonal (random matrices with controlled singular values), because
Xavier/He initializations are motivated by what SVD says about how a matrix
distorts vectors. Researchers also inspect trained models by SVD-ing attention
and MLP matrices to see what directions the model actually learned. The
observation that "most of the useful signal lives in a few dozen singular
values out of thousands" shows up again and again.

### 2. LoRA fine-tuning

LoRA (Low-Rank Adaptation) is the dominant way people fine-tune large models
on consumer hardware, and it is pure SVD-thinking.

The intuition: when you fine-tune a 7-billion-parameter model on your own data,
the *change* to the weights is low-rank. You are not teaching the model
English from scratch; you are nudging a few hundred directions. So instead of
training the full weight update `ΔW` (billions of numbers), you train
`ΔW = A · B` where `A` is tall-and-thin and `B` is short-and-wide, with inner
dimension `r` typically 8 or 16.

That is exactly the shape SVD produces: `U_k · S_k · V_kᵀ` is a rank-`k`
factorization of a matrix. LoRA skips the SVD and learns the two factors
directly, because that is cheaper, but the hypothesis it is built on —
*fine-tuning updates are low-rank* — is an SVD statement, and it has been
checked empirically by actually SVD-ing fine-tuned updates and seeing the
singular values collapse to near-zero after the first 10–20.

A LoRA adapter for a 7B model is about 10 MB. The base model is 14 GB. The
compression factor is roughly the ratio of singular values you kept to the
ones you ignored.

### 3. Compression and quantization

Shipping a 70B-parameter model to a phone needs it to shrink by 10× or 20×
without losing quality. One of the clean tricks: SVD each weight matrix,
truncate the small singular values, and store the two factors. Methods like
ASVD, SVD-LLM, and FWSVD are variations on this theme — mostly about which
singular values to trust, since naively truncating hurts the output more than
you would expect. The core move is the same: rank-`k` approximation.

### 4. Interpretability and model editing

When researchers want to understand what a transformer learned, SVD is one of
the first tools out. The direction of the largest singular value of an
attention output matrix often corresponds to an interpretable concept
("formality", "sentiment", "last-name-ness"). Editing a model by patching a
single singular direction is a live research area — ROME, MEMIT, and related
methods lean on this.

## The honest caveats

Three things that are easy to overstate:

**SVD alone is not how modern LLMs are trained.** Transformers are trained by
gradient descent on a next-token-prediction loss over trillions of tokens.
SVD is a tool used *around* the training — for initial embeddings in older
systems, for fine-tuning adapters, for compressing finished models, for
analysing them. It is not what makes GPT-style models work.

**"Low-rank" is a hypothesis, not a theorem, for neural network weights.** It
is true that fine-tuning updates look low-rank in practice. It is less true
for the base model weights themselves, where the useful rank is often
surprisingly high. Aggressive SVD compression of base weights loses quality
in ways that LoRA-style low-rank fine-tuning does not.

**The singular vectors are not inherently interpretable.** When you SVD a
co-occurrence table and the top pattern separates pets from baking, that is
real but partly lucky — the dataset is small and the structure is strong. On
Wikipedia-scale data the top singular directions mix dozens of concepts and
have to be rotated or probed to be read. "SVD finds meaning" is a useful
teaching claim but a misleading research one.

## The through-line

If you remember one thing: **word embeddings, LoRA fine-tunes, and model
compression are all the same move at different scales.** Each one says *there
is a huge matrix I cannot afford, but almost all of it is noise or
redundancy; let me keep the few directions that matter*. SVD is the
mathematically correct way to find those directions, and it is still, after
three decades, the sharpest tool we have for the job.

The `build` command in this repo runs that move on 19 words. A trillion-token
pretraining run runs a far messier version of it on 50,000 words at a time,
for billions of weights, for months. The hardware budget is different. The
idea is not.
