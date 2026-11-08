# wordvec

A small demo for explaining word embeddings to a non-technical audience.
You type an equation made of words, it does the arithmetic, and it tells you
which word the answer lands on.

```
$ uv run wordvec

word math> king - man + woman

  king - man + woman  =  ?

    queen              ########################....  0.852
    throne             #####################.......  0.766
    prince             #####################.......  0.759

  Best answer: QUEEN
```

## Running

```sh
uv run wordvec
```

The first run downloads the pretrained GloVe vectors (`glove-wiki-gigaword-50`,
about 65 MB) into `~/gensim-data` and caches them there.

## Commands

| Input | What it does |
| --- | --- |
| `full` | 14-step chained walkthrough: tour → build → context → attend → generate |
| `tour` | guided 4-step walkthrough, easiest idea first — start here if presenting |
| `build` | where the numbers come from: trains vectors from scratch on 16 sentences |
| Enter / `b` | advance / go back in whichever walkthrough is active |
| `show <word>` | prints the 50 numbers that represent `<word>` |
| `near <word>` | the words closest in meaning to `<word>` |
| `compare <a> <b>` | how related two words are, with a plain-English reading |
| `twins` | the near-identical word pairs, grouped by *why* they're near-identical |
| `context` | one word, two meanings — which sense survived? Motivates attention. |
| `attend` | a toy one-head attention: `apple pie` lands near fruit, `apple ceo` near tech |
| `generate` | 3-step walkthrough — bigram language model trained live on the 16 sentences |
| `king - man + woman` | word arithmetic; any number of `+`/`-` terms works |
| `help` | reprints the intro |
| `quit` | exit |

## Presenting it

The commands are deliberately ordered as a teaching sequence, and `tour` walks
an audience through it with a pause between each step:

1. **`show dog`** — a word is just a list of numbers. Nothing mysterious yet.
2. **`near dog`** — those numbers aren't random; nearby numbers mean nearby
   meanings. This is the idea to land *before* any arithmetic.
3. **`compare dog cat`** vs **`compare dog algebra`** — distance is meaningful.
4. **`king - man + woman`** — so is direction. That's the payoff.

## How words become numbers (`build`)

`build` is the answer to "but where do the 50 numbers come from?". Rather than
explaining it, it trains real vectors from scratch, live, on a 16-sentence
corpus small enough that an audience can check the arithmetic by hand.

It advances one step per command so you control the pace — each panel ends by
telling you what to type next. Every step is also addressable directly, by
number or name, for when a question sends you back two panels:

```
build            step 1
Enter            next step
b                previous step
build 2          jump to step 2
build counts     same thing by name (prefixes work: `build comp`)
counts           shorthand for the same panel
```

The five steps:

1. **The corpus** — 16 sentences, half about pets and half about baking. The
   program is told nothing about what any word means.
2. **Count neighbours** — for each word, tally the words appearing within 2
   places of it. The `dog` and `puppy` rows come out nearly identical, because
   those words keep the same company. *Each row is already a vector.*
3. **Rank the patterns (SVD)** — the 19×19 table has 19 patterns running
   through it. SVD lists them in order of how much of the table each one
   explains; the first two dominate and the tail is nearly zero.
4. **Compress** — keep only the top 2 patterns, so every word gets 2 numbers
   instead of 19. The real model keeps 50 from a 400,000-word vocabulary.
5. **Check it worked** — with 2 numbers per word you can plot them, and the
   pets and the baking have sorted themselves into opposite corners with no
   labels at all. `dog`/`puppy` scores 0.993, `dog`/`oven` only 0.366.

The last panel scales the idea back up: same recipe, 6 billion words of
Wikipedia, 50 numbers per word. It also states the simplification out loud —
real embeddings weight rare pairs differently or train a network to *predict*
neighbours rather than counting them, but "notice what appears together, then
compress" is the honest core.

Step 3 of the tour also sets up the honest caveat: `happy`/`sad` scores **0.689** while
`happy`/`joyful` scores only **0.555**, because opposites appear in all the same
sentences. The model knows how words are *used*, not what they *mean*.

## Two gotchas the program now handles

**Answers are limited to the 50,000 most common words.** The model knows 400,000,
and the rare tail is a trap: those words appeared in so few sentences that each
vector encodes one narrow context and nothing else, which makes them magnets for
any oddly-shaped result. Before this limit, `king - man` answered `vajiravudh`
(a real king of Siam, frequency rank 154,115). Set by `COMMON_WORDS` in the
source if you want to show the effect deliberately.

**Subtracting without adding a word back has no good answer.** `king - man`
leaves a leftover direction — "royalty with the generic-person part removed" —
and no English word means that, so you get a spray of obscure monarchs. The
analogy trick needs all three terms: take a step away from `man`, then land
somewhere by adding `woman` back. The program prints a warning in this case.

Related, and worth deciding how you want to frame it: `doctor - man + woman`
returns **nurse**. The vectors absorbed the biases of the text they were trained
on, which makes this a concrete demonstration of where training-data bias comes
from — but it lands better deliberately than by accident.

## How it works

Each word is a list of 50 numbers (a "vector") learned from how words are used
across Wikipedia. Words used in similar contexts get similar numbers, so
directions in that space line up with meaning: the step from `man` to `woman`
is roughly the same step as from `king` to `queen`.

The program scales each word's vector to the same length, adds and subtracts
them, then looks for the known words closest to the result. The input words are
excluded from the answer, since a word is always closest to itself.

## Chinese version (`wordvec-zh`)

```sh
uv run wordvec-zh
```

The same word arithmetic, but the vectors were learned from Chinese text, so
the math happens in Chinese. You don't need to read it: type English and it is
translated for you, and every answer comes back with pinyin and an English
meaning.

```
词向量> king - man + woman

  Translating your words:
    king       -> 国王 (guó wáng)
    man        -> 男人 (nán rén)
    woman      -> 女人 (nǚ rén)

    王后 (wáng hòu)          ##############......  0.705  queen
    爵士 (jué shì)           #############.......  0.656  knight; Sir
    王子 (wáng zǐ)           #############.......  0.651  prince; son of a king
```

You can also type Chinese directly (`国王 - 男人 + 女人`). Commands: equations,
`near`, `compare`, `show`, and `translate`, which uses **jieba** to cut a
Chinese sentence into words (written Chinese has no spaces, so this has to
happen before any lookup) and glosses each one.

| Piece | What it is |
| --- | --- |
| Vectors | [Tencent AI Lab Chinese embeddings](https://ai.tencent.com/ailab/nlp/en/embedding.html), light edition: 143,613 words x 200 numbers (Apache-2.0, 116 MB) |
| Dictionary | [CC-CEDICT](https://cc-cedict.org/) (CC BY-SA 4.0): English -> Chinese for your input, Chinese -> English and pinyin for the output |
| Word cutting | [jieba](https://github.com/fxsjy/jieba) |

Both downloads land in `~/.cache/wordvec` on first run.

**People's Daily vectors.** The [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
project publishes vectors trained on People's Daily (1946-2017), but only via
Baidu Pan, which can't be scripted. Download `sgns.renmin.word` by hand and run:

```sh
uv run wordvec-zh --vectors ~/Downloads/sgns.renmin.word
```

Things that differ from the English demo:

- **Translation is a guess.** English words map to several Chinese ones (`man`
  could be 男人, 汉子, 男士...). It picks the first dictionary sense, with a
  small `OVERRIDES` table for cases where that is wrong (`doctor` -> 博士, a PhD).
  Each equation prints what it picked and the runner-up options.
- **No `walking - walk + swim`.** Chinese doesn't inflect verbs, so the
  grammatical-tense analogies don't exist; `walking` just becomes 走 like `walk`.
- **Compounds are filtered out of answers.** Chinese builds words by gluing
  characters together, so the nearest neighbour of 医生 (doctor) is often 主治医生
  (attending doctor). Answers that contain an input word are dropped.
- **Walkthroughs are not ported.** `tour`, `build` etc. rely on an English toy
  corpus; only the interactive commands exist in Chinese so far.

## Going deeper: SVD

Step 3 of `build` uses SVD to compress the co-occurrence table. The same move
— keep the big singular values, throw the small ones away — shows up all over
modern LLMs, from GloVe embeddings to LoRA fine-tuning to on-device model
compression. [`SVD.md`](SVD.md) is a tutorial in the same style that walks
through why.
