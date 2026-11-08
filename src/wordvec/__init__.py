"""Word arithmetic demo: type `king - man + woman` and get back `queen`.

Meant to be run in front of a non-technical audience, so everything it prints
is plain English and every number comes with a picture.

The commands are ordered as a teaching sequence -- `tour` walks through it:
    show    a word is just a list of numbers
    near    similar words have similar numbers
    compare distance between two words is meaningful
    math    so is direction: king - man + woman = queen

`build` answers the follow-up question: where did those numbers come from?
It trains real vectors from scratch on a 16-sentence corpus, live.
"""

import random
import re
import readline  # noqa: F401 -- enables arrow-key editing and history in input()
import sys
from collections import Counter, defaultdict

import numpy as np

MODEL_NAME = "glove-wiki-gigaword-50"

# The vocabulary is sorted most-common-first, so this keeps answers to words a
# general audience has actually heard of. Without it, `king - man` answers
# "vajiravudh" (rank 154,115) -- a real king of Siam, but useless in a demo.
# Rare words were trained on so few sentences that their vectors encode one
# narrow context and nothing else, which makes them magnets for leftover vectors.
COMMON_WORDS = 50_000

BANNER = """
==============================================================
  WORD MATH  --  doing arithmetic on the meaning of words
==============================================================

Every word below has been turned into a list of 50 numbers,
learned by reading Wikipedia. Words used in similar ways ended
up with similar numbers -- which lets you add and subtract
meaning the way you would add and subtract money.

  full              THE WHOLE STORY in 14 steps -- tour + build
                    + context + attend + generate, chained. Press
                    Enter to advance, `b` to go back.
  tour              a shorter 4-step overview (easiest idea first)
  build             where the numbers come from: trains vectors
                    from scratch on 16 sentences, one step at a
                    time. Enter advances, `b` goes back, or jump
                    straight to one:
                      build 1  sentences    the text it learns from
                      build 2  counts       tally the neighbours
                      build 3  svd          rank the patterns
                      build 4  compress     squeeze to 2 numbers
                      build 5  plot         did it work?
  generate          same corpus, used to produce text: counts
                    which word follows which, then samples
                    sentences one word at a time:
                      generate 1  follows      what follows what
                      generate 2  probability  normalize
                      generate 3  sample       produce sentences

Or explore directly:

  show <word>       peek at the 50 numbers behind a word
  near <word>       the words closest in meaning to <word>
  compare <a> <b>   how related are two words?
  twins             which words are almost identical -- and why
  context           one word, two meanings: which sense survives?
  attend            a toy attention head that fixes it
  <equation>        the word math itself, for example:

      king - man + woman
      paris - france + italy
      walking - walk + swim
      bigger - big + small

  help              show this message
  quit              exit
"""

# Split on + and - whether or not they are padded with spaces.
SPLIT_RE = re.compile(r"\s*([+-])\s*")

# Thresholds picked by measuring real pairs in this model, not guessed:
# dog/cat 0.92, dog/puppy 0.78, dog/leash 0.48, dog/teaspoon 0.12, dog/algebra -0.05
# A deliberately tiny corpus for `build`. Two topics (pets, baking) with a few
# sentences bridging them, so the learned clusters come out distinct but not
# perfectly separable -- a clean split would give fake-looking scores of 1.000.
TOY_CORPUS = [
    "the dog barked at the stranger",
    "the puppy barked and wagged its tail",
    "the cat purred and licked its fur",
    "my dog wagged its tail at the vet",
    "the vet checked the cat and the puppy",
    "the kitten purred on my lap and wagged its tail",
    "she baked fresh bread in the oven",
    "he baked a sweet cake in the oven",
    "the bread was warm and fresh from the oven",
    "she ate a slice of sweet cake",
    "he ate warm bread with butter",
    "fresh pie baked in the oven tasted sweet",
    "the dog ate warm bread from my hand",
    "the kitten ate a slice of sweet cake",
    "my puppy ate the fresh pie",
    "the cat sat by the warm oven",
]
TOY_STOP = set("the a an and at in on its my he she it was with from of by".split())
TOY_WINDOW = 2
TOY_ROWS = ["dog", "puppy", "cat", "bread", "cake", "oven"]
TOY_COLS = ["barked", "wagged", "tail", "purred", "baked", "sweet", "warm", "ate"]
TOY_PLOT = ["dog", "puppy", "cat", "kitten", "barked", "tail",
            "bread", "cake", "pie", "oven", "baked", "sweet"]

# Pairs found by scoring every combination of the 8,000 most common words and
# then sorting the results into the four reasons two words end up near-identical.
TWIN_GROUPS = [
    (
        "GENUINE SYNONYMS -- different words, same meaning",
        [("protesters", "demonstrators"), ("movie", "film"), ("huge", "enormous")],
        "This is the case people expect: the model noticed these words are\n"
        "    swappable, so it filed them in almost the same place.",
    ),
    (
        "SAME SLOT -- near-identical numbers, but NOT the same meaning",
        [("monday", "tuesday"), ("three", "four"), ("nbc", "cbs")],
        "Monday and Tuesday are the most alike words in the whole model --\n"
        "    yet they are different days. They score high because they appear\n"
        "    in identical sentences, not because they mean the same thing.",
    ),
    (
        "TWO HALVES OF ONE NAME -- they only ever appear side by side",
        [("hong", "kong"), ("buenos", "aires"), ("kuala", "lumpur")],
        "'Hong' is never used without 'Kong', so they share one context\n"
        "    completely. High score, and not even whole words on their own.",
    ),
    (
        "OPPOSITES -- the one that surprises everybody",
        [("buy", "sell"), ("up", "down"), ("increase", "decrease")],
        "Opposites get high scores because they slot into the same\n"
        "    sentences: prices go up, prices go down. The model is reading\n"
        "    the company a word keeps -- which is usage, not meaning.",
    ),
]

RELATEDNESS = [
    (0.70, "very closely related"),
    (0.45, "clearly related"),
    (0.25, "loosely related"),
    (-0.15, "basically unrelated"),
]


def load_model():
    """Load the pretrained vectors, downloading them the first time."""
    import gensim.downloader as api

    print(f"Loading word vectors ({MODEL_NAME})... ", end="", flush=True)
    model = api.load(MODEL_NAME)
    print(f"ready. {len(model.index_to_key):,} words known.\n")
    return model


def parse(expression):
    """Turn 'king - man + woman' into [('+', 'king'), ('-', 'man'), ('+', 'woman')].

    Raises ValueError with an audience-friendly message on bad input.
    """
    parts = [p for p in SPLIT_RE.split(expression.strip()) if p != ""]
    if not parts:
        raise ValueError("Nothing to calculate.")
    if parts[0] in "+-":
        raise ValueError("The equation can't start with + or -.")

    for left, right in zip(parts, parts[1:]):
        if left in "+-" and right in "+-":
            raise ValueError("Two operators in a row -- try `king - man + woman`.")

    terms = [("+", parts[0].lower())]
    rest = parts[1:]
    if len(rest) % 2 != 0:
        raise ValueError("That equation looks unfinished -- a word is missing.")
    for sign, word in zip(rest[::2], rest[1::2]):
        terms.append((sign, word.lower()))
    return terms


def unit(model, word):
    """A word's vector, scaled to length 1 so no single word dominates."""
    vector = model[word]
    return vector / np.linalg.norm(vector)


def bar(score, width=28):
    """Draw a similarity score as a bar, so scores are comparable at a glance."""
    filled = max(0, min(width, round(score * width)))
    return "#" * filled + "." * (width - filled)


def describe(score):
    """Plain-English reading of a similarity score."""
    for threshold, label in RELATEDNESS:
        if score >= threshold:
            return label
    return "unrelated -- they point in opposite directions"


def known(model, word):
    """Check a word is in the vocabulary, complaining in plain English if not."""
    if word in model.key_to_index:
        return True
    print(f"  I don't know the word '{word}'.")
    print("  This demo only knows common English words. Try another.\n")
    return False


def missing_words(model, terms):
    return [word for _, word in terms if word not in model.key_to_index]


def solve(model, terms, topn=5):
    """Add and subtract the words, then find the words nearest the result."""
    query = np.zeros(model.vector_size, dtype=np.float32)
    for sign, word in terms:
        query += unit(model, word) if sign == "+" else -unit(model, word)

    inputs = {word for _, word in terms}
    # Ask for extras so we still have topn left after dropping the input words.
    candidates = model.most_similar(
        positive=[query], topn=topn + len(inputs), restrict_vocab=COMMON_WORDS
    )
    return [(word, score) for word, score in candidates if word not in inputs][:topn]


def show_equation(terms):
    rendered = terms[0][1]
    for sign, word in terms[1:]:
        rendered += f" {sign} {word}"
    return rendered


def handle_equation(model, expression):
    terms = parse(expression)

    unknown = missing_words(model, terms)
    if unknown:
        print(f"  I don't know the word(s): {', '.join(unknown)}")
        print("  This demo only knows common English words. Try another.\n")
        return

    print(f"\n  {show_equation(terms)}  =  ?\n")

    # Subtracting without adding a word back leaves a leftover direction rather
    # than a point near any real word, so the answers tend to look arbitrary.
    if len(terms) > 1 and all(sign == "-" for sign, _ in terms[1:]):
        subtracted = ", ".join(word for sign, word in terms[1:])
        print(f"  Heads up: you took {subtracted} away but didn't add anything")
        print("  back. English has no single word for what's left over, so these")
        print(f"  answers will look odd. Try `{show_equation(terms)} + <word>`.\n")

    results = solve(model, terms)
    best = results[0][0]
    for word, score in results:
        print(f"    {word:<18} {bar(score)}  {score:.3f}")
    print(f"\n  Best answer: {best.upper()}")
    print("  (the bars show how close each word is to the answer)\n")


def handle_near(model, word, topn=8):
    word = word.lower()
    if not known(model, word):
        return
    print(f"\n  Words used most like '{word}':\n")
    for other, score in model.most_similar(word, topn=topn, restrict_vocab=COMMON_WORDS):
        print(f"    {other:<18} {bar(score)}  {score:.3f}")
    print("\n  Nobody gave the computer a dictionary or a thesaurus. It worked")
    print(f"  this out purely from the company '{word}' keeps in a sentence.\n")


def handle_compare(model, first, second):
    first, second = first.lower(), second.lower()
    if not known(model, first) or not known(model, second):
        return
    score = float(model.similarity(first, second))
    print()
    print(f"    {first} <-> {second}")
    print(f"    {bar(score)}  {score:+.3f}  ({describe(score)})")


def handle_show(model, word):
    word = word.lower()
    if not known(model, word):
        return
    numbers = model[word]
    print(f"\n  '{word}' is stored as these {len(numbers)} numbers:\n")
    for start in range(0, len(numbers), 10):
        row = numbers[start : start + 10]
        print("    " + " ".join(f"{value:7.2f}" for value in row))
    print("\n  On their own they mean nothing. What matters is that similar")
    print("  words have similar numbers -- that is what makes the math work.\n")


def train_toy_vectors(dimensions=2):
    """Build vectors from TOY_CORPUS the honest way: count neighbours, then compress.

    Returns (vocab, index, counts, vectors). The counts matrix is what a word
    "is" before compression -- one row per word, one column per neighbour.
    """
    sentences = [re.findall(r"[a-z]+", line) for line in TOY_CORPUS]
    frequency = Counter(w for s in sentences for w in s if w not in TOY_STOP)
    vocab = sorted(w for w, n in frequency.items() if n >= 2)
    index = {w: i for i, w in enumerate(vocab)}

    counts = np.zeros((len(vocab), len(vocab)))
    for sentence in sentences:
        kept = [w for w in sentence if w in index]
        for position, word in enumerate(kept):
            first = max(0, position - TOY_WINDOW)
            last = min(len(kept), position + TOY_WINDOW + 1)
            for other in range(first, last):
                if other != position:
                    counts[index[word], index[kept[other]]] += 1

    # log1p stops common words from dominating; SVD keeps the strongest patterns.
    left, strength, _ = np.linalg.svd(np.log1p(counts), full_matrices=False)
    return vocab, index, counts, strength, left[:, :dimensions] * strength[:dimensions]


def scatter(points, width=58, height=15):
    """Plot {word: (x, y)} as ASCII, nudging labels vertically to avoid overlap."""
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)
    scale_x = (width - 12) / (high_x - low_x or 1)
    scale_y = (height - 2) / (high_y - low_y or 1)

    grid = [[" "] * width for _ in range(height)]
    for word, (x, y) in sorted(points.items()):
        column = int((x - low_x) * scale_x) + 1
        row = height - 2 - int((y - low_y) * scale_y)
        label = "*" + word
        for shift in range(height):
            candidate = row + (shift // 2) * (1 if shift % 2 else -1)
            if not 0 <= candidate < height:
                continue
            if all(grid[candidate][column + k] == " "
                   for k in range(len(label)) if column + k < width):
                for k, character in enumerate(label):
                    if column + k < width:
                        grid[candidate][column + k] = character
                break

    print("    +" + "-" * width + "+")
    for row in grid:
        print("    |" + "".join(row) + "|")
    print("    +" + "-" * width + "+")


def build_step_sentences(model, trained):
    """Step 1: the whole corpus, small enough to read aloud."""
    print("  Real models read billions of sentences. Ours reads 16, so you")
    print("  can check its work by hand:\n")
    for line in TOY_CORPUS:
        print(f"    {line}")
    print("\n  That is everything it will ever see. Nobody tells it what a")
    print("  single word means -- it only gets the raw text.")


def build_step_counts(model, trained):
    """Step 2: the co-occurrence tally, which is the actual mechanism."""
    _, index, counts, _, _ = trained
    print("  For every word, tally the words appearing within 2 places of it.")
    print("  Reading across: how often each row-word sat near each column-word.\n")
    print("        " + "".join(f"{c[:6]:>7}" for c in TOY_COLS))
    for row_word in TOY_ROWS:
        tally = "".join(f"{int(counts[index[row_word], index[c]]):>7}" for c in TOY_COLS)
        print(f"  {row_word:>6}" + tally)
    print("\n  That is the whole trick. Compare the 'dog' and 'puppy' rows --")
    print("  nearly the same, because those two words keep the same company.")
    print("  'cake' and 'oven' match each other, but not them.")
    print("\n  Each row is ALREADY a list of numbers describing a word.")


def build_step_svd(model, trained):
    """Step 3: SVD ranks patterns across the table so we can keep only the best."""
    _, _, _, strength, _ = trained
    print("  Nineteen numbers per word is too many. We want two -- but WHICH")
    print("  two? SVD answers that. It finds patterns across the whole table")
    print("  and ranks them by how much of the table each one explains.")
    print(f"  Here are all {len(strength)}, strongest first:\n")
    for position, value in enumerate(strength, start=1):
        print(f"    [{position:>2}]  {value:5.2f}  {'#' * int(value * 4)}")
    captured = 100 * (strength[:2] ** 2).sum() / (strength ** 2).sum()
    print(f"\n  The top two alone capture {captured:.0f}% of the whole table,")
    print("  and the tail trails off to nearly zero. Keep the top 2 and you")
    print("  have the dominant patterns in two numbers. That is the compression.")


def build_step_compress(model, trained):
    """Step 4: show the final 2 numbers per word, after keeping the top 2 patterns."""
    vocab, index, _, _, vectors = trained
    print(f"  Keeping just the top 2 patterns gives every word two numbers")
    print(f"  instead of {len(vocab)}. The loaded model keeps 50 patterns from")
    print("  a vocabulary 400,000 words wide -- same recipe, bigger scale.\n")
    for word in TOY_ROWS:
        x, y = vectors[index[word]]
        print(f"    {word:>6}  [{x:+.2f}, {y:+.2f}]")
    print("\n  Two numbers per word, learned from nothing but those 16 lines.")


def build_step_plot(model, trained):
    """Step 5: the payoff -- clusters nobody labelled, and scores to prove it."""
    _, index, _, _, vectors = trained
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    print("  Two numbers means we can draw them. Every word, positioned")
    print("  only by the company it keeps:\n")
    scatter({w: (float(vectors[index[w], 0]), float(vectors[index[w], 1]))
             for w in TOY_PLOT if w in index})
    print("\n  The pets sorted themselves into one corner and the baking into")
    print("  another. No one labelled them. Counting neighbours was enough.\n")
    print("  Scoring pairs it was never told about:\n")
    for first, second in [("dog", "puppy"), ("bread", "cake"), ("dog", "oven")]:
        score = float(normalized[index[first]] @ normalized[index[second]])
        print(f"    {f'{first} / {second}':<16} {bar(score)}  {score:+.3f}")
    print("\n  Same recipe, scaled up: 6 billion words of Wikipedia and 50")
    print("  numbers per word gives you the model running in this program --")
    print(f"  where dog / cat scores {model.similarity('dog', 'cat'):.3f}.")
    print("\n  (Real embeddings refine this: weighting rare pairs differently, or")
    print("  training a network to predict neighbours instead of counting them.")
    print("  But 'notice what appears together, then compress' is the idea.)")


def train_toy_bigrams():
    """Count which word follows which across TOY_CORPUS.

    Returns a dict mapping each token (including sentence boundary markers)
    to a Counter of the words that immediately followed it.
    """
    following = defaultdict(Counter)
    for line in TOY_CORPUS:
        tokens = ["<s>"] + re.findall(r"[a-z]+", line) + ["</s>"]
        for current, nxt in zip(tokens, tokens[1:]):
            following[current][nxt] += 1
    return following


def sample_bigram(following, max_length=15):
    """Sample a sentence from the bigram model by weighted random choice."""
    tokens = ["<s>"]
    for _ in range(max_length):
        choices = list(following[tokens[-1]].items())
        if not choices:
            break
        words, counts = zip(*choices)
        total = sum(counts)
        threshold = random.random() * total
        for word, count in zip(words, counts):
            threshold -= count
            if threshold <= 0:
                tokens.append(word)
                break
        if tokens[-1] == "</s>":
            break
    return " ".join(t for t in tokens if t not in ("<s>", "</s>"))


# Words whose follower lists are varied enough to be worth showing on-screen;
# others (e.g. 'wagged' always -> 'its') show up in the explanation sentence.
GENERATE_SHOW = ["<s>", "the", "dog", "ate", "warm", "sweet", "oven"]


def generate_step_follows(model, following):
    """Step 1: for each word, which word comes next, and how often."""
    print("  Reusing the same 16 sentences from `build`, but this time")
    print("  tally which word comes AFTER each word. '<s>' marks sentence")
    print("  start, '</s>' marks sentence end.\n")
    for word in GENERATE_SHOW:
        items = following[word].most_common()
        shown = ", ".join(f"{w}({n})" for w, n in items)
        label = f"'{word}' ->"
        print(f"    {label:<10}  {shown}")
    print("\n  That is the whole model: a dictionary of what-follows-what.")
    print("  '<s>' tells us how sentences start -- almost always 'the'.")
    print("  'oven' almost always ends them. 'wagged' only ever leads")
    print("  to 'its' (always 'wagged its tail').")


def generate_step_probability(model, following):
    """Step 2: normalize counts into a probability distribution for one word."""
    counts = following["the"]
    total = sum(counts.values())
    peak = max(counts.values())
    print("  Divide each count by the total to get a probability. For")
    print("  'the', here is the full distribution over what comes next:\n")
    for word, count in counts.most_common():
        probability = count / total
        print(f"    the -> {word:<10} {bar(count / peak)}  {int(round(probability * 100)):>3}%"
              f"   ({count}/{total})")
    print(f"\n  {total} pairs starting with 'the', {len(counts)} different nexts.")
    print("  'oven' is the single most common, but nothing is guaranteed --")
    print("  the model picks one at random, weighted by probability.")


def generate_step_generate(model, following):
    """Step 3: sample sentences from the bigram model, then set up the caveat."""
    print("  Start with '<s>', pick the next word by its probability, keep")
    print("  going until '</s>' or until it rambles too long. Six tries:\n")
    for position in range(1, 7):
        sentence = sample_bigram(following)
        print(f"    {position}. {sentence}")
    print("\n  Some are real sentences from the corpus. Some are plausible")
    print("  mash-ups the model invented. Some fall over immediately --")
    print("  'the oven' ends there because 'oven' usually ends a sentence,")
    print("  and the model can't see that the sentence has barely started.\n")
    print("  That is the fundamental limit: it only remembers ONE word.\n")
    print("-" * 62)
    print("  Same recipe as a real language model, with four upgrades:")
    print("    - ATTENTION over many previous words, not just one")
    print("    - word vectors (from `build`) instead of raw tokens, so")
    print("      'dog' and 'puppy' can share what the model knows")
    print("    - a trained network predicts the distribution, instead of")
    print("      a counts table")
    print("    - 50,000-word vocabulary, billions of sentences of training")
    print("  Pick-the-next-word-from-a-distribution is the whole engine.")
    print("-" * 62)


GENERATE_STEPS = [
    ("follows", "which word follows which", generate_step_follows),
    ("probability", "normalize counts into probabilities", generate_step_probability),
    ("sample", "sample sentences from the model", generate_step_generate),
]


# Each step is addressable by number or by name, so you can jump straight to a
# panel when someone asks about it mid-presentation.
BUILD_STEPS = [
    ("sentences", "the 16 sentences it learns from", build_step_sentences),
    ("counts", "count which words sit next to which", build_step_counts),
    ("svd", "rank the patterns in the table", build_step_svd),
    ("compress", "squeeze the table down", build_step_compress),
    ("plot", "did it work?", build_step_plot),
]


class Walkthrough:
    """A sequence of panels you page through: Enter advances, `b` goes back.

    Shared by `build` (which lazy-trains vectors on first use) and `tour`
    (which only needs the model). Steps take (model, data); data is whatever
    `data_factory` returns, or None for walkthroughs that don't need any.
    """

    def __init__(self, name, steps, header, data_factory=None):
        self.name = name
        self.steps = steps
        self.header = header
        self.step = 0
        self._data = None
        self._data_factory = data_factory

    @property
    def data(self):
        if self._data_factory is not None and self._data is None:
            self._data = self._data_factory()
        return self._data

    def resolve(self, argument):
        """Turn '2', 'counts' or '' into a step number. None if unrecognised."""
        if not argument:
            return 1
        if argument.isdigit():
            number = int(argument)
            return number if 1 <= number <= len(self.steps) else None
        for position, (step_name, _, _) in enumerate(self.steps, start=1):
            if step_name.startswith(argument):
                return position
        return None

    def show(self, model, number):
        self.step = number
        _, title, render = self.steps[number - 1]
        total = len(self.steps)
        # Clear screen + home cursor so each step is read on a fresh page.
        # Skipped when output isn't a tty (piped input, test runs) so logs
        # don't fill with escape sequences.
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print("\n" + "=" * 62)
        if number == 1 and self.header:
            print(f"  {self.header}")
        print(f"  STEP {number} of {total} -- {title}")
        print("=" * 62 + "\n")
        render(model, self.data)

        options = []
        if number < total:
            following = self.steps[number][0]
            options.append(f"Enter for step {number + 1} ({following})")
        if number > 1:
            options.append("`b` for back")
        if number == total:
            options.append(f"`{self.name} 1` to start over")
        print(f"\n  -- step {number}/{total} --  " + "  |  ".join(options) + "\n")

    def move(self, model, offset):
        target = self.step + offset
        if target < 1:
            print(f"  Already at step 1. Type `{self.name}` to show it again.\n")
        elif target > len(self.steps):
            print(f"  That was the last step. Type `{self.name} 1` to start over,")
            print("  or try an equation like `king - man + woman`.\n")
        else:
            self.show(model, target)


def describe_build_steps():
    lines = ["  Steps (type the number or the name):"]
    for position, (name, title, _) in enumerate(BUILD_STEPS, start=1):
        lines.append(f"    build {position}   {name:<10} {title}")
    return "\n".join(lines)


def context_body(model, _data=None):
    """The content of the `context` panel (minus the standalone banner).

    Shared by the standalone `context` command and by `full` step 10.
    """
    print("  Every word is ONE list of 50 numbers. But 'bank' means two")
    print("  things, 'apple' means two, 'bat' means two. A single vector")
    print("  can't be both -- so one of the meanings gets erased.\n")
    for word, note in [
        ("bank",  "the river shoreline is gone -- all finance"),
        ("apple", "the fruit is gone -- all tech"),
        ("bat",   "the flying mammal is gone -- all sports"),
    ]:
        print(f"    nearest to '{word}':")
        for other, score in model.most_similar(word, topn=6, restrict_vocab=COMMON_WORDS):
            print(f"      {other:<16} {bar(score)}  {score:.3f}")
        print(f"    -- {note}\n")
    print("  Whichever sense appears more often in the training text wins,")
    print("  and the other one disappears. The model can't decide per")
    print("  sentence -- it only ever has one vector per word.\n")
    print("  'python' is the rare exception: Wikipedia talks about the")
    print("  language AND the snake enough to keep both:\n")
    print("    nearest to 'python':")
    for other, score in model.most_similar("python", topn=8, restrict_vocab=COMMON_WORDS):
        print(f"      {other:<16} {bar(score)}  {score:.3f}")
    print("\n    -- php, perl, scripting, monty = the language and the show")
    print("    -- owl, mouse, monkey, rhino    = the snake")
    print("    All of it jammed into a single vector.\n")
    print("-" * 62)
    print("  The fix is called ATTENTION: let a word's numbers shift based")
    print("  on the words around it, so 'apple pie' becomes a different")
    print("  vector from 'apple ceo'. That is the next thing to build.")
    print("-" * 62)


def handle_context(model):
    """Standalone `context` command: banner + body."""
    print("\n" + "=" * 62)
    print("  ONE WORD, ONE VECTOR -- WHAT GETS LOST")
    print("=" * 62 + "\n")
    context_body(model)
    print()


def attend_pair(model, first, second):
    """A toy single-head attention: cosine-similarity scores -> softmax -> blend.

    Returns (weights_row_for_first, new_vector_for_first). Honest shortcut --
    real transformers use three learned projections (Q, K, V); we use the raw
    vectors for all three. The pedagogical idea -- 'mix a word with its
    neighbours, weighted by relevance' -- is the same.
    """
    vectors = np.stack([model[first], model[second]])
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = normed @ normed.T  # 2x2 cosine similarities
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    return weights[0], weights[0] @ vectors


def show_attend_example(model, first, second, sense):
    """Print attention weights and the top neighbours of the new vector."""
    weights, new_vector = attend_pair(model, first, second)
    print(f"  --- '{first} {second}' ---")
    print(f"  what '{first}' pays attention to:")
    for word, weight in zip((first, second), weights):
        print(f"    {word:<14} {bar(float(weight))}  {weight:.2f}")
    print(f"  nearest words to the NEW '{first}' vector:")
    neighbours = model.most_similar(positive=[new_vector], topn=12,
                                    restrict_vocab=COMMON_WORDS)
    kept = [(n, s) for n, s in neighbours if n not in (first, second)][:5]
    for other, score in kept:
        print(f"    {other:<16} {bar(score)}  {score:.3f}")
    print(f"    -- {sense}\n")


def attend_body(model, _data=None):
    """The content of the `attend` panel (minus the standalone banner).

    Shared by the standalone `attend` command and by `full` step 11.
    """
    print("  `context` showed what a single vector can't do: 'apple' is")
    print("  stuck meaning the company, 'crane' is stuck meaning the")
    print("  machine. Attention is the fix. For each word in a sentence,")
    print("  build a NEW vector that is a weighted blend of all the words")
    print("  -- so 'apple' near 'pie' mixes in some pie-ness, and 'apple'")
    print("  near 'ceo' mixes in some ceo-ness.\n")
    print("  Three steps, for each word in the sentence:")
    print("    1. score it against every other word (we use cosine")
    print("       similarity -- how aligned are their vectors already?)")
    print("    2. softmax those scores so they sum to 1")
    print("    3. new vector = weighted sum of all the word vectors\n")
    print("  Watch 'apple' land in two different neighbourhoods:\n")
    show_attend_example(model, "apple", "pie", "fruit, dessert, baking")
    show_attend_example(model, "apple", "ceo", "tech companies and executives")
    print("  Same word, two sentences, two different places in the space.")
    print("  One more, to show it is not a fluke:\n")
    show_attend_example(model, "crane", "bird", "animals -- the model found the BIRD")
    show_attend_example(model, "crane", "construction", "the MACHINE -- steel, tower, roof")
    print("-" * 62)
    print("  Honest caveat: real attention learns three projections of")
    print("  each vector (query, key, value) instead of using raw cosine")
    print("  similarity. But 'mix each word with its neighbours, weighted")
    print("  by relevance' is the whole idea -- and it is already enough")
    print("  to tell apple-the-fruit from apple-the-company.")
    print("-" * 62)


def handle_attend(model):
    """Standalone `attend` command: banner + body."""
    print("\n" + "=" * 62)
    print("  ATTENTION: LETTING WORDS READ THEIR NEIGHBOURS")
    print("=" * 62 + "\n")
    attend_body(model)
    print()


def handle_twins(model):
    """Which words are almost identical -- and the four reasons why."""
    print("\n" + "=" * 62)
    print("  WORDS THE MODEL THINKS ARE ALMOST THE SAME")
    print("=" * 62)
    print("\n  1.000 would mean identical. Everything below scores over 0.88.")
    for title, pairs, note in TWIN_GROUPS:
        print(f"\n  {title}\n")
        for first, second in pairs:
            score = float(model.similarity(first, second))
            label = f"{first} / {second}"
            print(f"    {label:<28} {bar(score)}  {score:.3f}")
        print(f"\n    {note}")
    print("\n" + "-" * 62)
    print("  The takeaway: a high score means two words get USED the same way.")
    print("  Only the first group actually means the same thing. That gap is")
    print("  the single most useful thing to understand about embeddings.")
    print("-" * 62 + "\n")


def tour_step_show(model, _data):
    print("  To a computer, 'dog' is not a word. It is this:")
    handle_show(model, "dog")
    print("  If someone asks where those numbers came from, `build` shows")
    print("  exactly that, trained live on 16 sentences.")


def tour_step_near(model, _data):
    print("  Those numbers aren't random. Words that appear in the same")
    print("  kinds of sentences ended up with similar numbers, so we can")
    print("  ask: which words are nearest to 'dog'?")
    handle_near(model, "dog")


def tour_step_compare(model, _data):
    print("  We can measure how close any two words are. 1.0 means")
    print("  interchangeable, 0.0 means nothing in common.")
    for first, second in [("dog", "cat"), ("dog", "leash"), ("dog", "algebra")]:
        handle_compare(model, first, second)
    print()
    print("  But the model only knows how words are USED, not what they")
    print("  mean. Watch what that does to opposites:")
    for first, second in [("happy", "joyful"), ("happy", "sad")]:
        handle_compare(model, first, second)
    print()
    print("  'happy' scores HIGHER against 'sad' than against 'joyful' --")
    print("  because opposites show up in all the same sentences.")
    print("  A good moment to mention that this is a real limitation.")


def tour_step_math(model, _data):
    print("  Here is the payoff. If distance carries meaning, so does")
    print("  direction: the step from 'man' to 'woman' turns out to be")
    print("  about the same step as from 'king' to 'queen'. So we can do")
    print("  arithmetic on meaning:")
    handle_equation(model, "king - man + woman")
    print("  Try your own now -- for example:")
    print("      paris - france + italy")
    print("      walking - walk + swim")


# Short names are what the footer previews ("Enter for step 2 (neighbours)"),
# so they read as the gist of each panel. Only BUILD_STEPS names work as bare
# shortcuts at the prompt, so these can be anything readable.
TOUR_STEPS = [
    ("numbers", "a word is just a list of numbers", tour_step_show),
    ("neighbours", "similar words get similar numbers", tour_step_near),
    ("distance", "the distance between words is meaningful", tour_step_compare),
    ("direction", "direction is meaningful too", tour_step_math),
]


def train_full():
    """Bundle every training artifact the full walkthrough needs."""
    return {"vectors": train_toy_vectors(), "bigrams": train_toy_bigrams()}


def _use(key, step_fn):
    """Adapt a step that wants a single artifact to the full walkthrough's bundle."""
    def wrapped(model, bundle):
        step_fn(model, bundle[key])
    wrapped.__name__ = f"full_{step_fn.__name__}"
    return wrapped


# The full walkthrough chains the four teaching sections into one 14-step tour:
# intuition (tour) -> mechanics (build) -> limitation (context + attend) ->
# generation (generate). Each step reuses the function from its home module.
FULL_STEPS = [
    ("numbers",     "a word is just a list of numbers",             tour_step_show),
    ("neighbours",  "similar words get similar numbers",            tour_step_near),
    ("distance",    "the distance between words is meaningful",     tour_step_compare),
    ("direction",   "direction is meaningful too: king - man + woman", tour_step_math),
    ("sentences",   "the 16 sentences we build from",               _use("vectors", build_step_sentences)),
    ("counts",      "count which words sit next to which",          _use("vectors", build_step_counts)),
    ("svd",         "rank the patterns in the table",               _use("vectors", build_step_svd)),
    ("compress",    "squeeze the table down to 2 numbers",          _use("vectors", build_step_compress)),
    ("plot",        "did it work?",                                 _use("vectors", build_step_plot)),
    ("limits",      "one vector per word -- what gets lost",        context_body),
    ("attend",      "attention: letting words read their neighbours", attend_body),
    ("follows",     "which word follows which",                     _use("bigrams", generate_step_follows)),
    ("probability", "normalize counts into probabilities",          _use("bigrams", generate_step_probability)),
    ("sample",      "sample sentences, and the honest caveat",      _use("bigrams", generate_step_generate)),
]


# Returned by execute() when the user asked to leave.
QUIT = object()


def make_walkthroughs():
    """The four walkthroughs, keyed by the command that starts each."""
    return {
        "build": Walkthrough("build", BUILD_STEPS, "WHERE THE NUMBERS COME FROM",
                             train_toy_vectors),
        "tour": Walkthrough("tour", TOUR_STEPS, "A GUIDED WALKTHROUGH"),
        "generate": Walkthrough("generate", GENERATE_STEPS,
                                "GENERATING TEXT ONE WORD AT A TIME",
                                train_toy_bigrams),
        "full": Walkthrough("full", FULL_STEPS,
                            "THE WHOLE STORY, END TO END", train_full),
    }


def execute(model, walks, active, line):
    """Run one command line, printing the result.

    `active` is whichever walkthrough was most recently shown: Enter and `b`
    advance / rewind it, so the presenter can just keep pressing Enter.
    Returns the walkthrough that is active afterwards, or QUIT. Shared by the
    terminal loop in main() and by the web app.
    """
    line = line.strip()
    if not line:
        if active is not None and active.step < len(active.steps):
            active.move(model, +1)
        return active

    command, _, argument = line.partition(" ")
    command = command.lower()
    words = argument.split()

    if command in {"quit", "exit", "q"}:
        print("Bye!")
        return QUIT
    if command in {"help", "?", "examples"}:
        print(BANNER)
        return active
    if command in walks:
        walk = walks[command]
        target = walk.resolve(argument.strip().lower())
        if target is None:
            print(f"  No such step: '{argument.strip()}'.")
            if command == "build":
                print(describe_build_steps())
            print()
            return active
        walk.show(model, target)
        return walk
    if command in {"next", "n"}:
        if active is not None:
            active.move(model, +1)
        else:
            print("  Nothing to advance. Type `tour` or `build` to start.\n")
        return active
    if command in {"back", "b", "prev"}:
        if active is not None:
            active.move(model, -1)
        else:
            print("  Nothing to go back to. Type `tour` or `build` to start.\n")
        return active
    # Jumping straight to a panel by name: `counts`, `plot`, ...
    if not words and any(command == name for name, _, _ in BUILD_STEPS):
        walks["build"].show(model, walks["build"].resolve(command))
        return walks["build"]
    if command == "twins":
        handle_twins(model)
    elif command == "context":
        handle_context(model)
    elif command == "attend":
        handle_attend(model)
    elif command == "near" and words:
        handle_near(model, words[0])
    elif command == "show" and words:
        handle_show(model, words[0])
    elif command == "compare":
        if len(words) == 2:
            handle_compare(model, words[0], words[1])
        else:
            print("  Give me exactly two words, like: compare dog cat\n")
    else:
        try:
            handle_equation(model, line)
        except ValueError as error:
            print(f"  {error}")
            print("  Try something like: king - man + woman\n")
    return active


def main() -> None:
    model = load_model()
    print(BANNER)
    walks = make_walkthroughs()
    active = None

    while True:
        try:
            line = input("word math> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        active = execute(model, walks, active, line)
        if active is QUIT:
            return


if __name__ == "__main__":
    sys.exit(main())
