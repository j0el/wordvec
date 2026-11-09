"""The guided walkthroughs for the Chinese demo: tour, build, generate, full,
plus the twins / context / attend panels.

These mirror the English ones in `wordvec`, with data chosen for Chinese:

  * the 16-sentence toy corpus is written in Chinese and cut into words by
    jieba before anything is counted -- so the very first step of "how words
    become numbers" is "decide where the words are", which English skips;
  * the ambiguous words are ones Chinese actually has (苹果 fruit/Apple,
    红旗 flag/car brand) rather than bank/bat/crane;
  * every number quoted in the explanatory text was measured on the real
    vectors, not assumed (the unrelated-pair floor is ~0.25 here, not ~0).

Steps are called as step(model, data). The dictionary they need for pinyin and
English glosses is set once by `init()`.
"""

import random
from collections import Counter, defaultdict

import numpy as np

from wordvec import (
    DESCENT_ROUNDS, MINI_TABLE, Walkthrough, bar, descend, descend_pair,
    descent_spec, figure, oriented_svd, rank_k,
)
from wordvec.chinese import (
    candidates, handle_compare, handle_equation, handle_near, handle_show,
    label, load_jieba, pad, print_results, width,
)

DICT = None
_CUTTER = None


def init(dictionary):
    global DICT
    DICT = dictionary


def cut(sentence):
    """Split a Chinese sentence into words with jieba (a private instance, so
    the toy corpus is cut the same way no matter what else the program does)."""
    global _CUTTER
    if _CUTTER is None:
        _CUTTER = load_jieba().Tokenizer()
    return _CUTTER.lcut(sentence)


# ---------------------------------------------------------------- toy corpus

# Same shape as the English corpus: two topics (pets, baking) with a few
# sentences bridging them, so the clusters come out distinct but not perfectly
# separable. Phrased so that jieba's default cutting gives clean words.
TOY_CORPUS = [
    ("狗对着陌生人叫", "The dog barks at a stranger."),
    ("小狗叫了又摇它的尾巴", "The puppy barked and wagged its tail."),
    ("猫舔它的毛又呼噜叫", "The cat licked its fur and purred."),
    ("我的狗朝兽医摇它的尾巴", "My dog wagged its tail at the vet."),
    ("兽医检查了猫和小狗", "The vet checked the cat and the puppy."),
    ("小猫在我腿上摇它的尾巴", "The kitten, on my lap, wagged its tail."),
    ("她用烤箱烤新鲜面包", "She baked fresh bread in the oven."),
    ("他用烤箱烤甜蛋糕", "He baked a sweet cake in the oven."),
    ("刚出炉的面包又热又新鲜", "The bread, just out, was warm and fresh."),
    ("她吃了一块甜蛋糕", "She ate a slice of sweet cake."),
    ("他吃热面包抹黄油", "He ate warm bread with butter."),
    ("烤箱里烤的新鲜馅饼是甜的", "The fresh pie baked in the oven was sweet."),
    ("狗吃了我手里的热面包", "The dog ate warm bread from my hand."),
    ("小猫吃了一块甜蛋糕", "The kitten ate a slice of sweet cake."),
    ("我的小狗吃了新鲜馅饼", "My puppy ate the fresh pie."),
    ("猫坐在热烤箱旁边", "The cat sat by the warm oven."),
]
TOY_STOP = set("的 了 着 对 朝 它 我 他 她 用 又 还 在 上 里 和 是 一块 刚 手里 旁边 坐在 出".split())
TOY_WINDOW = 2
TOY_ROWS = ["狗", "小狗", "猫", "面包", "蛋糕", "烤箱"]
TOY_COLS = ["叫", "摇", "兽医", "吃", "烤", "甜", "热", "新鲜"]
TOY_PLOT = ["狗", "小狗", "猫", "小猫", "叫", "尾巴",
            "面包", "蛋糕", "馅饼", "烤箱", "烤", "甜"]

# English for every word jieba produces from the corpus. Hand-written because
# CC-CEDICT offers several senses and these need exactly one.
TOY_EN = {
    "狗": "dog", "小狗": "puppy", "猫": "cat", "小猫": "kitten",
    "面包": "bread", "蛋糕": "cake", "烤箱": "oven", "馅饼": "pie",
    "叫": "bark/cry", "摇": "wag", "尾巴": "tail", "吃": "eat", "烤": "bake",
    "甜": "sweet", "热": "warm", "新鲜": "fresh", "兽医": "vet",
    "陌生人": "stranger", "舔": "lick", "毛": "fur", "呼噜": "purr",
    "检查": "check", "腿": "lap/leg", "出炉": "out of oven", "抹": "spread",
    "黄油": "butter", "的": "of", "了": "past marker", "着": "-ing marker", "对": "at",
    "又": "and/again", "它": "it", "我": "I/my", "他": "he", "她": "she",
    "用": "with", "在": "at/on", "上": "on", "里": "inside", "和": "and",
    "是": "is", "一块": "a slice", "刚": "just", "手里": "in hand",
    "旁边": "beside", "坐在": "sat at", "朝": "toward", "出": "out",
}


def en(word):
    return TOY_EN.get(word, "")


def tagged(word):
    """'狗 dog' -- a toy word with its English, for tables read by non-readers."""
    return f"{word} {en(word)}".strip()


def train_toy_vectors(dimensions=2):
    """Same recipe as the English `build`: cut, count neighbours, then compress.

    Returns (vocab, index, counts, singular values, vectors).
    """
    sentences = [cut(chinese) for chinese, _ in TOY_CORPUS]
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

    left, strength, _ = np.linalg.svd(np.log1p(counts), full_matrices=False)
    return vocab, index, counts, strength, left[:, :dimensions] * strength[:dimensions]


def scatter(points, columns=58, rows=15):
    """ASCII scatter plot that knows Chinese characters are two cells wide."""
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)
    scale_x = (columns - 10) / (high_x - low_x or 1)
    scale_y = (rows - 2) / (high_y - low_y or 1)

    grid = [[" "] * columns for _ in range(rows)]
    for word, (x, y) in sorted(points.items()):
        column = int((x - low_x) * scale_x) + 1
        row = rows - 2 - int((y - low_y) * scale_y)
        text = "*" + word
        span = width(text)
        for shift in range(rows):
            candidate = row + (shift // 2) * (1 if shift % 2 else -1)
            if not 0 <= candidate < rows:
                continue
            if all(grid[candidate][column + k] == " "
                   for k in range(span) if column + k < columns):
                cell = column
                for character in text:
                    if cell >= columns:
                        break
                    grid[candidate][cell] = character
                    step = width(character)
                    if step == 2 and cell + 1 < columns:
                        grid[candidate][cell + 1] = ""  # second half of a wide glyph
                    cell += step
                break

    print("    +" + "-" * columns + "+")
    for row in grid:
        print("    |" + "".join(row) + "|")
    print("    +" + "-" * columns + "+")


# --------------------------------------------------------------- build steps

def build_step_sentences(model, trained):
    print("  Real models read billions of sentences. Ours reads 16, so you")
    print("  can check its work by hand (English added for you; the program")
    print("  never sees it):\n")
    for number, (chinese, english) in enumerate(TOY_CORPUS, start=1):
        print(f"    {number:>2}. {pad(chinese, 26)} {english}")
    sentence = TOY_CORPUS[6][0]
    print("\n  One thing English never has to do first. Written Chinese has no")
    print("  spaces, so before it can count anything the program must decide")
    print("  where the words are. jieba cuts sentence 7 like this:\n")
    print(f"    {sentence}")
    print("    " + " | ".join(cut(sentence)))
    print("    " + " | ".join(en(w) or "?" for w in cut(sentence)))
    print("\n  It cuts all 16 the same way. After that, nobody tells the program")
    print("  what a single word means -- it only gets the raw text.")


# ------------------------------------------------ SVD and gradient descent
# Same teaching steps as the English `build`, on the Chinese table. The maths
# helpers and figure plumbing are shared (wordvec.oriented_svd, descend, ...).

ZH_PETS = set("狗 小狗 猫 小猫 叫 摇 尾巴 兽医 舔 毛 呼噜".split())
ZH_MINI_ROWS = ["狗", "小狗", "蛋糕"]
ZH_MINI_COLS = ["摇", "尾巴", "烤箱"]
# Step sizes for this table: a crawl, a good one, too much. The Chinese table is
# a little smaller than the English one, so it stays stable at a different size.
ZH_DESCENT_STEPS = [0.005, 0.08, 0.3]


def zh_grid(values, rows, cols, fmt="{:g}", blank_zero=False, cell=8):
    """A small labelled table whose columns line up despite wide characters."""
    print("  " + " " * 8 + "".join(pad(c, cell) for c in cols))
    for r, row in zip(rows, values):
        cells = "".join(pad("." if blank_zero and v == 0 else fmt.format(v), cell) for v in row)
        print("  " + pad(r, 8) + cells)


def zh_key(words):
    print("  Key: " + ", ".join(tagged(w) for w in words))


def build_step_counts(model, trained):
    vocab, index, counts, _, _ = trained
    sentence = TOY_CORPUS[1][0]
    pieces = cut(sentence)
    kept = [w for w in pieces if w in index]
    pairs = [(kept[i], kept[j]) for i in range(len(kept))
             for j in range(i + 1, min(len(kept), i + TOY_WINDOW + 1))]

    print("  We want a list of numbers for each word. Start with a table whose")
    print(f"  rows AND columns are the same {len(vocab)} words. Each cell answers:")
    print(f"  how often did these two words appear within {TOY_WINDOW} places of")
    print("  each other?\n")
    print("  Two rules keep the table meaningful:")
    print("    - skip filler words (的 'of', 了 'past marker', 又 'and', 它 'it' ...):")
    print("      they sit next to everything and tell us nothing")
    print("    - skip words seen only once: one sighting is not a pattern\n")
    print(f"  ONE SENTENCE BY HAND:  {sentence}  ({TOY_CORPUS[1][1]})")
    print("  jieba cuts it into words:  " + " | ".join(pieces))
    print("  Drop the filler and rare words, keep:  " + " ".join(kept))
    print("  Every pair within 2 places gets +1:")
    print("    " + "   ".join(f"{a}-{b}" for a, b in pairs))
    print("  Do that for all 16 sentences. A corner of the finished table:\n")

    corner = ["狗", "小狗", "摇", "尾巴", "烤箱", "面包"]
    picks = [index[w] for w in corner]
    zh_grid(counts[np.ix_(picks, picks)].astype(int), corner, corner, blank_zero=True)
    print()
    zh_key(corner)

    biggest = counts.max()
    top_pairs = sorted({tuple(sorted((vocab[i], vocab[j])))
                        for i, j in zip(*np.where(counts == biggest))})
    empty = int((counts == 0).sum())
    dog_row = [vocab[j] for j in np.nonzero(counts[index["狗"]])[0]]
    puppy_row = [vocab[j] for j in np.nonzero(counts[index["小狗"]])[0]]
    print("\n  What to notice:")
    print(f"    - the biggest count anywhere is {int(biggest)}: "
          + ", ".join(f"{a}-{b}" for a, b in top_pairs))
    print("    - it is a mirror image: if 摇 sits near 尾巴, 尾巴 sits near 摇.")
    print("      The diagonal is empty (a word is not its own neighbour)")
    print(f"    - it is mostly empty: {empty} of {counts.size} cells are 0. A word")
    print("      only touches the few words it actually sat beside")
    print(f"    - 狗 (dog) row     = {', '.join(dog_row)}")
    print(f"      小狗 (puppy) row = {', '.join(puppy_row)}")
    print("      different words, overlapping neighbours: similar rows")
    print("\n  Each row is ALREADY a list of numbers for a word. But it is long")
    print("  and mostly empty, which is what the next steps compress.")
    print("\n  One squash first: replace each count c with log(1 + c).")
    squash = ",  ".join(f"{c} -> {np.log1p(c):.2f}" for c in range(int(biggest) + 1))
    print(f"    {squash}")
    print("  The biggest pair now counts about twice a single sighting, not")
    print("  three times, so a few common pairs cannot drown out the rest.")


def build_step_layers(model, trained):
    left, strength, right = oriented_svd(MINI_TABLE)
    rows = [tagged(w) for w in ZH_MINI_ROWS]
    cols = [tagged(w) for w in ZH_MINI_COLS]
    print("  SVD takes a table of numbers and breaks it into LAYERS. Add the")
    print("  layers back together and you get the table again. Each layer is")
    print("  one simple pattern, with a STRENGTH saying how big it is.\n")
    print("  A tiny table you can check by hand (numbers made up):\n")
    zh_grid(MINI_TABLE.astype(int), ZH_MINI_ROWS, ZH_MINI_COLS)
    print()
    zh_key(ZH_MINI_ROWS + ZH_MINI_COLS)
    print("\n  Two patterns are easy to see:")
    print("    layer 1: the pets, with 狗 and 小狗 moving together (2 2 / 3 3)")
    print("    layer 2: the lone 4 for 蛋糕 (cake) and 烤箱 (oven)\n")
    print("  Every layer is built from three ingredients:")
    print("    - a COLUMN of numbers, one per row-word")
    print("    - a ROW of numbers, one per column-word")
    print("    - a STRENGTH")
    print("  and each cell of the layer is   strength x column x row.\n")
    layers = []
    for k in range(2):
        column, row = left[:, k], right[k]
        layer = strength[k] * np.outer(column, row)
        layers.append(layer)
        print(f"  LAYER {k + 1}   strength {strength[k]:.2f}")
        print(f"    column {np.round(column, 2) + 0.0}   row {np.round(row, 2) + 0.0}")
        if k == 0:
            print(f"    check one cell, 狗/摇:    {strength[0]:.2f} x {column[0]:.2f}"
                  f" x {row[0]:.2f} = {layer[0, 0]:.1f}")
        else:
            print(f"    check one cell, 蛋糕/烤箱: {strength[1]:.2f} x {column[2]:.2f}"
                  f" x {row[2]:.2f} = {layer[2, 2]:.1f}")
        zh_grid(np.round(layer, 1) + 0.0, ZH_MINI_ROWS, ZH_MINI_COLS, fmt="{:.1f}")
        print()
    print("  Add layer 1 + layer 2 and every cell matches the original table.\n")
    print("  Where does a strength come from? Square every cell of a layer, add")
    print("  them up, take the square root:")
    print(f"    layer 1:  sqrt(2^2 + 2^2 + 3^2 + 3^2) = sqrt(26) = {strength[0]:.2f}")
    print(f"    layer 2:  sqrt(4^2) = {strength[1]:.2f}")
    print("  A bigger strength means more of the table lives in that layer.\n")
    print("  SVD finds the layers automatically, even when patterns overlap")
    print("  and are not this clean, and lists them strongest first. Keep")
    print("  only the strongest few and you rebuild most of the table from")
    print("  far fewer numbers. Next: do exactly this to the real table.")

    figure({"kind": "svd-mini", "rows": rows, "cols": cols,
            "table": MINI_TABLE.tolist(),
            "layers": [{"strength": round(float(strength[k]), 2),
                        "column": (np.round(left[:, k], 2) + 0.0).tolist(),
                        "row": (np.round(right[k], 2) + 0.0).tolist(),
                        "matrix": (np.round(layers[k], 2) + 0.0).tolist()} for k in range(2)]})


def build_step_strengths(model, trained):
    vocab, index, counts, _, _ = trained
    table = np.log1p(counts)
    left, strength, right = oriented_svd(table, flip_by={1: index["摇"]})
    energy = strength ** 2
    total = energy.sum()
    share = 100 * energy / total
    size = len(vocab)

    print(f"  The real table: {size} words x {size} neighbours, with each")
    print("  count squashed by log(1 + count). Same recipe as the tiny one:")
    print(f"  SVD splits it into {len(strength)} layers. Their strengths,")
    print("  strongest first:\n")
    for position, value in enumerate(strength, start=1):
        print(f"    [{position:>2}]  {value:5.2f}  {'#' * int(value * 4)}")
    print(f"\n  So {strength[0]:.2f} is simply the strength of layer 1, {strength[1]:.2f} of layer 2.")
    print("  Same meaning as before: square every cell of the layer, add up,")
    print("  square root. Squaring the strengths gives each layer's share of")
    print("  the table's total size:")
    print(f"    total = {total:.1f}  (the sum of the squares of all {counts.size} cells)")
    print(f"    layer 1:  {strength[0]:.2f}^2 = {energy[0]:.1f}   -> {share[0]:.0f}% of the table")
    print(f"    layer 2:  {strength[1]:.2f}^2 = {energy[1]:.1f}   -> {share[1]:.0f}% of the table")
    print(f"    together: {share[:2].sum():.0f}%.  (About half the table, not all of it --")
    print("    two layers is a rough sketch, good enough to draw.)\n")

    first = np.argsort(-np.abs(left[:, 0]))[:5]
    pos = np.argsort(-left[:, 1])[:4]
    neg = np.argsort(left[:, 1])[:3]
    print("  What are the layers? Each gives every word a score:")
    print("    layer 1 scores highest on " + ", ".join(tagged(vocab[i]) for i in first) + ".")
    print("      Not a topic: it measures how much company a word keeps.")
    print("      The baking and eating words sit next to more words, so they score big.")
    print("    layer 2 is the real split:")
    print("      one side   " + ", ".join(f"{tagged(vocab[i])} ({left[i, 1]:+.2f})" for i in pos))
    print("      other side " + ", ".join(f"{tagged(vocab[i])} ({left[i, 1]:+.2f})" for i in neg))
    print("      Pets on one side, baking on the other. Nobody told it.\n")
    print("  SVD is the PRECISE method: it computes the best possible answer")
    print(f"  in one go. The catch is size. This table is {size} x {size}. The real")
    print(f"  Chinese model's is {len(model.index_to_key):,} x {len(model.index_to_key):,}, and doing SVD on that is")
    print("  slow and needs enormous memory. Real models use a second method")
    print("  that is faster and almost as good: gradient descent. Next.")

    order = list(np.argsort(-left[:, 1]))
    keep = [rank_k(left, strength, right, k)[np.ix_(order, order)] for k in range(len(strength) + 1)]
    figure({"kind": "svd-real",
            "words": [tagged(vocab[i]) for i in order],
            "pets": [vocab[i] in ZH_PETS for i in order],
            "strengths": np.round(strength, 2).tolist(),
            "share": np.round(share, 1).tolist(),
            "table": np.round(table[np.ix_(order, order)], 2).tolist(),
            "recon": [np.round(m, 2).tolist() for m in keep]})


def build_step_descent(model, trained):
    vocab, index, counts, _, _ = trained
    count = int(counts[index["摇"], index["尾巴"]])
    target = float(np.log1p(count))
    step = 0.2
    frames = descend_pair(target, step, 60)
    print("  Gradient descent finds good numbers by trial and correction:")
    print("  guess, measure how wrong, nudge to be a little less wrong, repeat.\n")
    print("  ONE PAIR, BY HAND. 摇 (wag) and 尾巴 (tail) sat together")
    print(f"  {count} times, which the squash turns into a")
    print(f"  TARGET of log(1 + {count}) = {target:.2f}.\n")
    print("  Each word gets one number to start with; say 0.5 each. The model's")
    print("  GUESS for the pair is the two numbers multiplied:")
    print(f"    0.5 x 0.5 = {frames[0][2]:.2f}.  Far below {target:.2f}, so the ERROR is "
          f"{frames[0][3]:+.2f}.\n")
    print("  The nudge: if the guess is too low, both numbers should go up.")
    print(f"      new number = old number - {step} x error x the OTHER number")
    print("  A big error gives a big nudge; a small error, a small one.\n")
    print("    round     摇 (wag)   尾巴 (tail)   guess    error")
    for r in [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 60]:
        a, b, guess, error = frames[r]
        error = error if abs(error) >= 0.0005 else 0.0
        print(f"    {r:>5}      {a:6.3f}      {b:6.3f}    {guess:6.3f}  {error:+7.3f}")
    print("\n  The nudges shrink as the error shrinks: quick at first, then")
    print("  easing in -- like walking down a slope that flattens at the bottom.")
    print("  (The 'slope' is exactly what the nudge formula measures.)\n")
    print("  Notice the numbers are not unique: 1.18 x 1.18 and 2 x 0.69 both")
    print("  make 1.39. Any numbers that multiply right will do.")

    figure({"kind": "gd-pair", "target": round(target, 3), "step": step,
            "names": ["摇 wag", "尾巴 tail"],
            "frames": [[round(a, 3), round(b, 3), round(g, 3), round(e, 3)]
                       for a, b, g, e in frames]})


def build_step_learning(model, trained):
    vocab, index, counts, _, _ = trained
    table = np.log1p(counts)
    left, strength, right = oriented_svd(table, flip_by={1: index["摇"]})
    best = float(((table - rank_k(left, strength, right, 2)) ** 2).sum())
    steps = ZH_DESCENT_STEPS
    runs = {step: descend(table, step, DESCENT_ROUNDS) for step in steps}
    good_step = steps[1]
    good = runs[good_step]
    small, _, big = steps

    print(f"  Now the whole table at once, all {len(vocab)} words together:\n")
    print("    1. Give every word two random numbers.")
    print("    2. For every pair, multiply the two words' numbers and compare")
    print("       with the squashed count in the table. Add up all the errors")
    print("       (squared, so over and under both hurt) into one score.")
    print("    3. Nudge every number a little to lower that score. A word is")
    print("       nudged by its errors, each weighted by the partner's numbers --")
    print("       the same rule as the single pair, summed over its pairs.")
    print("    4. Repeat.\n")
    print(f"  Total error as it learns (step size {good_step}):\n")
    for r in [0, 5, 10, 20, 40, 80, 120]:
        print(f"    round {r:>3}   {good[r][0]:6.1f}  {'#' * int(good[r][0] / 2)}")
    print(f"\n  SVD's best possible answer scores {best:.1f}, so gradient descent")
    print("  arrives at about the same place SVD computes directly. It just")
    print("  walks there, never doing the big calculation. That is why it")
    print("  scales: each nudge needs only a few pairs at a time.\n")
    print("  The STEP SIZE matters:")
    print(f"    {small}  too small: after 120 rounds the error is still {runs[small][-1][0]:.1f}")
    print(f"    {good_step}  good: settles at {good[-1][0]:.1f}")
    print(f"    {big}  too big: overshoots, and bounces around {runs[big][-1][0]:.1f}")
    print("  Real training spends a great deal of effort choosing it.\n")
    print("  This is how nearly all AI is trained, including large language")
    print("  models: the guess comes from a huge network instead of a product")
    print("  of two numbers, but it is the same loop -- guess, measure the")
    print("  error, nudge, repeat.")

    figure(descent_spec([tagged(w) for w in vocab], [w in ZH_PETS for w in vocab],
                        best, runs, good_step))


def build_step_compress(model, trained):
    vocab, index, _, _, vectors = trained
    print("  Keeping just the top 2 layers gives every word two numbers")
    print(f"  instead of {len(vocab)}: its score in layer 1 and its score in layer 2,")
    print("  each multiplied by that layer's strength. (The other layers are")
    print("  small and mostly quirks of these 16 sentences.)\n")
    for word in TOY_ROWS:
        x, y = vectors[index[word]]
        print(f"    {pad(word, 6)} {en(word):<7} [{x:+.2f}, {y:+.2f}]")
    print("\n  Two numbers per word, learned from nothing but those 16 lines.")
    print("  Gradient descent lands on the same picture (possibly rotated or")
    print("  flipped), so the next step holds whichever way you got here.\n")
    print(f"  The loaded Chinese model keeps {model.vector_size} numbers per word from a")
    print(f"  vocabulary {len(model.index_to_key):,} words wide -- same recipe, bigger scale.")


def build_step_plot(model, trained):
    _, index, _, _, vectors = trained
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    print("  Two numbers means we can draw them. Every word, positioned")
    print("  only by the company it keeps:\n")
    scatter({w: (float(vectors[index[w], 0]), float(vectors[index[w], 1]))
             for w in TOY_PLOT if w in index})
    print("\n  Key: " + ", ".join(tagged(w) for w in TOY_PLOT))
    print("\n  The pets sorted themselves into one corner and the baking into")
    print("  another. No one labelled them. Counting neighbours was enough.\n")
    print("  Scoring pairs it was never told about:\n")
    for first, second in [("狗", "小狗"), ("面包", "蛋糕"), ("狗", "烤箱")]:
        score = float(normalized[index[first]] @ normalized[index[second]])
        name = f"{first} / {second}  ({en(first)} / {en(second)})"
        print(f"    {pad(name, 28)} {bar(score)}  {score:+.3f}")
    print("\n  Same recipe, scaled up: billions of words of Chinese text and")
    print(f"  {model.vector_size} numbers per word gives you the model running in this")
    print(f"  program -- where 狗 / 猫 (dog / cat) scores {model.similarity('狗', '猫'):.3f}.")
    print("\n  (Real embeddings refine this: weighting rare pairs differently, or")
    print("  training a network to predict neighbours instead of counting them.")
    print("  But 'notice what appears together, then compress' is the idea.)")


BUILD_STEPS = [
    ("sentences", "the 16 sentences it learns from (and cutting them)", build_step_sentences),
    ("counts", "count which words sit next to which", build_step_counts),
    ("layers", "SVD: break a table into layers", build_step_layers),
    ("strengths", "SVD: where 5.05 and 3.17 come from", build_step_strengths),
    ("descent", "gradient descent: learn by small corrections", build_step_descent),
    ("learning", "gradient descent on the whole table", build_step_learning),
    ("compress", "squeeze the table down", build_step_compress),
    ("plot", "did it work?", build_step_plot),
]


# ------------------------------------------------------------ generate steps

def train_toy_bigrams():
    """Count which word follows which, over the jieba-cut corpus."""
    following = defaultdict(Counter)
    for chinese, _ in TOY_CORPUS:
        tokens = ["<s>"] + cut(chinese) + ["</s>"]
        for current, nxt in zip(tokens, tokens[1:]):
            following[current][nxt] += 1
    return following


def sample_bigram(following, max_length=15):
    """Sample a sentence by weighted random choice. Chinese joins with no spaces."""
    tokens = ["<s>"]
    for _ in range(max_length):
        choices = list(following[tokens[-1]].items())
        if not choices:
            break
        words, counts = zip(*choices)
        threshold = random.random() * sum(counts)
        for word, count in zip(words, counts):
            threshold -= count
            if threshold <= 0:
                tokens.append(word)
                break
        if tokens[-1] == "</s>":
            break
    return "".join(t for t in tokens if t not in ("<s>", "</s>"))


GENERATE_SHOW = ["<s>", "烤箱", "吃", "热", "甜", "尾巴"]


def generate_step_follows(model, following):
    print("  Reusing the same 16 sentences from `build`, but this time")
    print("  tally which word comes AFTER each word. '<s>' marks sentence")
    print("  start, '</s>' marks sentence end. (Words are jieba's cuts.)\n")
    for word in GENERATE_SHOW:
        items = following[word].most_common(5)
        shown = ", ".join(f"{w}({n})" for w, n in items)
        if len(following[word]) > len(items):
            shown += ", ..."
        head = f"'{word}' ->" + (f"  [{en(word)}]" if en(word) else "")
        print(f"    {pad(head, 18)} {shown}")
    print("\n  Key: " + ", ".join(tagged(w) for w in
                                  ["吃", "热", "甜", "尾巴", "的", "了"]))
    print("\n  That is the whole model: a dictionary of what-follows-what.")
    print("  '<s>' shows how sentences start, and '尾巴' (tail) always ends")
    print("  one -- in this corpus every tail comes last.")


def generate_step_probability(model, following):
    counts = following["<s>"]
    total = sum(counts.values())
    peak = max(counts.values())
    print("  Divide each count by the total to get a probability. For")
    print("  '<s>' -- the start of a sentence -- here is the full distribution")
    print("  over what the FIRST word is:\n")
    for word, count in counts.most_common():
        probability = count / total
        name = f"<s> -> {word}" + (f"  ({en(word)})" if en(word) else "")
        print(f"    {pad(name, 24)} {bar(count / peak)}  {int(round(probability * 100)):>3}%"
              f"   ({count}/{total})")
    print(f"\n  {total} sentences, {len(counts)} different ways to start. Nothing is")
    print("  guaranteed -- the model picks one at random, weighted by")
    print("  probability, then does the same again for the word after it.")


def generate_step_sample(model, following):
    print("  Start with '<s>', pick the next word by its probability, keep")
    print("  going until '</s>' or until it rambles too long. Six tries:\n")
    for position in range(1, 7):
        print(f"    {position}. {sample_bigram(following)}")
    print("\n  Some are real sentences from the corpus. Some are plausible")
    print("  mash-ups the model invented. Some fall over immediately -- it")
    print("  only ever sees ONE word back, so it cannot tell that a sentence")
    print("  has barely started.\n")
    print("  That is the fundamental limit: it only remembers ONE word.\n")
    print("-" * 62)
    print("  Same recipe as a real language model, with four upgrades:")
    print("    - ATTENTION over many previous words, not just one")
    print("    - word vectors (from `build`) instead of raw tokens, so")
    print("      狗 (dog) and 小狗 (puppy) can share what the model knows")
    print("    - a trained network predicts the distribution, instead of")
    print("      a counts table")
    print("    - a huge vocabulary, billions of sentences of training")
    print("  Pick-the-next-word-from-a-distribution is the whole engine.")
    print("-" * 62)


GENERATE_STEPS = [
    ("follows", "which word follows which", generate_step_follows),
    ("probability", "normalize counts into probabilities", generate_step_probability),
    ("sample", "sample sentences from the model", generate_step_sample),
]


# ---------------------------------------------------- twins / context / attend

# Found by scoring every pair among the 8,000 most common multi-character
# words, then sorting what came out on top into four reasons for the match.
TWIN_GROUPS = [
    (
        "GENUINE SYNONYMS -- different words, same meaning",
        [("患者", "病人", "patient", "patient"), ("宿舍", "寝室", "dormitory", "dorm room"), ("击败", "战胜", "defeat", "overcome")],
        "患者/病人 (patient), 宿舍/寝室 (dormitory), 击败/战胜 (defeat):\n"
        "    the model noticed these are swappable, so it filed them together.",
    ),
    (
        "SAME SLOT -- near-identical numbers, but NOT the same meaning",
        [("周四", "周二", "Thursday", "Tuesday"), ("上午", "下午", "morning", "afternoon"), ("三年", "五年", "3 years", "5 years")],
        "Thursday/Tuesday, morning/afternoon, 3 years/5 years. They score\n"
        "    high because they appear in identical sentences, not because\n"
        "    they mean the same thing.",
    ),
    (
        "PEERS -- same job, different name",
        [("皇马", "巴萨", "Real Madrid", "Barcelona"), ("杨幂", "赵丽颖", "actress Yang Mi", "actress Zhao Liying"), ("库里", "杜兰特", "Stephen Curry", "Kevin Durant")],
        "Real Madrid/Barcelona, two film actresses, two basketball stars.\n"
        "    Names that fill the same role get the same company -- so the\n"
        "    model knows what KIND of thing each is, not which one it is.",
    ),
    (
        "OPPOSITES -- the one that surprises everybody",
        [("左手", "右手", "left hand", "right hand"), ("妻子", "丈夫", "wife", "husband"), ("上行", "下行", "upstream/up", "downstream/down")],
        "Left/right hand is the single most alike pair among these 8,000\n"
        "    words. Opposites slot into the same sentences, so the model is\n"
        "    reading the company a word keeps -- usage, not meaning.",
    ),
]


def handle_twins(model, _data=None):
    scores = [float(model.similarity(p[0], p[1])) for _, pairs, _ in TWIN_GROUPS for p in pairs]
    floor = int(min(scores) * 100) / 100
    print("\n" + "=" * 62)
    print("  WORDS THE MODEL THINKS ARE ALMOST THE SAME")
    print("=" * 62)
    print(f"\n  1.000 would mean identical. Everything below scores over {floor:.2f}.")
    for title, pairs, note in TWIN_GROUPS:
        print(f"\n  {title}\n")
        for first, second, english_a, english_b in pairs:
            score = float(model.similarity(first, second))
            name = f"{first} / {second}"
            meaning = f"{english_a} / {english_b}"
            print(f"    {pad(name, 14)} {pad(meaning, 36)} {bar(score, 20)}  {score:.3f}")
        print(f"\n    {note}")
    print("\n" + "-" * 62)
    print("  The takeaway: a high score means two words get USED the same way.")
    print("  Only the first group actually means the same thing. That gap is")
    print("  the single most useful thing to understand about embeddings.")
    print("-" * 62 + "\n")


def neighbours(model, word, topn=6):
    print_results(DICT, candidates(model, model[word] / np.linalg.norm(model[word]),
                                   [word], topn))


def context_body(model, _data=None):
    print("  Every word is ONE list of numbers. But 苹果 means two things")
    print("  (the fruit AND Apple the company), 小米 means two (millet AND")
    print("  Xiaomi), 奔驰 means two (galloping AND Mercedes-Benz). A single")
    print("  vector can't be both -- so one of the meanings gets erased.\n")
    for word, note in [
        ("苹果", "the fruit is gone -- all tech"),
        ("小米", "the millet is gone -- all phones"),
        ("奔驰", "the galloping is gone -- all luxury cars"),
    ]:
        print(f"    nearest to {label(DICT, word)}  [{DICT.gloss(word, 30)}]:")
        neighbours(model, word)
        print(f"    -- {note}\n")
    print("  Whichever sense appears more often in the training text wins,")
    print("  and the other one disappears. The model can't decide per")
    print("  sentence -- it only ever has one vector per word.\n")
    print("  (The English labels above were corrected by hand: the raw dictionary")
    print("  calls 宝马 'precious horse' and 小米 'millet', because it only knows")
    print("  the ordinary meaning. The model never saw a dictionary.)\n")
    print("  红旗 is the exception: the text talks about red flags AND the")
    print("  Hongqi car brand enough to keep both:\n")
    print(f"    nearest to {label(DICT, '红旗')}:")
    neighbours(model, "红旗", topn=10)
    print("\n    -- 军旗, 旗帜, 国旗 = flags")
    print("    -- 东风, 吉利, 一汽, 奔驰 = car makers")
    print("    All of it jammed into a single vector.\n")
    print("-" * 62)
    print("  The fix is called ATTENTION: let a word's numbers shift based")
    print("  on the words around it, so 苹果 next to 香蕉 (banana) becomes a")
    print("  different vector from 苹果 next to 手机 (phone). That is next.")
    print("-" * 62)


def handle_context(model, _data=None):
    print("\n" + "=" * 62)
    print("  ONE WORD, ONE VECTOR -- WHAT GETS LOST")
    print("=" * 62 + "\n")
    context_body(model)
    print()


def attend_pair(model, first, second):
    """A toy single-head attention: cosine scores -> softmax -> blend.

    Real transformers use three learned projections (Q, K, V); this uses the
    raw vectors for all three. The idea -- mix a word with its neighbours,
    weighted by relevance -- is the same.
    """
    vectors = np.stack([model[first], model[second]])
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = normed @ normed.T
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    return weights[0], weights[0] @ vectors


def show_attend_example(model, first, second, sense):
    weights, new_vector = attend_pair(model, first, second)
    print(f"  --- {label(DICT, first)} + {label(DICT, second)}  "
          f"({DICT.gloss(first, 14)} + {DICT.gloss(second, 14)}) ---")
    print(f"  what {first} pays attention to:")
    for word, weight in zip((first, second), weights):
        print(f"    {pad(word, 10)} {bar(float(weight))}  {weight:.2f}")
    print(f"  nearest words to the NEW {first} vector:")
    print_results(DICT, candidates(model, new_vector, [first, second], 5))
    print(f"    -- {sense}\n")


def attend_body(model, _data=None):
    print("  `context` showed what a single vector can't do: 苹果 is stuck")
    print("  meaning the company, 红旗 is stuck half-and-half. Attention is")
    print("  the fix. For each word in a sentence, build a NEW vector that is")
    print("  a weighted blend of all the words -- so 苹果 near 香蕉 mixes in")
    print("  some banana-ness, and 苹果 near 手机 mixes in some phone-ness.\n")
    print("  Three steps, for each word in the sentence:")
    print("    1. score it against every other word (we use cosine")
    print("       similarity -- how aligned are their vectors already?)")
    print("    2. softmax those scores so they sum to 1")
    print("    3. new vector = weighted sum of all the word vectors\n")
    print("  Watch 苹果 land in two different neighbourhoods:\n")
    show_attend_example(model, "苹果", "香蕉", "fruit")
    show_attend_example(model, "苹果", "手机", "phone makers and tech companies")
    print("  Same word, two sentences, two different places in the space.")
    print("  One more, to show it is not a fluke:\n")
    show_attend_example(model, "红旗", "国旗", "flags -- the model found the BANNER")
    show_attend_example(model, "红旗", "汽车", "car makers -- the model found the CAR")
    print("-" * 62)
    print("  Honest caveat: real attention learns three projections of")
    print("  each vector (query, key, value) instead of using raw cosine")
    print("  similarity. But 'mix each word with its neighbours, weighted")
    print("  by relevance' is the whole idea -- and it is already enough")
    print("  to tell apple-the-fruit from Apple-the-company.")
    print("-" * 62)


def handle_attend(model, _data=None):
    print("\n" + "=" * 62)
    print("  ATTENTION: LETTING WORDS READ THEIR NEIGHBOURS")
    print("=" * 62 + "\n")
    attend_body(model)
    print()


# ---------------------------------------------------------------------- tour

def tour_step_show(model, _data):
    print("  To a computer, 狗 (dog) is not a word. It is this:")
    handle_show(model, DICT, "dog")
    print("  If someone asks where those numbers came from, `build` shows")
    print("  exactly that, trained live on 16 sentences.")


def tour_step_near(model, _data):
    print("  Those numbers aren't random. Words that appear in the same")
    print("  kinds of sentences ended up with similar numbers, so we can")
    print("  ask: which words are nearest to 狗 (dog)?")
    handle_near(model, DICT, "dog")


def tour_step_compare(model, _data):
    print("  We can measure how close any two words are. Higher means more")
    print("  interchangeable. (In this model even unrelated words score")
    print("  around 0.25, so treat that as the floor.)")
    for first, second in [("狗", "猫"), ("狗", "骨头"), ("狗", "代数")]:
        handle_compare(model, DICT, first, second)
    print("  But the model only knows how words are USED, not what they")
    print("  mean. Watch what that does to opposites:\n")
    for first, second in [("高兴", "快乐"), ("高兴", "难过")]:
        handle_compare(model, DICT, first, second)
    print("  高兴 (happy) scores HIGHER against 难过 (sad) than against 快乐")
    print("  (joyful) -- because opposites show up in all the same sentences.")
    print("  A good moment to mention that this is a real limitation.")


def tour_step_math(model, _data):
    print("  Here is the payoff. If distance carries meaning, so does")
    print("  direction: the step from 'man' to 'woman' turns out to be")
    print("  about the same step as from 'king' to 'queen'. So we can do")
    print("  arithmetic on meaning:")
    handle_equation(model, DICT, "king - man + woman")
    print("  Try your own now -- for example:")
    print("      beijing - china + japan")
    print("      paris - france + italy")


TOUR_STEPS = [
    ("numbers", "a word is just a list of numbers", tour_step_show),
    ("neighbours", "similar words get similar numbers", tour_step_near),
    ("distance", "the distance between words is meaningful", tour_step_compare),
    ("direction", "direction is meaningful too", tour_step_math),
]


# ---------------------------------------------------------------------- full

def train_full():
    return {"vectors": train_toy_vectors(), "bigrams": train_toy_bigrams()}


def _use(key, step_fn):
    def wrapped(model, bundle):
        step_fn(model, bundle[key])
    wrapped.__name__ = f"full_{step_fn.__name__}"
    return wrapped


FULL_STEPS = [
    ("numbers",     "a word is just a list of numbers",             tour_step_show),
    ("neighbours",  "similar words get similar numbers",            tour_step_near),
    ("distance",    "the distance between words is meaningful",     tour_step_compare),
    ("direction",   "direction is meaningful too: king - man + woman", tour_step_math),
    ("sentences",   "the 16 sentences we build from",               _use("vectors", build_step_sentences)),
    ("counts",      "count which words sit next to which",          _use("vectors", build_step_counts)),
    ("layers",      "SVD: break a table into layers",               _use("vectors", build_step_layers)),
    ("strengths",   "SVD: where 5.05 and 3.17 come from",           _use("vectors", build_step_strengths)),
    ("descent",     "gradient descent: learn by small corrections", _use("vectors", build_step_descent)),
    ("learning",    "gradient descent on the whole table",          _use("vectors", build_step_learning)),
    ("compress",    "squeeze the table down to 2 numbers",          _use("vectors", build_step_compress)),
    ("plot",        "did it work?",                                 _use("vectors", build_step_plot)),
    ("limits",      "one vector per word -- what gets lost",        context_body),
    ("attend",      "attention: letting words read their neighbours", attend_body),
    ("follows",     "which word follows which",                     _use("bigrams", generate_step_follows)),
    ("probability", "normalize counts into probabilities",          _use("bigrams", generate_step_probability)),
    ("sample",      "sample sentences, and the honest caveat",      _use("bigrams", generate_step_sample)),
]


def build_walkthroughs():
    """The four walkthroughs, keyed by the command that starts each."""
    return {
        "build": Walkthrough("build", BUILD_STEPS, "HOW WE TRAIN A MODEL",
                             train_toy_vectors),
        "tour": Walkthrough("tour", TOUR_STEPS, "A GUIDED WALKTHROUGH"),
        "generate": Walkthrough("generate", GENERATE_STEPS,
                                "GENERATING TEXT ONE WORD AT A TIME",
                                train_toy_bigrams),
        "full": Walkthrough("full", FULL_STEPS, "THE WHOLE STORY, END TO END",
                            train_full),
    }
