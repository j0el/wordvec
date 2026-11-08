"""Word arithmetic in Chinese -- for people who don't read Chinese.

The same idea as the English demo (`king - man + woman = queen`), but the
vectors were learned from Chinese text, so the math happens entirely in Chinese.
Two helpers make that usable without knowing a word of it:

    CC-CEDICT   a Chinese-English dictionary. It translates what you type into
                Chinese, and translates every answer back, with pinyin so you
                can say the word out loud.
    jieba       Chinese is written with no spaces between words, so a program
                has to *cut* a sentence into words before it can look anything
                up. jieba is that cutter (try `translate 我喜欢吃苹果`).

You can type English (`king - man + woman`) or Chinese (`国王 - 男人 + 女人`).
"""

import argparse
import collections
import logging
import re
import readline  # noqa: F401 -- enables arrow-key editing and history in input()
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np

from wordvec import COMMON_WORDS, bar, parse, unit
from wordvec.datafiles import CEDICT_FILE, ensure_cedict, ensure_vectors

BANNER = """
==============================================================
  词向量  WORD MATH IN CHINESE
==============================================================

The same word arithmetic as the English demo, but the numbers
behind each word were learned from Chinese text. You do not
need to read Chinese: type English and it is translated for
you, and every answer comes back with pinyin and an English
meaning.

  full              THE WHOLE STORY in 14 steps -- tour + build
                    + context + attend + generate, chained. Press
                    Enter to advance, `b` to go back.
  tour              a shorter 4-step overview (easiest idea first)
  build             where the numbers come from: cuts 16 Chinese
                    sentences into words (jieba), then trains vectors
                    from scratch, one step at a time:
                      build 1  sentences    the text it learns from
                      build 2  counts       tally the neighbours
                      build 3  svd          rank the patterns
                      build 4  compress     squeeze to 2 numbers
                      build 5  plot         did it work?
  generate          same corpus, used to produce text:
                      generate 1  follows      what follows what
                      generate 2  probability  normalize
                      generate 3  sample       produce sentences

Or explore directly:

  <equation>        English or Chinese, for example:

      king - man + woman
      beijing - china + japan
      国王 - 男人 + 女人

  near <word>       the words used most like <word>
  compare <a> <b>   how related are two words?
  show <word>       peek at the numbers behind a word
  twins             which words are almost identical -- and why
  context           one word, two meanings: which sense survives?
  attend            a toy attention head that fixes it
  translate <text>  English -> Chinese, or cut a Chinese sentence
                    into words (jieba) and translate each one:
                      translate happy
                      translate 我喜欢吃苹果
  help              show this message
  quit              exit
"""

# Chinese is full of words with no clean one-word English equivalent, so the
# dictionary's first guess is sometimes wrong for a demo ("doctor" comes back as
# 博士, a PhD). These were chosen by looking at what the dictionary returned.
OVERRIDES = {
    "doctor": "医生", "big": "大", "small": "小", "puppy": "小狗",
    "brother": "哥哥", "sister": "姐姐", "uncle": "叔叔", "aunt": "阿姨",
    "grandmother": "奶奶", "grandfather": "爷爷", "queen": "王后",
    "prince": "王子", "princess": "公主", "child": "孩子", "children": "孩子",
    "people": "人们", "car": "汽车", "money": "钱", "student": "学生",
}

# Brand and place names whose dictionary entry only gives the ordinary meaning
# (宝马 is "precious horse" in CC-CEDICT). Shown instead of the dictionary gloss.
GLOSS_OVERRIDES = {
    "宝马": "BMW (lit. precious horse)", "三星": "Samsung (lit. three stars)",
    "小米": "Xiaomi; also millet", "苹果": "apple; also Apple Inc.",
    "奔驰": "Mercedes-Benz; also to gallop", "红米": "Redmi (Xiaomi phones)",
    "高通": "Qualcomm", "一加": "OnePlus", "雷军": "Lei Jun (Xiaomi founder)",
    "吉利": "Geely (car maker); also lucky", "一汽": "FAW (car maker)",
    "东风": "Dongfeng (car maker); also east wind",
    "红旗": "red flag; also Hongqi (car brand)",
    "路虎": "Land Rover", "奔驰车": "Mercedes-Benz car",
    "杨幂": "Yang Mi (actress)", "赵丽颖": "Zhao Liying (actress)",
}

# Dictionary senses that describe something other than the word's meaning.
SKIP_SENSES = ("surname", "variant of", "old variant", "abbr.", "see ", "CL:",
               "also pr", "Taiwan pr", "used in", "erhua", "(Tw)", "archaic",
               "Kangxi radical", "radical in Chinese")

# Thresholds measured on this model: 20,000 random pairs of common words have a
# median similarity of 0.25 (95th percentile 0.41), so "unrelated" is ~0.25
# here, not ~0 as in the English GloVe vectors. 狗/猫 scores 0.75, 狗/骨头 0.45.
RELATEDNESS = [
    (0.65, "very closely related"),
    (0.45, "clearly related"),
    (0.33, "loosely related"),
]


def describe(score):
    for threshold, text in RELATEDNESS:
        if score >= threshold:
            return text
    return "basically unrelated -- about what two random words score"


HANZI_RE = re.compile(r"[一-鿿]")
ONLY_HANZI_RE = re.compile(r"^[一-鿿]+$")
CEDICT_LINE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/$")
SYLLABLE_RE = re.compile(r"^([A-Za-zü]+?)([1-5])$")
CLASSIFIER_RE = re.compile(r"\s*\(CL:[^)]*\)")
PLACE_SUFFIX_RE = re.compile(r" (municipality|province|city|county|district)$")

TONE_MARKS = {
    "a": "āáǎàa", "e": "ēéěèe", "i": "īíǐìi", "o": "ōóǒòo",
    "u": "ūúǔùu", "ü": "ǖǘǚǜü",
}

# Endings tried, in order, when an English word isn't in the dictionary as typed
# ("walking" -> "walk", "bigger" -> "big", "dogs" -> "dog").
SUFFIXES = (("ies", "y"), ("ing", ""), ("ing", "e"), ("ier", "y"), ("er", ""),
            ("er", "e"), ("ed", ""), ("ed", "e"), ("s", ""))


def load_jieba():
    """Import jieba quietly.

    jieba's own source has regex strings that newer Pythons flag with a
    SyntaxWarning on first compile. They are harmless and not ours to fix, but
    they would otherwise print into the middle of a presentation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import jieba

    jieba.setLogLevel(logging.WARNING)  # hides "Building prefix dict..."
    return jieba


# --------------------------------------------------------------- downloading

def load_model(path=None):
    """Load the Chinese vectors, downloading the default set the first time."""
    from gensim.models import KeyedVectors

    if path is None:
        path = ensure_vectors()
    path = Path(path)
    print(f"Loading Chinese word vectors ({path.name})... ", end="", flush=True)
    binary = path.suffix == ".bin"
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    print(f"ready. {len(model.index_to_key):,} words known.\n")
    return model


# ---------------------------------------------------------------- dictionary

def pinyin_marks(numbered):
    """'guo2 wang2' -> 'guó wáng'; CC-CEDICT writes tones as trailing digits."""
    out = []
    for syllable in numbered.replace("u:", "ü").split():
        match = SYLLABLE_RE.match(syllable)
        if not match:
            out.append(syllable)
            continue
        letters, tone = match.group(1).lower(), int(match.group(2))
        # Tone mark goes on a/e if present, the o of "ou", otherwise the last vowel.
        if "a" in letters:
            target = letters.index("a")
        elif "e" in letters:
            target = letters.index("e")
        elif "ou" in letters:
            target = letters.index("o")
        else:
            vowels = [i for i, ch in enumerate(letters) if ch in TONE_MARKS]
            target = vowels[-1] if vowels else None
        if target is None:
            out.append(letters)
            continue
        marked = TONE_MARKS[letters[target]][tone - 1]
        out.append(letters[:target] + marked + letters[target + 1:])
    return " ".join(out)


def normalise_sense(sense):
    """Reduce a dictionary definition to the bare English word it defines."""
    sense = re.sub(r"\(.*?\)", "", sense).strip().lower()
    return re.sub(r"^(to|a|an|the) ", "", sense)


class Dictionary:
    """CC-CEDICT, indexed both ways: Chinese -> English and English -> Chinese."""

    def __init__(self, model):
        ensure_cedict()

        self.model = model
        self.entries = collections.defaultdict(list)   # hanzi -> [(pinyin, [senses])]
        self.reverse = collections.defaultdict(list)   # english -> [(penalty, sense_idx, rank, hanzi)]
        for line in CEDICT_FILE.read_text(encoding="utf-8").splitlines():
            match = CEDICT_LINE.match(line)
            if not match:
                continue
            _, simplified, numbered, senses = match.groups()
            senses = senses.split("/")
            self.entries[simplified].append((pinyin_marks(numbered), senses))
            rank = model.key_to_index.get(simplified)
            if rank is None or rank >= COMMON_WORDS:
                continue
            for index, sense in enumerate(senses):
                if sense.startswith(SKIP_SENSES):
                    continue
                keys = {normalise_sense(sense)}
                # "Paris, capital of France" should be found under "paris".
                proper = sense[:1].isupper()
                if ", " in sense and proper:
                    name = normalise_sense(sense.split(", ")[0])
                    keys.add(name)
                    keys.add(PLACE_SUFFIX_RE.sub("", name))
                # A proper noun like France is better served by the full word
                # 法国 than by the bare character 法, so demote one-character matches.
                penalty = int(proper and len(simplified) == 1)
                for key in keys:
                    if key and len(key.split()) <= 3:
                        self.reverse[key].append((penalty, index, rank, simplified))

    def pinyin(self, hanzi):
        if hanzi in self.entries:
            return self.entries[hanzi][0][0]
        return ""

    def gloss(self, hanzi, limit=44):
        """Short English meaning of a Chinese word, falling back to its parts."""
        if hanzi in GLOSS_OVERRIDES:
            return self._clip(GLOSS_OVERRIDES[hanzi], limit)
        senses = self._senses(hanzi)
        if senses:
            return self._clip("; ".join(senses[:2]), limit)
        # Not a dictionary word (e.g. a compound like 主治医生): gloss its pieces.
        jieba = load_jieba()

        parts = list(dict.fromkeys(
            p for p in jieba.cut_for_search(hanzi) if p != hanzi))
        pieces = [(self._senses(p) or ["?"])[0] for p in parts]
        if parts and any(self._senses(p) for p in parts):
            return self._clip("~ " + " + ".join(pieces), limit)
        return "(no dictionary entry)"

    def _senses(self, hanzi):
        # Prefer an ordinary-word entry (apple) over a proper-noun one (Apple Inc.).
        entries = sorted(self.entries.get(hanzi, []), key=lambda e: e[1][0][:1].isupper())
        for _, senses in entries:
            good = [CLASSIFIER_RE.sub("", s) for s in senses if not s.startswith(SKIP_SENSES)]
            if good:
                # Plain meanings first; "(dialect) ..." and "(literary) ..." last.
                return sorted(good, key=lambda sense: sense.startswith("("))
        return []

    @staticmethod
    def _clip(text, limit):
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def to_chinese(self, english):
        """Best Chinese word for an English word, plus runners-up. None if unknown."""
        word = english.lower().strip()
        if word in OVERRIDES and OVERRIDES[word] in self.model.key_to_index:
            first = OVERRIDES[word]
            return first, self._alternatives(word, first)
        candidates = self.reverse.get(word)
        if not candidates:
            for ending, replacement in SUFFIXES:
                if word.endswith(ending) and len(word) > len(ending) + 1:
                    candidates = self.reverse.get(word[: -len(ending)] + replacement)
                    if candidates:
                        break
        if not candidates:
            return None
        ranked = []
        for *_, hanzi in sorted(candidates):
            if hanzi not in ranked:
                ranked.append(hanzi)
        return ranked[0], ranked[1:4]

    def _alternatives(self, word, chosen):
        ranked = [c[-1] for c in sorted(self.reverse.get(word, [])) if c[-1] != chosen]
        return list(dict.fromkeys(ranked))[:3]


# ------------------------------------------------------------------- display

def width(text):
    """Terminal columns used by `text` -- Chinese characters take two."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, columns):
    return text + " " * max(0, columns - width(text))


def label(dictionary, hanzi):
    """'国王 (guó wáng)' -- the Chinese word with a way to say it."""
    pinyin = dictionary.pinyin(hanzi)
    return f"{hanzi} ({pinyin})" if pinyin else hanzi


def normalise_input(text):
    """Accept full-width punctuation Chinese keyboards produce."""
    return (text.replace("＋", "+").replace("－", "-").replace("—", "-")
            .replace("　", " "))


# ---------------------------------------------------------------- resolution

class Term:
    def __init__(self, typed, hanzi, alternatives=()):
        self.typed = typed
        self.hanzi = hanzi
        self.alternatives = list(alternatives)


def resolve(model, dictionary, text):
    """Turn what the user typed (English or Chinese) into a word in the model."""
    if HANZI_RE.search(text):
        if text in model.key_to_index:
            return Term(text, text)
        jieba = load_jieba()

        pieces = jieba.lcut(text)
        if len(pieces) > 1:
            raise ValueError(
                f"'{text}' is really {len(pieces)} words ({' + '.join(pieces)}). "
                "Put one word in each slot of the equation."
            )
        raise ValueError(f"I don't know the Chinese word '{text}'.")

    found = dictionary.to_chinese(text)
    if found is None:
        raise ValueError(
            f"I couldn't find a Chinese word for '{text}'. Try a simpler English "
            "word, or type the Chinese directly."
        )
    hanzi, alternatives = found
    return Term(text, hanzi, alternatives)


def explain_terms(dictionary, terms):
    """Show how each typed word was translated -- the user can't check this."""
    print("\n  Translating your words:")
    for term in terms:
        if term.typed == term.hanzi:
            print(f"    {pad(label(dictionary, term.hanzi), 22)}  {dictionary.gloss(term.hanzi)}")
            continue
        extra = ""
        if term.alternatives:
            extra = "   (other options: " + ", ".join(term.alternatives) + ")"
        print(f"    {term.typed:<10} -> {label(dictionary, term.hanzi)}{extra}")
    print()


def candidates(model, query, exclude, topn):
    """Nearest common Chinese words to `query`, dropping boring echoes.

    Skipped: punctuation and Latin-letter entries, the input words, and words
    that merely contain an input word (医生 -> 主治医生 "attending physician").
    """
    found = model.most_similar(positive=[query], topn=topn * 12 + 40,
                               restrict_vocab=COMMON_WORDS)
    kept = []
    for word, score in found:
        if not ONLY_HANZI_RE.match(word):
            continue
        if any(word == e or e in word or word in e for e in exclude):
            continue
        kept.append((word, score))
        if len(kept) == topn:
            break
    return kept


def print_results(dictionary, results):
    for hanzi, score in results:
        name = pad(label(dictionary, hanzi), 24)
        print(f"    {name} {bar(score, 20)}  {score:.3f}  {dictionary.gloss(hanzi)}")


# ------------------------------------------------------------------ commands

def handle_equation(model, dictionary, expression):
    expression = normalise_input(expression)
    parsed = parse(expression)
    terms = [(sign, resolve(model, dictionary, text)) for sign, text in parsed]

    explain_terms(dictionary, [t for _, t in terms])

    shown = terms[0][1].hanzi
    for sign, term in terms[1:]:
        shown += f" {sign} {term.hanzi}"
    print(f"  {shown}  =  ?\n")

    if len(terms) > 1 and all(sign == "-" for sign, _ in terms[1:]):
        print("  Heads up: you took words away but didn't add anything back, so")
        print("  there is no single word for what's left over. Try adding one.\n")

    query = np.zeros(model.vector_size, dtype=np.float32)
    for sign, term in terms:
        vector = unit(model, term.hanzi)
        query += vector if sign == "+" else -vector

    if np.linalg.norm(query) < 1e-3:
        print("  Those words cancel out exactly (the same word was added and taken")
        print("  away), so there is nothing left to look up. Try different words.\n")
        return

    results = candidates(model, query, [t.hanzi for _, t in terms], topn=5)
    print_results(dictionary, results)
    best = results[0][0]
    print(f"\n  Best answer: {best}  {dictionary.pinyin(best)}  --  {dictionary.gloss(best)}")
    print("  (the bars show how close each word is to the answer)\n")


def lookup(model, dictionary, text):
    """Resolve one word for near/compare/show, printing a translation note."""
    term = resolve(model, dictionary, normalise_input(text).strip().lower())
    if term.typed != term.hanzi:
        print(f"\n  '{term.typed}' -> {label(dictionary, term.hanzi)}")
    return term.hanzi


def handle_near(model, dictionary, text, topn=8):
    word = lookup(model, dictionary, text)
    print(f"\n  Words used most like {label(dictionary, word)}:\n")
    print_results(dictionary, candidates(model, unit(model, word), [word], topn))
    print("\n  Nobody gave the computer a dictionary or a thesaurus -- the English")
    print("  above comes from CC-CEDICT, not from the model. The model worked out")
    print(f"  which words are related purely from the company {word} keeps.\n")


def handle_compare(model, dictionary, first, second):
    a = lookup(model, dictionary, first)
    b = lookup(model, dictionary, second)
    score = float(model.similarity(a, b))
    print()
    print(f"    {label(dictionary, a)} <-> {label(dictionary, b)}")
    print(f"    {dictionary.gloss(a, 24)} <-> {dictionary.gloss(b, 24)}")
    print(f"    {bar(score)}  {score:+.3f}  ({describe(score)})\n")


def handle_show(model, dictionary, text):
    word = lookup(model, dictionary, text)
    numbers = model[word]
    print(f"\n  {label(dictionary, word)} is stored as {len(numbers)} numbers. The first 20:\n")
    for start in range(0, 20, 10):
        print("    " + " ".join(f"{v:7.2f}" for v in numbers[start:start + 10]))
    print("\n  On their own they mean nothing. What matters is that similar")
    print("  words have similar numbers -- that is what makes the math work.\n")


def handle_translate(model, dictionary, text):
    jieba = load_jieba()

    text = normalise_input(text).strip()
    if HANZI_RE.search(text):
        pieces = [p for p in jieba.lcut(text) if p.strip()]
        print(f"\n  jieba cut that into {len(pieces)} words:\n")
        for piece in pieces:
            if HANZI_RE.search(piece):
                print(f"    {pad(piece, 10)} {pad(dictionary.pinyin(piece), 16)} {dictionary.gloss(piece)}")
            else:
                print(f"    {piece}")
        print("\n  Written Chinese has no spaces, so this cutting step has to happen")
        print("  first -- the model only knows words, not raw strings of characters.\n")
        return
    print()
    for english in text.split():
        found = dictionary.to_chinese(english)
        if found is None:
            print(f"    {english:<12} (no Chinese word found)")
            continue
        hanzi, others = found
        extra = f"   also: {', '.join(others)}" if others else ""
        print(f"    {english:<12} {label(dictionary, hanzi)}{extra}")
    print()


# ----------------------------------------------------------------- main loop

def execute(model, dictionary, walks, active, line):
    """Run one command line, printing the result; same contract as
    wordvec.execute(). Returns the active walkthrough or wordvec.QUIT."""
    from wordvec import QUIT
    from wordvec import chinese_lessons as lessons

    line = line.strip()
    if not line:
        if active is not None and active.step < len(active.steps):
            active.move(model, +1)
        return active

    command, _, argument = line.partition(" ")
    command = command.lower()
    words = argument.split()

    try:
        if command in {"quit", "exit", "q"}:
            print("Bye! 再见!")
            return QUIT
        if command in {"help", "?"}:
            print(BANNER)
        elif command in walks:
            walk = walks[command]
            target = walk.resolve(argument.strip().lower())
            if target is None:
                print(f"  No such step: '{argument.strip()}'.\n")
            else:
                walk.show(model, target)
                return walk
        elif command in {"next", "n"}:
            if active is not None:
                active.move(model, +1)
            else:
                print("  Nothing to advance. Type `tour` or `build` to start.\n")
        elif command in {"back", "b", "prev"}:
            if active is not None:
                active.move(model, -1)
            else:
                print("  Nothing to go back to. Type `tour` or `build` to start.\n")
        elif not words and any(command == n for n, _, _ in lessons.BUILD_STEPS):
            walks["build"].show(model, walks["build"].resolve(command))
            return walks["build"]
        elif command == "twins":
            lessons.handle_twins(model)
        elif command == "context":
            lessons.handle_context(model)
        elif command == "attend":
            lessons.handle_attend(model)
        elif command == "near" and words:
            handle_near(model, dictionary, words[0])
        elif command == "show" and words:
            handle_show(model, dictionary, words[0])
        elif command == "compare":
            if len(words) == 2:
                handle_compare(model, dictionary, words[0], words[1])
            else:
                print("  Give me exactly two words, like: compare dog cat\n")
        elif command in {"translate", "tr", "seg"} and words:
            handle_translate(model, dictionary, argument)
        else:
            handle_equation(model, dictionary, line)
    except ValueError as error:
        print(f"  {error}")
        print("  Try something like: king - man + woman\n")
    return active


def load_all(vectors_path=None):
    """Load vectors + dictionary + walkthroughs. Returns (model, dictionary, walks)."""
    from wordvec import chinese_lessons as lessons

    model = load_model(vectors_path)
    dictionary = Dictionary(model)
    lessons.init(dictionary)
    return model, dictionary, lessons.build_walkthroughs()


def main(argv=None) -> None:
    from wordvec import QUIT

    parser = argparse.ArgumentParser(
        prog="wordvec-zh", description="Word arithmetic in Chinese.")
    parser.add_argument(
        "--vectors", metavar="FILE",
        help="use your own word2vec-format vectors instead of the Tencent set, "
             "e.g. the People's Daily 'sgns.renmin.word' from "
             "github.com/Embedding/Chinese-Word-Vectors (.bin = binary, "
             "anything else = text, .bz2 is fine)")
    args = parser.parse_args(argv)

    model, dictionary, walks = load_all(args.vectors)
    print(BANNER)
    active = None

    while True:
        try:
            line = input("词向量> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 再见!")
            return
        active = execute(model, dictionary, walks, active, line)
        if active is QUIT:
            return


if __name__ == "__main__":
    sys.exit(main())
