"""Where the data files live and how to fetch them.

Kept free of heavy imports (no numpy, no gensim at import time) so the Docker
build can run `python datafiles.py` in its own cached layer, before the rest of
the source is copied in, and bake the models into the image.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

ENGLISH_MODEL = "glove-wiki-gigaword-50"

CACHE_DIR = Path(os.environ.get("WORDVEC_CACHE_DIR")
                 or Path.home() / ".cache" / "wordvec")

# Tencent AI Lab's Chinese embeddings (light edition): 143,613 words x 200
# numbers, learned from billions of words of news, web pages and novels.
# Apache-2.0, mirrored on Hugging Face by the text2vec project.
VECTORS_URL = (
    "https://huggingface.co/shibing624/text2vec-word2vec-tencent-chinese/"
    "resolve/main/light_Tencent_AILab_ChineseEmbedding.bin"
)
VECTORS_FILE = CACHE_DIR / "tencent_light.bin"

# CC-CEDICT is community-maintained and licensed CC BY-SA 4.0.
CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
CEDICT_FILE = CACHE_DIR / "cedict_ts.u8"


def download(url, destination, label):
    """Fetch `url` to `destination` with a one-line progress readout."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {label}...")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            print(f"\r  {min(done, total) / 1e6:6.1f} / {total / 1e6:.1f} MB",
                  end="", flush=True)

    urllib.request.urlretrieve(url, partial, progress)
    partial.rename(destination)
    print()


def ensure_vectors():
    """Download the Chinese vectors if they are not cached yet."""
    if not VECTORS_FILE.exists():
        download(VECTORS_URL, VECTORS_FILE,
                 "Chinese word vectors (Tencent AI Lab, 116 MB)")
    return VECTORS_FILE


def ensure_cedict():
    """Download and unpack CC-CEDICT if it is not cached yet."""
    if not CEDICT_FILE.exists():
        archive = CACHE_DIR / "cedict.zip"
        download(CEDICT_URL, archive, "CC-CEDICT Chinese-English dictionary (4 MB)")
        with zipfile.ZipFile(archive) as bundle:
            CEDICT_FILE.write_bytes(bundle.read("cedict_ts.u8"))
        archive.unlink()
    return CEDICT_FILE


def main():
    """Fetch everything both programs need (used at image build time)."""
    import gensim.downloader as api

    api.load(ENGLISH_MODEL, return_path=True)  # download only, don't load
    ensure_vectors()
    ensure_cedict()


if __name__ == "__main__":
    main()
