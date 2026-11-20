"""Count a fitted character vocabulary in raw prompts without materializing every substring."""

import re
from functools import lru_cache
from itertools import chain
from operator import itemgetter

import numpy as np
from scipy.sparse import csr_matrix

try:
    import ahocorasick
except ImportError:
    ahocorasick = None


SPACES = re.compile(r"\s\s+")
MATCH_ID = itemgetter(1)


@lru_cache(maxsize=4)
def character_matcher(vectorizer):
    matcher = ahocorasick.Automaton(ahocorasick.STORE_INTS)
    for term, index in vectorizer.vocabulary_.items():
        matcher.add_word(term, int(index))
    matcher.make_automaton()

    @lru_cache(maxsize=8192)
    def word_ids(word):
        return tuple(map(MATCH_ID, matcher.iter(" " + word + " ")))

    return matcher, word_ids


def character_tfidf(vectorizer, texts):
    """Apply the existing panel's TF-IDF weights to its char or char_wb vocabulary."""
    if ahocorasick is None:
        return vectorizer.transform(texts)
    matcher, word_ids = character_matcher(vectorizer)
    preprocess = vectorizer.build_preprocessor()
    width = len(vectorizer.vocabulary_)
    columns, values, pointers = [], [], [0]
    for text in texts:
        value = SPACES.sub(" ", preprocess(text))
        if vectorizer.analyzer == "char":
            ids = map(MATCH_ID, matcher.iter(value))
        else:
            ids = chain.from_iterable(map(word_ids, value.split()))
        counts = np.bincount(np.fromiter(ids, dtype=np.int32), minlength=width)
        indices = np.flatnonzero(counts)
        columns.append(indices)
        values.append(counts[indices])
        pointers.append(pointers[-1] + len(indices))
    counts = csr_matrix(
        (np.concatenate(values).astype(vectorizer.dtype), np.concatenate(columns), pointers),
        shape=(len(texts), width),
    )
    return vectorizer._tfidf.transform(counts, copy=False)
