"""Vector representations of questions, question segments and topic cards, cached as one .npz file.

Default backend: LSA (TF-IDF plus truncated SVD), fully local and deterministic. A neural backend
(sentence-transformers) is supported when the model can be downloaded; the build environment of this
repository could not reach the model hub, see ADR-7 in the README.

The LSA model itself (vocabulary, idf, SVD components) is stored next to the vectors as plain arrays,
so a question that is not in the cache is projected into the same space with transform(), instead of
refitting the space under weights that were trained on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vde.data.cleaning import tokenize
from vde.data.loader import load_questions
from vde.data.segments import parse
from vde.knowledge import topic_cards
from vde.knowledge.base import artifact_path, write_provenance
from vde.paths import ANNEX_DIR

FILE = "embeddings.npz"
DEFAULT_ST_MODEL = "intfloat/multilingual-e5-small"


@dataclass
class Embeddings:
    ids: np.ndarray
    vectors: np.ndarray
    lsa: dict | None = None

    def __post_init__(self) -> None:
        self.index = {str(k): i for i, k in enumerate(self.ids)}

    def add(self, keys: list[str], texts: list[str]) -> None:
        """Project texts that are not cached with the stored LSA model."""
        keys_texts = [(k, t) for k, t in zip(keys, texts) if k not in self.index]
        if not keys_texts:
            return
        if self.lsa is None:
            raise KeyError("no stored LSA model to project new texts: run `vde build embeddings`")
        new = transform([t for _, t in keys_texts], self.lsa)
        start = len(self.ids)
        self.ids = np.concatenate([self.ids, np.array([k for k, _ in keys_texts])])
        self.vectors = np.vstack([self.vectors, new.astype(np.float32)])
        for i, (k, _) in enumerate(keys_texts):
            self.index[k] = start + i

    def get(self, key: str) -> np.ndarray:
        if key not in self.index:
            raise KeyError(f"no cached embedding for {key!r}: run `vde build embeddings`")
        return self.vectors[self.index[key]]

    def matrix(self, keys: list[str]) -> np.ndarray:
        return np.vstack([self.get(k) for k in keys]) if keys else np.zeros((0, self.vectors.shape[1]))

    def has(self, key: str) -> bool:
        return key in self.index


def question_key(qid: str) -> str:
    return f"q:{qid}"


def segment_key(qid: str, seg_key: str) -> str:
    return f"s:{qid}:{seg_key}"


def card_key(tema_id: int) -> str:
    return f"card:{tema_id}"


def load() -> Embeddings:
    data = np.load(artifact_path(FILE), allow_pickle=False)
    lsa = None
    if "lsa_terms" in data.files:
        lsa = {"terms": data["lsa_terms"], "idf": data["lsa_idf"], "components": data["lsa_components"]}
    return Embeddings(ids=data["ids"], vectors=data["vectors"].astype(np.float32), lsa=lsa)


def transform(texts: list[str], lsa: dict) -> np.ndarray:
    """Same pipeline as _lsa (TF-IDF with sublinear tf and l2 norm, then SVD, then l2 norm), no refit."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vocabulary = {str(t): i for i, t in enumerate(lsa["terms"])}
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                                 sublinear_tf=True, vocabulary=vocabulary)
    vectorizer.idf_ = np.asarray(lsa["idf"], dtype=float)
    return normalize(vectorizer.transform(texts) @ np.asarray(lsa["components"], dtype=float).T)


def question_texts(parsed) -> tuple[list[str], list[str]]:
    """Cache keys and texts of one parsed question: the full text and every segment."""
    keys, texts = [question_key(parsed.id)], [parsed.full_text]
    for seg in parsed.segments():
        keys.append(segment_key(parsed.id, seg.key))
        texts.append(seg.text)
    return keys, texts


def build(backend: str = "lsa", dimensions: int = 256, seed: int = 0, model: str = DEFAULT_ST_MODEL,
          annex_dir=ANNEX_DIR) -> str:
    keys, texts = _corpus(annex_dir)
    extra = {}
    if backend == "lsa":
        vectors, info, extra = _lsa(texts, dimensions, seed)
    elif backend == "st":
        vectors, info = _sentence_transformers(texts, model)
    else:
        raise ValueError(f"unknown embedding backend {backend!r}")
    target = artifact_path(FILE)
    np.savez_compressed(target, ids=np.array(keys), vectors=vectors.astype(np.float32), **extra)
    write_provenance(target, produced_by=f"embedding backend {backend}", llm_used=False, texts=len(keys), **info)
    return str(target)


def _corpus(annex_dir) -> tuple[list[str], list[str]]:
    keys, texts = [], []
    for tid, card in sorted(topic_cards.load().items()):
        keys.append(card_key(tid))
        texts.append(card.text)
    for question in load_questions(annex_dir):
        k, t = question_texts(parse(question))
        keys += k
        texts += t
    return keys, texts


def _lsa(texts: list[str], dimensions: int, seed: int) -> tuple[np.ndarray, dict, dict]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=dimensions, random_state=seed, algorithm="randomized", n_iter=7)
    vectors = normalize(svd.fit_transform(matrix))
    info = {"backend": "lsa", "dimensions": dimensions, "vocabulary": len(vectorizer.vocabulary_),
            "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4), "seed": seed}
    terms = np.empty(len(vectorizer.vocabulary_), dtype=object)
    for term, i in vectorizer.vocabulary_.items():
        terms[i] = term
    model = {"lsa_terms": terms.astype(str), "lsa_idf": vectorizer.idf_.astype(np.float64),
             "lsa_components": svd.components_.astype(np.float32)}
    return vectors, info, model


def _sentence_transformers(texts: list[str], model: str) -> tuple[np.ndarray, dict]:
    from sentence_transformers import SentenceTransformer  # optional dependency, offline step only

    encoder = SentenceTransformer(model, device="cpu")
    vectors = encoder.encode([f"query: {t}" for t in texts], batch_size=32, normalize_embeddings=True)
    return np.asarray(vectors), {"backend": "st", "model": model}
