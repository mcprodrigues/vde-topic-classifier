"""Deterministic text cleaning: HTML to plain text, accent folding and tokenization."""

import html
import re
import unicodedata

STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de do da dos das no na nos nas em ao aos à às pelo pela pelos pelas
    por para com sem sob sobre entre ate até que qual quais quando como se sua seu suas seus ser
    e ou nao não mais menos muito ja já foi sao são esta está estao estão isso esse essa este
    ele ela eles elas lhe lhes quem cujo cuja onde tem têm ter ha há caso assim tambem também
    apenas somente todo toda todos todas outro outra outros outras mesmo mesma assinale opcao
    opção alternativa correta incorreta afirmar afirmativa afirmativas item itens julgue seguir
    abaixo acima conforme acordo respeito considerando considere certo errado
    """.split()
)

_BREAK_TAGS = re.compile(r"<\s*(?:br|/p|/div|/li|p|div|li)\b[^>]*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def html_to_text(raw: str) -> str:
    """Strip tags, keep line breaks and unescape entities."""
    if not raw:
        return ""
    text = _BREAK_TAGS.sub("\n", raw)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text).replace("\xa0", " ").replace("\\n", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def fold(text: str) -> str:
    """Lowercase and remove accents, keeping the original spacing."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Folded word tokens of length >= 2, numbers included."""
    tokens = [tok for tok in _NON_WORD.split(fold(text)) if len(tok) >= 2]
    if keep_stopwords:
        return tokens
    folded_stop = _folded_stopwords()
    return [tok for tok in tokens if tok not in folded_stop]


def collapse(text: str) -> str:
    """Single line version of a text, for previews and full-text similarity."""
    return _SPACES.sub(" ", (text or "").replace("\n", " ")).strip()


_FOLDED_STOPWORDS: frozenset | None = None


def _folded_stopwords() -> frozenset:
    global _FOLDED_STOPWORDS
    if _FOLDED_STOPWORDS is None:
        _FOLDED_STOPWORDS = frozenset(fold(word) for word in STOPWORDS)
    return _FOLDED_STOPWORDS
