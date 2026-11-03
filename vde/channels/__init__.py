"""Evidence channels. Each one maps a question to scores over canonical topics."""

from __future__ import annotations

from vde.channels.bin_prior import BinPriorChannel
from vde.channels.duplicates import DuplicateChannel
from vde.channels.interfaces import Channel
from vde.channels.knn import KnnChannel
from vde.channels.law_article import CodeArticleChannel, LawArticleChannel, SpecialLawChannel
from vde.channels.lexical import LexicalChannel, TitleLexicalChannel
from vde.channels.prototypes import PrototypeChannel
from vde.channels.segmented import SegmentChannel
from vde.channels.subject import SubjectChannel

REGISTRY: dict[str, type[Channel]] = {
    cls.name: cls
    for cls in (LawArticleChannel, SpecialLawChannel, CodeArticleChannel, LexicalChannel, TitleLexicalChannel,
                PrototypeChannel, KnnChannel, SubjectChannel, DuplicateChannel, BinPriorChannel, SegmentChannel)
}


def build_channels(res, names: list[str]) -> dict[str, Channel]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; known: {sorted(REGISTRY)}")
    return {name: REGISTRY[name](res) for name in names}
