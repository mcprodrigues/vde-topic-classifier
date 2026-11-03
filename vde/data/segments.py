"""Split a question into stem, command, numbered assertions and alternatives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vde.data.cleaning import collapse, fold, html_to_text
from vde.data.loader import Question

ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
MIN_SEGMENT_CHARS = 25

_SEPARATORS = r"[\)\.\-:\u2013\u2014]"  # \u2013 and \u2014 are dash escapes understood by re
_LINE_ASSERTION = re.compile(r"^\(?(" + "|".join(ROMAN[::-1]) + r")\s*" + _SEPARATORS + r"\s*(\S.*)$")
_INLINE_ASSERTION = re.compile(r"(?:^|\s)\(?(" + "|".join(ROMAN[::-1]) + r")\s*" + _SEPARATORS + r"\s+")
_NEGATED = re.compile(r"\b(incorret|errad|exceto|nao e corret|falsa|nao se aplica|nao configura)")
_TRUE_FALSE = {"certo", "errado", "verdadeiro", "falso", "c", "e"}


@dataclass(frozen=True)
class Segment:
    key: str
    kind: str
    text: str


@dataclass
class ParsedQuestion:
    id: str
    stem: str
    command: str
    assertions: list[str]
    alternatives: dict[str, str]
    kind: str
    negated: bool
    origin_discipline: str
    origin_subject: str
    is_new_import: bool
    full_text: str = field(init=False)

    def __post_init__(self) -> None:
        parts = [self.stem, *self.assertions, self.command, *self.alternatives.values()]
        self.full_text = collapse("\n".join(p for p in parts if p))

    @property
    def length(self) -> int:
        return len(self.full_text)

    def segments(self) -> list[Segment]:
        """Informative segments: the stem with its command, each assertion, each long alternative."""
        out = [Segment("stem", "stem", collapse(f"{self.stem} {self.command}"))]
        out += [Segment(f"as:{i + 1}", "assertion", collapse(t)) for i, t in enumerate(self.assertions)]
        if self.kind != "assertions":
            out += [Segment(f"alt:{k}", "alternative", collapse(t)) for k, t in self.alternatives.items()]
        return [s for s in out if len(s.text) >= MIN_SEGMENT_CHARS or s.kind == "stem"]


def parse(question: Question) -> ParsedQuestion:
    """Clean the HTML and split the question into its parts."""
    stem_text = html_to_text(question.stem_html)
    stem, assertions = _split_assertions(stem_text)
    alternatives = {k: collapse(html_to_text(v)) for k, v in question.alternatives.items()}
    command = collapse(html_to_text(question.command))
    kind = _question_kind(alternatives, assertions)
    negated = bool(_NEGATED.search(fold(f"{stem} {command}")))
    return ParsedQuestion(
        id=question.id,
        stem=stem,
        command=command,
        assertions=assertions,
        alternatives=alternatives,
        kind=kind,
        negated=negated,
        origin_discipline=question.origin_discipline,
        origin_subject=question.origin_subject,
        is_new_import=question.is_new_import,
    )


def _split_assertions(text: str) -> tuple[str, list[str]]:
    lines = text.split("\n")
    stem_lines, assertions = [], []
    for line in lines:
        match = _LINE_ASSERTION.match(line.strip())
        if match:
            assertions.append(match.group(2).strip())
        elif assertions:
            assertions[-1] = f"{assertions[-1]} {line.strip()}"
        else:
            stem_lines.append(line)
    if len(assertions) >= 2:
        return collapse("\n".join(stem_lines)), assertions
    return _split_inline_assertions(collapse(text))


def _split_inline_assertions(text: str) -> tuple[str, list[str]]:
    matches = list(_INLINE_ASSERTION.finditer(text))
    sequential, expected = [], 0
    for match in matches:
        if expected < len(ROMAN) and match.group(1) == ROMAN[expected]:
            sequential.append(match)
            expected += 1
    if len(sequential) < 2:
        return text, []
    stem = text[: sequential[0].start()].strip()
    bounds = [m.end() for m in sequential] + [len(text)]
    starts = [m.start() for m in sequential[1:]] + [len(text)]
    assertions = [text[bounds[i]: starts[i]].strip() for i in range(len(sequential))]
    return stem, [a for a in assertions if a]


def _question_kind(alternatives: dict[str, str], assertions: list[str]) -> str:
    values = {fold(v).strip(" .") for v in alternatives.values()}
    if alternatives and values <= _TRUE_FALSE:
        return "true_false"
    if len(assertions) >= 2:
        return "assertions"
    return "multiple_choice"
