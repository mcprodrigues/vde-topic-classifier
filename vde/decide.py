"""publish, review or out, with a short reason (ADR-4).

out comes only from explicit rules. Anything that merely looks out of the tree goes to review, so a
question never disappears from the catalog because of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from vde.engine import Prediction
from vde.resources import Resources

OUT_OF_SCOPE_DISCIPLINES = frozenset({"Direito Penal Militar", "Direito Processual Penal Militar"})


@dataclass(frozen=True)
class Decision:
    id: str
    decision: str
    central: int | None
    extras: tuple[int, ...]
    confidence: float
    reason: str


def decide(pred: Prediction, res: Resources, threshold: float) -> Decision:
    if pred.qid in res.gold:
        return _gold_decision(pred, res)
    out_reason = out_rule(pred, res)
    if out_reason:
        return Decision(pred.qid, "fora", None, (), round(pred.confidence, 4), out_reason)
    if pred.choice is None:
        return Decision(pred.qid, "revisar", None, (), 0.0, "nenhum canal trouxe evidência")
    extras = tuple(e for e in pred.extras if e != pred.choice.node)
    if pred.confidence >= threshold:
        return Decision(pred.qid, "publicar", pred.choice.node, extras, round(pred.confidence, 4), "")
    return Decision(pred.qid, "revisar", pred.choice.node, extras, round(pred.confidence, 4),
                    review_reason(pred, res))


def out_rule(pred: Prediction, res: Resources) -> str:
    """Explicit rules only: near duplicate of a labeled out, or an out anchor with no in scope anchor."""
    in_scope = any(a.in_scope for a in pred.anchors)
    if pred.duplicate is not None and pred.duplicate.is_out:
        return f"quase duplicata de {pred.duplicate.labeled_id}, que a equipe marcou como fora"
    if pred.out_reasons and not in_scope:
        return "cita norma sem tema ativo na árvore: " + ", ".join(pred.out_reasons)
    discipline = res.parsed[pred.qid].origin_discipline
    if discipline in OUT_OF_SCOPE_DISCIPLINES and not in_scope:
        return f"disciplina de origem sem tema na árvore ({discipline})"
    return ""


def review_reason(pred: Prediction, res: Resources) -> str:
    tree = res.tree
    choice = pred.choice
    parts = [f"confiança {pred.confidence:.2f} abaixo do limiar"]
    if choice.runner_up is not None and choice.runner_up_mass > 0.15:
        parts.append(f"dúvida entre {choice.branch} ({tree.title(choice.branch)}) e "
                     f"{choice.runner_up} ({tree.title(choice.runner_up)})")
    if pred.features.get("segment_dispersion", 0) >= 0.5:
        parts.append("alternativas espalhadas por vários temas")
    if pred.features.get("anchor_conflict"):
        parts.append("lei citada aponta para outro ramo")
    if pred.features.get("knn_out_share", 0) >= 0.3:
        parts.append("vizinhas rotuladas como fora")
    return "; ".join(parts)


def _gold_decision(pred: Prediction, res: Resources) -> Decision:
    gold = res.gold[pred.qid]
    if gold.is_out:
        return Decision(pred.qid, "fora", None, (), 1.0, "folha-verdade: fora da taxonomia")
    return Decision(pred.qid, "publicar", gold.central, gold.extras, 1.0, "folha-verdade")
