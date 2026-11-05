"""publish, review or out, with a short reason (ADR-4).

out comes only from explicit rules. Anything that merely looks out of the tree goes to review, so a
question never disappears from the catalog because of a guess.

A question with no real evidence never publishes, whatever its confidence (evidence_gate): embedding
channels always return candidates, even for an empty text, so they do not count as evidence on their own.
"""

from __future__ import annotations

from dataclasses import dataclass

from vde.engine import Prediction
from vde.resources import Resources

OUT_OF_SCOPE_DISCIPLINES = frozenset({"Direito Penal Militar", "Direito Processual Penal Militar"})
SLICES = ("none", "clean", "anchored")


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
    gate = evidence_gate(pred.text_evidence, pred.choice.branch_mass, res.config["decision"]["min_branch_mass"])
    if gate:
        return Decision(pred.qid, "revisar", pred.choice.node, extras, round(pred.confidence, 4),
                        f"sem evidência suficiente ({gate}); confiança {pred.confidence:.2f}")
    slice_name = res.config["decision"].get("slice", "none")
    if pred.confidence >= threshold and not eligible(pred.features, slice_name):
        return Decision(pred.qid, "revisar", pred.choice.node, extras, round(pred.confidence, 4),
                        f"fora da faixa de publicação ({slice_name}): " + review_reason(pred, res))
    if pred.confidence >= threshold:
        return Decision(pred.qid, "publicar", pred.choice.node, extras, round(pred.confidence, 4), "")
    return Decision(pred.qid, "revisar", pred.choice.node, extras, round(pred.confidence, 4),
                    review_reason(pred, res))


def evidence_gate(text_evidence: bool, branch_mass: float, min_branch_mass: float) -> str:
    """Why a question may not publish for lack of evidence, or "" when it may.

    Two pre declared conditions, independent of the confidence model: at least one text based channel has
    a score, and the chosen branch holds at least `min_branch_mass` of the probability.
    """
    reasons = []
    if not text_evidence:
        reasons.append("nenhum termo, lei, artigo ou assunto reconhecido no texto")
    if branch_mass < min_branch_mass:
        reasons.append(f"massa do ramo {branch_mass:.2f} abaixo de {min_branch_mass:.2f}")
    return "; ".join(reasons)


def eligible(features: dict, slice_name: str = "none") -> bool:
    """Pre declared publication slices, evaluated out of fold on the same feature rows (ADR-11).

    none: every question may publish. clean: no labeled out neighbour, no law pointing elsewhere and
    alternatives that stay on one subject. anchored: clean plus a legal anchor or channel agreement.
    """
    if slice_name not in SLICES:
        raise ValueError(f"unknown slice {slice_name!r}")
    if slice_name == "none":
        return True
    clean = (features["knn_out_share"] == 0 and features["anchor_conflict"] == 0
             and features["segment_dispersion"] < 0.5)
    if slice_name == "clean":
        return bool(clean)
    anchored = features["special_law_hit"] or features["code_article_hit"] or features["agreement"] >= 0.8
    return bool(clean and anchored)


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
    """The team's label, moved to the canonical twin: a shadow node never reaches the publication."""
    gold = res.gold[pred.qid]
    if gold.is_out:
        return Decision(pred.qid, "fora", None, (), 1.0, "folha-verdade: fora da taxonomia")
    tree = res.tree
    central = tree.canonical(gold.central) if gold.central in tree else gold.central
    extras = []
    for extra in gold.extras:
        canon = tree.canonical(extra) if extra in tree else extra
        if canon != central and canon not in extras:
            extras.append(canon)
    return Decision(pred.qid, "publicar", central, tuple(extras), 1.0, "folha-verdade")
