"""The classification engine: channels, aggregation, node choice and the correctness model.

fit() learns only from the labeled ids it receives, so cross validation can call it per fold.
The correctness model is trained on inner out of fold features, never on in sample scores. The deployed
engine can add a recalibration fitted on the outer out of fold rows (see pipeline.train_model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vde.aggregate import FEATURES, CandidateScorer, CorrectnessModel
from vde.calibrate import Platt
from vde.channels import build_channels
from vde.channels.duplicates import DuplicateChannel, DuplicateMatch
from vde.channels.knn import KnnChannel
from vde.channels.law_article import Anchor, CodeArticleChannel, LawArticleChannel, SpecialLawChannel
from vde.channels.lexical import LexicalChannel
from vde.channels.prototypes import PrototypeChannel
from vde.channels.segmented import SegmentAnalyzer
from vde.data.loader import GoldLabel
from vde.knowledge.canonical_tree import learn_twins
from vde.resources import Resources
from vde.tree import NodeChoice, choose_node, pick_extras

INNER_FOLDS = 5


@dataclass
class Prediction:
    qid: str
    p: dict[int, float]
    choice: NodeChoice | None
    features: dict
    confidence: float
    extras: list[int]
    anchors: tuple[Anchor, ...]
    out_reasons: list[str]
    duplicate: DuplicateMatch | None
    channel_tops: dict[str, int] = field(default_factory=dict)
    # True when at least one text based channel (BM25, cited law or article, subject, duplicate) has a score.
    text_evidence: bool = True


class Engine:
    def __init__(self, res: Resources, channel_names: list[str] | None = None, node_mode: str | None = None):
        self.res = res
        self.config = res.config
        self.node_mode = node_mode or self.config["decision"].get("node_mode", "branch")
        self.rho = float(self.config["decision"]["rho"])
        names = channel_names or list(self.config["channels"]["enabled"])
        self.channels = build_channels(res, names)
        self.proposing = {n for n, c in self.channels.items() if c.proposes_candidates}
        # Helpers used for features and rules, whatever channels are enabled.
        self.law = self._shared(LawArticleChannel)
        self.special = self._shared(SpecialLawChannel)
        self.code = self._shared(CodeArticleChannel)
        self.knn = self._shared(KnnChannel)
        self.duplicates = self._shared(DuplicateChannel)
        self.segments = SegmentAnalyzer(res, self._shared(LexicalChannel), self._shared(PrototypeChannel))
        self.prototypes = self._shared(PrototypeChannel)
        agg = self.config["aggregate"]
        self.scorer = CandidateScorer(names, agg["l2"], agg["max_weight"], agg.get("transform", "log"))
        self.correctness = CorrectnessModel(agg["correctness_c"])
        self.recalibration: Platt | None = None
        self.fit_info: dict = {}

    def fit(self, labeled_ids: list[str], seed: int = 0) -> dict:
        labeled = {q: self.res.gold[q] for q in labeled_ids}
        rows, ys = [], []
        rng = np.random.default_rng(seed)
        order = rng.permutation(labeled_ids)
        for k in range(INNER_FOLDS):
            inner_test = list(order[k::INNER_FOLDS])
            held_out = set(inner_test)
            inner_train = {q: labeled[q] for q in labeled_ids if q not in held_out}
            self._fit_scorer(inner_train)
            for qid in inner_test:
                pred = self._predict_one(qid)
                rows.append(pred.features)
                ys.append(self.is_hit(pred, labeled[qid]))
        self.correctness.fit(rows, ys)
        self.fit_info = self._fit_scorer(labeled)
        self.fit_info["inner_hit_rate"] = float(np.mean(ys)) if ys else 0.0
        return self.fit_info

    def predict(self, qids: list[str]) -> list[Prediction]:
        preds = [self._predict_one(q) for q in qids]
        conf = self.correctness.predict([p.features for p in preds]) if preds else []
        if self.recalibration is not None and len(conf):
            conf = self.recalibration.predict(conf)
        for pred, c in zip(preds, conf):
            pred.confidence = float(c) if pred.choice is not None else 0.0
        return preds

    def is_hit(self, pred: Prediction, gold: GoldLabel) -> bool:
        if gold.is_out or pred.choice is None:
            return False
        return pred.choice.node in self.res.tree.correct_set(gold.central, gold.extras)

    def branch_hit(self, pred: Prediction, gold: GoldLabel) -> bool:
        if gold.is_out or pred.choice is None:
            return False
        return pred.choice.branch in self.res.tree.correct_set(gold.central, gold.extras)

    def export_state(self) -> dict:
        state = {"scorer": self.scorer.to_dict(), "correctness": self.correctness.to_dict(),
                 "node_mode": self.node_mode, "rho": self.rho}
        if self.recalibration is not None:
            state["recalibration"] = self.recalibration.to_dict()
        return state

    def load_state(self, state: dict, labeled_ids: list[str]) -> None:
        """Restore trained weights. Label dependent channels are re-indexed from the gold sheet."""
        self.scorer = CandidateScorer.from_dict(state["scorer"])
        self.correctness = CorrectnessModel.from_dict(state["correctness"])
        self.rho = float(state.get("rho", self.rho))
        self.recalibration = Platt.from_dict(state["recalibration"]) if "recalibration" in state else None
        labeled = {q: self.res.gold[q] for q in labeled_ids}
        self.res.tree.twins = learn_twins(self.res.tree, labeled)
        self._fit_channels(labeled)

    def _shared(self, cls):
        for channel in self.channels.values():
            if type(channel) is cls:
                return channel
        return cls(self.res)

    def _fit_channels(self, labeled: dict[str, GoldLabel]) -> None:
        seen = set()
        for channel in list(self.channels.values()) + [self.knn, self.duplicates]:
            if channel.label_dependent and id(channel) not in seen:
                channel.fit(labeled)
                seen.add(id(channel))

    def _fit_scorer(self, labeled: dict[str, GoldLabel]) -> dict:
        # The twin rule is learned from labels, so it is refitted on every training part (no fold leakage).
        self.res.tree.twins = learn_twins(self.res.tree, labeled)
        self._fit_channels(labeled)
        designs = []
        for qid, gold in labeled.items():
            if gold.is_out:
                continue
            table = self._table(qid)
            correct = self.res.tree.correct_set(gold.central, gold.extras)
            designs.append(self.scorer.design(table, self.res.tree, self.proposing, correct))
        return self.scorer.fit(designs)

    def _table(self, qid: str) -> dict[str, dict[int, float]]:
        return {name: channel.score(qid) for name, channel in self.channels.items()}

    def _predict_one(self, qid: str) -> Prediction:
        table = self._table(qid)
        design = self.scorer.design(table, self.res.tree, self.proposing)
        p = self.scorer.distribution(design)
        choice = choose_node(p, self.res.tree, self.rho, self.node_mode)
        tops = {n: self.res.tree.branch_of(max(s, key=s.get)) for n, s in table.items()
                if s and n in self.proposing}
        text_evidence = any(table[n] for n in self.proposing if not self.channels[n].embedding_based)
        extras = pick_extras(p, choice, self.res.tree, self.config["decision"]["extras_min_mass"],
                             self.segments.branches(qid)) if choice else []
        return Prediction(
            qid=qid, p=p, choice=choice, features=self._features(qid, choice, tops), confidence=0.0,
            extras=extras, anchors=self.law.anchors(qid), out_reasons=self.law.out_reasons(qid),
            duplicate=self.duplicates.match(qid), channel_tops=tops, text_evidence=text_evidence,
        )

    def _features(self, qid: str, choice: NodeChoice | None, tops: dict[str, int]) -> dict:
        parsed = self.res.parsed[qid]
        if choice is None:
            return {name: 0.0 for name in FEATURES}
        mass = min(max(choice.branch_mass, 1e-4), 1 - 1e-4)
        special = self.special.in_scope_branches(qid)
        code = self.code.in_scope_branches(qid)
        anchors = special | code
        dup = self.duplicates.match(qid)
        return {
            "branch_logit": math.log(mass / (1 - mass)),
            "margin": choice.branch_mass - choice.runner_up_mass,
            "agreement": (sum(b == choice.branch for b in tops.values()) / len(tops)) if tops else 0.0,
            "special_law_hit": float(choice.branch in special),
            "code_article_hit": float(choice.branch in code),
            "anchor_conflict": float(bool(anchors) and choice.branch not in anchors),
            "knn_out_share": self.knn.out_share(qid),
            "segment_support": self.segments.support(qid, choice.branch),
            "segment_dispersion": self.segments.dispersion(qid),
            "true_false": float(parsed.kind == "true_false"),
            "assertions": float(parsed.kind == "assertions"),
            "negated": float(parsed.negated),
            "log_length": math.log1p(parsed.length),
            "top_cosine": self.prototypes.top_cosine(qid),
            "descended": float(choice.descended),
            "bin_branch": float(self.res.tree.is_bin(choice.branch)),
            "duplicate_hit": float(dup is not None and not dup.is_out),
        }

