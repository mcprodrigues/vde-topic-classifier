"""Choose the node to publish in, not only the leaf (direction 1, ADR-3).

Under the case rule, P(hit | level 1 node T) = p(T) + sum p(children of T), which is never smaller
than P(hit | child c) = p(c) + p(T). So we publish at the best level 1 branch and descend to a
subtopic only when it holds at least rho of the branch mass. Level 0 is never an option.
"""

from __future__ import annotations

from dataclasses import dataclass

# level1_bins: level 1 everywhere except in a bin topic, where it descends to a subtopic like "branch".
MODES = ("branch", "leaf", "level1", "level1_bins")


@dataclass(frozen=True)
class NodeChoice:
    node: int
    branch: int
    branch_mass: float
    runner_up: int | None
    runner_up_mass: float
    descended: bool


def branch_masses(p: dict[int, float], tree) -> dict[int, float]:
    masses: dict[int, float] = {}
    for node, prob in p.items():
        top = tree.branch_of(node)
        masses[top] = masses.get(top, 0.0) + prob
    return masses


def choose_node(p: dict[int, float], tree, rho: float, mode: str = "branch") -> NodeChoice | None:
    if not p:
        return None
    if mode not in MODES:
        raise ValueError(f"unknown node mode {mode!r}")
    masses = branch_masses(p, tree)
    ranked = sorted(masses.items(), key=lambda kv: -kv[1])
    runner_up, runner_mass = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    if mode == "leaf":
        return _leaf_choice(p, tree, masses)
    branch, mass = ranked[0]
    if mode == "level1" or (mode == "level1_bins" and not tree.is_bin(branch)):
        return NodeChoice(branch, branch, mass, runner_up, runner_mass, False)
    children = {n: q for n, q in p.items() if n != branch and tree.level1_of(n) == branch}
    if children and mass > 0:
        child, child_p = max(children.items(), key=lambda kv: kv[1])
        if child_p / mass >= rho:
            return NodeChoice(child, branch, mass, runner_up, runner_mass, True)
    return NodeChoice(branch, branch, mass, runner_up, runner_mass, False)


def pick_extras(p: dict[int, float], choice: NodeChoice, tree, min_mass: float,
                segment_branches: dict[int, int]) -> list[int]:
    """Other level 1 branches with enough mass, or supported by at least two segments."""
    masses = branch_masses(p, tree)
    extras = [b for b, m in masses.items() if b != choice.branch and m >= min_mass]
    extras += [b for b, count in segment_branches.items()
               if b != choice.branch and count >= 2 and b not in extras and masses.get(b, 0.0) >= min_mass / 4]
    return sorted(extras, key=lambda b: -masses.get(b, 0.0))[:3]


def _leaf_choice(p: dict[int, float], tree, masses: dict[int, float]) -> NodeChoice:
    """Ablation E3: publish the single most probable node, with no branch reasoning."""
    node, prob = max(p.items(), key=lambda kv: kv[1])
    branch = tree.branch_of(node)
    if tree.level1_of(node) != branch:
        node = branch
    others = sorted(((b, m) for b, m in masses.items() if b != branch), key=lambda kv: -kv[1])
    runner_up, runner_mass = others[0] if others else (None, 0.0)
    return NodeChoice(node, branch, masses[branch], runner_up, runner_mass, tree.level(node) == 2)
