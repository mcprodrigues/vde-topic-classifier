"""Command line: vde publish | vde evaluate | vde build <artifact> | vde diagnose."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ARTIFACTS = ["canonical_tree", "law_index", "code_atlas", "topic_cards", "subject_map", "embeddings", "model", "all"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    start = time.time()
    if args.command == "publish":
        from vde.pipeline import publish  # read only path: loaders and the trained model, nothing else

        decisions = publish(Path(args.out) if args.out else None, strict_manifest=not args.skip_manifest)
        counts = {k: sum(d.decision == k for d in decisions) for k in ("publicar", "revisar", "fora")}
        print(f"publicacao.csv: {len(decisions)} linhas {counts}")
    elif args.command == "build":
        _build(args.artifact, args.source)
    elif args.command == "evaluate":
        from vde.report import evaluate

        report = evaluate(ablations=args.ablations)
        fc = report["forecast"]
        print(f"cobertura {fc['coverage']:.3f}, precisao prevista {fc['precision']}, limiar {report['threshold']:.3f}")
    elif args.command == "diagnose":
        from vde.config import load_config
        from vde.resources import Resources
        from vde.shift import describe

        res = Resources.load(load_config())
        new = [q for q in res.question_ids if res.parsed[q].is_new_import]
        old = [q for q in res.question_ids if not res.parsed[q].is_new_import]
        print(describe(res, old, new))
    print(f"[{time.time() - start:.1f}s]", file=sys.stderr)
    return 0


def _build(artifact: str, source: str) -> None:
    from vde.config import load_config
    from vde.knowledge.base import refresh_manifest
    from vde.knowledge.registry import BUILD_ORDER, builders

    config = load_config()
    names = BUILD_ORDER + ["model"] if artifact == "all" else [artifact]
    table = builders(config, source)
    for name in names:
        if name == "model":
            from vde.pipeline import train_model

            model = train_model(config)
            print(f"model: threshold {model['threshold']:.3f}")
        else:
            print(f"{name}: {table[name]()}")
    refresh_manifest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vde", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="regenerate outputs/publicacao.csv without any LLM")
    pub.add_argument("--out", help="output path (default outputs/publicacao.csv)")
    pub.add_argument("--skip-manifest", action="store_true", help="do not stop when artifacts differ from the manifest")
    build = sub.add_parser("build", help="rebuild a versioned artifact (offline step)")
    build.add_argument("artifact", choices=ARTIFACTS)
    build.add_argument("--source", choices=["session", "llm"], default="session",
                       help="llm writes a proposal to artifacts/proposals for human review")
    ev = sub.add_parser("evaluate", help="metrics, figures, shift report and previsao.txt")
    ev.add_argument("--ablations", action="store_true")
    sub.add_parser("diagnose", help="compare the bank with the new import")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
