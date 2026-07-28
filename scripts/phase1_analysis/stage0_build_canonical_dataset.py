#!/usr/bin/env python3
"""
Phase 1 real-data analysis -- Stage 0: build the canonical dataset manifest.

Loads both real Phase 1 source directories (src/analysis/phase1_canonical.py::CANONICAL_SOURCES),
keeps each directory's designated domain only (tower_of_hanoi from the 2026-07-22 run,
textworld from the 2026-07-24 re-collection -- see that module's docstring for why), asserts the
result matches the frozen preregistered design (1500 episodes, 750/domain, 250/domain/stage, 5
holdout instances/domain), and writes a lightweight JSON manifest (episode_id/domain/source_dir
triples + a content hash) that every later stage reads through instead of re-deriving the
domain/directory selection itself.

Idempotent: same two source directories -> same manifest (episode ordering is sorted before
hashing, so the content_hash is stable across repeated runs regardless of filesystem iteration
order).

Usage:
  python scripts/phase1_analysis/stage0_build_canonical_dataset.py \
      --output data/results/phase1_analysis/stage0/canonical_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phase1_canonical import (  # noqa: E402
    CANONICAL_SOURCES,
    assert_canonical_invariants,
    build_canonical_dataset,
)


def _content_hash(episode_ids: list[str]) -> str:
    joined = "\n".join(sorted(episode_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_manifest(sources: dict[str, Path] | None = None) -> dict[str, Any]:
    ds = build_canonical_dataset(sources)
    assert_canonical_invariants(ds)  # raises loudly on any deviation -- do not catch
    entries = [
        {
            "episode_id": e.get("episode_id"),
            "domain": e.get("domain"),
            "source_dir": e.get("_source_dir"),
        }
        for e in ds.episodes
    ]
    entries.sort(key=lambda r: (str(r["domain"]), str(r["episode_id"])))
    return {
        "selection_rule": (
            "tower_of_hanoi from data/results/phase1/phase1_20260722_091125 (valid); "
            "textworld from data/results/phase1/textworld_regen_20260724 (valid re-collection "
            "after commit b47e35d fixed a silent unwinnable-stub fallback that had corrupted "
            "45/50 textworld instances in the 2026-07-22 run's textworld half -- that half is "
            "excluded entirely, not merged). See docs/consistency_log.md for the dated entry."
        ),
        "sources": {k: str(v) for k, v in ds.sources.items()},
        "n_episodes": len(entries),
        "content_hash": _content_hash([str(e["episode_id"]) for e in entries]),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        default="data/results/phase1_analysis/stage0/canonical_manifest.json",
        help="Manifest output path.",
    )
    args = parser.parse_args()

    try:
        manifest = build_manifest(dict(CANONICAL_SOURCES))
    except AssertionError as exc:
        print(f"Stage 0 FAILED -- canonical dataset invariant violated: {exc}", file=sys.stderr)
        return 1

    out_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 0 OK -- {manifest['n_episodes']} episodes, manifest written to {out_path}")
    print(f"  content_hash={manifest['content_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
