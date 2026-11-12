"""Select a compact calibration corpus preserving rare observable contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from flylatro.analysis.corpus import CalibrationCorpus
from flylatro.analysis.corpus_selection import select_calibration_corpus
from flylatro.analysis.coverage import CoveragePolicy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, required=True)
    parser.add_argument("--states-per-stratum", type=int, default=8)
    parser.add_argument("--coverage-policy", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report_path = args.report or args.output.with_suffix(".selection.json")
    destinations = (args.output, args.output.with_suffix(args.output.suffix + ".manifest.json"), report_path)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("corpus, manifest and selection report need distinct paths")
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"selection destination already exists: {path}")
    corpus, report = select_calibration_corpus(
        CalibrationCorpus.load(args.source),
        states=args.states,
        states_per_stratum=args.states_per_stratum,
        policy=CoveragePolicy.load(args.coverage_policy) if args.coverage_policy else None,
    )
    corpus.save(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(report_path),
        "states": len(corpus),
        "sha256": corpus.sha256,
        "status": report["status"],
        "deficiencies": report["coverage"]["deficiency_summary"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
