"""Materialise a self-contained showcase reproducibility bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Sequence

from flylatro.replay.bundle import ReplayBundle, sha256_file


def package_showcase(
    source_bundle: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
    config: Path,
) -> ReplayBundle:
    source = ReplayBundle.open(source_bundle)
    checkpoint_hash = sha256_file(checkpoint)
    expected = source.identity.checkpoint_sha256
    if expected is not None and checkpoint_hash != expected:
        raise ValueError("checkpoint does not match replay identity")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    shutil.copytree(source.path, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copies = {
        "checkpoint.pt": checkpoint,
        "experiment.toml": config,
    }
    for name, source_path in copies.items():
        destination = output_dir / name
        shutil.copy2(source_path, destination)
        manifest["files"][name] = {"sha256": sha256_file(destination)}
    readme = output_dir / "README.md"
    identity = source.identity
    episode = manifest.get("metadata", {}).get("episode", {})
    readme.write_text(
        "# Flylatro showcase replay\n\n"
        f"- Balatro seed: `{identity.balatro_seed}`\n"
        f"- Simulator seed: `{identity.simulator_seed}`\n"
        f"- Result: `{'win' if episode.get('won') else 'not recorded as win'}`\n"
        f"- Maximum Ante: `{episode.get('ante')}`\n"
        f"- Checkpoint SHA-256: `{checkpoint_hash}`\n"
        f"- Connectome SHA-256: `{identity.connectome_hash}`\n"
        f"- Encoder SHA-256: `{identity.encoder_hash}`\n"
        f"- Feature extractor SHA-256: `{identity.feature_extractor_hash}`\n"
        f"- Fly backend: `{identity.fly_backend_version}`\n"
        f"- Fly dynamics SHA-256: `{identity.fly_dynamics_hash}`\n"
        f"- Balatro simulator: `{identity.simulator_version}`\n"
        f"- Policy: `{identity.policy_version}`\n\n"
        "Verify from the repository root:\n\n"
        "```bash\n"
        "flylatro-replay --bundle PATH_TO_THIS_DIRECTORY "
        "--config PATH_TO_THIS_DIRECTORY/experiment.toml --heavy\n"
        "```\n",
        encoding="utf-8",
    )
    manifest["files"][readme.name] = {"sha256": sha256_file(readme)}
    manifest.setdefault("metadata", {})["packaged_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return ReplayBundle.open(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    packaged = package_showcase(
        args.bundle,
        args.output_dir,
        checkpoint=args.checkpoint,
        config=args.config,
    )
    print(
        json.dumps(
            {"bundle": str(packaged.path), "files": sorted(packaged.manifest["files"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
