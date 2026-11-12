"""Read-only KC->MBON source coverage audit against the local unthresholded export."""
from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from flylatro.fly.flywire_artifact import FlyWireArtifact, sha256_file


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "artefacts/lr-sensitivity-20261009/source-edge-coverage.json"
    if output.exists():
        raise FileExistsError(output)
    artifact = FlyWireArtifact.load(root / "data/flywire/flywire_fafb_v783.npz")
    kc = artifact.root_ids[artifact.kenyon_indices]
    mbon = artifact.root_ids[artifact.mbon_indices]
    counts = {}
    started = time.monotonic()
    for filename in ("connections_princeton.csv.gz", "connections_princeton_no_threshold.csv.gz"):
        pairs = defaultdict(int)
        rows = 0
        path = root / "data" / filename
        for chunk in pd.read_csv(path, usecols=["pre_root_id", "post_root_id", "syn_count"], chunksize=250000):
            selected = np.isin(chunk.pre_root_id.to_numpy(), kc) & np.isin(chunk.post_root_id.to_numpy(), mbon)
            for pre, post, count in chunk.loc[selected].itertuples(index=False, name=None):
                pairs[(int(pre), int(post))] += int(count)
            rows += len(chunk)
        counts[filename] = {"sha256": sha256_file(path), "rows": rows,
            "kc_mbon_pairs": len(pairs), "kc_mbon_synapses": sum(pairs.values()),
            "thresholds": {str(n): {"edges": sum(v >= n for v in pairs.values()),
                                   "synapses": sum(v for v in pairs.values() if v >= n)} for n in (1, 2, 5, 10)}}
        print(json.dumps({"source": filename, **counts[filename]}), flush=True)
    output.write_text(json.dumps({"version": "local-source-edge-coverage-v1", "artifact_sha256": artifact.manifest["artifact_sha256"],
        "sources": counts, "cpu_wall_seconds": time.monotonic() - started,
        "interpretation": "Source coverage measurement only; no topology or historical evidence was changed."}, indent=2) + "\n")
