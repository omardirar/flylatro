"""Preserve CPU forensic evidence for raw snapshot order and first-step reward."""
import json
from pathlib import Path
import pickle

import numpy as np

from flylatro.analysis.resume_cli import state_digest
from flylatro.env.balatro_sim import BalatroSimAdapter
from flylatro.env.snapshot_identity import canonical_snapshot
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.reinforcement import ReinforcementConfig, ReinforcementMapper
from run_lr_pilots import OUT, ROOT


def main():
    destination = OUT / "resume-serialization-and-reward-forensics.json"
    if destination.exists():
        raise FileExistsError(destination)
    source = ROOT / "runs/ante1-guarded-1000-20261009/mapping-000-replicate-000-plastic_real"
    checkpoint = source / "plastic-checkpoint-stage-000000000100.pkl"
    payload = pickle.loads(checkpoint.read_bytes())
    trainer = payload["trainer"]
    env = BalatroSimAdapter(1, win_ante=1)
    env.reset(trainer["training_seeds"])
    env.restore_all(trainer["environment_snapshots"])
    original = trainer["environment_snapshots"][0]
    restored = env.snapshot_all()[0]
    restored_observations = env.observe()[0]
    action_record = json.loads((source / "training-actions.jsonl").read_text().splitlines()[100])
    actions = {key: np.asarray(value, dtype=np.int64) for key, value in action_record["actions"].items()}
    # Preserve the public pinned action dtypes, including int8/int16 heads.
    from flylatro.env.upstream_contract import ACTION_SPEC
    actions = {key: value.astype(ACTION_SPEC[key][1]) for key, value in actions.items()}
    step = env.step(actions)
    mapper = ReinforcementMapper(ReinforcementConfig(**payload["experiment_config"]["reinforcement"]))
    pulse = mapper.map(step.infos[0]["reward_components"], step.infos[0])
    recorded = json.loads((source / "synthetic-reinforcement-events.jsonl").read_text().splitlines()[100])["synthetic_reinforcement_channels"][0]
    expected = {"synthetic_appetitive": pulse.appetitive, "synthetic_aversive": pulse.aversive, "events": list(pulse.events)}
    report = {"version": "resume-snapshot-reward-forensics-v1", "safe_checkpoint_sha256": sha256_file(checkpoint),
        "raw_snapshot_equal_after_restore": original == restored,
        "canonical_snapshot_equal_after_restore": canonical_snapshot(original, simulator_version=env.simulator_version) == canonical_snapshot(restored, simulator_version=env.simulator_version),
        "saved_observable_state_matches_restore": state_digest(trainer["observations"]) == state_digest(restored_observations),
        "historical_first_resumed_decision": 101, "recorded_pulse": recorded, "correctly_enriched_pulse": expected,
        "pulse_agrees": expected == recorded,
        "source": {"commit": "38ae214317009952db4d22a98dc0765cef79370a", "serialization": "bincode 1; Snapshot.run.rng states HashMap<String,f64> order is not semantic", "paths": ["sim/py/src/env.rs", "sim/core/src/run.rs", "sim/core/src/rng.rs"]},
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            ROOT / "src/flylatro/env/balatro_sim.py", ROOT / "src/flylatro/env/snapshot_identity.py")}}
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({key: report[key] for key in ("raw_snapshot_equal_after_restore", "canonical_snapshot_equal_after_restore", "pulse_agrees", "recorded_pulse", "correctly_enriched_pulse")}))


if __name__ == "__main__":
    main()
