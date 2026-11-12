"""Run bounded recovery stages, retaining the existing GPU authority/ledger.

This driver never launches a long training session or unlocks final-test seeds.
Stages are explicit and output directories are immutable experiment records.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import sys
import time

from run_lr_pilots import ROOT, gpu_command, module, write

OUT = ROOT / "artefacts/autonomous-recovery-20261010"
CONFIG = ROOT / "configs/exploratory-20261010-combined-short-trace.toml"
CHECKPOINT = ROOT / "runs/exploratory-20261010-combined-short-trace/plastic-checkpoint-final.pkl"


def paired_screen(trained, reference):
    """Validate fixed model/seed identity while allowing declared rule changes."""
    from flylatro.evaluation.compare import MATCHED_FIELDS
    import numpy as np
    for report in (trained, reference):
        if (report.get('version') != 'frozen-plastic-evaluation-v2'
                or report.get('learning_disabled') is not True
                or report.get('training_action_schedule_replayed') is not False
                or report.get('action_policy') != 'independent-fixed-motor-choices-no-training-action-replay'
                or report.get('seed_stream') != 'validation'):
            raise ValueError('Screen requires frozen independently acting validation evaluations')
    for field in ('seeds', 'win_ante'):
        if trained[field] != reference[field]:
            raise ValueError(f'Screen {field} differs')
    for field in MATCHED_FIELDS:
        if field in ('reinforcement_mapping_sha256', 'plasticity_rule_sha256'):
            continue  # These are the explicitly varied engineering factors.
        if trained['components'][field] != reference['components'][field]:
            raise ValueError(f'Screen fixed model field differs: {field}')
    seeds = trained['seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Screen requires nonempty unique seeds')
    for report in (trained, reference):
        if [row['seed'] for row in report['episodes']] != seeds:
            raise ValueError('Screen episode order differs from declared seeds')
    pairs = list(zip(trained['episodes'], reference['episodes'], strict=True))
    if any(a['balatro_seed'] != b['balatro_seed'] for a,b in pairs):
        raise ValueError('Screen simulator run seeds differ')
    if any(not np.isfinite([row['blinds_beaten'], row['decisions'], row['episode_return']]).all()
           for pair in pairs for row in pair):
        raise ValueError('Screen outcome measurements must be finite')
    return {
        'mean_blinds_gain': sum(a['blinds_beaten']-b['blinds_beaten'] for a,b in pairs)/len(pairs),
        'win_rate_gain': sum(int(a['won'])-int(b['won']) for a,b in pairs)/len(pairs),
        'mean_return_gain': sum(a['episode_return']-b['episode_return'] for a,b in pairs)/len(pairs),
        'trained_only_wins': sum(a['won'] and not b['won'] for a,b in pairs),
        'reference_only_wins': sum(b['won'] and not a['won'] for a,b in pairs),
        'episodes': len(pairs),
    }


def evaluate(config, checkpoint, destination, label, *, initial=False, episodes=128, batch_size=8, offset=10000):
    command = module("flylatro.evaluation.plastic_cli", "--config", config,
        "--checkpoint", checkpoint, "--output-dir", destination,
        "--episodes", episodes, "--batch-size", batch_size, "--seed-offset", offset,
        "--win-ante", 1, "--seed-stream", "validation", "--heavy", "--progress")
    if initial:
        command.append("--initial-weights")
    gpu_command(command, f"autonomous-recovery-{label}")
    return json.loads((destination / "summary.json").read_text())


def first_evaluations():
    serial_dir, batch_dir = OUT / "parity-serial-8", OUT / "parity-batch8-8"
    serial = evaluate(CONFIG, CHECKPOINT, serial_dir, "parity-serial-8", episodes=8, batch_size=1)
    batch = evaluate(CONFIG, CHECKPOINT, batch_dir, "parity-batch8-8", episodes=8, batch_size=8)
    def trajectories(path):
        return sorted((json.loads(line) for line in (path / "transitions.jsonl").read_text().splitlines()),
            key=lambda row: (row['seed'], row['decision_id']))
    parity = {"episodes_identical": serial['episodes'] == batch['episodes'],
        "transitions_identical": trajectories(serial_dir) == trajectories(batch_dir),
        "frozen_weight_identical": serial['plastic_weight_sha256'] == batch['plastic_weight_sha256']}
    write(OUT / "frozen-batch-parity.json", parity)
    width = 8 if all(parity.values()) else 1
    initial = evaluate(CONFIG, CHECKPOINT, OUT / "initial-validation-128", "initial-validation-128", initial=True, batch_size=width)
    trained = evaluate(CONFIG, CHECKPOINT, OUT / "refined-validation-128", "refined-validation-128", batch_size=width)
    assert initial['seeds'] == trained['seeds']
    pairs = list(zip(trained['episodes'], initial['episodes']))
    comparison = {
        "kind": "exploratory-paired-screen-not-confirmatory", "episodes": len(pairs),
        "seeds": trained['seeds'], "win_ante": 1, "batch_size": width,
        "trained": trained['metrics'], "initial": initial['metrics'],
        "trained_only_wins": sum(a['won'] and not b['won'] for a,b in pairs),
        "initial_only_wins": sum(b['won'] and not a['won'] for a,b in pairs),
        "mean_blinds_beaten_delta": sum(a['blinds_beaten']-b['blinds_beaten'] for a,b in pairs)/len(pairs),
        "mean_return_delta": sum(a['episode_return']-b['episode_return'] for a,b in pairs)/len(pairs),
        "episodes_identical": trained['episodes'] == initial['episodes'],
        "trajectories_identical": trajectories(OUT / 'initial-validation-128') == trajectories(OUT / 'refined-validation-128'),
        "interpretation": "Frozen screening evidence only; independent training seeds and confirmation cohort remain required.",
    }
    write(OUT / "refined-versus-initial.json", comparison)
    return comparison


def screening_pilots(status, status_path):
    from flylatro.learning.config import PlasticExperimentConfig
    # Separate interventions first. The combined-short-trace checkpoint was
    # already evaluated in the first stage and is never retrained/overwritten.
    arms = ['original', 'weak-failure', 'short-trace', 'lower-lr', 'combined']
    baseline = json.loads((OUT / 'initial-validation-128/summary.json').read_text())
    parity = json.loads((OUT / 'frozen-batch-parity.json').read_text())
    width = 8 if all(parity.values()) else 1
    status['arms'] = {}
    for arm in arms:
        config_path = ROOT / f'configs/exploratory-20261010-{arm}.toml'
        config = PlasticExperimentConfig.load(config_path)
        run_dir = ROOT / f'runs/exploratory-20261010-{arm}'
        row = {'config': str(config_path), 'config_sha256': config.sha256,
            'run_dir': str(run_dir), 'status': 'calibrating'}
        status['arms'][arm] = row
        write(status_path, status)
        calibration = OUT / f'{arm}-integrated-calibration.json'
        code = gpu_command(module('flylatro.analysis.calibration_cli', '--config', config_path,
            '--decisions', 16, '--mode', 'integrated', '--output', calibration, '--heavy'),
            f'autonomous-recovery-{arm}-calibration-16', allowed=(0, 2))
        row['calibration'] = str(calibration)
        if code != 0:
            row['status'] = 'calibration-failed'
            write(status_path, status)
            continue
        row['status'] = 'training'
        write(status_path, status)
        code = gpu_command(module('flylatro.learning.cli', '--config', config_path,
            '--run-dir', run_dir, '--heavy', '--no-dashboard'),
            f'autonomous-recovery-{arm}-training-1000', allowed=(0, 2))
        training = json.loads((run_dir / 'run-summary.json').read_text())
        row['training_summary'] = str(run_dir / 'run-summary.json')
        if code != 0 or training['status'] != 'completed':
            row['status'] = 'training-health-failed'
            write(status_path, status)
            continue
        manifest = json.loads((run_dir / 'run-manifest.json').read_text())
        assert manifest['config_sha256'] == config.sha256
        assert training['environment_decisions'] == 1000
        row['status'] = 'evaluating'
        write(status_path, status)
        result = evaluate(config_path, run_dir / 'plastic-checkpoint-final.pkl',
            OUT / f'{arm}-validation-128', f'{arm}-validation-128', batch_size=width)
        comparison = paired_screen(result, baseline)
        row.update(status='screened', evaluation=str(OUT / f'{arm}-validation-128/summary.json'),
            metrics=result['metrics'], **comparison)
        write(status_path, status)
    return status['arms']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['first-evaluations', 'screening-pilots'], required=True)
    parser.add_argument('--after-first-stage', action='store_true',
        help='queue screening pilots behind the verified live first-evaluation driver')
    args = parser.parse_args()
    if args.after_first_stage and args.stage != 'screening-pilots':
        parser.error('--after-first-stage only applies to screening pilots')
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / f'{args.stage}-execution.json'
    if status_path.exists():
        raise FileExistsError(f'Inspect the existing stage and its live process before continuing: {status_path}')
    status = {'stage': args.stage, 'status': 'running', 'pid': os.getpid(),
        'started_at': datetime.now(timezone.utc).isoformat()}
    write(status_path, status)
    try:
        if args.after_first_stage:
            prerequisite_path = OUT / 'first-evaluations-execution.json'
            while True:
                prerequisite = json.loads(prerequisite_path.read_text())
                if prerequisite['status'] == 'completed':
                    break
                if prerequisite['status'] != 'running':
                    raise RuntimeError('First evaluation stage failed; screening was not started')
                # A status file alone does not prove a live prerequisite.
                # Never restart it or run GPU work alongside a missing driver.
                os.kill(int(prerequisite['pid']), 0)
                status.update(status='waiting-for-first-stage', prerequisite_pid=prerequisite['pid'])
                write(status_path, status)
                time.sleep(5)
            status['status'] = 'running'
            write(status_path, status)
        status['result'] = first_evaluations() if args.stage == 'first-evaluations' else screening_pilots(status, status_path)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)


if __name__ == '__main__':
    main()
