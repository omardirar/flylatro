"""Bounded local-plasticity study of play-now versus discarding for a draw."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from card_learning.common import identity, write_json
from card_learning.motor import CardExploration, install_motor
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack, resolve_path
from discard_strategy import StrategyStudy, VERSION, evaluate_choices
from run_local_normalisation import verify
from strategy_puzzles import PuzzleCorpus, Reference, benchmark, build


def read(path):
    return json.loads(Path(path).read_text())


def prepare(config_path, output, *, steps=2000, replicates=3, per_suit=3, draws=128, repeats=2):
    if min(steps, replicates, per_suit, draws, repeats) < 1: raise ValueError('positive budgets required')
    output = output.resolve(); config_path = config_path.resolve()
    config = PlasticExperimentConfig.load(config_path)
    if (config.environment.allow_skip_blinds or config.plasticity.version != 'three-factor-choice-gated-v1'
            or config.plasticity.dopamine_decay != 0 or config.reinforcement.run_failure_pulse != .01):
        raise ValueError('unchanged no-skip, choice-gated, decision-local outcome configuration required')
    output.mkdir(parents=True, exist_ok=False)
    corpus = build(output/'puzzles.json', per_suit=per_suit)
    paths = [config_path, corpus.path]
    paths += [Path(__file__).with_name(name) for name in ('run_discard_strategy.py','discard_strategy.py','strategy_transfer.py',
        'strategy_puzzles.py','local_normalisation.py','run_local_normalisation.py','run_card_membership.py',
        'run_training_bottlenecks.py','train_discard_play_curriculum.py')]
    for path in (config.fly.artifact_path if config.fly.backend == 'flywire' else '',
                 config.fly.motor_mapping_path, config.calibration.corpus_path):
        if path: paths.append(resolve_path(config, path))
    write_json(output/'declaration.json', {'version': VERSION, 'implementation': identity(),
        'input_sha256': {str(p.resolve()):sha256_file(p) for p in paths}, 'config':str(config_path),
        'steps':steps,'replicates':replicates,'reference_draws':draws,'repeats':repeats,
        'stages':sorted(set([min(1000,steps),steps])), 'per_suit':per_suit,
        'arms':['original','normalised','frozen'],'fresh_unit_weights':True,
        'reinforcement':asdict(config.reinforcement),'plasticity':asdict(config.plasticity),
        'assistance':['two curated discard alternatives, both exactly four cards','reference chooses PLAY cards from current visible hand'],
        'learning_scope':'action type PLAY/DISCARD and DISCARD membership only; helper PLAY gets no fresh eligibility',
        'hidden_futures':'native redeterminize after policy choice; public observations invariant; independent train/benchmark/transfer banks',
        'reward':'actual native progress, blind clear and failure; no direct discard bonus or hand-type label',
        'success_criteria':{'minimum_expected_clear_gain_vs_initial':.05,'minimum_positive_learners':2,
            'minimum_play_fraction_contrast_playnow_minus_drawneeded':.10},
        'free_membership_draws':32, 'fly_play_transfer_draws':8,
        'limits':'assisted three-choice strategy test; fixed discard count; no full-game mastery claim',
        'final_test_used':False})
    write_json(output/'status.json',{'status':'prepared'})


def free_membership(study, rows, reference, draws):
    values = []
    for row in rows:
        per_lane = []
        for discard in row['discard_sets_native']:
            per_lane.append(float(np.mean([reference.outcome(row['case_index'],discard,('free-transfer',i))['cleared']
                                          for i in range(draws)])))
        values.append(per_lane)
    result = {}
    for role,name in enumerate(('original','normalised','frozen')):
        matrix=np.asarray(values)[:,study.roles==role]
        result[name]={'per_learner':matrix.mean(axis=0).tolist(),'mean':float(matrix.mean())}
    return {'summary':result,'values':values,'decisions':rows,'draws':draws,
        'scope':'unrestricted discard membership at fixed count four; PLAY remains assisted'}


def summary(study, initial, final):
    output={}
    for name in ('original','normalised','frozen'):
        a=np.asarray(initial[name]['estimated_clear_probability']);b=np.asarray(final[name]['estimated_clear_probability'])
        contrast=float(np.mean(final[name]['play-now']['play_fraction'])-np.mean(final[name]['draw-needed']['play_fraction']))
        output[name]={'initial_expected_clear':float(a.mean()),'trained_expected_clear':float(b.mean()),
            'change_per_learner':(b-a).tolist(),'mean_change':float((b-a).mean()),
            'playnow_play_fraction':final[name]['play-now']['play_fraction'],
            'drawneeded_play_fraction':final[name]['draw-needed']['play_fraction'],
            'conditional_type_contrast':contrast,'reference_regret':final[name]['mean_reference_regret']}
    n=output['normalised']
    checks={'gain_at_least_five_points':n['mean_change']>=.05,
        'gain_in_multiple_learners':int(np.count_nonzero(np.asarray(n['change_per_learner'])>0))>=2,
        'conditional_type_contrast':n['conditional_type_contrast']>=.10}
    output['advance']={'checks':checks,'assisted_strategy_signal':all(checks.values()),
        'full_game_strategy_demonstrated':False}
    return output


def execute(output, *, heavy=False):
    output=output.resolve(); declaration=read(output/'declaration.json');verify(declaration)
    if (output/'report.json').exists():return read(output/'report.json')
    corpus=PuzzleCorpus(output/'puzzles.json')
    if not (output/'benchmark.json').exists():
        write_json(output/'status.json',{'status':'benchmarking'})
        reference=benchmark(corpus,output/'benchmark.json',draws=declaration['reference_draws'])
        write_json(output/'benchmark-manifest.json',{'sha256':sha256_file(output/'benchmark.json'),
            'declaration_sha256':sha256_file(output/'declaration.json')})
    else:reference=read(output/'benchmark.json')
    receipt=read(output/'benchmark-manifest.json')
    if (receipt['sha256']!=sha256_file(output/'benchmark.json')
            or receipt['declaration_sha256']!=sha256_file(output/'declaration.json')):
        raise ValueError('reference benchmark changed')
    config=PlasticExperimentConfig.load(Path(declaration['config']));config.require_heavy_opt_in(heavy)
    stack=install_motor(build_plastic_stack(config),CardExploration(.05))
    if not stack.agent.motor.mapping.routing.action_conditioned:raise ValueError('separate PLAY/DISCARD mapping required')
    study=StrategyStudy(stack,corpus,replicates=declaration['replicates'])
    started=time.perf_counter(); log_path=output/'training.jsonl';log_path.touch(exist_ok=True)
    if (output/'checkpoint.json').exists():study.restore(output)
    elif log_path.stat().st_size:raise ValueError('uncheckpointed output exists; preserve it')
    def evaluate(stage):
        path=output/f'evaluation-{stage:07d}.json'
        if not path.exists():
            rows={split:study.decisions(split,declaration['repeats'],initial=stage==0) for split in ('training','validation')}
            write_json(path,{'step':stage,'rows':rows,
                'summary':{split:evaluate_choices(study,data,reference) for split,data in rows.items()},'health':study.health()})
    if not (output/'evaluation-0000000.json').exists():evaluate(0)
    if study.steps in declaration['stages']:evaluate(study.steps)
    try:
        with log_path.open('ab') as stream:
            while study.steps<declaration['steps']:
                row=study.train_step();stream.write((json.dumps(row,sort_keys=True)+'\n').encode())
                if study.steps%100==0 or study.steps in declaration['stages']:
                    stream.flush();health=study.health()
                    write_json(output/'status.json',{'status':'training',**health})
                    print(json.dumps({'step':study.steps,'budget_error':health['max_post_budget_relative_error_per_lane']}),flush=True)
                if study.steps%250==0 or study.steps in declaration['stages']:
                    stream.flush();study.save(output,stream.tell())
                if study.steps in declaration['stages']:evaluate(study.steps)
        write_json(output/'status.json',{'status':'fresh-evaluation','steps':study.steps})
        fresh={label:study.decisions('fresh',declaration['repeats'],initial=label=='initial') for label in ('initial','trained')}
        fresh_summary={label:evaluate_choices(study,rows,reference) for label,rows in fresh.items()}
        write_json(output/'fresh-evaluation.json',{'rows':fresh,'summary':fresh_summary})
        evaluator=Reference(corpus)
        free={label:free_membership(study,study.decisions('fresh',declaration['repeats'],initial=label=='initial',free_cards=True),
            evaluator,declaration['free_membership_draws']) for label in ('initial','trained')}
        write_json(output/'free-membership-transfer.json',free)
        from strategy_transfer import fly_transfer
        write_json(output/'fly-play-transfer.json',fly_transfer(study,draws=declaration['fly_play_transfer_draws']))
        verify(declaration)
        result={'version':VERSION,'status':'completed','steps':study.steps,
            'roles':study.roles.tolist(),'replicate_ids':study.ids.tolist(),'health':study.health(),
            'summary':summary(study,fresh_summary['initial'],fresh_summary['trained']),
            'declaration_sha256':sha256_file(output/'declaration.json'),
            'assisted':True,'wall_seconds_this_invocation':time.perf_counter()-started,'final_test_used':False}
        write_json(output/'report.json',result);write_json(output/'status.json',{'status':'completed','steps':study.steps})
        return result
    except BaseException as error:
        write_json(output/'status.json',{'status':'stopped','steps':study.steps,'error':f'{type(error).__name__}: {error}'})
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('artefacts/learnable-motor-20261028-v2/config.toml'))
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--replicates',type=int,default=3)
    p.add_argument('--per-suit',type=int,default=3);p.add_argument('--draws',type=int,default=128)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--heavy',action='store_true');a=p.parse_args(argv)
    output=a.output_dir.resolve()
    if not output.exists():prepare(a.config,output,steps=a.steps,replicates=a.replicates,per_suit=a.per_suit,draws=a.draws,repeats=a.repeats)
    if not a.prepare_only:print(json.dumps(execute(output,heavy=a.heavy)['summary'],indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
