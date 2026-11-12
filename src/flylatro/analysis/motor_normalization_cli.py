"""Fit reward-free 250 ms motor normalization and test it on reserved states."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from flylatro.analysis.card_information import measure_learner
from flylatro.analysis.card_selection import plastic_hashes
from flylatro.analysis.card_window import integration_window
from flylatro.analysis.card_window_pilot import write_json
from flylatro.analysis.corpus import build_calibration_corpus
from flylatro.analysis.evidence import load_corpus
from flylatro.analysis.motor_normalization import (
    VERSION, CRITERION, assess, evaluate_cached, fit_fixed_normalization, heldout_cases,
    membership_hash,
)
from flylatro.fly.flywire_artifact import sha256_file
from flylatro.fly.mushroom_body.state import state_numpy
from flylatro.learning.checkpoints import load_plastic_checkpoint, load_plastic_manifest
from flylatro.learning.choice_credit_pilot import implementation_hash
from flylatro.learning.config import (
    PlasticExperimentConfig, build_plastic_environment, build_plastic_stack, fly_dynamics_payload,
    payload_sha256, resolve_path,
)
from flylatro.seeds import SeedPlan, derive_seed


def prepare(config, checkpoint, output, *, states=20, repeats=4, fit_repeats=4, duration=250):
    if output.exists():
        raise FileExistsError(output)
    if min(states, fit_repeats) < 1 or repeats < 2 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("positive counts/duration and at least two test repeats required")
    if config.fly.backend != "flywire" or config.fly.mode != "mbon_direct" or config.environment.backend != "balatro_sim":
        raise ValueError("normalization study requires FlyWire MBON-direct and the real simulator")
    if config.plasticity.version != "three-factor-choice-gated-v1":
        raise ValueError("normalization study primary learners use choice-specific credit")
    if config.environment.allow_skip_blinds or config.reinforcement.run_failure_pulse != .01:
        raise ValueError("normalization study keeps skips disabled and failure pulse 0.01")
    calibration = load_corpus(config)
    seeds = SeedPlan().seeds("validation", 8, offset=10_000)
    env = build_plastic_environment(replace(config, environment=replace(config.environment, num_envs=1)))
    test = build_calibration_corpus(env, environment_seeds=seeds, states_per_seed=64,
                                    navigation_seed=25_301_000, store_snapshots=True)
    cases = heldout_cases(calibration, test, states=states)
    fit_seed, test_seed = 25_200_000, 25_300_000
    fit_seeds = [derive_seed("motor-normalization-fit", fit_seed, repeat, index)
                 for repeat in range(fit_repeats) for index in range(len(calibration))]
    test_seeds = [derive_seed("card-information-flow", test_seed, case["state_index"], repeat)
                  for case in cases for repeat in range(repeats)]
    if set(fit_seeds) & set(test_seeds):
        raise ValueError("fit/test neural seeds overlap")
    source = {"config": str(config.source_path.resolve()), "config_sha256": sha256_file(config.source_path),
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
              "motor_file": str(resolve_path(config, config.fly.motor_mapping_path)),
              "motor_file_sha256": sha256_file(resolve_path(config, config.fly.motor_mapping_path)),
              "calibration_file":str(resolve_path(config,config.calibration.corpus_path)),
              "calibration_file_sha256":sha256_file(resolve_path(config,config.calibration.corpus_path))}
    declaration = {"version": VERSION, "implementation_sha256": implementation_hash(), "source": source,
        "duration_ms": float(duration), "source_training_duration_ms": config.fly.duration_ms,
        "calibration_corpus_sha256": calibration.sha256, "calibration_states": len(calibration),
        "test_corpus_sha256": test.sha256, "test_environment_seeds": list(seeds),
        "test_navigation_seed":25_301_000, "fit_activity_seed":fit_seed,"test_activity_seed":test_seed,
        "fit_neural_seeds": fit_seeds, "test_neural_seeds": sorted(set(test_seeds)),
        "fit_repeats":fit_repeats, "test_repeats":repeats,"trained_learners":config.environment.num_envs,
        "fit_policy":"one shared normalization; initial unit weights only; existing median/IQR and scale floor; all fixed pools; no reselection",
        "case_plan":[{"state_index":c["state_index"],"state_hash":c["state_hash"],"run_seed":test.source_run_seed[c["state_index"]],"slot":c["slot"]} for c in cases],
        "criterion":CRITERION,"final_test_seeds_used":False,
        "limitations":"held out from this calibration; diagnostic input interventions, not gameplay; three trained learners; no hand-quality labels used"}
    output.mkdir(parents=True,exist_ok=False)
    test.save(output/"reserved-states.npz")
    write_json(output/"declaration.json",declaration)
    return declaration,calibration,test,cases


def check_sources(declaration):
    for key in ("config","checkpoint","motor_file","calibration_file"):
        if sha256_file(Path(declaration["source"][key]))!=declaration["source"][key+"_sha256"]:
            raise ValueError("normalization study source changed")
    if implementation_hash()!=declaration["implementation_sha256"]:
        raise ValueError("normalization study implementation changed")


def run(config,checkpoint,output,declaration,calibration,test,cases):
    report={"version":VERSION,"status":"running","learning_disabled":True,
            "declaration_sha256":sha256_file(output/"declaration.json"),"learners":[]}
    save=lambda:write_json(output/"report.json",report)
    save()
    started=time.perf_counter()
    try:
        check_sources(declaration)
        stack=build_plastic_stack(config)
        load_plastic_checkpoint(checkpoint,stack.trainer,expected_components=stack.components,
                                require_matching_exposure_budget=False)
        source=stack.agent.motor.mapping
        state=stack.agent.plasticity.state
        before=plastic_hashes(state)
        if not np.all(state_numpy(state.initial_efficacy)==1):
            raise ValueError("calibration requires the checkpoint's initial unit efficacies")
        report.update(source_manifest=load_plastic_manifest(checkpoint),source_components=stack.components,
                      source_motor=source.to_manifest(),pool_membership_sha256=membership_hash(source))
        rows,activity,seeds=[],[],[]
        exact_fit=0
        with integration_window(stack.agent.processor,declaration["duration_ms"]):
            for repeat in range(declaration["fit_repeats"]):
                for index in range(len(calibration)):
                    obs,_=calibration.row(index)
                    seed=derive_seed("motor-normalization-fit",declaration["fit_activity_seed"],repeat,index)
                    def sample():
                        return np.asarray(stack.agent.processor.process(obs,fly_seeds=(seed,),
                            efficacy=state.initial_efficacy[0:1]).output_activity).copy()
                    values=sample()
                    if repeat==0:
                        if not np.array_equal(values,sample()):raise ValueError("calibration exact repeat failed")
                        exact_fit+=1
                    activity.append(values[0]);rows.append(stack.agent.motor.pool_activity(values)[0]);seeds.append(seed)
                print(json.dumps({"stage":"fit","completed_repeat":repeat+1}),flush=True)
            if seeds!=declaration["fit_neural_seeds"]:
                raise ValueError("calibration seed schedule differs from declaration")
            fit_path=output/"fit-neural-activity.npz"
            np.savez_compressed(fit_path,output_hz=np.stack(activity),pool_hz=np.stack(rows),neural_seeds=np.asarray(seeds,dtype=np.int64))
            provenance={"duration_ms":declaration["duration_ms"],"corpus_sha256":calibration.sha256,
                "fit_neural_activity_sha256":sha256_file(fit_path),"initial_efficacy_sha256":before["initial_efficacy"],
                "samples":len(rows),"repeats":declaration["fit_repeats"]}
            candidate=fit_fixed_normalization(source,np.stack(rows),provenance=provenance)
            candidate.save(output/"fresh-motor.json")
            report.update(candidate_motor=candidate.to_manifest(),fit_exact_repeat_checks=exact_fit,
                fit_neural_activity_sha256=sha256_file(fit_path),
                analysis_dynamics=fly_dynamics_payload(replace(config,fly=replace(config.fly,duration_ms=declaration["duration_ms"]))))
            save()
            for index in range(state.learners):
                result=measure_learner(stack.agent,index,cases,repeats=declaration["test_repeats"],
                    seed=declaration["test_activity_seed"],output_dir=output,label="choice",
                    progress=lambda weights,row,slot:print(json.dumps({"stage":"heldout","learner":index,"weights":weights,"state":row,"slot":slot}),flush=True))
                result["normalization_comparison"]=evaluate_cached(output,result,source,candidate,cases)
                report["learners"].append(result)
                save()
        if plastic_hashes(state)!=before:
            raise ValueError("normalization study changed frozen plastic state")
        if stack.agent.motor.mapping.sha256!=source.sha256:
            raise ValueError("normalization study changed source decoder")
        check_sources(declaration)
        assessment=assess(report["learners"],minimum_kc_active=config.safety.minimum_kc_active_fraction,
                          minimum_motor_active=config.safety.minimum_motor_active_fraction)
        assessment.update(comparability="PASS",selected_test_states_disjoint=True,
            selected_test_run_seeds_disjoint=True,calibration_test_neural_seeds_disjoint=True,
            declaration_sha256=report["declaration_sha256"])
        write_json(output/"assessment.json",assessment)
        report.update(status="completed",sources_unchanged=True,plastic_state_unchanged=True,
                      pool_memberships_unchanged=True,wall_seconds=time.perf_counter()-started)
        save()
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        save()
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("artefacts/choice-credit-20261016-pilot-v1/configs/choice.toml"))
    parser.add_argument("--checkpoint",type=Path,default=Path("artefacts/choice-credit-20261016-pilot-v1/choice/plastic-checkpoint-final.pkl"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--states",type=int,default=20)
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--fit-repeats",type=int,default=4)
    parser.add_argument("--duration-ms",type=float,default=250)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--heavy",action="store_true")
    args=parser.parse_args(argv)
    config=PlasticExperimentConfig.load(args.config)
    if not args.prepare_only:config.require_heavy_opt_in(args.heavy)
    declaration,calibration,test,cases=prepare(config,args.checkpoint,args.output_dir,
        states=args.states,repeats=args.repeats,fit_repeats=args.fit_repeats,duration=args.duration_ms)
    if not args.prepare_only:run(config,args.checkpoint,args.output_dir,declaration,calibration,test,cases)
    print(json.dumps({"output_dir":str(args.output_dir),"prepared_only":args.prepare_only}))
    return 0


if __name__=="__main__":raise SystemExit(main())
