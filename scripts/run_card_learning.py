"""Card-decision audits, subset-exploration pilots, and a one-pair curriculum."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='replay frozen native evaluation decisions on CPU')
    audit.add_argument('--transitions', type=Path, action='append', required=True)
    audit.add_argument('--output-dir', type=Path, required=True)
    audit.add_argument('--native-check-states', type=int, default=20)
    corpus = commands.add_parser('build-pairs', help='build rank-balanced training/validation opening snapshots on CPU')
    corpus.add_argument('--output', type=Path, required=True)
    corpus.add_argument('--train-per-rank', type=int, default=8)
    corpus.add_argument('--validation-per-rank', type=int, default=4)
    corpus.add_argument('--training-offset', type=int, default=6000)
    corpus.add_argument('--validation-offset', type=int, default=60000)
    pilot = commands.add_parser('pilot', help='declare four matched conditions; GPU execution needs --heavy')
    pilot.add_argument('--config', type=Path, required=True)
    pilot.add_argument('--output-dir', type=Path, required=True)
    pilot.add_argument('--steps-per-learner', type=int, default=20000)
    pilot.add_argument('--replicates', type=int, default=3)
    pilot.add_argument('--episodes', type=int, default=256)
    pilot.add_argument('--batch-size', type=int, default=4)
    pilot.add_argument('--probability', type=float, default=.05)
    pilot.add_argument('--pair-corpus', type=Path)
    pilot.add_argument('--prepare-only', action='store_true')
    pilot.add_argument('--heavy', action='store_true')
    run = commands.add_parser('run', help='execute an already prepared pilot serially')
    run.add_argument('--pilot-dir', type=Path, required=True)
    run.add_argument('--heavy', action='store_true')
    for name in ('train', 'evaluate'):
        worker = commands.add_parser(name, help='condition-bound existing trainer/evaluator; remaining flags are forwarded', add_help=False)
        worker.add_argument('--condition', type=Path, required=True)
    pair_eval = commands.add_parser('pair-evaluate', help='compare a learner with its own initial weights on held-out pairs')
    pair_eval.add_argument('--condition', type=Path, required=True)
    pair_eval.add_argument('--checkpoint', type=Path, required=True)
    pair_eval.add_argument('--output', type=Path, required=True)
    pair_eval.add_argument('--learner-index', type=int, default=0)
    pair_eval.add_argument('--repeats', type=int, default=4)
    pair_eval.add_argument('--heavy', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    if args.command in ('train', 'evaluate'):
        from card_learning.experiment import worker
        return worker(args.condition, args.command, remaining)
    if remaining:
        parser.error(f'unrecognized arguments: {remaining}')
    if args.command == 'audit':
        from card_learning.audit import audit_files
        audit_files(args.transitions, args.output_dir, native_check_states=args.native_check_states)
    elif args.command == 'build-pairs':
        from card_learning.pairs import build_corpus
        build_corpus(args.output, train_per_rank=args.train_per_rank, validation_per_rank=args.validation_per_rank,
            training_offset=args.training_offset, validation_offset=args.validation_offset)
    elif args.command == 'run':
        from card_learning.experiment import execute
        if not args.heavy:
            parser.error('pilot execution requires --heavy')
        execute(args.pilot_dir.resolve())
    elif args.command == 'pilot':
        from card_learning.experiment import prepare, execute
        if not args.prepare_only and not args.heavy:
            parser.error('pilot execution requires --heavy; use --prepare-only while another GPU experiment is active')
        declaration = prepare(args.config, args.output_dir, steps=args.steps_per_learner, replicates=args.replicates,
            episodes=args.episodes, batch=args.batch_size, probability=args.probability, pair_corpus=args.pair_corpus)
        print(f"Prepared {len(declaration['arms'])} conditions in {args.output_dir}")
        if not args.prepare_only:
            execute(args.output_dir.resolve())
    else:
        from card_learning.pair_evaluation import evaluate
        evaluate(args.condition, args.checkpoint, args.output, learner=args.learner_index, repeats=args.repeats, heavy=args.heavy)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
