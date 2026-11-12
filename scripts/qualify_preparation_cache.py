"""Cold/warm artifact/layout cache timing and exact full-state qualification."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time

from flylatro.learning.config import PlasticExperimentConfig, build_plastic_stack
from qualify_execution import run_steps

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--steps', type=int, default=16)
p.add_argument('--heavy', action='store_true')
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
config = PlasticExperimentConfig.load(a.config)
config.require_heavy_opt_in(a.heavy)
import torch
rows = []
reference = None
previous = {name: os.environ.get(name) for name in ('FLYLATRO_ARTIFACT_CACHE_DIR', 'FLYLATRO_LAYOUT_CACHE_DIR')}
try:
    for mode in ('disabled', 'cold', 'warm'):
        for name, folder in [('FLYLATRO_ARTIFACT_CACHE_DIR', 'artifacts'), ('FLYLATRO_LAYOUT_CACHE_DIR', 'layouts')]:
            if mode == 'disabled':
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(a.output_dir / folder)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        stack = build_plastic_stack(config)
        torch.cuda.synchronize()
        startup = time.perf_counter() - start
        peak_startup = torch.cuda.max_memory_allocated()
        backend = stack.agent.processor.backend
        start = time.perf_counter()
        trajectory = run_steps(stack, a.steps)
        seconds = time.perf_counter() - start
        if reference is None:
            reference = trajectory
        exact = trajectory == reference
        cache = dict(backend.layout_preparation)
        if mode == 'warm' and not all(cache.values()):
            raise AssertionError('warm run did not reuse both fixed graph layouts')
        rows.append({'mode': mode, 'startup_seconds': startup,
            'startup_peak_allocated_bytes': peak_startup,
            'allocated_bytes': torch.cuda.memory_allocated(),
            'exact_uncached_trajectory': exact, 'layout_preparation': cache,
            'diagnostic_seconds': seconds, 'components': stack.components})
        print(mode, round(startup, 3), exact, cache, flush=True)
        del stack, backend
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
finally:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
(a.output_dir / 'qualification.json').write_text(json.dumps({'steps': a.steps, 'results': rows,
    'scope': 'one persistent process; first construction also loads CUDA/extension; compare preparation stages, not cold-process startup speed'}, indent=2) + '\n')
raise SystemExit(0 if all(r['exact_uncached_trajectory'] for r in rows) else 2)
