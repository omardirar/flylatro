"""Full-trainer integration with strict hashes for out-of-tree research code."""
from pathlib import Path

from flylatro.fly.flywire_artifact import sha256_file
from local_normalisation import install_normalisation as _install


def install_normalisation(stack):
    stack = _install(stack)
    # Historical implementation hashes deliberately cover production/harness
    # sources only. Explicitly bind both new research modules for this factory.
    paths = (Path(__file__), Path(__file__).with_name('local_normalisation.py'))
    stack.components['normalisation_research_sources'] = {p.name: sha256_file(p) for p in paths}
    return stack


def build_normalised_stack(config, *, condition=None, corpus=None, **kwargs):
    if condition is None:
        from flylatro.learning.config import build_plastic_stack
        stack = build_plastic_stack(config, **kwargs)
    else:
        from card_learning.experiment import build_condition_stack
        stack = build_condition_stack(config, condition, corpus, **kwargs)
    return install_normalisation(stack)
