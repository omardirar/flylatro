"""Frozen evaluation for plastic-brain and legacy policy experiments."""

from typing import Any

__all__ = ["EvaluationResult", "evaluate_policy"]


def __getattr__(name: str) -> Any:
    """Keep legacy exports without importing the PPO policy on package import."""

    if name in __all__:
        from flylatro.evaluation.evaluator import EvaluationResult, evaluate_policy

        return {
            "EvaluationResult": EvaluationResult,
            "evaluate_policy": evaluate_policy,
        }[name]
    raise AttributeError(name)
