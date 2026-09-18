"""Diagnostics for neural representations and learned fly synapses."""

from flylatro.analysis.plasticity import plasticity_diagnostics, synaptic_change_report
from flylatro.analysis.representation import representation_diagnostics

__all__ = [
    "plasticity_diagnostics",
    "representation_diagnostics",
    "synaptic_change_report",
]
