"""Defer SIGINT until a completed, audited decision boundary."""
from contextlib import contextmanager
import signal
import threading


class TrainingInterrupted(Exception):
    pass


@contextmanager
def decision_boundary_interrupt():
    requested = threading.Event()
    main = threading.current_thread() is threading.main_thread()
    if main:
        previous = signal.signal(signal.SIGINT, lambda *_: requested.set())
    try:
        yield requested
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
