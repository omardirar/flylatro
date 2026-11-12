"""Bounded synchronous UTF-8 buffering with explicit checkpoint barriers."""
from __future__ import annotations
import math
from pathlib import Path
import time


class BufferedTextWriter:
    def __init__(self, path: Path, *, max_bytes: int = 262144, flush_seconds: float = 1.0):
        if max_bytes < 1 or not math.isfinite(flush_seconds) or flush_seconds < 0:
            raise ValueError('positive buffer size and finite non-negative flush interval required')
        self.max_bytes = max_bytes
        self.flush_seconds = flush_seconds
        self._stream = path.open('ab', buffering=0)
        self._buffer = bytearray()
        self._last_flush = time.monotonic()
        self.closed = False
        self._error = None

    def write(self, text: str):
        if self.closed:
            raise ValueError('write to closed log')
        if self._error is not None:
            raise self._error
        data = text.encode('utf-8')
        if len(self._buffer) + len(data) > self.max_bytes:
            self.flush()
        if len(data) >= self.max_bytes:
            self._write_all(data)
        else:
            self._buffer.extend(data)
        self.maybe_flush()
        return len(text)

    def _write_all(self, data):
        try:
            view = memoryview(data)
            while view:
                written = self._stream.write(view)
                if written is None or written <= 0:
                    raise OSError('audit writer made no progress')
                view = view[written:]
        except Exception as error:
            self._error = error
            raise

    def maybe_flush(self):
        if time.monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self):
        if self.closed:
            raise ValueError('flush of closed log')
        if self._error is not None:
            raise self._error
        self._write_all(self._buffer)
        self._buffer.clear()
        self._stream.flush()
        self._last_flush = time.monotonic()

    def close(self):
        if not self.closed:
            try:
                if self._error is None:
                    self.flush()
            finally:
                self.closed = True
                self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
