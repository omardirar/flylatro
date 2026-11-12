import json
import signal
from pathlib import Path

import pytest
from flylatro.telemetry.buffered import BufferedTextWriter
from flylatro.learning.cli import main
import flylatro.learning.cli as cli


def test_bounded_unicode_order_and_close(tmp_path):
    path = tmp_path / 'events'
    with BufferedTextWriter(path, max_bytes=16, flush_seconds=999) as writer:
        for text in ['α\n', 'beta\n', 'x' * 40, '\nfinal\n']:
            writer.write(text)
            assert len(writer._buffer) <= 16
    assert path.read_text() == 'α\nbeta\n' + 'x' * 40 + '\nfinal\n'


def test_partial_write_failure_not_retried_on_close(tmp_path):
    writer = BufferedTextWriter(tmp_path / 'events', flush_seconds=999)
    writer._stream.close()
    class Broken:
        calls = 0
        def write(self, data):
            self.calls += 1
            if self.calls == 1:
                return 2
            raise OSError('disk failure')
        def close(self):
            pass
    broken = writer._stream = Broken()
    writer.write('abcdef')
    with pytest.raises(OSError, match='disk failure'):
        writer.flush()
    writer.close()
    assert broken.calls == 2


def test_checkpoint_barrier_and_interrupt_boundary(tmp_path, monkeypatch):
    real_build = cli.build_plastic_stack
    def build(config):
        stack = real_build(config)
        step = stack.trainer.step
        def request_interrupt_mid_decision():
            signal.raise_signal(signal.SIGINT)
            return step()
        stack.trainer.step = request_interrupt_mid_decision
        return stack
    monkeypatch.setattr(cli, 'build_plastic_stack', build)
    real_save = cli.save_plastic_checkpoint
    def save(path, trainer, **kwargs):
        for name in ('training-actions', 'synthetic-reinforcement-events', 'metrics'):
            rows = [json.loads(line) for line in (tmp_path / f'{name}.jsonl').read_text().splitlines()]
            assert len(rows) == trainer.state.environment_decisions == 1
        return real_save(path, trainer, **kwargs)
    monkeypatch.setattr(cli, 'save_plastic_checkpoint', save)
    assert main(['--config', 'configs/plastic-smoke.toml', '--run-dir', str(tmp_path),
        '--no-dashboard', '--no-tensorboard', '--checkpoint-every-decisions', '1',
        '--log-flush-seconds', '999']) == 130
    summary = json.loads((tmp_path / 'run-summary.json').read_text())
    assert summary['status'] == 'interrupted'
    assert Path(summary['checkpoint']).name == 'plastic-checkpoint-interrupted.pkl'
    assert not (tmp_path / 'plastic-checkpoint-final.pkl').exists()
