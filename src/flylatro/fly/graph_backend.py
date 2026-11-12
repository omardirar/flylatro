"""CUDA Graph replay of one neural decision or delay-aligned timestep blocks."""
from __future__ import annotations
import time
import numpy as np
from flylatro.fly.execution import NativePlasticFlyWireBackend, CuSparsePlasticFlyWireBackend, EventPlasticFlyWireBackend
from flylatro.fly.torch_backend import TorchFlyActivity


class GraphPlasticFlyWireBackend(NativePlasticFlyWireBackend):
    backend_version = 'flywire-shiu-cuda-native-graph-plastic-kc-mbon-v2'
    block_cycles = 0  # zero captures a complete decision

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graphs = {}
        self._graph_signature = None
        self.capture_seconds = 0.0
        self.capture_count = 0

    def _discard_graphs(self):
        if self._graphs:
            self._synchronize()
            self._graphs.clear()
        self._graph_signature = None

    def reset(self, batch_size, seeds=None):
        if self._state is not None and self._state[0].shape[0] != batch_size:
            self._discard_graphs()
        super().reset(batch_size, seeds)

    def set_plastic_efficacy(self, efficacy):
        shape = tuple(efficacy.shape)
        if len(shape) == 1:
            shape = (1, shape[0])
        if shape != tuple(self._efficacy.shape):
            self._discard_graphs()
        super().set_plastic_efficacy(efficacy)

    def _run_steps(self, steps):
        p = self.parameters
        g, delay, spikes, voltage, refractory, counts = self._state
        noise = self._PoissonSource(self, self._probability, steps)
        for step in range(steps):
            poisson = noise.draw(step) * p.poisson_scale
            recurrent = self._recurrent(spikes)
            self._lif_step(g, delay, spikes, voltage, refractory, counts,
                poisson, recurrent, step % delay.shape[0], int(round(p.refractory_ms / p.dt_ms)))

    def _capture(self, lengths):
        torch = self.torch
        started = time.perf_counter()
        saved_state = [t.clone() for t in self._state]
        saved_rng = [g.get_state() for g in self._generators]
        def restore():
            for target, saved in zip(self._state, saved_state, strict=True):
                target.copy_(saved)
            for generator, saved in zip(self._generators, saved_rng, strict=True):
                generator.set_state(saved)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            self._run_steps(min(lengths))
        torch.cuda.current_stream(self.device).wait_stream(stream)
        restore()
        self._synchronize()
        # Separate pools avoid cross-graph lifetime assumptions for repeated blocks.
        for length in sorted(set(lengths)):
            graph = torch.cuda.CUDAGraph()
            for generator in self._generators:
                graph.register_generator_state(generator)
            with torch.cuda.graph(graph, stream=stream), torch.no_grad():
                self._run_steps(length)
            self._graphs[length] = graph
            restore()
        self._synchronize()
        self.capture_seconds += time.perf_counter() - started
        self.capture_count += 1

    def simulate_tensor(self, stimulus, duration_ms):
        if self.record_events or self.profile_components:
            raise ValueError('graph execution requires record_events=False and profile_components=False; select cuda_native for detailed recording/profiling')
        if self.poisson_method != 'per-step-per-row-v1':
            raise ValueError('graph execution currently requires per-step-per-row-v1 RNG')
        if self._state is None:
            raise RuntimeError('reset must be called before simulate')
        if duration_ms <= 0:
            raise ValueError('duration_ms must be positive')
        torch = self.torch
        steps = int(round(duration_ms / self.parameters.dt_ms))
        if steps < 1 or tuple(stimulus.rates_hz.shape) != tuple(self._state[0].shape):
            raise ValueError('invalid simulation duration or stimulus shape')
        block = self.block_cycles * self._state[1].shape[0] if self.block_cycles else steps
        lengths = [block] * (steps // block)
        if steps % block:
            lengths.append(steps % block)
        signature = (steps, tuple(self._state[0].shape), self._efficacy.data_ptr(),
                     tuple(t.data_ptr() for t in self._state))
        if signature != self._graph_signature:
            self._discard_graphs()
            self._probability = torch.empty_like(self._state[0])
            self._graph_signature = signature
        self._probability.copy_(torch.tensor(np.asarray(stimulus.rates_hz),
            dtype=torch.float32))
        self._probability.mul_(self.parameters.dt_ms / 1000.0).clamp_(0, 1)
        if not self._graphs:
            self._capture(lengths)
        for length in lengths:
            self._graphs[length].replay()
        self._synchronize()
        return TorchFlyActivity(
            spike_counts=self._state[5].index_select(1, self._readout_device),
            final_voltage=self._state[3].index_select(1, self._readout_device),
            duration_ms=duration_ms, neuron_ids=self._readout_roots_device)


class BlockGraphPlasticFlyWireBackend(GraphPlasticFlyWireBackend):
    backend_version = 'flywire-shiu-cuda-native-graph-blocks-plastic-kc-mbon-v2'
    block_cycles = 4


class CuSparseBlockGraphPlasticFlyWireBackend(GraphPlasticFlyWireBackend, CuSparsePlasticFlyWireBackend):
    backend_version = 'flywire-shiu-cuda-native-cusparse-graph-blocks-plastic-kc-mbon-v1'
    block_cycles = 4


class EventBlockGraphPlasticFlyWireBackend(GraphPlasticFlyWireBackend, EventPlasticFlyWireBackend):
    backend_version = 'flywire-shiu-cuda-native-event-graph-blocks-plastic-kc-mbon-v1'
    block_cycles = 4
