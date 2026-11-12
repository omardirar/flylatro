"""Explicit execution selection, separate from canonical connectome identity."""
from __future__ import annotations

from flylatro.fly.plastic_backend import PlasticTorchFlyWireBackend

EXECUTION_BACKENDS = ('torch_reference', 'torch_csr', 'cuda_native', 'cuda_native_graph',
                      'cuda_native_graph_blocks', 'cuda_native_cusparse',
                      'cuda_native_cusparse_graph_blocks', 'cuda_native_event',
                      'cuda_native_event_graph_blocks')


class CSRPlasticFlyWireBackend(PlasticTorchFlyWireBackend):
    """FP32 CSR recurrence with checked int32 indices and independent plasticity.

    Conversion occurs after the reference constructor removes plastic values
    from fixed recurrence. Source connectivity identity is never recomputed
    from the execution layout. Reduction order is conservatively versioned.
    """
    backend_version = 'flywire-shiu-torch-csr32-plastic-kc-mbon-v1'

    def _initialize_connectivity(self, pre, post, weights, procedure):
        from flylatro.fly.prepared_cache import layout_identity, read_layout, validate_csr_layout
        self.topology_procedure = procedure
        self._fixed_layout_identity = layout_identity(self)
        cached = read_layout(self._fixed_layout_identity)
        self.layout_preparation = {'incoming_hit': cached is not None}
        if cached is None:
            return super()._initialize_connectivity(pre, post, weights, procedure)
        metadata, arrays = cached
        if metadata['neurons'] != self.neuron_count:
            raise ValueError('cached graph neuron count differs from artifact')
        validate_csr_layout(arrays, self.neuron_count, plastic_edges=self.plastic_topology.edge_count)
        torch = self.torch
        self.weights = torch.sparse_csr_tensor(
            torch.tensor(arrays['crow'], dtype=torch.int32, device=self.device),
            torch.tensor(arrays['col'], dtype=torch.int32, device=self.device),
            torch.tensor(arrays['values'], dtype=torch.float32, device=self.device),
            size=(self.neuron_count, self.neuron_count), device=self.device, check_invariants=True)
        self._plastic_positions = torch.tensor(arrays['positions'], dtype=torch.int64, device=self.device)
        self.connectivity_hash = metadata['connectivity_hash']
        self._prepared_fixed_plastic_separation = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch = self.torch
        if self.weights.layout == torch.sparse_csr:
            return
        coo = self.weights.coalesce()
        check_int32_capacity(self.neuron_count, coo._nnz())
        indices = coo.indices()
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.neuron_count):
            raise ValueError('sparse index outside the configured neuron range')
        csr = coo.to_sparse_csr()
        self.weights = torch.sparse_csr_tensor(
            csr.crow_indices().to(torch.int32), csr.col_indices().to(torch.int32),
            # A sparse values view retains its parent COO and 64-bit indices.
            # Own the payload so discarded layouts can actually be freed.
            csr.values().clone(), size=csr.shape, device=self.device, check_invariants=True,
        )
        from flylatro.fly.prepared_cache import write_layout
        if self._fixed_layout_identity is not None:
            write_layout(self._fixed_layout_identity, {
                'crow': self.weights.crow_indices().cpu().numpy(),
                'col': self.weights.col_indices().cpu().numpy(),
                'values': self.weights.values().cpu().numpy(),
                'positions': self._plastic_positions.cpu().numpy(),
            }, connectivity_hash=self.connectivity_hash, neurons=self.neuron_count)


class NativePlasticFlyWireBackend(CSRPlasticFlyWireBackend):
    """Experimental fused FP32 LIF; fixed recurrence uses PyTorch cuSPARSE."""
    backend_version = 'flywire-shiu-cuda-native-plastic-kc-mbon-v2'

    def __init__(self, *args, **kwargs):
        if not str(kwargs.get('device', 'cpu')).startswith('cuda'):
            raise ValueError('cuda_native requires a CUDA device')
        from flylatro.fly.native_build import load_native
        self.native, self.native_identity = load_native()
        super().__init__(*args, **kwargs)
        import numpy as np
        post = self.plastic_topology.post_indices
        order = np.argsort(post, kind='stable')
        posts, counts = np.unique(post, return_counts=True)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        self._plastic_order = self.torch.as_tensor(order, dtype=self.torch.int64, device=self.device)
        self._plastic_offsets = self.torch.as_tensor(offsets, dtype=self.torch.int64, device=self.device)
        self._plastic_posts_unique = self.torch.as_tensor(posts, dtype=self.torch.int64, device=self.device)

    def _recurrent(self, spikes):
        with self._timed('recurrent_fixed'):
            fixed = self.torch.sparse.mm(self.weights, spikes.T).T.contiguous()
        with self._timed('recurrent_plastic'):
            self.native.plastic_add(fixed, spikes, self._efficacy, self._plastic_anatomical,
                self._plastic_pre, self._plastic_order, self._plastic_offsets, self._plastic_posts_unique)
        return fixed

    def reset(self, batch_size, seeds=None):
        seeds = tuple(range(batch_size)) if seeds is None else seeds
        if batch_size < 1 or len(seeds) != batch_size:
            raise ValueError('one fly seed required per positive batch row')
        if self._state is None or self._state[0].shape[0] != batch_size:
            return super().reset(batch_size, seeds)
        conductance, delay, spikes, voltage, refractory, counts = self._state
        conductance.zero_()
        delay.zero_()
        spikes.zero_()
        voltage.fill_(self.parameters.resting_mv)
        refractory.fill_(int(round(self.parameters.refractory_ms / self.parameters.dt_ms)))
        counts.zero_()
        for generator, seed in zip(self._generators, seeds, strict=True):
            generator.manual_seed(int(seed))

    def set_plastic_efficacy(self, efficacy):
        torch = self.torch
        tensor = torch.as_tensor(efficacy, dtype=torch.float32, device=self.device).detach()
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        if tensor.ndim != 2 or tensor.shape[1] != self.plastic_topology.edge_count:
            raise ValueError('efficacy batch does not match plastic topology')
        if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError('efficacy must be finite and non-negative')
        if self._efficacy.shape != tensor.shape:
            self._efficacy = torch.empty_like(tensor)
        self._efficacy.copy_(tensor)

    def _lif_step(self, conductance, delay, spikes, voltage, refractory, counts,
                  poisson, recurrent, delay_slot, refractory_steps):
        p = self.parameters
        target = (delay_slot + delay.shape[0] - 1) % delay.shape[0]
        self.native.lif_step(conductance, spikes, voltage, refractory, counts,
            poisson.contiguous(), recurrent.contiguous(), delay[delay_slot], delay[target],
            1.0 - p.dt_ms / p.synaptic_tau_ms, p.dt_ms / p.membrane_tau_ms,
            p.resting_mv, p.reset_mv, p.threshold_mv, p.synapse_scale_mv, refractory_steps)
        return conductance, spikes, voltage, refractory


class CuSparsePlasticFlyWireBackend(NativePlasticFlyWireBackend):
    """Explicit reusable cuSPARSE plans, row-major batched operands, FP32 LIF.

    Fixed recurrence and plastic reduction remain separate. Descriptor/input/
    workspace addresses persist across decisions and graph replay. This has its
    own dynamics version because sparse algorithms can change reduction order.
    """
    backend_version = 'flywire-shiu-cuda-native-cusparse-plastic-kc-mbon-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from flylatro.fly.sparse_build import load_sparse
        self._sparse_module, sparse_identity = load_sparse()
        self.native_identity = {**self.native_identity, 'sparse_plan': sparse_identity}
        self._sparse_plan = None
        self._fixed_input = None

    def reset(self, batch_size, seeds=None):
        super().reset(batch_size, seeds)
        if self._fixed_input is None or self._fixed_input.shape[1] != batch_size:
            self._sparse_plan = None
            self._fixed_input = self.torch.empty(
                (self.neuron_count, batch_size), dtype=self.torch.float32, device=self.device)
            self._sparse_plan = self._sparse_module.SparsePlan(
                self.weights.crow_indices(), self.weights.col_indices(), self.weights.values(),
                self._fixed_input, batch_size == 1, 3 if batch_size == 1 else 4, True, True)

    def _recurrent(self, spikes):
        with self._timed('recurrent_fixed'):
            self._fixed_input.T.copy_(spikes)
            fixed = self._sparse_plan.run().T.contiguous()
        with self._timed('recurrent_plastic'):
            self.native.plastic_add(fixed, spikes, self._efficacy, self._plastic_anatomical,
                self._plastic_pre, self._plastic_order, self._plastic_offsets, self._plastic_posts_unique)
        return fixed


def check_integral_event_weights(post, values, neurons):
    """Prove exact FP32 accumulation for every subset/order of binary spikes."""
    import numpy as np
    if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
        raise ValueError('event execution requires exact integral fixed weights; no rounding permitted')
    bound = float(np.bincount(post, weights=np.abs(values), minlength=neurons).max())
    if bound >= 2**24:
        raise ValueError('event fixed weights exceed exact FP32 integral accumulation bound')
    return bound


def check_integral_event_csr(crow, values):
    """Validate the same integral bound without constructing COO row indices."""
    import numpy as np
    if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
        raise ValueError('event execution requires exact integral fixed weights; no rounding permitted')
    nonempty = np.flatnonzero(np.diff(crow) > 0)
    bound = float(np.add.reduceat(np.abs(values).astype(np.float64), crow[nonempty]).max()) if len(nonempty) else 0.0
    if bound >= 2**24:
        raise ValueError('event fixed weights exceed exact FP32 integral accumulation bound')
    return bound


class EventPlasticFlyWireBackend(NativePlasticFlyWireBackend):
    """Experimental active-outgoing propagation for validated integral weights.

    Compaction order may vary. Every partial anatomical sum is an exactly
    representable integer, so its FP32 atomic result cannot vary with ordering.
    Plastic efficacy contributions retain the separate native fixed-tree rule.
    """
    backend_version = 'flywire-shiu-cuda-native-event-plastic-kc-mbon-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from flylatro.fly.event_build import load_event
        self._event_module, event_identity = load_event()
        self.native_identity = {**self.native_identity, 'event_propagation': event_identity}
        self.maximum_absolute_incoming_sum = check_integral_event_csr(
            self.weights.crow_indices().cpu().numpy(), self.weights.values().cpu().numpy())
        from flylatro.fly.prepared_cache import layout_identity, read_layout, write_layout, validate_csr_layout
        identity = layout_identity(self, direction='outgoing')
        cached = read_layout(identity)
        self.layout_preparation['outgoing_hit'] = cached is not None
        if cached is None:
            outgoing = self.weights.to_sparse_coo().transpose(0, 1).coalesce().to_sparse_csr()
            self._outgoing_crow = outgoing.crow_indices().to(self.torch.int32)
            self._outgoing_posts = outgoing.col_indices().to(self.torch.int32)
            self._outgoing_values = outgoing.values().clone()
            if identity is not None:
                write_layout(identity, {'crow': self._outgoing_crow.cpu().numpy(),
                    'col': self._outgoing_posts.cpu().numpy(), 'values': self._outgoing_values.cpu().numpy()},
                    connectivity_hash=self.connectivity_hash, neurons=self.neuron_count)
        else:
            metadata, arrays = cached
            if metadata['neurons'] != self.neuron_count or metadata['connectivity_hash'] != self.connectivity_hash:
                raise ValueError('cached outgoing graph differs from incoming source')
            validate_csr_layout(arrays, self.neuron_count)
            self._outgoing_crow = self.torch.tensor(arrays['crow'], dtype=self.torch.int32, device=self.device)
            self._outgoing_posts = self.torch.tensor(arrays['col'], dtype=self.torch.int32, device=self.device)
            self._outgoing_values = self.torch.tensor(arrays['values'], dtype=self.torch.float32, device=self.device)
        self._event_output = None

    def reset(self, batch_size, seeds=None):
        super().reset(batch_size, seeds)
        if self._event_output is None or self._event_output.shape[0] != batch_size:
            self._event_output = self.torch.empty_like(self._state[0])
            self._event_active = self.torch.empty_like(self._state[0], dtype=self.torch.int32)
            self._event_counts = self.torch.empty((batch_size,), dtype=self.torch.int32, device=self.device)

    def _recurrent(self, spikes):
        with self._timed('recurrent_fixed'):
            self._event_module.event_step(self._outgoing_crow, self._outgoing_posts,
                self._outgoing_values, spikes, self._event_active, self._event_counts, self._event_output)
        with self._timed('recurrent_plastic'):
            self.native.plastic_add(self._event_output, spikes, self._efficacy, self._plastic_anatomical,
                self._plastic_pre, self._plastic_order, self._plastic_offsets, self._plastic_posts_unique)
        return self._event_output

    def propagate_once(self, activity):
        import numpy as np
        values = np.asarray(activity)
        if not np.all((values == 0) | (values == 1)):
            raise ValueError('event propagation requires binary spike activity')
        batch = 1 if values.ndim == 1 else values.shape[0]
        if self._event_output is None or self._event_output.shape[0] != batch:
            self.reset(batch)
        return super().propagate_once(activity)


def check_int32_capacity(neurons: int, entries: int) -> None:
    if not 0 < neurons <= 2**31 - 1 or not 0 <= entries <= 2**31 - 1:
        raise ValueError('CSR graph dimensions and row offsets must fit signed int32')


def backend_class(execution: str):
    if execution == 'torch_reference':
        return PlasticTorchFlyWireBackend
    if execution == 'torch_csr':
        return CSRPlasticFlyWireBackend
    if execution == 'cuda_native':
        return NativePlasticFlyWireBackend
    if execution == 'cuda_native_cusparse':
        return CuSparsePlasticFlyWireBackend
    if execution == 'cuda_native_cusparse_graph_blocks':
        from flylatro.fly.graph_backend import CuSparseBlockGraphPlasticFlyWireBackend
        return CuSparseBlockGraphPlasticFlyWireBackend
    if execution == 'cuda_native_event':
        return EventPlasticFlyWireBackend
    if execution == 'cuda_native_event_graph_blocks':
        from flylatro.fly.graph_backend import EventBlockGraphPlasticFlyWireBackend
        return EventBlockGraphPlasticFlyWireBackend
    if execution in ('cuda_native_graph', 'cuda_native_graph_blocks'):
        from flylatro.fly.graph_backend import GraphPlasticFlyWireBackend, BlockGraphPlasticFlyWireBackend
        return GraphPlasticFlyWireBackend if execution == 'cuda_native_graph' else BlockGraphPlasticFlyWireBackend
    raise ValueError(f'unknown fly execution backend: {execution}')


def build_backend(execution: str, *args, **kwargs):
    return backend_class(execution)(*args, **kwargs)


def implementation_identity(execution: str) -> dict:
    cls = backend_class(execution)
    return {'execution': execution, 'version': cls.backend_version,
            'sparse_layout': 'outgoing-csr' if 'event' in execution else 'csr' if execution != 'torch_reference' else 'coo',
            'index_bits': 32 if execution != 'torch_reference' else 64,
            'precision': 'float32',
            'algorithm': ('outgoing-active-compaction-FP32-integral-atomic-v1' if 'event' in execution else
                'cuSPARSE-SpMV-CSR-ALG2-batch1-SpMM-CSR-ALG1-row-major-preprocess' if 'cusparse' in execution else 'torch.sparse.mm'),
            'graph_mode': ('delay-aligned-blocks' if execution.endswith('_blocks') else 'whole-decision' if execution.endswith('_graph') else 'none'), 'native_build': None,
            'rng_method': 'per-step-per-row-v1',
            'plastic_reduction': 'fixed-tree128-v1' if execution.startswith('cuda_native') else 'torch-scatter-add'}


def execution_identity_hash(execution):
    """Preflight identity without loading the graph or compiling an extension."""
    if execution == 'torch_reference':
        return None
    import hashlib
    import json
    identity = implementation_identity(execution)
    if execution.startswith('cuda_native'):
        from flylatro.fly.native_build import describe_native
        _, identity['native_build'] = describe_native()
        if 'cusparse' in execution:
            from flylatro.fly.sparse_build import describe_sparse
            _, sparse = describe_sparse()
            identity['native_build']['sparse_plan'] = sparse
        if 'event' in execution:
            from flylatro.fly.event_build import describe_event
            _, event = describe_event()
            identity['native_build']['event_propagation'] = event
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
