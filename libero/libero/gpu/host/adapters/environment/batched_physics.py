"""Explicit same-model native batch entry; no per-world owner dispatch.

Each bundle has one shared static model and W world-major dynamic states.
Different static models must be grouped into separate batches; matching NV alone
is insufficient. This is a physics driver, not a production task-factory grant.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


def stack_reset_states(states):
    """Cold/action-boundary data preparation from complete singleton states."""
    states = tuple(states)
    if not states:
        raise ValueError("nonempty reset states required")
    keys = set(states[0])
    result = {}
    for state in states:
        if set(state) != keys:
            raise ValueError("all worlds require the same complete reset fields")
    for key in keys:
        values = [np.asarray(state[key]) for state in states]
        if any(value.dtype != np.float64 or not value.ndim or value.shape[0] != 1 or
               value.shape != values[0].shape for value in values):
            raise ValueError("float64 singleton world-major reset field required: " + key)
        result[key] = np.ascontiguousarray(np.concatenate(values, axis=0))
    return result


def descriptor_key(built):
    """Exact static identity for dispatch grouping, not only shape compatibility."""
    metadata = built.metadata
    # Metadata contains source/decoder/name/topology/read-set identity; reject
    # unserializable metadata instead of replacing it with process-local IDs.
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def group_model_descriptors(built_models):
    """Stable groups of input indices sharing an exact static descriptor."""
    groups = {}
    for index, built in enumerate(built_models):
        groups.setdefault(descriptor_key(built), []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


@dataclass
class NativeBatchedPhysics:
    """One enqueue sequence with native grid.y world ownership."""
    bundle: object

    def __post_init__(self):
        self.nworld = self.bundle.runtime.euler.capacity
        if self.nworld < 1 or self.bundle.runtime.intent.shape != (self.nworld,):
            raise ValueError("actual native batch owner required")

    def stage_action(self, action):
        if self.bundle.controller is None:
            raise ValueError("controller was not prepared")
        self.bundle.controller.stage_action(action)

    def enqueue_substep(self, *, policy_step=False, force_refresh=False):
        # There is deliberately no loop over worlds here. All functional plans
        # consume the same W-major arena and one mask entry per world.
        self.bundle.runtime.enqueue_substep(policy_step=policy_step,
                                           force_refresh=force_refresh)

    def enqueue_action(self, action):
        self.stage_action(action)
        for substep in range(self.bundle.runtime.substeps):
            self.enqueue_substep(policy_step=(substep == 0))

    def download_state(self):
        import warp as wp
        wp.synchronize_stream(self.bundle.stream)
        e = self.bundle.runtime.euler
        return {name: getattr(e, name).numpy() for name in
                ("qpos", "qvel", "time", "ctrl", "qacc", "qacc_warmstart")}

    def close(self):
        self.bundle.close()


def prepare_native_batch(built, *, states, **kwargs):
    from libero.gpu.host.factory import prepare_cuda_physics
    states = tuple(states)
    state = stack_reset_states(states)
    W = len(states)
    if "nworld" in kwargs or "state" in kwargs:
        raise ValueError("batch states determine nworld and state")
    bundle = prepare_cuda_physics(built, state=state, nworld=W, **kwargs)
    return NativeBatchedPhysics(bundle)