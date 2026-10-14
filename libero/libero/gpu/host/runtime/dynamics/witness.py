"""Per-stage epoch stamps and consumer witnesses for the P06-D entries.

Each world carries a state epoch (bumped by qpos/qvel/time writes) and a
forward sequence (bumped on every forward call, even at unchanged state).  A successful stage
writes stamp[stage][w] = scratch_epoch[w]; a failed world keeps the previous
stamp so downstream gates reject stale consumers.  Consumer witnesses record
the exact producer epoch a stage read, so an offline auditor can prove the
dataflow matches the original source order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rules import FIELD_GROUPS

STAGE_ORDER = (
    'position', 'velocity_prefix', 'bias', 'control', 'actuation', 'acceleration',
)

# Which stages each stage reads (source-order predecessors whose outputs it
# consumes).  'control' consumes post-forward physics plus live bias.
STAGE_READS = {
    'position': (),
    'velocity_prefix': ('position',),
    'bias': ('position', 'velocity_prefix'),
    'control': ('position', 'velocity_prefix', 'bias'),
    'actuation': ('position', 'velocity_prefix', 'control'),
    'acceleration': ('position', 'velocity_prefix', 'bias', 'actuation'),
}

STAGE_WRITES = {stage: FIELD_GROUPS[stage] for stage in STAGE_ORDER}


@dataclass
class ConsumerWitness:
    """What one stage consumed, per world slot (device arrays, no host copy)."""
    stage: str
    producer_stamps: dict = field(default_factory=dict)  # name -> device int64[N]
    producer_epoch_ref: dict = field(default_factory=dict)  # name -> epoch array

    def as_host_record(self, world=None):
        out = {'stage': self.stage, 'consumers': {}}
        for name, arr in self.producer_stamps.items():
            data = np.array(arr.numpy(), copy=True)
            out['consumers'][name] = int(data[world]) if world is not None else data.tolist()
        return out


class StageWitness:
    """Host-side ledger plus device epoch arrays for one DynamicsStages run."""

    def __init__(self, capacity: int, device='cuda:0'):
        import warp as wp
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive int capacity required')
        self.capacity = capacity
        self.device = device
        self.state_epoch = wp.zeros(capacity, dtype=wp.int64, device=device)
        self.scratch_epoch = wp.zeros(capacity, dtype=wp.int64, device=device)
        self.scratch_state_epoch = wp.zeros(capacity, dtype=wp.int64, device=device)
        self.stamps = {stage: wp.zeros(capacity, dtype=wp.int64, device=device)
                       for stage in STAGE_ORDER}
        self.versions = {stage: wp.zeros(capacity, dtype=wp.int64, device=device)
                         for stage in STAGE_ORDER}
        self.read_stamps = {stage: {p: wp.zeros(capacity, dtype=wp.int64, device=device)
                                    for p in STAGE_READS[stage]} for stage in STAGE_ORDER}
        self.read_versions = {stage: {p: wp.zeros(capacity, dtype=wp.int64, device=device)
                                      for p in STAGE_READS[stage]} for stage in STAGE_ORDER}
        self.read_mask = {stage: wp.zeros(capacity, dtype=wp.int32, device=device)
                          for stage in STAGE_ORDER}
        self.valid = {stage: wp.zeros(capacity, dtype=wp.int32, device=device)
                      for stage in STAGE_ORDER}
        self.first_error_stage = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.first_error_code = wp.zeros(capacity, dtype=wp.int32, device=device)
        self._events = []

    def bind_events(self, events):
        self._events = list(events)

    @property
    def events(self):
        return tuple(self._events)

    def record(self, stage, kind, **payload):
        self._events.append({'stage': stage, 'kind': kind, **payload})

    def host_snapshot(self):
        """Diagnostic only; never called between production kernels."""
        return {
            'state_epoch': np.array(self.state_epoch.numpy(), copy=True),
            'scratch_epoch': np.array(self.scratch_epoch.numpy(), copy=True),
            'scratch_state_epoch': np.array(self.scratch_state_epoch.numpy(), copy=True),
            'versions': {s: np.array(a.numpy(), copy=True) for s,a in self.versions.items()},
            'read_stamps': {s: {p: np.array(a.numpy(), copy=True) for p,a in v.items()} for s,v in self.read_stamps.items()},
            'read_versions': {s: {p: np.array(a.numpy(), copy=True) for p,a in v.items()} for s,v in self.read_versions.items()},
            'read_mask': {s: np.array(a.numpy(), copy=True) for s,a in self.read_mask.items()},
            'stamps': {s: np.array(a.numpy(), copy=True)
                       for s, a in self.stamps.items()},
            'valid': {s: np.array(a.numpy(), copy=True)
                      for s, a in self.valid.items()},
            'first_error_stage': np.array(self.first_error_stage.numpy(), copy=True),
            'first_error_code': np.array(self.first_error_code.numpy(), copy=True),
            'events': list(self._events),
        }
