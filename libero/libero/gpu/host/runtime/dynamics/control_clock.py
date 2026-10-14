"""Control-refresh clock for the two P06-D control entries.

Source: robosuite 1.4.0 osc.py:211-357 and base_controller.py:125-161.  A
controller update runs a full sim.forward only on new_update (or force); a
skipped refresh must keep the previous pose/velocity/Jacobian/mass caches
exactly.  Goal writes happen at policy_step (set_goal); the gripper command
accumulates every control call; torque compensation reads the LIVE qfrc_bias
each run_controller call and must not be substituted with the cached bias.
The clock records those distinctions so a consumer witness can prove which
cache generation and which bias generation a control result used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REFRESH = 1
SKIP = 0


@dataclass
class ControlClock:
    """Host ledger of control-time events; device state lives in OSC workspaces.

    Fields
    ------
    refresh_count:      full cache refreshes (new_update or force) observed
    skip_count:         run_controller calls that reused the last cache
    goal_writes:        policy_step set_goal events
    gripper_accum:      control calls that accumulated gripper command
    ctrl_version:       monotonically increasing actuation-ctrl write counter
    cache_version:      version of the effective pose/vel/J/M cache
    bias_live_reads:    live qfrc_bias reads (torque compensation)
    bias_cache_version: version of the cached bias that OSC consumed
    events:             ordered raw event records
    """
    refresh_count: int = 0
    skip_count: int = 0
    goal_writes: int = 0
    gripper_accum: int = 0
    ctrl_version: int = 0
    cache_version: int = 0
    bias_live_reads: int = 0
    bias_cache_version: int = 0
    events: list = field(default_factory=list)

    def on_update(self, *, force_refresh: bool, world=None):
        kind = 'refresh' if force_refresh else 'skip'
        if force_refresh:
            self.refresh_count += 1
            self.cache_version += 1
        else:
            self.skip_count += 1
        self.events.append({'event': 'controller_update', 'kind': kind,
                            'cache_version': self.cache_version,
                            'world': world})
        return REFRESH if force_refresh else SKIP

    def on_set_goal(self, *, world=None):
        self.goal_writes += 1
        self.events.append({'event': 'set_goal', 'goal_write_index': self.goal_writes,
                            'world': world})

    def on_gripper(self, *, world=None):
        self.gripper_accum += 1
        self.events.append({'event': 'gripper_accum', 'count': self.gripper_accum,
                            'world': world})

    def on_ctrl_write(self, *, world=None):
        self.ctrl_version += 1
        self.events.append({'event': 'ctrl_write', 'ctrl_version': self.ctrl_version,
                            'world': world})
        return self.ctrl_version

    def on_bias(self, *, live: bool, cache_version: int | None = None, world=None):
        """Record the bias generation a control computation consumed.

        live=True is the torque-compensation path (reads current qfrc_bias);
        live=False records consumption of the cached OSC-inputs bias with its
        generation number.  Mixing them is a witness failure, not a rounding
        difference.
        """
        if live:
            self.bias_live_reads += 1
            self.events.append({'event': 'bias_read', 'live': True,
                                'count': self.bias_live_reads, 'world': world})
        else:
            if cache_version is None:
                raise ValueError('cached bias read requires cache_version')
            self.bias_cache_version = int(cache_version)
            self.events.append({'event': 'bias_read', 'live': False,
                                'cache_version': int(cache_version), 'world': world})

    def assert_ctrl_consumed(self, ctrl_version: int):
        if int(ctrl_version) != self.ctrl_version:
            raise ValueError(f'ctrl version mismatch: consumed {ctrl_version} '
                             f'but current is {self.ctrl_version}')

    def assert_cache_consumed(self, cache_version: int):
        if int(cache_version) != self.cache_version:
            raise ValueError(f'cache version mismatch: consumed {cache_version} '
                             f'but current is {self.cache_version}')

    def as_dict(self):
        return {
            'refresh_count': self.refresh_count,
            'skip_count': self.skip_count,
            'goal_writes': self.goal_writes,
            'gripper_accum': self.gripper_accum,
            'ctrl_version': self.ctrl_version,
            'cache_version': self.cache_version,
            'bias_live_reads': self.bias_live_reads,
            'bias_cache_version': self.bias_cache_version,
            'events': list(self.events),
        }
