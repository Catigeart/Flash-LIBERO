"""MuJoCo/robosuite source order; this module performs no physics itself."""
from dataclasses import dataclass
from typing import Callable


FORWARD_ORDER = (
    'position_mass', 'collision', 'constraint_position', 'transmission',
    'sensor_position', 'velocity', 'constraint_reference', 'bias',
    'sensor_velocity', 'actuation_acceleration', 'constraint_solve',
    'sensor_acceleration',
)


@dataclass(frozen=True)
class ForwardEntry:
    name: str
    launch: Callable
    source: str


class PreparedForwardPipeline:
    """Explicit complete-forward entries, bound once before graph capture.

    ``launch(mask, stream)`` must consume the actual device mask. For fixed
    zero-cardinality source stages the registered entry may do no work, but
    its source qualification remains explicit. This is not an adapter which
    silently replaces missing stages with no-ops.
    """
    def __init__(self, entries, *, validate, begin, publish):
        self.entries = tuple(entries)
        if tuple(e.name for e in self.entries) != FORWARD_ORDER:
            raise ValueError('complete forward entries must match CPU_FLOW order')
        if any(not callable(e.launch) or not e.source for e in self.entries):
            raise TypeError('each forward stage requires a callable and source identity')
        if not all(callable(f) for f in (validate, begin, publish)):
            raise TypeError('forward guards and device publication required')
        self._validate, self._begin, self._publish = validate, begin, publish

    def enqueue(self, mask, stream):
        self._validate()
        self._begin(mask, stream)
        for entry in self.entries:
            entry.launch(mask, stream)
        self._publish(mask, stream)

    def source_manifest(self):
        return [{'stage': e.name, 'source': e.source} for e in self.entries]


def enqueue_substep(forward, controller, integrate, observe, mask, stream,
                    *, policy_step=False, force_refresh=False):
    """One source substep, including controller-owned conditional forward.

    Controller.set_goal/update and run_controller/update must use the same
    actual full-forward callback when their device new_update predicate
    requests it. A skipped controller refresh leaves the cache untouched.
    Integration leaves forward scratch at its pre-integration epoch.
    """
    forward.enqueue(mask, stream)
    controller.enqueue(policy_step=policy_step, force_refresh=force_refresh)
    forward.enqueue(mask, stream)
    integrate()
    observe()


def substeps_per_action(control_timestep, outer_model_timestep):
    # Mirror RobotEnv.step int(control_timestep / model_timestep), without
    # conflating the outer loop timestep with model.opt.timestep.
    if control_timestep <= 0 or outer_model_timestep <= 0:
        raise ValueError('positive outer timestep required')
    return int(control_timestep / outer_model_timestep)
