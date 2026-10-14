"""Resident adapter for the licensed Euler kernel; no host substep transfers.

The runtime owns state/sensor epochs: after launch it must commit its state
version only for active worlds with Euler status zero. Forward and sensor
versions stay unchanged. No .numpy(), upload, CPU mass solve, or CPU qpos
integration occurs in launch. Model licensing remains EulerWorkspace's job.
"""
from __future__ import annotations

_STATE = ('active', 'qpos', 'qvel', 'time', 'qM', 'qfrc_smooth',
          'qfrc_constraint', 'qacc', 'qH', 'qHDiagInv')
_HELD = ('ctrl', 'qacc_warmstart', 'sensordata')


def _dtype_name(array):
    return getattr(array.dtype, '__name__', str(array.dtype))


def _span(array):
    ptr = getattr(array, 'ptr', None)
    if ptr is None:
        return None
    size = 1
    for length in array.shape:
        size *= length
    return (int(ptr), int(ptr) + size * (4 if _dtype_name(array) == 'int32' else 8))


class ResidentEulerAdapter:
    """Bind real device arrays to Euler and its independent damped workspace.

    Construct at model-load time, bind once outside Graph capture, then launch
    on the same stream as preceding forward stages. Kernel launch publishes
    qpos/qvel/time and damped qH/qHDiagInv atomically per successful world;
    qM, qacc, ctrl, warmstart and sensordata remain held. An inactive/failed
    world retains its external state. Diagnostic scratch may be cleared.
    """

    def __init__(self, model=None, capacity=1, device='cuda:0', *, workspace=None):
        if workspace is None:
            from .kernel import EulerWorkspace
            workspace = EulerWorkspace(model, capacity, device)
        self.workspace = workspace
        self.capacity = int(workspace.capacity)
        self.device = workspace.device
        self._bound = False

    @property
    def status(self):
        return self.workspace.status

    @property
    def used_acceleration(self):
        return self.workspace.qacc_used

    def bind(self, *, forward_factors=(), **fields):
        """Validate and adopt buffers without copying; mandatory fields exact.

        Held fields are optional because Euler never reads or writes them;
        binding them makes ownership explicit. Shapes include world axis.
        Mass damping buffers must not alias forward mass/factor buffers.
        No rebinding may occur during capture, replay, or an in-flight launch.
        """
        if not set(_STATE).issubset(fields) or set(fields) - set(_STATE + _HELD):
            raise ValueError('Euler resident fields mismatch')
        w = self.workspace
        spans = []
        for name, array in fields.items():
            dst = getattr(w, name)
            if dst is None:
                raise ValueError('empty preserved field must be omitted: ' + name)
            expected = 'int32' if name == 'active' else 'float64'
            if tuple(array.shape) != tuple(dst.shape) or _dtype_name(array) != expected:
                raise TypeError('Euler device schema: ' + name)
            if str(array.device) != str(dst.device):
                raise ValueError('Euler device mismatch: ' + name)
            if hasattr(array, 'is_contiguous') and not array.is_contiguous:
                raise ValueError('Euler requires contiguous arrays: ' + name)
            span = _span(array)
            if span is not None:
                for old_name, old in spans:
                    if max(span[0], old[0]) < min(span[1], old[1]):
                        raise ValueError('Euler buffers overlap: ' + old_name + '/' + name)
                spans.append((name, span))
        # Factors owned by forward must remain separate from Euler qH outputs.
        for name, span in spans:
            if name not in ('qH', 'qHDiagInv'):
                continue
            for factor in forward_factors:
                other = _span(factor)
                if other and max(span[0], other[0]) < min(span[1], other[1]):
                    raise ValueError('Euler output aliases forward factor: ' + name)
        # Reject caller buffers aliasing private mass scratch or commit scratch.
        private = [getattr(w, n) for n in ('qpos_after', 'qvel_after', 'time_after', 'qacc_used')]
        private += [getattr(w.mass, n) for n in ('qm', 'rhs', 'ld', 'inv', 'solution')]
        for name, span in spans:
            for scratch in private:
                other = _span(scratch)
                if other and max(span[0], other[0]) < min(span[1], other[1]):
                    raise ValueError('Euler external buffer aliases private scratch: ' + name)
        for name, array in fields.items():
            setattr(w, name, array)
        self._bound = True
        return self

    @property
    def bound_roots(self):
        """Actual resident arrays; not snapshots or copied state."""
        if not self._bound:
            raise RuntimeError('bind resident state before querying roots')
        return {name: getattr(self.workspace, name) for name in _STATE + _HELD}

    def launch(self, stream=None):
        if not self._bound:
            raise RuntimeError('bind resident state before Euler launch')
        self.workspace.launch(stream=stream)

    def action_boundary_snapshot(self, fields=('qpos', 'qvel', 'time', 'status')):
        """Explicit diagnostic/action-boundary transfer, never called by launch."""
        allowed = set(_STATE + _HELD + ('status', 'qacc_used', 'damped'))
        if set(fields) - allowed:
            raise ValueError('unknown Euler snapshot field')
        return {name: getattr(self.workspace, name).numpy() for name in fields
                if getattr(self.workspace, name) is not None}

    def memory_bytes(self):
        """Current bound plus private-array bytes, deduplicated by pointer.

        This is array storage only: CUDA context, module and allocator overhead
        require separate executor measurement. It is not an observed peak.
        """
        seen = set()
        total = 0
        for owner in (self.workspace, self.workspace.mass):
            for array in vars(owner).values():
                if not hasattr(array, 'shape') or not hasattr(array, 'dtype'):
                    continue
                span = _span(array)
                if span is not None and span not in seen:
                    seen.add(span)
                    total += span[1] - span[0]
        return total

