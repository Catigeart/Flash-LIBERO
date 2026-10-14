"""Cold device-memory policy. Free memory is a snapshot, not an allocation grant."""
from dataclasses import asdict, dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class MemoryBudget:
    """Optional process budget and headroom, in bytes.

    The limit applies to allocations still to come at the planning boundary.
    Existing allocations are already excluded by CUDA's free-memory query.
    """
    limit_bytes: int | None = None
    reserve_bytes: int = 512 * 1024**2
    safety_fraction: float = 0.10

    def __post_init__(self):
        for name in ('limit_bytes', 'reserve_bytes'):
            value = getattr(self, name)
            if value is None and name == 'limit_bytes':
                continue
            if type(value) is not int or value < (1 if name == 'limit_bytes' else 0):
                raise ValueError(name + ' must be an integer byte count')
        if (isinstance(self.safety_fraction, bool) or
                not isinstance(self.safety_fraction, (int, float)) or
                not isfinite(self.safety_fraction) or not 0 <= self.safety_fraction < 1):
            raise ValueError('safety_fraction must be finite and in [0, 1)')


def normalize_memory_budget(policy=None):
    if policy is None:
        return MemoryBudget()
    if isinstance(policy, MemoryBudget):
        return policy
    if isinstance(policy, dict):
        return MemoryBudget(**policy)
    raise TypeError('memory_budget must be MemoryBudget, a dictionary, or None')


def resolve_memory_budget(policy=None, *, device='cuda:0', pending_bytes=0):
    """Query the selected device after context/current-owner initialization."""
    import warp as wp
    policy = normalize_memory_budget(policy)
    if type(pending_bytes) is not int or pending_bytes < 0:
        raise ValueError('pending_bytes must be nonnegative')
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError('adaptive memory budgeting requires a CUDA device')
    free, total = int(selected.free_memory), int(selected.total_memory)
    if not 0 <= free <= total or total <= 0:
        raise ValueError('invalid CUDA memory snapshot')
    available = min(free, policy.limit_bytes) if policy.limit_bytes is not None else free
    headroom = max(policy.reserve_bytes, ceil(available * policy.safety_fraction))
    reserved = pending_bytes + headroom
    if available <= reserved:
        raise MemoryError(f'GPU memory budget: available={available}, pending={pending_bytes}, '
                          f'headroom={headroom}, device={selected}')
    return dict(schema='adaptive-device-memory-budget-v1', device=str(selected),
                free_bytes=free, total_bytes=total, available_bytes=available,
                pending_bytes=pending_bytes, headroom_bytes=headroom,
                reserved_bytes=reserved, collision_budget_bytes=available-reserved,
                policy=asdict(policy),
                estimate_scope='pending allocation estimate plus headroom; CUDA OOM remains possible')


def estimate_pending_bytes(metadata, nworld, *, rows, contacts):
    """Conservative planning allowance for S/N/Post/Euler/control/observations.

    This is an estimate, not a complete allocator ledger. D/F owners already
    exist when C is planned and must not be charged a second time. Include
    preservation/event copies, construction scratch and optional image scenes.
    """
    d = metadata['dimensions']
    nv, nq, nb = (int(d[k]) for k in ('nv', 'nq', 'nbody'))
    per_world = 8 * (8*rows*nv + 96*rows + 192*contacts + 32*nv*nv +
                     256*nv + 64*nq + 256*nb)
    cameras = metadata.get('camera_names', ()) if metadata.get('use_camera_obs') else ()
    per_world += 8 * len(cameras) * (128*nb + 64*int(d['ngeom']) + 32*int(d['nsite']))
    # Rendering buffers and host-returned images are separate from physics;
    # reserve room for graphics attachments rather than counting RGB alone.
    for h, w in zip(metadata.get('camera_heights', ())[:len(cameras)],
                    metadata.get('camera_widths', ())[:len(cameras)]):
        per_world += int(h)*int(w)*32
    return int(nworld)*per_world + 128*1024**2
