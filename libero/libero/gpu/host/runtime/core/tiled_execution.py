"""Same-stream tiled phase ownership. No device arithmetic or GPU feature probe.

Prepare and commit are world coordinators (one CTA/world); worker tile functions
own disjoint private spans. A kernel boundary orders writes across CTAs. Public
output, epochs and semantic counters belong exclusively to commit. Failure keeps
public output/epoch intact. Bindings and graph capture remain NativePlan-owned.
"""
from dataclasses import dataclass


def validate_tile_stages(work_tiles, stages):
    if isinstance(work_tiles, bool) or not isinstance(work_tiles, int) or not 1 <= work_tiles <= 2147483647:
        raise ValueError('work_tiles must be a positive int32')
    if work_tiles > 1 and (not stages or any(s[2] != 'tile' for s in stages)):
        raise ValueError('multi-CTA plans require exclusively explicit tile ABI stages')


@dataclass(frozen=True)
class TiledPhaseContract:
    name: str
    work_tiles: int
    tile_extent: int
    output_scope: str = 'private_until_commit'
    error_order: str = 'priority_then_tile'

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('phase name required')
        validate_tile_stages(self.work_tiles, [('worker', [], 'tile')])
        if isinstance(self.tile_extent, bool) or not isinstance(self.tile_extent, int) or self.tile_extent < 1:
            raise ValueError('positive tile_extent required')
        if self.output_scope != 'private_until_commit' or self.error_order != 'priority_then_tile':
            raise ValueError('unsupported publication/error contract')

    def as_dict(self):
        return dict(vars(self), protocol='tiled-private-commit-v1', world_axis='y', tile_axis='x',
                    synchronization='same_stream_kernel_boundary', coordinator_tiles=1)


class TiledExecution:
    """Prepared three-phase launch. No allocation, host count readback or sync.

    The descriptor proves launch structure, not correctness of arbitrary device
    code. Prepare must clear tile error slots and preflight once; workers must
    honor live counts/admission and write only private owned spans; commit must
    merge errors before copying/publishing. Captured plans retain their buffers.
    """
    def __init__(self, prepare, workers, commit, contract):
        if not isinstance(contract, TiledPhaseContract):
            raise TypeError('TiledPhaseContract required')
        plans=(prepare, workers, commit)
        if any(p.nworld != prepare.nworld or p.device != prepare.device for p in plans):
            raise ValueError('all phases require identical worlds/device')
        if prepare.work_tiles != 1 or commit.work_tiles != 1 or workers.work_tiles != contract.work_tiles:
            raise ValueError('prepare/commit must be unique coordinators and workers must match tiles')
        modes=workers.manifest.get('stage_execution_modes', [])
        if not modes or any(mode != 'tile' for mode in modes):
            raise ValueError('worker phase requires explicit tile ABI')
        self.plans=plans
        self.prepare, self.workers, self.commit = plans
        self.contract=contract
        self.manifest=dict(contract.as_dict(), phases=[p.manifest for p in plans],
                           device_ownership_proof=False)
        self._sealed_structure=self._structure()

    def _structure(self):
        return (id(self.contract), tuple(sorted(self.contract.as_dict().items())),
                tuple(id(p) for p in self.plans),
                tuple((id(p), p.nworld, p.work_tiles, p.block_size,
                       p.dynamic_shared_bytes, tuple(p.manifest.get('stage_execution_modes', ())))
                      for p in (self.prepare, self.workers, self.commit)))

    def validate_bindings(self):
        if self._structure() != self._sealed_structure:
            raise ValueError('tiled phase structure changed; rebuild execution and graph')
        for plan in (self.prepare, self.workers, self.commit):
            plan.validate_bindings()

    def launch(self, stream=None):
        # Prevalidate all three before launching the first: host binding failure
        # cannot leave an incomplete device transaction queued.
        self.validate_bindings()
        if stream is None:
            import warp as wp
            stream=wp.get_stream(self.prepare.device)
        for plan in (self.prepare, self.workers, self.commit):
            plan.launch(stream=stream)
