"""Cold execution candidates and hardware/version keyed tuning records.

No GPU queries, autotuning, strategy activation or performance certification are
performed here. Capability is separate from a particular linked kernel's launch
qualification. The mathematical kernels do not depend on these policy tables.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DeviceCapabilities:
    name: str
    compute_capability: tuple
    sm_count: int
    max_threads_per_block: int
    shared_memory_per_block_bytes: int
    shared_memory_per_sm_bytes: int
    registers_per_sm: int
    warp_size: int = 32
    cooperative_launch: bool = False
    cluster_launch: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'compute_capability', tuple(self.compute_capability))
        if len(self.compute_capability) != 2 or any(type(x) is not int or x < 0 for x in self.compute_capability):
            raise ValueError('compute capability must be a major/minor pair')
        if not self.name or any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in (
            'sm_count', 'max_threads_per_block', 'shared_memory_per_block_bytes',
            'shared_memory_per_sm_bytes', 'registers_per_sm', 'warp_size')):
            raise ValueError('explicit positive device limits required')

    @property
    def architecture(self):
        # Known CC families only: do not label Ada SM89 as Ampere, or
        # silently certify an unknown future minor under a known major.
        cc = self.compute_capability
        if cc in ((8, 0), (8, 6), (8, 7)):
            return 'ampere'
        if cc == (8, 9):
            return 'ada'
        if cc == (9, 0):
            return 'hopper'
        if cc in ((10, 0), (10, 3), (11, 0), (12, 0), (12, 1)):
            return 'blackwell'
        return 'other'

    @property
    def supports_cluster(self):
        return self.compute_capability[0] >= 9 and self.cluster_launch


@dataclass(frozen=True)
class RoutingThresholds:
    # None means unresolved: never invent a measured crossover from snapshots.
    multi_cta_rows: int | None = None
    multi_cta_contacts: int | None = None
    multi_cta_queries: int | None = None
    tiled_copy_bytes: int | None = None

    def __post_init__(self):
        if any(value is not None and (type(value) is not int or value < 1)
               for value in asdict(self).values()):
            raise ValueError('thresholds must be positive integers or unresolved None')


@dataclass(frozen=True)
class ExecutionConfig:
    strategy: str = 'single_cta'
    block_threads: int = 64
    contact_tile: int = 32
    row_tile: int = 32
    # Newton worker/cluster candidate count; capacity-tiled S/C launchers
    # derive grid.x from their own entity capacity and tile, not this field.
    workers_per_world: int = 1
    cluster_blocks: int = 1
    thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)
    shared_budget_bytes: int = 48 * 1024
    tuning_status: str = 'candidate'

    def __post_init__(self):
        if self.strategy not in ('single_cta', 'multi_cta', 'cluster'):
            raise ValueError('unknown execution strategy')
        if type(self.block_threads) is not int or self.block_threads not in (32, 64, 128, 256, 512, 1024):
            raise ValueError('supported whole-warp block size required')
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in (
            'contact_tile', 'row_tile', 'workers_per_world', 'cluster_blocks', 'shared_budget_bytes')):
            raise ValueError('positive execution dimensions and budget required')
        if self.strategy == 'single_cta' and (self.workers_per_world != 1 or self.cluster_blocks != 1):
            raise ValueError('single CTA has exactly one block per world')
        if self.strategy != 'cluster' and self.cluster_blocks != 1:
            raise ValueError('cluster size belongs only to the cluster strategy')
        if self.strategy == 'cluster' and self.workers_per_world != self.cluster_blocks:
            raise ValueError('workers must match total cluster blocks, including the solver hub')
        if self.tuning_status not in ('candidate', 'measured', 'selected'):
            raise ValueError('explicit tuning state required')
        if not isinstance(self.thresholds, RoutingThresholds):
            raise TypeError('RoutingThresholds required')

    def capability_qualification(self, device):
        issues = []
        if self.block_threads > device.max_threads_per_block:
            issues.append('threads exceed device block limit')
        if self.shared_budget_bytes > device.shared_memory_per_block_bytes:
            issues.append('shared budget exceeds queried block limit')
        if self.strategy == 'cluster' and not device.supports_cluster:
            issues.append('cluster unavailable on this device')
        return dict(capability_compatible=not issues, issues=issues,
                    launch_qualified=False,
                    pending='inspect compiled kernel attributes and occupancy; cluster shape needs runtime launch qualification')


DEFAULT_NEWTON_BLOCK_THREADS = 128  # User-selected production default; generic CTA defaults stay unchanged.

def newton_execution_config(block_size=None, execution_config=None):
    """Resolve user-selected default or explicit single CTA Newton launch shape.

    Mathematical/Dx specializations currently qualify only 32/64/128 threads.
    Multi CTA Newton needs a different ownership algorithm and is rejected here.
    """
    if execution_config is not None and not isinstance(execution_config, ExecutionConfig):
        raise TypeError('ExecutionConfig required for Newton')
    config = execution_config or ExecutionConfig(
        block_threads=DEFAULT_NEWTON_BLOCK_THREADS if block_size is None else block_size,
        tuning_status="selected" if block_size is None else "candidate")
    if config.strategy != 'single_cta' or config.block_threads not in (32,64,128):
        raise ValueError('Newton supports single CTA with 32/64/128 threads')
    if block_size is not None and block_size != config.block_threads:
        raise ValueError('conflicting Newton block_size and execution_config')
    return config


def candidate_configs(device):
    """Ordered experiments, never automatic production choices."""
    configs = [ExecutionConfig(block_threads=threads) for threads in (32, 64, 128)]
    configs.extend(ExecutionConfig(strategy='multi_cta', block_threads=128,
                                  workers_per_world=workers) for workers in (2, 4))
    if device.supports_cluster:
        configs.extend(ExecutionConfig(strategy='cluster', block_threads=128,
                                      workers_per_world=blocks, cluster_blocks=blocks) for blocks in (2, 4))
    return tuple(config for config in configs
                 if config.capability_qualification(device)['capability_compatible'])


@dataclass(frozen=True)
class BuildIdentity:
    compiler_version: str
    mathdx_version: str
    kernel_version: str
    target_arch: str
    compile_options: tuple = ('--fmad=true', '--prec-div=true', '--prec-sqrt=true')
    driver_version: str = 'unknown'

    def __post_init__(self):
        object.__setattr__(self, 'compile_options', tuple(self.compile_options))
        if any(not getattr(self, key) for key in (
            'compiler_version', 'mathdx_version', 'kernel_version', 'target_arch')):
            raise ValueError('compiler/MathDx/kernel/target identities required')


def tuning_cache_key(device, build, *, phase, workload_family):
    if not phase or not workload_family:
        raise ValueError('phase and model/workload family required')
    payload = dict(schema_version=1, device=asdict(device), build=asdict(build),
                   phase=phase, workload_family=workload_family)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_tuning_record(path, *, device, build, phase, workload_family, config, evidence):
    """Write an explicitly requested cold record; no benchmark is launched."""
    if config.tuning_status != 'candidate' and not evidence:
        raise ValueError('measured/selected configurations require an evidence reference')
    record = dict(schema_version=1,
        cache_key=tuning_cache_key(device, build, phase=phase, workload_family=workload_family),
        device=asdict(device), build=asdict(build), phase=phase, workload_family=workload_family,
        config=asdict(config), evidence=evidence, launch_qualified=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(target)
    return record


def load_tuning_record(path, *, device, build, phase, workload_family):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = tuning_cache_key(device, build, phase=phase, workload_family=workload_family)
    if record.get('schema_version') != 1 or record.get('cache_key') != expected:
        raise ValueError('stale or foreign platform/build/workload tuning record')
    values = dict(record['config'])
    values['thresholds'] = RoutingThresholds(**values['thresholds'])
    config = ExecutionConfig(**values)
    if config.tuning_status != 'candidate' and not record.get('evidence'):
        raise ValueError('measured/selected record lacks evidence')
    if not config.capability_qualification(device)['capability_compatible']:
        raise ValueError('cached configuration exceeds device capability')
    return config, record
