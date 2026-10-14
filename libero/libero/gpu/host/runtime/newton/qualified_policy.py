"""Cold selection and holders for already-qualified independent Newton modules.

No CUDA/Warp import, compilation, input download or automatic persistent-core
selection happens here. A domain names an exact tested module and input recipe.
Unknown domains and mixed/negative timings select the retained baseline. The
caller remains responsible for restoring that recipe into fresh physical owners.
"""
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import hashlib
import json
import math
import shutil
import uuid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class LeafSelection:
    domain: str
    variant: str
    reason: str
    catalog_sha256: str
    record_json: str = ''

    @property
    def candidate(self):
        return self.variant == 'candidate'

    @property
    def record(self):
        return json.loads(self.record_json) if self.record_json else None

    def manifest(self):
        return dict(domain=self.domain, variant=self.variant, reason=self.reason,
                    catalog_sha256=self.catalog_sha256,
                    scope='independent shaderfunction only', public_core_adopted=False)


class QualifiedNewtonCatalog:
    """An immutable, source-bound cold catalog supplied by the root integrator."""
    def __init__(self, path, expected_sha256):
        data = Path(path).read_bytes()
        if digest(data) != expected_sha256:
            raise ValueError('Newton qualification catalog changed')
        catalog = json.loads(data)
        if catalog.get('schema') != 'newton-qualified-leaves-v1':
            raise ValueError('Newton qualification catalog schema')
        self.sha256 = expected_sha256
        self._records = {d['domain']: canonical(d) for d in catalog['domains']}
        self._cold_history = canonical(catalog['cold_history_domains'])
        self._cold_source_sha256 = catalog['cold_history_source_sha256']
        if len(self._records) != len(catalog['domains']):
            raise ValueError('duplicate Newton domain')

    def select(self, domain, *, input_identity, module_identity, dimensions,
               scope='independent_shaderfunction'):
        text = self._records.get(domain)
        if text is None:
            return LeafSelection(domain, 'baseline', 'unqualified domain', self.sha256)
        record = json.loads(text)
        reason = None
        if scope != 'independent_shaderfunction':
            reason = 'independent function receipt does not qualify a public persistent core'
        elif canonical(input_identity) != canonical(record['input_identity']):
            reason = 'input recipe/source differs from the qualified domain'
        elif canonical(dimensions) != canonical(record['dimensions']):
            reason = 'capacity/world/work dimensions differ'
        elif canonical(module_identity) != canonical(record['module_identity']):
            reason = 'shaderfunction/bridge/ABI differs'
        elif not record['mathematics_passed'] or not record['work_passed']:
            reason = 'independent mathematical/work qualification missing'
        else:
            ratios = record['round_ratios']
            if (len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios)
                    or not all(r < 1 for r in ratios)):
                reason = 'mixed or negative paired rounds retain baseline'
        return LeafSelection(domain, 'baseline' if reason else 'candidate',
                             reason or 'three positive paired rounds in this exact domain',
                             self.sha256, text)

    def select_history(self, *, nv, worlds, rows, maximum_slots,
                       constructor_source_sha256, diagnostic_profile):
        """Only the explicit fast NV43 W1/W2 history-constructor receipt."""
        label = f'NS09/nv{nv}/w{worlds}/rows{rows}/slots{maximum_slots}'
        for d in json.loads(self._cold_history):
            if (nv, worlds, rows, maximum_slots) == (
                    d['nv'], d['worlds'], d['physical_history_rows'], d['maximum_slots']):
                if diagnostic_profile != 'fast':
                    reason = 'compatible caller retains history allocation'
                elif constructor_source_sha256 != self._cold_source_sha256:
                    reason = 'constructor source differs'
                elif not all(math.isfinite(r) and 0 < r < 1 for r in d['timing']['round_ratios']):
                    reason = 'history domain has no three positive paired rounds'
                else:
                    return LeafSelection(label, 'candidate', 'explicit cold fast constructor',
                                         self.sha256, canonical(dict(d, id='NS09')))
                return LeafSelection(label, 'baseline', reason, self.sha256)
        return LeafSelection(label, 'baseline', 'unqualified history capacity', self.sha256)


def _layout(array, ctype, width):
    itemsize = 4 if ctype == 'int' else 8
    expected = 'int32' if ctype == 'int' else 'float64'
    dtype = getattr(array.dtype, '__name__', str(array.dtype))
    if dtype != expected or tuple(array.shape) != (1, width):
        raise ValueError('qualified Newton owner dtype/shape differs')
    if tuple(array.strides) != (width * itemsize, itemsize):
        raise ValueError('qualified Newton owners require tested contiguous layout')
    if not int(array.ptr):
        raise ValueError('qualified Newton null owner')
    return (int(array.ptr), tuple(array.shape), tuple(array.strides), dtype,
            str(array.device), itemsize * width)


class QualifiedShaderHolder:
    """One independent call with new physical library and retained array owners.

    `validate_input` is the existing caller's recipe/epoch validator. It must
    restore/validate input values, not infer them from shapes. This class seals
    physical layouts and the source/binary ABI; it never reads device values.
    A successful or failed launch consumes the lease, so later iterations cannot
    inherit a native-input or factor-readiness credential. CUDA's existing
    within-call stop/error/held contracts remain the qualified module's job.
    """
    def __init__(self, selection, module_root, private_root, owners, stream, *,
                 generation, validate_input, loader=ct.CDLL):
        record = selection.record
        if record is None or record['id'] == 'NS09':
            raise ValueError('a qualified shaderfunction domain is required')
        if not callable(validate_input):
            raise TypeError('existing input/epoch validator is required')
        self.selection = selection
        self.record = record
        self._record_json = canonical(record)
        self.generation = generation
        self.validate_input = validate_input
        self.owners = dict(owners)
        self._owner_refs = tuple(owners.values())
        self.stream = stream
        self._retired = False
        fields = record['fields']
        if set(owners) != {f[0] for f in fields}:
            raise ValueError('qualified Newton Args owner set differs')
        self._layouts = {n: _layout(owners[n], t, width) for n, t, width in fields}
        self._owner_ids = {n: id(array) for n, array in owners.items()}
        devices = {v[4] for v in self._layouts.values()}
        if len(devices) != 1 or str(stream.device) not in devices:
            raise ValueError('qualified Newton stream/device differs')
        self._stream = (str(stream.device), int(stream.cuda_stream))
        # Every owned interval was independent in the positive tested domains.
        intervals = sorted((v[0], v[0] + v[5]) for v in self._layouts.values())
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('qualified Newton owner intervals overlap')
        self._validate_values()
        spec = record['modules'][selection.variant]
        root = Path(module_root).resolve()
        for key in ('source', 'bridge', 'binary'):
            path = (root / spec[key]['path']).resolve()
            if not path.is_relative_to(root) or digest(path.read_bytes()) != spec[key]['sha256']:
                raise ValueError('qualified Newton module bytes changed: ' + key)
        target = Path(private_root) / ('newton-' + uuid.uuid4().hex)
        target.mkdir(parents=True, exist_ok=False)
        binary = target / 'private.so'
        shutil.copyfile(root / spec['binary']['path'], binary)
        if digest(binary.read_bytes()) != spec['binary']['sha256']:
            raise ValueError('qualified Newton private binary copy changed')
        self.private_binary = binary
        stat = binary.stat()
        self.private_identity = (stat.st_dev, stat.st_ino, stat.st_size)
        self.library = loader(str(binary))
        self._library_ref = self.library
        self.Arg = type('QualifiedNewtonArgs', (ct.Structure,),
                        {'_fields_': [(n, ct.c_void_p) for n, _, _ in fields]})
        self.args = self.Arg(*(ct.c_void_p(owners[n].ptr) for n, _, _ in fields))
        self.library.launch.argtypes = [self.Arg, ct.c_int, ct.c_int, ct.c_void_p]
        self.library.launch.restype = ct.c_int

    def _validate_values(self):
        # True is explicit; a validator returning None is not an admission.
        if self.validate_input(self.record['input_identity'], self.generation) is not True:
            raise ValueError('qualified Newton input/epoch not admitted')

    def validate(self, generation):
        if self._retired or generation != self.generation:
            raise ValueError('qualified Newton invocation lease expired')
        if canonical(self.record) != self._record_json or self.library is not self._library_ref:
            raise ValueError('qualified Newton ABI/library binding changed')
        if {n: id(array) for n, array in self.owners.items()} != self._owner_ids:
            raise ValueError('qualified Newton physical owner replaced')
        current = {n: _layout(self.owners[n], t, width)
                   for n, t, width in self.record['fields']}
        if current != self._layouts:
            raise ValueError('qualified Newton physical owner changed')
        if any(getattr(self.args, n) != self._layouts[n][0] for n in self._layouts):
            raise ValueError('qualified Newton Args pointer changed')
        if (str(self.stream.device), int(self.stream.cuda_stream)) != self._stream:
            raise ValueError('qualified Newton stream changed')
        stat = self.private_binary.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size) != self.private_identity:
            raise ValueError('qualified Newton private library owner changed')
        if digest(self.private_binary.read_bytes()) != self.record['modules'][self.selection.variant]['binary']['sha256']:
            raise ValueError('qualified Newton private library bytes changed')
        self._validate_values()

    def launch_once(self, generation):
        """No compile, restore, sync, timing or scalar arithmetic is added."""
        self.validate(generation)
        try:
            result = self.library.launch(self.args, 1, 1, self.stream.cuda_stream)
            if result:
                raise RuntimeError(('qualified independent Newton launch', result))
        finally:
            self._retired = True

    def manifest(self):
        return dict(**self.selection.manifest(), generation=self.generation,
                    private_binary=str(self.private_binary),
                    binary_sha256=self.record['modules'][self.selection.variant]['binary']['sha256'],
                    physical_identity=list(self.private_identity),
                    owners=list(self._layouts), stream=list(self._stream),
                    invocation_only=True, retired=self._retired,
                    owner_contract=self.record['owner_contract'],
                    compile_invocations=0, arithmetic_or_timer_changed=False)


def qualified_cold_history_factory(factory, *, explicitly_fast, create,
                                   profile_context):
    """Retain the existing explicit fast ABI while avoiding allocate-then-retire.

    `create` constructs all workspace/solver owners inside the cold context.
    The ordinary compatible caller takes its original factory unchanged. The
    tested NV43 W1/W2 constructor gain is reported separately by the catalog;
    this factory does not assert that other shapes gained or grant solver math.
    """
    if type(explicitly_fast) is not bool:
        raise TypeError('cold history profile must be explicit')
    if not explicitly_fast:
        return factory()
    with profile_context('fast'):
        return create()
