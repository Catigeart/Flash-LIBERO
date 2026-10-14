"""Explicit cold publication from one strictly admitted original collision job.

This does not load disk receipts, compile unknown specifications or grant launch
permission. A permit publishes the actual private loader output once, after the
live bank and source transaction validate the full product. Existing CDLL owners
remain held on failure/revoke; their original runtime contracts still apply.
"""
import copy
import ctypes as ct
import hashlib
import os
from pathlib import Path
import threading

from .build_product_receipt import CompleteProductFacts, canonical, pin


class _ProductPermit:
    __slots__ = ()


class HeldProductIssuer:
    """One explicit role/module/current authority, never a generic cache gate."""

    def __init__(self, bank, authority, *, module_key, role, cache_complete_facts=False):
        from ...build.held_native_job import LeasedHeldNativeJob, _compiler_binding, encoded
        from ...runtime.collision.job_sources import JobSourceAuthority, BorrowedJobFiles
        if type(bank) is not LeasedHeldNativeJob or type(authority) is not JobSourceAuthority:
            raise TypeError('actual strict held bank and job source authority required')
        if authority.bank is not bank or bank.phase != 'run':
            raise ValueError('same original successful run job required')
        pin(module_key)
        if role != 'collision_before_detect':
            raise ValueError('only explicit before-detect cold product scope is supported')
        if type(cache_complete_facts) is not bool:
            raise TypeError('explicit complete-facts cache choice must be boolean')
        bank._assert_scope()
        entry = bank.manifest['modules'][module_key]
        if entry['status'] != 'ready' or entry['description']['name'] != role:
            raise ValueError('exact ready original module/role required')
        self._bank = bank
        self._authority = authority
        self._types = (LeasedHeldNativeJob, JobSourceAuthority, BorrowedJobFiles)
        self._module_key = module_key
        self._role = role
        self._job_id = bank.manifest['job_id']
        self._root = str(bank.root)
        self._entry = entry
        self._entry_image = canonical(entry)
        self._cache_complete_facts = cache_complete_facts
        self._cache_choice = cache_complete_facts
        self._compiler_binding = _compiler_binding
        self._encoded = encoded
        self._view = authority.borrow('A06:' + module_key)
        self._view_generation = self._view.generation
        self._view_owner = self._view
        self._lock = threading.RLock()
        self._closed = False
        self._permits = {}
        self._held = []
        self._ctypes = (ct.CDLL, ct._CFuncPtr, ct.cast, ct.c_void_p)
        self._facts_type = CompleteProductFacts
        self._permit_type = _ProductPermit
        self._owner_type = _PrivateOwner
        self._source_path = os.path.abspath(__file__)
        self._source_sha = hashlib.sha256(Path(self._source_path).read_bytes()).hexdigest()
        self._method_tokens = tuple((name, value, value.__code__, value.__defaults__,
                                    copy.deepcopy(value.__kwdefaults__))
            for name, value in type(self).__dict__.items() if callable(value)
            and hasattr(value, '__code__'))
        self._validator = self._assert
        self._helper_tokens = tuple((name, globals()[name], getattr(globals()[name], '__code__', None))
            for name in ('canonical', 'pin'))
        self._facts_methods = tuple((name, getattr(CompleteProductFacts, name).__func__,
                                     getattr(CompleteProductFacts, name).__func__.__code__)
            for name in ('inspect',))
        self._parser_source_path = os.path.abspath(self._facts_methods[0][1].__globals__['__file__'])
        self._parser_source_sha = hashlib.sha256(Path(self._parser_source_path).read_bytes()).hexdigest()
        self._source_methods = tuple((cls, name, value, self._compiler_binding(value))
            for cls in (JobSourceAuthority, BorrowedJobFiles)
            for name, value in cls.__dict__.items() if callable(value) and hasattr(value, '__code__'))
        self._source_wrapped = tuple((value, value.__wrapped__, self._compiler_binding(value.__wrapped__))
            for _, _, value, _ in self._source_methods if hasattr(value, '__wrapped__'))
        with self._view.verification():
            self._assert()
            bank.validate(entry)
            self._facts = CompleteProductFacts.inspect(canonical(entry['manifest']['build_receipt']))
            self._facts_owner = self._facts
            self._facts_values = tuple(vars(self._facts).values())
            self._assert()

    def _assert(self):
        bank = self._bank
        if self._closed:
            raise OSError('cold product issuer revoked')
        if (type(bank) is not self._types[0] or type(self._authority) is not self._types[1]
                or self._authority.bank is not bank or self._view is not self._view_owner
                or type(self._view) is not self._types[2] or self._view.authority is not self._authority
                or self._view.generation != self._view_generation):
            raise OSError('cold product source/physical owner changed')
        if (str(bank.root) != self._root or bank.manifest['job_id'] != self._job_id
                or bank.phase != 'run' or bank.manifest['modules'].get(self._module_key) is not self._entry
                or canonical(self._entry) != self._entry_image):
            raise OSError('cold product original complete job/specification changed')
        for name, value, code, defaults, kwdefaults in self._method_tokens:
            actual = type(self).__dict__.get(name)
            if (actual is not value or actual.__code__ is not code or actual.__defaults__ is not defaults
                    or actual.__kwdefaults__ != kwdefaults
                    or getattr(getattr(self, name), '__func__', None) is not value):
                raise OSError('cold product issuer method changed: ' + name)
        validator = self._validator
        original_validator = next(value for name, value, *_ in self._method_tokens if name == '_assert')
        if (getattr(validator, '__self__', None) is not self
                or getattr(validator, '__func__', None) is not original_validator):
            raise OSError('cold product frozen validator binding changed')
        for name, value, code in self._helper_tokens:
            if globals().get(name) is not value or getattr(value, '__code__', None) is not code:
                raise OSError('cold product issuer helper changed: ' + name)
        if (ct.CDLL, ct._CFuncPtr, ct.cast, ct.c_void_p) != self._ctypes:
            raise OSError('cold product loader/export provider changed')
        if (CompleteProductFacts is not self._facts_type or _ProductPermit is not self._permit_type
                or _PrivateOwner is not self._owner_type):
            raise OSError('cold product receipt/permit/private owner type changed')
        if self._cache_complete_facts is not self._cache_choice:
            raise OSError('cold product facts-cache policy changed')
        if hasattr(self, '_facts') and (self._facts is not self._facts_owner
                or tuple(vars(self._facts).values()) != self._facts_values):
            raise OSError('cold complete product facts changed')
        for cls, name, value, token in self._source_methods:
            actual = cls.__dict__.get(name)
            current = self._compiler_binding(actual)
            if (actual is not value or any(x is not y for x, y in zip(current[:4], token[:4]))
                    or current[4:] != token[4:]):
                raise OSError('cold product source authority method changed: ' + name)
        for value, wrapped, token in self._source_wrapped:
            current = self._compiler_binding(wrapped)
            if (value.__wrapped__ is not wrapped or any(x is not y for x, y in zip(current[:4], token[:4]))
                    or current[4:] != token[4:]):
                raise OSError('cold product wrapped source authority method changed')
        for name, value, code in self._facts_methods:
            actual = getattr(CompleteProductFacts, name).__func__
            if actual is not value or actual.__code__ is not code:
                raise OSError('cold product complete receipt parser changed')
        if self._view.digest(self._source_path, self._source_sha) != self._source_sha:
            raise OSError('cold product issuer source changed')
        if self._view.digest(self._parser_source_path, self._parser_source_sha) != self._parser_source_sha:
            raise OSError('cold complete product parser source changed')

    def _check_product(self, held):
        self._assert()
        bank = self._bank
        bank.validate(self._entry)
        library, manifest = held['library'], held['manifest']
        if type(library) is not self._ctypes[0]:
            raise TypeError('actual private ctypes CDLL loader output required')
        if canonical(manifest) != held['manifest_image']:
            raise OSError('cold private manifest changed')
        # Reuse value facts only after the complete current entry/manifest and
        # bank readset proofs above. Neither branch reuses an authority result.
        facts = (self._facts if self._cache_complete_facts else
                 CompleteProductFacts.inspect(canonical(manifest['build_receipt'])))
        matches = [row for row in bank._private_records if row['owner'] is library]
        if len(matches) != 1:
            raise OSError('actual held private CDLL owner required')
        row = matches[0]
        if row is not held['record'] or canonical(row['file']) != held['file_image']:
            raise OSError('cold product physical ownership record changed')
        physical = row['file']
        declared = manifest['held_native_job']
        expected = copy.deepcopy(self._entry['manifest'])
        expected['library'] = physical['path']
        expected['held_native_job'] = dict(job_id=self._job_id, module_key=self._module_key,
            physical_inode=physical['inode'], hardlink_count=1)
        if (manifest != expected or declared != expected['held_native_job']
                or physical['hardlink_count'] != 1 or physical['sha256'] != facts.product_sha256
                or physical['bytes'] != facts.product_bytes or not row['handle']
                or library._handle != row['handle']
                or os.path.abspath(library._name) != physical['path']):
            raise OSError('complete actual private product/build/module identity changed')
        current = self._view.pin_record(physical['path'], physical['sha256'])
        if current != (physical['resolved'], physical['bytes'], physical['sha256'],
                       (physical['device'], physical['inode'], physical['hardlink_count'])):
            raise OSError('current private product digest/extent/physical identity changed')
        for symbol in ('launch', 'configure_shared'):
            function = getattr(library, symbol)  # Preserve actual loader's first export error.
            owners = getattr(function, '_objects', {})
            if (not isinstance(function, self._ctypes[1])
                    or not any(owner is library for owner in owners.values())):
                raise OSError('real private CDLL export owner required: ' + symbol)
            address = self._ctypes[2](function, self._ctypes[3]).value
            if not address:
                raise OSError('real nonnull private CDLL export required: ' + symbol)
            token = (function, address)
            prior = held['exports'].setdefault(symbol, token)
            if prior[0] is not function or prior[1] != address:
                raise OSError('private CDLL export binding changed: ' + symbol)
        self._assert()

    def issue(self, original, source, private_build_root, **options):
        """Load only the pinned run specification; mint after successful postcheck."""
        with self._lock:
            held = None
            try:
                validator = self._validator
                fixed = next(row for row in self._method_tokens if row[0] == '_assert')
                if (getattr(validator, '__self__', None) is not self
                        or getattr(validator, '__func__', None) is not fixed[1]
                        or fixed[1].__code__ is not fixed[2]):
                    raise OSError('cold product frozen validator binding/code changed')
                with self._view.verification():
                    self._validator()
                    # Keep the bank's compiler-before-specification error order,
                    # but reject another ready role specification before load.
                    current = self._compiler_binding(original)
                    prior = self._bank._compiler_tokens[self._role]
                    if (any(x is not y for x, y in zip(current[:4], prior[:4]))
                            or current[4:] != prior[4:]):
                        raise ValueError('held original complete compiler method changed')
                    requested = dict(options)
                    device = requested.get('device', 'cuda:0')
                    requested['device'] = str(device)
                    if hasattr(device, 'arch'):
                        requested['device_arch'] = int(device.arch)
                    description = dict(name=self._role, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                       options=self._encoded(requested))
                    if description != self._entry['description']:
                        raise ValueError('exact pinned complete product specification required')
                    library, manifest = self._bank.compile(original, self._role, source,
                                                          private_build_root, **options)
                    held = dict(library=library, manifest=copy.deepcopy(manifest),
                        manifest_image=canonical(manifest), exports={}, status='held_unpublished')
                    self._held.append(held)
                    matches = [row for row in self._bank._private_records if row['owner'] is library]
                    if len(matches) != 1:
                        raise OSError('actual held private CDLL owner required')
                    held['record'] = matches[0]
                    held['file_image'] = canonical(matches[0]['file'])
                    self._view.bind_plan(_PrivateOwner(library, manifest))
                    self._check_product(held)
                    self._check_product(held)
                    self._validator()
                # No permit exists during load or the outer source postcheck.
                permit = _ProductPermit()
                self._permits[permit] = held
                held['status'] = 'issued'
                return permit
            except BaseException:
                if held is not None:
                    held['status'] = 'failed_preserved'
                raise

    def consume(self, permit):
        """Return the actual private loader output once after current revalidation."""
        with self._lock:
            if type(permit) is not _ProductPermit or permit not in self._permits:
                raise ValueError('this issuer live opaque product permit required')
            held = self._permits[permit]
            if held['status'] != 'issued':
                raise ValueError('cold product permit is no longer publishable')
            held['status'] = 'consuming'
            try:
                validator = self._validator
                fixed = next(row for row in self._method_tokens if row[0] == '_assert')
                if (getattr(validator, '__self__', None) is not self
                        or getattr(validator, '__func__', None) is not fixed[1]
                        or fixed[1].__code__ is not fixed[2]):
                    raise OSError('cold product frozen validator binding/code changed')
                with self._view.verification():
                    self._validator()
                    self._check_product(held)
                    result = (held['library'], copy.deepcopy(held['manifest']))
                    self._check_product(held)
                    self._validator()
                held['status'] = 'consumed'
                return result
            except BaseException:
                held['status'] = 'failed_preserved'
                raise

    def revoke(self, permit):
        with self._lock:
            if type(permit) is not _ProductPermit or permit not in self._permits:
                raise ValueError('this issuer live opaque product permit required')
            self._permits[permit]['status'] = 'revoked'

    def close(self):
        with self._lock:
            self._closed = True
            for held in self._held:
                if held['status'] in ('issued', 'consuming'):
                    held['status'] = 'revoked'
            self._view.invalidate('cold product issuer released', disable=True)
            # Never unload/delete held objects or close another owner/parent.


class _PrivateOwner:
    """Explicit CDLL/manifest binding for BorrowedJobFiles, not a runtime plan."""
    __slots__ = ('library', 'manifest')

    def __init__(self, library, manifest):
        self.library, self.manifest = library, manifest
