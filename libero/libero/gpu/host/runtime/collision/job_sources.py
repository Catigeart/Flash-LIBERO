"""One explicitly admitted job's source facts, never execution/owner grants."""
from contextlib import contextmanager
import hashlib,json,os,sys,threading
from pathlib import Path
from types import MappingProxyType

def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class JobSourceAuthority:
    def __init__(self,bank,*,original_source_root,source_files,overlay_files=None,scope_id='CG06-source-auth-r05'):
        from ..core.cold_credentials import FileDigestLeasePool
        bank._assert_scope()
        if bank.phase!='run' or bank.manifest['status']!='prepared':raise ValueError('original strict prepared job required')
        self.bank=bank;self._bank_owner=bank;self.scope_id=str(scope_id);self.root=Path(original_source_root).resolve()
        self.job_id=bank.manifest['job_id'];self.closed=False;self._lock=threading.RLock();self._local=threading.local()
        self.shared=FileDigestLeasePool();self._shared_owner=self.shared;self._shared_type=type(self.shared)
        self._sdk={};self._source={};self._aliases={};self._module_images={}
        for key,entry in bank.manifest['modules'].items():
            image=_canonical(entry)
            if bank._entry_images.get(key)!=image:raise ValueError('original complete module admission changed')
            self._module_images[key]=image
            receipt=entry['manifest']['build_receipt'];identity=receipt['identity']
            rows=identity['dependencies']+identity['toolchain']['components']
            if identity.get('archive') is not None:rows=rows+[identity['archive']]
            rows=rows+[entry['source'],entry['binary']]
            for row in rows:
                path=os.path.abspath(row['path']);fact=(int(row['bytes']),row['sha256'])
                if path in self._sdk and self._sdk[path]!=fact:raise ValueError('inconsistent original source/build readset')
                self._sdk[path]=fact
        for relative,digest in dict(source_files).items():
            path=os.path.abspath(self.root/relative)
            if os.path.commonpath((str(self.root),path))!=str(self.root):raise ValueError('source pin outside fixed original root')
            self._source[path]=digest
        for path,digest in dict(overlay_files or {}).items():self._source[os.path.abspath(path)]=digest
        self._source[os.path.abspath(__file__)]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._sdk=MappingProxyType(self._sdk);self._source=MappingProxyType(self._source)
        self.stats=dict(transactions=0,snapshots=0,dedup_hits=0,current_resolves=0,post_checks=0,strict_reads=0)
        bank._add_guard_file(__file__)
        self._methods=tuple((name,getattr(type(self),name),getattr(type(self),name).__code__) for name in ('transaction','snapshot','borrow','close'))

    def _assert(self):
        if self.closed:raise OSError('job source authority closed')
        if self.bank is not self._bank_owner or self.shared is not self._shared_owner or type(self.shared) is not self._shared_type:
            raise OSError('job source authority owner changed')
        if self.bank.manifest['job_id']!=self.job_id or set(self.bank.manifest['modules'])!=set(self._module_images):
            raise OSError('original complete job/module identity changed')
        for key,image in self._module_images.items():
            if _canonical(self.bank.manifest['modules'][key])!=image:raise OSError('original complete source/build/policy changed')
        for name,function,code in self._methods:
            actual=getattr(type(self),name)
            if actual is not function or actual.__code__ is not code:raise OSError('job source authentication method changed')

    @contextmanager
    def transaction(self):
        with self._lock:
            depth=getattr(self._local,'depth',0)
            if depth:
                self._local.depth=depth+1
                try:yield
                finally:self._local.depth=depth
                return
            with self.bank._operation(),self.shared.verification():
                self._assert();self._local.depth=1;self._local.cache={};self.stats['transactions']+=1
                try:
                    yield
                    # Re-check actual current identity/alias/lease and namespace
                    # after this single seal/admit transaction, never globally.
                    bridges=set()
                    for pool,path,row,lease in self._local.cache.values():
                        if lease is None:continue
                        actual=self._current(path,pool,lease)
                        self.stats['post_checks']+=1
                        if actual[:2]!=row[:2] or actual[2]!=row[3] or pool.digest(path,row[2])!=row[2]:raise OSError('job source/build changed during authentication: '+path)
                        if getattr(lease,'bridge',None) is not None:bridges.add(lease.bridge)
                    for bridge in bridges:
                        if bridge.process.poll() is not None or Path('/proc/self/mountinfo').read_bytes()!=bridge._namespace:
                            raise OSError('job source namespace/helper changed during authentication')
                    self._assert()
                finally:self._local.depth=0;self._local.cache={}

    def _lease(self,pool,path):
        return pool._leases.get(os.path.abspath(path))

    def _current(self,path,pool,lease):
        info=Path(path).stat()
        provider=sys.modules[type(pool).__module__]
        bridge_type=getattr(provider,'_BridgeLease',())
        # Only the formally tested WSL Windows provider proves all ancestors
        # non-reparse and no-write/delete while this exact namespace is live.
        if bridge_type and isinstance(lease,bridge_type) and lease.valid():
            resolved=self._aliases.get(path)
            if resolved is None:
                resolved=str(Path(path).resolve());self._aliases[path]=resolved;self.stats['current_resolves']+=1
        else:
            resolved=str(Path(path).resolve());self.stats['current_resolves']+=1
        return (resolved,info.st_size,(info.st_dev,info.st_ino,info.st_nlink))

    def snapshot(self,path,*,private_pool=None,expected=None):
        path=os.path.abspath(path)
        if not getattr(self._local,'depth',0):
            with self.transaction():return self.snapshot(path,private_pool=private_pool,expected=expected)
        if path in self._sdk:
            pool=self.bank._pool_owner;required=self._sdk[path][1]
        elif path in self._source:
            pool=self.shared;required=self._source[path]
        else:
            if private_pool is None:raise OSError('unknown source requires independent runtime proof')
            pool=private_pool;required=expected
            if required is None:
                required=hashlib.sha256(Path(path).read_bytes()).hexdigest();self.stats['strict_reads']+=1
        key=(id(pool),path)
        prior=self._local.cache.get(key)
        if prior is not None and prior[3] is not None and prior[3].valid():
            self.stats['dedup_hits']+=1
            return prior[2]
        content=pool.digest(path,required)
        lease=self._lease(pool,path)
        resolved,size,identity=self._current(path,pool,lease)
        if path in self._sdk:
            expected_identity=self.bank._file_identities.get(path)
            if expected_identity!=(resolved,*identity):raise OSError('original shared source/build physical identity changed: '+path)
        row=(resolved,size,content,identity)
        self._local.cache[key]=(pool,path,row,lease);self.stats['snapshots']+=1
        return row

    def borrow(self,label='runtime'):return BorrowedJobFiles(self,label)

    def bind_registry(self,registry):
        """Explicit CPU-tested bridge; current CG factory uses cold admission.

        No Admission object is cached. Existing registry status/scope/report and
        all wrapper/native owner validators still execute on every submission.
        """
        self._assert();prior=registry._source_leases;prior.invalidate('explicit job source binding')
        view=self.borrow('registry');registry._source_leases=view;return view

    def state(self):
        return dict(scope=self.scope_id,job_id=self.job_id,closed=self.closed,stats=dict(self.stats),
            shared_project=self.shared.state(),parent_sdk=self.bank._pool_owner.state())

    def close(self):
        with self._lock:
            self.closed=True;self.shared.invalidate('job source scope closed',disable=True)
            # Parent SDK bank owns its own handles and explicit close boundary.

class BorrowedJobFiles:
    def __init__(self,authority,label):
        from ..core.cold_credentials import FileDigestLeasePool
        self.authority=authority;self._owner=authority;self.label=label;self.local=FileDigestLeasePool();self._local_owner=self.local
        self.revoked=False;self.generation=0;self._private={}
        self._text={}

    def _assert(self):
        if self.revoked:raise OSError('runtime source view revoked')
        if self.authority is not self._owner or self.local is not self._local_owner:raise OSError('runtime source view owner changed')
        if not getattr(self.authority._local,'depth',0):self.authority._assert()

    @property
    def _leases(self):
        # Compatibility with H01 registry's all-unsupported strict fast path.
        if self.revoked:return {'revoked':None}
        return self.authority.bank._pool_owner._leases or self.authority.shared._leases or self.local._leases

    @contextmanager
    def verification(self):
        self._assert()
        with self.authority.transaction(),self.local.verification():yield

    def pin_record(self,path,expected=None):
        self._assert();key=os.path.abspath(path)
        if key in self._private:
            library,handle,digest=self._private[key]
            if library._handle!=handle or os.path.abspath(library._name)!=key:
                raise OSError('runtime private CDLL owner changed')
            if expected is None:expected=digest
        row=self.authority.snapshot(path,private_pool=self.local,expected=expected)
        if expected is None:
            fixed=self.authority._sdk.get(key,(None,None))[1] or self.authority._source.get(key)
            if fixed is not None and row[2]!=fixed:
                raise OSError('fixed job source/build version changed: '+key)
        return row

    def digest(self,path,expected=None):return self.pin_record(path,expected)[2]

    def text_digest(self,path):
        row=self.pin_record(path);key=(os.path.abspath(path),row[2])
        if key not in self._text:self._text[key]=hashlib.sha256(Path(path).read_text().encode()).hexdigest()
        return self._text[key]

    def bind_plan(self,plan):
        """Each actual private product/CDLL owner must come from this bank load."""
        self._assert();path=os.path.abspath(plan.manifest['library']);library=plan.library
        found=[row for row in self.authority.bank._private_records if row['file']['resolved']==str(Path(path).resolve())]
        if len(found)!=1 or found[0]['owner'] is not library or found[0]['handle']!=library._handle:
            raise ValueError('independent actual private plan/CDLL owner required')
        self._private[path]=(library,library._handle,found[0]['file']['sha256'])

    def state(self):
        return dict(scope=self.authority.scope_id,job_id=self.authority.job_id,borrower=self.label,revoked=self.revoked,
            generation=self.generation,private=self.local.state(),shared=self.authority.state())

    def invalidate(self,reason='runtime released',disable=False):
        self.local.invalidate(reason,disable=disable);self.revoked=True;self.generation+=1
        # Never revoke another runtime's SDK/project authority or parent bank.
