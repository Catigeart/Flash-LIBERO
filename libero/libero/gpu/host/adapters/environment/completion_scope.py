"""Reuse a successful completion only within an explicitly controlled boundary.

No CUDA kernel or numerical status precedence is implemented here. The actual
reader owns synchronization and the ordered runtime/foundation/newton/forward
packet. Its complete(record) method must reject stale request/owner/stream and
model generation. This scope never caches a failed check.
"""
from contextlib import contextmanager
from threading import RLock


class ControlledCompletion:
    def __init__(self, reader, identity):
        if not callable(identity) or not callable(getattr(reader,'read',None)) or not callable(getattr(reader,'complete',None)):
            raise TypeError('typed status reader and actual identity required')
        self.reader,self.identity=reader,identity
        self._lock=RLock();self._depth=0;self._record=None;self._key=None
        self._serial=0;self._mutations=0;self._closed=False

    def _open(self):
        if self._closed:raise RuntimeError('controlled completion closed')

    @classmethod
    def for_recovery_reader(cls,reader,submission_identity):
        """Adapt RP04 without including its read-incremented packet request.

        ``submission_identity`` is the controlled host action/write ordinal,
        with every enqueue/mutation routed through invalidate(). This method
        does not grant exclusive writer discipline to an unprotected owner.
        Current measured recovery qualification is W1 only. Keep W2 on its
        original reader callback; matching metadata never widens that scope.
        """
        if getattr(reader,'nworld',None)!=1:
            raise ValueError('controlled recovery completion qualified for W1 only')
        if not callable(getattr(reader,'validate_owner',None)) or not callable(submission_identity):
            raise TypeError('RP04 owner validation and actual submission identity required')
        reader.validate_owner()
        methods=(reader.read,reader.complete,reader.validate_owner)
        def identity():
            current=(reader.read,reader.complete,reader.validate_owner)
            if any(getattr(method,'__func__',method) is not getattr(saved,'__func__',saved)
                    or getattr(method,'__self__',None) is not getattr(saved,'__self__',None)
                    for method,saved in zip(current,methods)):
                raise ValueError('boundary reader method changed')
            # Qualified RP04 read and complete each perform their original full
            # validate_owner walk. Repeating it in identity adds no protection.
            # Safe lookup keeps the cold closed sentinel's original error on
            # the delegated read/complete path after reader.release().
            return (id(reader.owner),id(reader.stream),id(getattr(reader.bundle,'model_spec',None)),
                    reader.model_identity,submission_identity())
        return cls(reader,identity)

    def invalidate(self):
        with self._lock:
            self._open()
            self._serial+=1;self._record=None;self._key=None

    def mutate(self,callback,*args,**kwargs):
        """One controlled enqueue/write leaf under the boundary's same lock.

        Invalidate before even a partial submission. Reentrant checks during
        the callback cannot publish reusable records; final invalidation also
        covers a callback which checks, enqueues again, then raises. External
        writes do not acquire a cached-completion permission from this method.
        """
        if not callable(callback):raise TypeError('actual mutation callback required')
        with self._lock:
            self._open();self.invalidate();self.identity()
            # A writer callback may run before another read. Keep the actual
            # RP04 live-owner guard on that path, including its closed sentinel;
            # read/complete's existing guards cover the completion-only path.
            validate_owner=getattr(self.reader,'validate_owner',None)
            if callable(validate_owner):validate_owner()
            self._mutations+=1
            try:return callback(*args,**kwargs)
            finally:
                self._mutations-=1
                self.invalidate()

    def check_void(self):
        """Compatibility callback: consume the typed record, return old None."""
        self.check()

    def release(self):
        """Cold local release after the actual owner's successful close."""
        with self._lock:
            if self._closed:return
            if self._mutations:raise RuntimeError('completion mutation still active')
            self._record=self._key=None;self._closed=True
            self.reader=self.identity=None

    @contextmanager
    def scope(self):
        # The lock spans the whole operation, preventing a competing submission
        # through this owner. External submissions require the same protocol.
        with self._lock:
            self._open()
            outer=self._depth==0
            if outer:self._record=None;self._key=None
            self._depth+=1
            try:yield self
            finally:
                self._depth-=1
                if outer:self._record=None;self._key=None

    def check(self):
        with self._lock:
            self._open()
            key=(self._serial,self.identity())
            if self._depth and not self._mutations and self._record is not None and key==self._key:
                try:return self.reader.complete(self._record)
                except BaseException:
                    self._record=self._key=None
                    raise
            self._record=self._key=None
            record=self.reader.read()
            result=self.reader.complete(record)
            # Read may itself reveal a changed request identity. Never reuse it
            # across that change, even when the reader accepted completion.
            if self._depth and not self._mutations and self.identity()==key[1]:
                self._record,self._key=record,key
            return result
