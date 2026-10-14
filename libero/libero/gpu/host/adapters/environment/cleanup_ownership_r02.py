"""Typed ownership for failed GPU releases; no physics or acceptance inference."""
from threading import RLock
class ReleaseOwner:
    def __init__(self,owner,method_name='close'):
        if method_name not in ('close','release') or not callable(getattr(owner,method_name,None)):
            raise TypeError('actual close/release owner required')
        self.owner=owner;self.method_name=method_name;self.released=False;self._lock=RLock()
    def release(self):
        with self._lock:
            if self.released:return
            getattr(self.owner,self.method_name)()
            self.released=True;self.owner=None
    close=release

def attach_cleanup(error,lease):
    if not isinstance(lease,ReleaseOwner):raise TypeError('typed GPU release owner required')
    old=tuple(getattr(error,'pending_gpu_cleanup',()))
    if not all(isinstance(item,ReleaseOwner) for item in old):
        old=()
    if not any(item is lease or (not item.released and item.owner is lease.owner and item.method_name==lease.method_name) for item in old):error.pending_gpu_cleanup=old+(lease,)
    return error
