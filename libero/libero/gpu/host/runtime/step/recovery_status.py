"""One same-stream typed download at a synchronous recovery public boundary."""
import warp as wp
from types import SimpleNamespace
from libero.gpu.host.runtime.production.boundary_status import WIDTH, validate_boundary_packet

@wp.kernel(enable_backward=False)
def _gather(request:wp.int64, runtime_error:wp.array(dtype=wp.int32),
        foundation_error:wp.array(dtype=wp.int32), newton_error:wp.array(dtype=wp.int32),
        forward_error:wp.array(dtype=wp.int32), model:wp.array(dtype=wp.int64),
        expected_model:wp.array(dtype=wp.int64), model_ready:wp.array(dtype=wp.int32),
        state:wp.array(dtype=wp.int64), expected_state:wp.array(dtype=wp.int64),
        forward:wp.array(dtype=wp.int64), expected_forward:wp.array(dtype=wp.int64),
        n_attempt:wp.array(dtype=wp.int64), n_written:wp.array(dtype=wp.int64),
        expected_newton:wp.array(dtype=wp.int64), integrations:wp.array(dtype=wp.int64),
        step_valid:wp.array(dtype=wp.int32), forward_valid:wp.array(dtype=wp.int32),
        n_valid:wp.array(dtype=wp.int32), packet:wp.array2d(dtype=wp.int64)):
    w=wp.tid()
    packet[w,0]=request;packet[w,1]=wp.int64(w)
    packet[w,2]=wp.int64(runtime_error[w]);packet[w,3]=wp.int64(foundation_error[w])
    packet[w,4]=wp.int64(newton_error[w]);packet[w,5]=wp.int64(forward_error[w])
    packet[w,6]=model[w];packet[w,7]=expected_model[w];packet[w,8]=wp.int64(model_ready[w])
    packet[w,9]=state[w];packet[w,10]=expected_state[w]
    packet[w,11]=forward[w];packet[w,12]=expected_forward[w]
    packet[w,13]=n_attempt[w];packet[w,14]=n_written[w];packet[w,15]=expected_newton[w]
    packet[w,16]=integrations[w];packet[w,17]=wp.int64(step_valid[w])
    packet[w,18]=wp.int64(forward_valid[w]);packet[w,19]=wp.int64(n_valid[w])

def _sources(bundle):
    r=bundle.runtime
    return (r.first_error,bundle.foundation.errors,bundle.newton.ws.err_code,r.forward.error,
        bundle.device_model_epoch,bundle.expected_model_epoch,bundle.model_ready,
        r.state_epoch,bundle.expected_state,bundle.dynamics.forward_epoch,r.forward.expected_forward,
        bundle.newton.n_attempt,bundle.newton.n_written,r.forward.expected_newton,
        r.integration_count,r.valid,r.forward.valid,bundle.newton.valid)

def _token(array):
    return (id(array),array.ptr,array.shape,array.strides,array.dtype,array.device)

class RecoveryBoundaryStatus:
    def __init__(self,bundle):
        self.bundle=bundle;self.owner=bundle.runtime;self.stream=bundle.stream
        self.model_spec=bundle.model_spec;self.model_identity=self.model_spec.model_sha256
        self.sources=_sources(bundle)
        self.tokens=tuple(_token(a) for a in self.sources)
        self.nworld=self.owner.euler.capacity
        if any(a.shape!=(self.nworld,) or a.device!=self.stream.device for a in self.sources):
            raise ValueError("boundary source world extent/device differs")
        i32={0,1,2,3,6,15,16,17}
        if any(a.dtype!=(wp.int32 if i in i32 else wp.int64) for i,a in enumerate(self.sources)):
            raise ValueError("boundary source scalar dtype differs")
        self.packet=wp.empty((self.nworld,WIDTH),dtype=wp.int64,device=self.stream.device)
        self.packet_token=_token(self.packet)
        self.request=0
        self._completed_record=None

    def validate_owner(self):
        if (getattr(self.bundle,"closed",False) or self.bundle.runtime is not self.owner or
                self.bundle.stream is not self.stream or
                self.bundle.model_spec is not self.model_spec or
                self.bundle.model_spec.model_sha256!=self.model_identity or
                tuple(_token(a) for a in _sources(self.bundle))!=self.tokens or
                _token(self.packet)!=self.packet_token):
            raise ValueError("boundary source owner changed")

    def synchronize_and_check(self):
        self.validate_owner()
        if self.stream.is_capturing:
            raise ValueError("synchronous public boundary cannot occur inside capture")
        if self.request>=2**63-1:
            raise ValueError("boundary request overflow")
        self.request+=1
        self._completed_record=None
        wp.launch(_gather,dim=self.nworld,inputs=[wp.int64(self.request),*self.sources,self.packet],
                  device=self.stream.device,stream=self.stream)
        # The existing synchronous public API retains its final wait. Delayed
        # device/launch failures propagate here before any public state escapes.
        wp.synchronize_stream(self.stream)
        record=validate_boundary_packet(self.packet.numpy(),owner=self.owner,stream=self.stream,
            request=self.request,model_identity=self.model_identity)
        self._completed_record=record
        return record

    def read(self):
        return self.synchronize_and_check()

    def complete(self,record):
        self.validate_owner()
        # read already waited and checked all sources. ControlledCompletion
        # may consume that exact completed record within its no-enqueue lock
        # scope; every write/enqueue must invalidate its separate serial.
        if (record is not self._completed_record or record.owner is not self.owner or
                record.stream is not self.stream or record.request!=self.request or
                record.model_identity!=self.model_identity):
            raise ValueError("foreign or stale completed boundary record")
        return record

    def release(self):
        # Called after the real bundle close has completed its existing wait.
        # A callback surviving cold replacement must not retain the old arena.
        # The closed sentinel uses the original owner guard without changing
        # any live read/gather/complete path.
        self._completed_record=None
        self.bundle=SimpleNamespace(closed=True)
        self.owner=None;self.stream=None;self.model_spec=None
        self.sources=();self.tokens=()
        self.packet=None;self.packet_token=None
