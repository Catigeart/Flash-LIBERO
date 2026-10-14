"""Diagnostic lifetime count of actual successful resident Euler commits."""
import warp as wp
from .runtime import ResidentStepRuntime, _token


@wp.kernel(enable_backward=False)
def _count_completed(valid:wp.array(dtype=wp.int32), count:wp.array(dtype=wp.int64)):
    w=wp.tid()
    if valid[w]==1:count[w]=count[w]+wp.int64(1)


class CountedResidentStepRuntime(ResidentStepRuntime):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.integration_count=wp.zeros(self.euler.capacity,dtype=wp.int64,device=self.stream.device)
        self._count_ref=self.integration_count
        self._count_token=_token(self.integration_count)

    def _integrate(self):
        if self.integration_count is not self._count_ref or _token(self.integration_count)!=self._count_token:
            raise ValueError('actual integration count owner changed')
        super()._integrate()
        wp.launch(_count_completed,dim=self.euler.capacity,inputs=[self.valid,self.integration_count],
                  device=self.stream.device,stream=self.stream)
