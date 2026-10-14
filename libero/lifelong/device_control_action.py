"""New explicit device action ABI for original OSC native action staging.

The unchanged CPU/NumPy stage_action remains the public default. This standalone
method is a proposal for the runtime_adapter owner and independent GPU testing.
"""

def stage_action_tensor(self, action, consumer_stream, *, torch_module, warp_module, completion_type):
    torch, wp = torch_module, warp_module
    self.validate()
    if action.dtype != torch.float64 or tuple(action.shape) != (self.capacity, 7):
        raise ValueError('action must be float64 (physical_worlds,7)')
    if action.device.type != 'cuda' or consumer_stream is None or consumer_stream.device != action.device:
        raise ValueError('matching CUDA action stream required')
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError('nonfinite policy action')
    source = wp.from_torch(action, dtype=wp.float64)
    if source.device != self.controller.action.device:
        raise ValueError('native action owner device mismatch')
    event = torch.cuda.Event()
    event.record(consumer_stream)
    native_stream = torch.cuda.ExternalStream(int(self.stream.cuda_stream), device=action.device)
    native_stream.wait_event(event)
    with wp.ScopedStream(self.stream):
        wp.copy(self.controller.action, source, stream=self.stream)
    done = torch.cuda.Event()
    done.record(native_stream)
    return completion_type(done, (self, self.controller.action, source, action))
