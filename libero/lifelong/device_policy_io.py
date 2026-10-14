"""Explicit borrowed tensor IO; never changes the existing NumPy API."""


class DevicePolicyBatch:
    """Lease + policy data. Keep open until the consumer completion is recorded."""
    def __init__(self, lease, data):
        self.lease, self.data, self.closed = lease, data, False

    def validate(self):
        if self.closed:
            raise RuntimeError('device policy batch closed')
        self.lease.validate()

    def release(self, completion):
        self.validate()
        # Consumer event/stream lifetime belongs to the explicit lease provider.
        self.lease.release(completion)
        self.closed = True


def raw_device_obs_to_tensor_obs(lease, task_emb, cfg, *, torch_module, process_obs, consumer_stream):
    """lease.resolve_tensors waits producer->consumer and returns batched fields.

    Provider must validate exact owner/generation/world ranges and reject stale
    arrays. Process_obs is the SAME configured robomimic transform as NumPy IO.
    No from_numpy, cpu(), implicit device transfer or global synchronize.
    """
    lease.validate()
    fields = lease.resolve_tensors(torch_module, consumer_stream)
    keys = [name for values in cfg.data.obs.modality.values() for name in values]
    data = {'obs': {}, 'task_emb': task_emb.repeat(lease.nworld, 1)}
    for name in keys:
        value = fields[cfg.data.obs_key_mapping[name]]
        if value.shape[0] != lease.nworld or str(value.device) != str(lease.device):
            raise ValueError('device observation batch/device mismatch')
        # Original per-world transform followed by stack produces contiguous
        # NCHW/low-dim batches; retain that layout for identical vendor kernels.
        data['obs'][name] = process_obs(value, obs_key=name).float().contiguous()
    if str(task_emb.device) != str(lease.device):
        raise ValueError('task embedding must already reside on observation device')
    lease.validate()
    return DevicePolicyBatch(lease, data)


def submit_action_tensor(batch, action, sink, consumer_stream):
    batch.validate()
    if tuple(action.shape) != (batch.lease.nworld, sink.action_dim) or str(action.device) != str(batch.lease.device):
        raise ValueError('device action batch/device mismatch')
    # sink must check finite values on device, preserve first error, and retain
    # action storage until its event completes. No CPU scalar validation here.
    completion = sink.submit_tensor(action, consumer_stream, batch.lease)
    batch.release(completion)
    return completion
