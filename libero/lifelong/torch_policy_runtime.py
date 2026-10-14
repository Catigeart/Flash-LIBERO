"""Opt-in real Torch observation/action ownership ABI; no policy math rewrite.

A controlled producer owns private tensor snapshots. Held leases prevent publish
or reset; producer/consumer events retain payloads until native action staging is
complete. CPU mode exists solely as an independent protocol oracle.
"""
from threading import RLock


def tensor_token(value):
    return (id(value), value.data_ptr(), str(value.device), value.dtype,
            tuple(value.shape), tuple(value.stride()), value._version)


class TorchCompletion:
    def __init__(self, event, retained):
        self.event, self.retained, self.completed = event, tuple(retained), False

    def query(self):
        return self.completed or self.event is None or self.event.query()

    def wait(self):
        if self.event is not None:
            self.event.synchronize()
        self.completed = True
        self.retained = ()


def record_completion(torch_module, stream, retained=()):
    event = None
    if stream is not None:
        event = torch_module.cuda.Event()
        event.record(stream)
    return TorchCompletion(event, retained)


class TorchObservationSource:
    """Private owned raw observation fields, exact publication generations."""
    def __init__(self, fields, *, owner, generation, torch_module, producer_stream=None):
        self.torch, self.owner = torch_module, owner
        self.generation, self.closed = generation, False
        self._lock, self._held, self._pending = RLock(), set(), []
        self._publish(fields, producer_stream)

    def _publish(self, fields, stream):
        if not fields:
            raise ValueError('observation fields required')
        values = list(fields.values())
        if not all(isinstance(x, self.torch.Tensor) and x.ndim >= 1 for x in values):
            raise TypeError('batched real Torch tensors required')
        self.nworld, self.device = values[0].shape[0], values[0].device
        if self.nworld <= 0 or any(x.shape[0] != self.nworld or x.device != self.device for x in values):
            raise ValueError('observation batch/device mismatch')
        if self.device.type == 'cuda' and (stream is None or stream.device != self.device):
            raise ValueError('matching producer CUDA stream required')
        if stream is None:
            self._fields = {k: x.detach().clone() for k, x in fields.items()}
        else:
            # Inputs handed over by the calling Torch stream become ready
            # before private producer clones; publication records their event.
            stream.wait_stream(self.torch.cuda.current_stream(self.device))
            with self.torch.cuda.stream(stream):
                self._fields = {k: x.detach().clone() for k, x in fields.items()}
        self._tokens = {k: tensor_token(x) for k, x in self._fields.items()}
        self._ready = record_completion(self.torch, stream, self._fields.values())

    def _validate(self):
        if self.closed:
            raise RuntimeError('observation source closed')
        if {k: tensor_token(x) for k, x in self._fields.items()} != self._tokens:
            raise RuntimeError('observation storage or content changed outside publication')

    def acquire(self):
        with self._lock:
            self._validate()
            pending = []
            for completion in self._pending:
                if completion.query():
                    completion.wait()
                else:
                    pending.append(completion)
            self._pending = pending
            lease = TorchObservationLease(self)
            self._held.add(lease)
            return lease

    def _idle(self):
        self._validate()
        if self._held:
            raise RuntimeError('held observation lease prevents publication/reset')
        for completion in self._pending:
            if not completion.query():
                raise RuntimeError('consumer completion pending')
            completion.wait()
        self._pending.clear()

    def publish(self, fields, *, owner, generation, producer_stream=None):
        with self._lock:
            self._idle()
            if owner is not self.owner:
                raise RuntimeError('observation owner changed')
            if generation == self.generation:
                raise ValueError('fresh observation generation required')
            self._publish(fields, producer_stream)
            self.generation = generation

    def close(self):
        with self._lock:
            if self.closed:
                return
            if self._held:
                raise RuntimeError('held observation lease prevents close')
            # Cleanup remains available after a failed validation; wait only
            # the explicit producer/consumer events retaining these tensors.
            for completion in self._pending:
                completion.wait()
            self._pending.clear()
            self._ready.wait()
            self.closed = True
            self._fields.clear()


class TorchObservationLease:
    def __init__(self, source):
        self.source, self.owner = source, source.owner
        self.generation = source.generation
        self.nworld, self.device = source.nworld, source.device
        self.fields = dict(source._fields)
        self.closed = False

    def validate(self):
        if self.closed:
            raise RuntimeError('observation lease released')
        self.source._validate()
        if self.source.owner is not self.owner or self.source.generation != self.generation:
            raise RuntimeError('observation owner/generation changed')
        if self not in self.source._held:
            raise RuntimeError('observation lease not held')

    def resolve_tensors(self, torch_module, consumer_stream):
        with self.source._lock:
            self.validate()
            if torch_module is not self.source.torch:
                raise ValueError('foreign Torch provider')
            if self.device.type == 'cuda':
                if consumer_stream is None or consumer_stream.device != self.device:
                    raise ValueError('matching consumer CUDA stream required')
                consumer_stream.wait_event(self.source._ready.event)
                for field in self.fields.values():
                    field.record_stream(consumer_stream)
            return dict(self.fields)

    def release(self, completion):
        with self.source._lock:
            self.validate()
            if not isinstance(completion, TorchCompletion):
                raise TypeError('actual retained completion required')
            if self.device.type == 'cuda' and (completion.event is None or completion.event.device != self.device):
                raise TypeError('matching recorded CUDA completion required')
            self.source._pending.append(completion)
            self.source._held.remove(self)
            self.closed = True

    def abort(self, consumer_stream=None):
        with self.source._lock:
            if self.closed:
                return
            if self.device.type == 'cuda' and (consumer_stream is None or consumer_stream.device != self.device):
                raise ValueError('matching consumer stream required to abort CUDA lease')
            # Error cleanup must not require a still-valid field/owner token.
            self.source._pending.append(record_completion(self.source.torch, consumer_stream, self.fields.values()))
            self.source._held.discard(self)
            self.closed = True


class ExactRgbNormalizer:
    """Cold CPU table reproduces the admitted original uint8 RGB processor bits.

    The native source transformation is separable for all 256 uint8 inputs.
    Runtime performs integer gather and original HWC->CHW indexing only. Custom
    modality processors require a separate ABI and are not silently replaced.
    """
    def __init__(self, torch_module, original_process_obs, rgb_keys, device):
        self.torch, self.rgb_keys = torch_module, frozenset(rgb_keys)
        sample = torch_module.arange(256, dtype=torch_module.int64).to(torch_module.uint8)
        sample = sample.reshape(1, 256, 1).repeat(1, 1, 3)
        key = next(iter(self.rgb_keys))
        table = original_process_obs(sample, obs_key=key)[0, 0].clone()
        self.table = table.to(device)
        self.token = tensor_token(self.table)
        self.original = original_process_obs

    def __call__(self, value, *, obs_key):
        if obs_key not in self.rgb_keys:
            return self.original(value, obs_key=obs_key)
        if tensor_token(self.table) != self.token:
            raise RuntimeError('RGB normalization table changed')
        if value.dtype != self.torch.uint8 or value.shape[-1] != 3:
            raise ValueError('admitted uint8 HWC RGB required')
        out = self.table[value.long()]
        dims = list(range(out.ndim))
        return out.permute(*dims[:-3], dims[-1], dims[-3], dims[-2])


class TorchActionStagingSink:
    """FP64 private action staging then the original validated native consumer.

    Float32->float64 conversion matches original np.asarray(dtype=float64)
    exactly. Finite reduction returns only a scalar status before mutation.
    The native consumer validates its own physical owner and copies unchanged
    action bytes into its retained controller.action allocation.
    """
    def __init__(self, *, nworld, action_dim, device, owner, generation,
                 torch_module, validate_owner, consume_tensor):
        if not callable(validate_owner) or not callable(consume_tensor):
            raise TypeError('native owner validation and consumer required')
        self.nworld, self.action_dim = nworld, action_dim
        self.device, self.owner, self.generation = torch_module.device(device), owner, generation
        self.torch, self.validate_owner, self.consume_tensor = torch_module, validate_owner, consume_tensor
        self._pending, self.closed = [], False

    def submit_tensor(self, action, consumer_stream, lease):
        if self.closed:
            raise RuntimeError('action sink closed')
        lease.validate()
        self.validate_owner(self.owner, self.generation)
        if lease.owner is not self.owner:
            raise RuntimeError('foreign observation/action physical owner')
        if tuple(action.shape) != (self.nworld, self.action_dim):
            raise ValueError('action must be float64 (physical_worlds,7)')
        if action.device != self.device:
            raise ValueError('device action device mismatch')
        if action.dtype not in (self.torch.float32, self.torch.float64):
            raise ValueError('float32/float64 policy action required')
        # GPU .item() is a bounded status synchronization, never full action D2H.
        if not bool(self.torch.isfinite(action).all().item()):
            raise ValueError('nonfinite policy action')
        staging = action.to(dtype=self.torch.float64).contiguous().clone()
        self.validate_owner(self.owner, self.generation)
        completion = self.consume_tensor(staging, consumer_stream)
        if not isinstance(completion, TorchCompletion):
            raise TypeError('native action consumer must return retained completion')
        completion.retained += (action, staging, lease)
        self._pending = [c for c in self._pending if not c.query()]
        self._pending.append(completion)
        return completion

    def close(self):
        for completion in self._pending:
            completion.wait()
        self._pending.clear()
        self.closed = True

class TorchPolicyExecutor:
    """Explicit native policy->held observation->validated action consumer."""
    def __init__(self, *, policy, source, sink, cfg, task_emb, process_obs,
                 consumer_stream, torch_module, policy_io):
        self.policy, self.source, self.sink = policy, source, sink
        self.cfg, self.task_emb, self.process_obs = cfg, task_emb, process_obs
        self.stream, self.torch, self.io = consumer_stream, torch_module, policy_io
        if source.owner is not sink.owner or source.nworld != sink.nworld:
            raise ValueError('policy source/action sink owner or batch mismatch')
        if task_emb.device != source.device:
            raise ValueError('task embedding must already reside on observation device')
        self._embedding_token = tensor_token(task_emb)

    def step(self):
        from contextlib import nullcontext
        if tensor_token(self.task_emb) != self._embedding_token:
            raise RuntimeError('task embedding changed')
        lease = self.source.acquire()
        context = self.torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        try:
            with context:
                batch = self.io.raw_device_obs_to_tensor_obs(lease, self.task_emb, self.cfg,
                    torch_module=self.torch, process_obs=self.process_obs, consumer_stream=self.stream)
                action = self.policy.get_action_tensor(batch.data)
                return self.io.submit_action_tensor(batch, action, self.sink, self.stream)
        except BaseException:
            # Record work already enqueued before relinquishing producer holds.
            if not lease.closed:
                try:
                    lease.abort(self.stream)
                except BaseException:
                    pass  # preserve the original policy/provider first error
            raise

    def reset(self):
        with self.source._lock:
            self.source._idle()
            self.sink.validate_owner(self.sink.owner, self.sink.generation)
            self.policy.reset()
