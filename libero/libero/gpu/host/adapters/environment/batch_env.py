"""Same-task LIBERO environments over one native world-major GPU physics arena.

Fixtures are sampled once at construction and shared by every world. Reset uses
the task's official initialization bank (or a supplied bank); it never constructs
singleton physics environments. Goal logic/render packaging runs on the host at
action boundaries. All worlds' physics advances in one native batch.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from threading import RLock
from pathlib import Path

import numpy as np


def _state_bank(values, nq, nv):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != 1+nq+nv:
        raise ValueError(f'init_states must have shape (M, {1+nq+nv}), M >= 1')
    if not np.isfinite(result).all():
        raise ValueError('initial states must be finite')
    return np.array(result, copy=True, order='C')


def _clone_shell(shell):
    # The model-only sim deliberately raises on every attribute access,
    # including deepcopy probes. Bypass the public __new__ routing as well:
    # reconstructing it with deepcopy would allocate a single-world GPU proxy.
    clone = object.__new__(type(shell))
    memo = {id(shell): clone, id(shell.sim): shell.sim}
    clone.__dict__.update(deepcopy(shell.__dict__, memo))
    return clone


def _prepare_shared_fixtures(built, seed):
    """Original host placement sampling, before static model intake/compilation."""
    from libero.gpu.model.factory import native_model_metadata
    from .task_driver import _PLACEMENT_LOCK
    from robosuite.utils.errors import RandomizationError
    shell, model = built.domain_shell, built.native_model
    with _PLACEMENT_LOCK:
        previous = np.random.get_state()
        np.random.seed(seed)
        try:
            for attempt in range(100):
                try:
                    placements = shell.placement_initializer.sample()
                    break
                except RandomizationError:
                    if attempt == 99:
                        raise
        finally:
            np.random.set_state(previous)
    for position, quaternion, obj in placements.values():
        if obj.name in shell.fixtures_dict:
            body = shell.obj_body_id[obj.name]
            model.body_pos[body] = position
            model.body_quat[body] = quaternion
    # Recompute the actual native source identity after cold fixture writes.
    fresh = native_model_metadata(model)
    built.metadata.update(fresh)


class _WorldBoundary:
    """Read-only task/predicate view into the owner's shared boundary snapshot."""
    physics_backend = 'gpu'

    def __init__(self, owner, world):
        self.owner, self.world = owner, world

    def read_fields(self, names):
        owner = self.owner
        owner._open()
        snapshot = owner._snapshot_cache
        if snapshot is None:
            owner._check()
            snapshot = owner._snapshot()
        return {name: snapshot[name][self.world] for name in names}

    def update_visual_parameters(self, changes):
        owner = self.owner
        owner._open()
        owner.observer.update_visual_parameters(changes, world=self.world, stream=owner.stream)

    def _unsupported(self, *args, **kwargs):
        from .facade import UnsupportedGpuOperation
        raise UnsupportedGpuOperation('mutate/advance the shared arena through GpuBatchEnv')

    write_fields = forward = step = _unsupported

    def close(self):
        self.owner = None


class GpuBatchEnv:
    """Explicit synchronous batch API, with leading environment axis.

    ``step(actions)`` advances all worlds together and returns
    ``(dict[str, ndarray], rewards, successes, infos)``. There is no auto-reset.
    ``reset(id=...)`` and ``set_init_state(..., id=...)`` can select worlds.
    Flat initialization states follow LIBERO's [time, qpos, qvel] convention.
    """
    physics_backend = 'gpu'
    is_async = False

    def __init__(self, bddl_file_name, num_envs=8, *, init_states=None,
                 seed=0, device='cuda:0', memory_budget=None,
                 strict_build_receipts=False, **environment_options):
        if type(num_envs) is not int or not 1 <= num_envs <= 128:
            raise ValueError('num_envs must be an integer in [1, 128]')
        from libero.gpu.host.runtime.core.memory_budget import normalize_memory_budget
        policy = normalize_memory_budget(memory_budget)
        if type(strict_build_receipts) is not bool:
            raise ValueError('strict_build_receipts must be bool')
        self._lock = RLock()
        self._closed = self._failed = False
        self._snapshot_cache = None
        self._physics = self.observer = self._status = None
        self._facades, self.shells = [], []
        self.num_envs = self.env_num = num_envs
        self._rngs = []
        self.seed(seed)
        from libero.gpu.model import build_task_model
        from .batched_physics import prepare_native_batch
        from libero.gpu.host.runtime.step.recovery_final_event_environment_factory import default_integration_state
        options = dict(environment_options)
        for key in ('physics_backend', 'nworld', 'state', 'stream'):
            if key in options:
                raise ValueError(f'{key} is controlled by GpuBatchEnv')
        if options.get('has_renderer', False) or options.get('hard_reset', False):
            raise ValueError('GpuBatchEnv requires offscreen rendering and fixed-model soft reset')
        options.setdefault('use_camera_obs', False)
        options.update(has_renderer=False, has_offscreen_renderer=bool(options['use_camera_obs']), hard_reset=False)
        self.built = build_task_model(bddl_file_name, **options)
        _prepare_shared_fixtures(self.built, seed)
        model = self.built.native_model
        if init_states is None:
            from libero.libero import get_libero_path
            from libero.libero.benchmark.init_state_io import load_init_states
            path = Path(bddl_file_name)
            init_states = load_init_states(Path(get_libero_path('init_states')) /
                                          path.parent.name / (path.stem + '.pruned_init'))
        self.initial_states = _state_bank(init_states, int(model.nq), int(model.nv))
        self._nq, self._nv = int(model.nq), int(model.nv)
        initial = self.initial_states[np.arange(num_envs) % len(self.initial_states)]
        states = []
        for row in initial:
            state = default_integration_state(model)
            state['time'][:] = row[0]
            state['qpos'][0] = row[1:1+self._nq]
            state['qvel'][0] = row[1+self._nq:]
            states.append(state)
        try:
            self._physics = prepare_native_batch(self.built, states=states,
                device=device, memory_budget=policy, strict_build_receipts=strict_build_receipts)
            self._assemble()
            self.reset(init_states=initial)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.gpu_cleanup_errors = [repr(cleanup)]
            raise

    def _assemble(self):
        import warp as wp
        from .facade import GpuSimFacade
        from .resident_runtime import fixed_root_descriptors
        from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
        from libero.gpu.host.adapters.observations.runtime_binding import FixedRuntimeObservationBinding
        from libero.gpu.host.runtime.step.recovery_status import RecoveryBoundaryStatus
        bundle = self._physics.bundle
        self.stream, self.runtime = bundle.stream, bundle.runtime
        self.action_dim = bundle.controller.action_dim
        self.control_freq = self.built.metadata['control_freq']
        self.substeps = self.runtime.substeps
        self.memory_budget = bundle.memory_budget
        self._fields = fixed_root_descriptors(bundle)
        self._status = RecoveryBoundaryStatus(bundle)
        self.observer = GpuObservationAdapter(self.built.native_model, self.built.metadata,
                                             self.num_envs, device=self.stream.device)
        self._binding = FixedRuntimeObservationBinding(bundle, self.observer,
                                                       expected_observe=self.runtime.observe).install()
        self._selection = wp.zeros(self.num_envs, dtype=wp.int32, device=self.stream.device)
        self._all = np.ones(self.num_envs, dtype=np.int32)
        template = self.built.domain_shell
        # Clone before installing any live CUDA/controller owners on a shell.
        self.shells = [template] + [_clone_shell(template) for _ in range(self.num_envs-1)]
        for world, shell in enumerate(self.shells):
            facade = GpuSimFacade(_WorldBoundary(self, world), self.built.facade_metadata())
            self._facades.append(facade)
            shell.sim, shell.physics_backend = facade, 'gpu'
            self.observer.bind_host_environment(shell, world=world)
            # Default task predicates use their own shell's object-state graph.
            # Keep CPU step/reset shims forbidden; this owner handles lifecycle.
        self._closed = False

    def _open(self):
        if self._closed:
            raise RuntimeError('GPU batch environment is closed')
        if self._failed:
            raise RuntimeError('GPU batch execution failed; close and rebuild required')

    def __len__(self):
        return self.num_envs

    def _ids(self, ids):
        if ids is None:
            return tuple(range(self.num_envs))
        if isinstance(ids, (int, np.integer)) and not isinstance(ids, bool):
            ids = [ids]
        if isinstance(ids, bool):
            raise ValueError('invalid world IDs')
        ids = tuple(ids)
        if (len(set(ids)) != len(ids) or any(isinstance(i, bool) or
                not isinstance(i, (int, np.integer)) or not 0 <= i < self.num_envs for i in ids)):
            raise ValueError('invalid or duplicate world IDs')
        return tuple(map(int, ids))

    def seed(self, seed=None):
        with self._lock:
            self._open()
            if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
                seeds = [int(seed)+i for i in range(self.num_envs)]
            elif seed is None:
                seeds = [None]*self.num_envs
            else:
                raise ValueError('seed must be an integer or None')
            self._rngs = [np.random.RandomState(s) for s in seeds]
            return seeds

    @contextmanager
    def _mutation(self):
        self._snapshot_cache = None
        try:
            yield
        except BaseException:
            self._failed = True
            raise
        finally:
            self._snapshot_cache = None

    def _check(self):
        record = self._status.synchronize_and_check()
        self._integrations = np.asarray([row[16] for row in record.rows], dtype=np.int64)
        return record

    def _snapshot(self):
        # One download per field for the entire batch, including contact pairs.
        from libero.gpu.host.adapters.observations.boundary import GOAL_FIELDS
        snapshot = {}
        for name in (*GOAL_FIELDS, 'time', 'qvel'):
            if name == 'contact':
                continue
            field = self._fields[name]
            snapshot[name] = field.array.numpy().reshape((self.num_envs,)+field.shape)
        transport = self.runtime.forward.transport
        pairs = {name: transport.inputs[name].numpy().reshape(self.num_envs, -1)
                 for name in ('geom1', 'geom2')}
        contacts = []
        for world, count in enumerate(snapshot['ncon']):
            count = int(count)
            if not 0 <= count <= pairs['geom1'].shape[1]:
                raise RuntimeError('contact count exceeds batch capacity')
            contacts.append({name: value[world, :count] for name, value in pairs.items()})
        snapshot['contact'] = contacts
        return snapshot

    def _observations(self, ids):
        values = self.observer.observations_many(ids,
            image_convention=self.built.metadata.get('image_convention', 'opengl'))
        if not values:
            return {}
        return {name: np.stack([obs[name] for obs in values]) for name in values[0]}

    def _activate_all(self):
        import warp as wp
        with wp.ScopedStream(self.stream):
            self.runtime.intent.assign(self._all)

    def step(self, actions):
        with self._lock:
            self._open()
            actions = np.asarray(actions, dtype=np.float64)
            if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
                raise ValueError(f'actions must have shape ({self.num_envs}, {self.action_dim}) and be finite')
            if any(shell.done for shell in self.shells):
                raise ValueError('reset terminated worlds before stepping the batch')
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            with self._mutation():
                before = self._integrations.copy()
                self._activate_all()
                self._physics.enqueue_action(np.ascontiguousarray(actions))
                self._check()
                if not np.all(self._integrations-before == self.substeps):
                    raise RuntimeError('GPU batch did not commit every requested physics substep')
                self._snapshot_cache = self._snapshot()
                rewards, successes, infos = [], [], []
                for world, (shell, facade) in enumerate(zip(self.shells, self._facades)):
                    shell.timestep += 1
                    shell.cur_time += shell.control_timestep
                    with boundary_predicates(shell, facade):
                        reward, _, info = shell._post_action(actions[world])
                        success = shell._check_success()
                    rewards.append(reward)
                    successes.append(success)
                    infos.append(dict(info, env_id=world))
                obs = self._observations(tuple(range(self.num_envs)))
                return obs, np.asarray(rewards), np.asarray(successes, dtype=bool), infos

    def reset(self, id=None, *, seed=None, init_states=None):
        with self._lock:
            self._open()
            ids = self._ids(id)
            if seed is not None:
                self.seed(seed)
            if init_states is not None:
                rows = self._validate_states(init_states, ids)
            else:
                rows = (np.stack([self.initial_states[self._rngs[i].randint(len(self.initial_states))]
                                  for i in ids]) if ids else np.empty((0, 1+self._nq+self._nv)))
            with self._mutation():
                self._restore(rows, ids, reset=True)
                for i in ids:
                    shell = self.shells[i]
                    shell.timestep, shell.cur_time, shell.done = 0, 0.0, False
                    shell._obs_cache = {}
                return self._observations(ids)

    def _validate_states(self, states, ids):
        rows = np.asarray(states, dtype=np.float64)
        if rows.shape != (len(ids), 1+self._nq+self._nv) or not np.isfinite(rows).all():
            raise ValueError('one finite LIBERO flat state required per selected world')
        return rows.copy()

    def set_init_state(self, states, id=None):
        with self._lock:
            self._open()
            ids = self._ids(id)
            rows = self._validate_states(states, ids)
            with self._mutation():
                self._restore(rows, ids, reset=False)
                return self._observations(ids)

    def _restore(self, rows, ids, *, reset):
        import warp as wp
        from libero.gpu.host.runtime.step.shape_environment import _written
        if not ids:
            return
        mask = np.zeros(self.num_envs, dtype=np.int32)
        mask[list(ids)] = 1
        e, bundle = self.runtime.euler, self._physics.bundle
        roots = {'time': e.time, 'qpos': e.qpos, 'qvel': e.qvel}
        payload = {'time': rows[:, 0], 'qpos': rows[:, 1:1+self._nq],
                   'qvel': rows[:, 1+self._nq:]}
        if reset:
            roots.update(ctrl=e.ctrl, qacc_warmstart=e.qacc_warmstart,
                         qfrc_applied=bundle.dynamics.qfrc_applied,
                         xfrc_applied=bundle.dynamics.xfrc_applied)
            for name in set(roots)-set(payload):
                payload[name] = np.zeros((len(ids), roots[name].size//self.num_envs))
        transfers = []
        with wp.ScopedStream(self.stream):
            self._selection.assign(mask)
            for name, array in roots.items():
                source = wp.array(np.ascontiguousarray(payload[name]).reshape(-1),
                                  dtype=wp.float64, device=self.stream.device)
                transfers.append(source)
                width = array.size//self.num_envs
                for row, world in enumerate(ids):
                    wp.copy(array, source, dest_offset=world*width, src_offset=row*width,
                            count=width, stream=self.stream)
            wp.launch(_written, self.num_envs, inputs=[self._selection, self.runtime.state_epoch,
                bundle.expected_state, bundle.foundation.state_valid,
                self.runtime.forward.valid, self.runtime.valid], device=self.stream.device, stream=self.stream)
            if reset:
                bundle.controller.initialize_ctrl_from_reset(self._selection, e.ctrl, stream=self.stream)
                self.observer.reset(self._selection, stream=self.stream)
            self.runtime.forward.enqueue(self._selection, self.stream)
            if reset:
                bundle.controller.bootstrap_from_current_physics(self._selection, stream=self.stream)
            record = self._check()
            if any(record.rows[i][18] != 1 for i in ids):
                raise RuntimeError('GPU batch reset did not publish every selected forward')
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            self._snapshot_cache = self._snapshot()
            try:
                for world in ids:
                    shell = self.shells[world]
                    with boundary_predicates(shell, self._facades[world]):
                        shell._check_success()
                        shell._post_process()
            finally:
                self._snapshot_cache = None
            self.observer.sample(self._binding.fields(), force=True,
                                 world_mask=self._selection, stream=self.stream)
            self._check()
        # Transfers remain alive until the synchronous restore boundary above.

    def get_sim_state(self):
        with self._lock:
            self._open()
            self._check()
            e = self.runtime.euler
            return np.concatenate((e.time.numpy().reshape(-1, 1),
                                   e.qpos.numpy().reshape(self.num_envs, self._nq),
                                   e.qvel.numpy().reshape(self.num_envs, self._nv)), axis=1)

    def diagnostics(self):
        with self._lock:
            self._open()
            self._check()
            return dict(physics_backend='gpu', nworld=self.runtime.euler.capacity,
                        model_sha256=self.built.metadata['model_sha256'],
                        substeps=self.substeps, memory_budget=self.memory_budget,
                        actual_integration_count=self.runtime.integration_count.numpy().tolist(),
                        first_error=self.runtime.first_error.numpy().tolist())

    def check_success(self):
        with self._lock:
            self._open()
            from libero.gpu.host.adapters.observations.boundary import boundary_predicates
            self._check()
            self._snapshot_cache = self._snapshot()
            try:
                result = []
                for shell, facade in zip(self.shells, self._facades):
                    with boundary_predicates(shell, facade):
                        result.append(shell._check_success())
                return np.asarray(result, dtype=bool)
            finally:
                self._snapshot_cache = None

    def close(self):
        with self._lock:
            if self._closed:
                return
            stream = getattr(self, 'stream', None)
            scope = nullcontext()
            if stream is not None:
                import warp as wp
                scope = wp.ScopedStream(stream)
            with scope:
                # Release graphics/facade borrows before the shared physics owner.
                for facade in self._facades:
                    facade.close()
                self._facades.clear()
                if self.observer is not None:
                    self.observer.close()
                if self._physics is not None:
                    self._physics.close()
                if self._status is not None:
                    self._status.release()
                self._snapshot_cache = None
                self.shells.clear()
                self._fields = {}
                self._binding = self.observer = self.runtime = self._physics = None
                self._selection = self._status = None
                import gc
                gc.collect()
                if stream is not None:
                    wp.synchronize_stream(stream)
                self._closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
