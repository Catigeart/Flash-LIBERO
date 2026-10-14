"""Explicit CPU-reference or GPU task entry for training and policy evaluation.

No policy/checkpoint transformations live here. GPU assembly remains an explicit
provider, with its own model admission; registration does not grant acceptance.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib
import numpy as np


def import_callable(reference):
    module, separator, name = reference.partition(':')
    if not separator or not module or not name:
        raise ValueError('expected an explicit module:callable reference')
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError('provider is not callable: ' + reference)
    return value


@dataclass(frozen=True)
class TaskSelection:
    benchmark: str
    task_id: int
    name: str
    language: str
    problem_folder: str
    bddl_path: str
    initial_states_path: str


def resolve_task(benchmark_name, task_id, task_order_index=0):
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    benchmark = get_benchmark(benchmark_name)(task_order_index)
    if not 0 <= task_id < benchmark.get_num_tasks():
        raise ValueError('task_id outside the actual benchmark task registry')
    task = benchmark.get_task(task_id)
    return TaskSelection(benchmark_name, task_id, task.name, task.language,
                         task.problem_folder,
                         str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                         str(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file))


def load_task_states(selection):
    from libero.libero.benchmark.init_state_io import load_init_states
    states = np.asarray(load_init_states(selection.initial_states_path))
    if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
        raise ValueError('official initialization states must be a finite nonempty matrix')
    return states


def make_task_environment(selection, *, backend, runtime_factory=None,
                          environment_options=None, constructor=None,
                          install_gpu=None):
    """Training callers may use this directly; no policy update is performed.

    constructor/install_gpu injection is a CPU protocol-test seam, never an
    implicit fallback. Normal execution uses the real original ControlEnv API.
    """
    if backend not in ('cpu', 'gpu'):
        raise ValueError('backend must be cpu or gpu')
    options = dict(environment_options or {})
    if 'physics_backend' in options or 'bddl_file_name' in options:
        raise ValueError('backend and task identity are explicit arguments')
    if backend == 'gpu':
        if not callable(runtime_factory):
            raise ValueError('GPU backend requires an explicit actual runtime factory')
        if install_gpu is None:
            from libero.libero.envs.bddl_utils import get_problem_info
            from libero.gpu.host.adapters.environment.factory import install_model_only_gpu_factory
            problem = get_problem_info(selection.bddl_path)['problem_name']
            install_model_only_gpu_factory(problem, runtime_factory, replace=True)
        else:
            install_gpu(selection, runtime_factory)
    elif runtime_factory is not None:
        raise ValueError('CPU reference must not receive a GPU runtime provider')
    if constructor is None:
        from libero.libero.envs.env_wrapper import ControlEnv
        constructor = ControlEnv
    env = constructor(bddl_file_name=selection.bddl_path,
                      physics_backend=backend, **options)
    try:
        if getattr(env, 'physics_backend', None) != backend:
            raise RuntimeError('constructed environment backend differs from request')
        for name in ('seed', 'reset', 'set_init_state', 'step', 'close'):
            if not callable(getattr(env, name, None)):
                raise TypeError('environment lacks original API: ' + name)
    except BaseException as error:
        try:
            env.close()
        except BaseException as close_error:
            error.backend_cleanup_error = repr(close_error)
        raise
    return env


def evaluate_episode(env, policy, selection, initial_state, *, seed, max_steps,
                     settle_steps=5):
    """Original success-done semantics, bounded rollout, five default settle calls.

    Policy owns tensor preprocessing and recurrent state. It receives the actual
    observation dictionary unchanged. No policy inference is supplied by physics.
    """
    if not callable(policy) or max_steps < 1 or settle_steps < 0:
        raise ValueError('callable policy and positive rollout length required')
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(initial_state)
    reset_policy = getattr(policy, 'reset', None)
    if callable(reset_policy):
        reset_policy()
    task = getattr(env, 'env', env)
    dimension = getattr(task, 'action_dim', None)
    if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
        raise TypeError('actual environment must declare its action dimension')
    for _ in range(settle_steps):
        env.step(np.zeros(dimension, dtype=np.float64))
    # Preserve original evaluation's observation after set_init_state during
    # settle; the first policy input is not silently changed by this adapter.
    success = False
    reward = None
    info = None
    for step in range(max_steps):
        action = np.asarray(policy(observation, selection, step), dtype=np.float64)
        if action.shape != (dimension,) or not np.isfinite(action).all():
            raise ValueError('policy returned invalid actual action shape or nonfinite values')
        observation, reward, done, info = env.step(action)
        if np.ndim(done) != 0:
            raise ValueError('single-world evaluation requires scalar success-done')
        success = success or bool(done)
        if success:
            break
    return {'success': success, 'policy_steps': step + 1, 'seed': seed,
            'settle_steps': settle_steps, 'last_reward': float(reward),
            'backend': env.physics_backend}


def close_preserving_failure(env, error=None):
    try:
        env.close()
    except BaseException as close_error:
        if error is None:
            raise
        error.backend_cleanup_error = repr(close_error)
