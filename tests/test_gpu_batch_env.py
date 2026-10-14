"""Native batch orchestration and state validation, with injected physics."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host/adapters/environment/batch_env.py'
spec = importlib.util.spec_from_file_location('_batch_api_tests', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@contextmanager
def predicates(*args):
    yield


class BatchEnvTests(unittest.TestCase):
    def test_shell_clone_preserves_raw_type_and_rebinds_task_state(self):
        class Task:
            def __new__(cls):
                raise AssertionError('public constructor must not run for a cold clone')
            def success(self):
                return self.state.env is self
        shell = object.__new__(Task)
        shell.sim = object()
        shell.state = SimpleNamespace(env=shell)
        shell.callback = shell.success
        clone = module._clone_shell(shell)
        self.assertIs(type(clone), Task)
        self.assertIs(clone.sim, shell.sim)
        self.assertIsNot(clone.state, shell.state)
        self.assertTrue(clone.callback())
        self.assertIs(clone.callback.__self__, clone)

    def environment(self):
        env = module.GpuBatchEnv.__new__(module.GpuBatchEnv)
        env.num_envs = env.env_num = 4
        env.action_dim = 7
        env.substeps = 2
        env._integrations = np.zeros(4, dtype=np.int64)
        env._lock = RLock()
        env._closed = env._failed = False
        env._snapshot_cache = None
        env._all = np.ones(4, dtype=np.int32)
        env._facades = [None]*4
        env.calls = []
        env.runtime = SimpleNamespace(intent=SimpleNamespace(assign=lambda x: env.calls.append('mask')))
        env._activate_all = lambda: env.runtime.intent.assign(env._all)
        env._physics = SimpleNamespace(enqueue_action=lambda x: env.calls.append(('batch', x.copy())))
        def complete():
            env.calls.append('check')
            env._integrations += env.substeps
        env._check = complete
        env._snapshot = lambda: {'qpos': np.zeros((4, 2))}
        env._observations = lambda ids: {'value': np.array(ids)}
        env.shells = []
        for i in range(4):
            shell = SimpleNamespace(timestep=0, cur_time=0., done=False, control_timestep=.05)
            shell._post_action = lambda a: (float(a[0]), False, {})
            shell._check_success = lambda i=i: i == 2
            env.shells.append(shell)
        return env

    def test_one_physics_enqueue_for_four_worlds_and_ordered_outputs(self):
        env = self.environment()
        actions = np.zeros((4, 7))
        actions[:, 0] = [1, 2, 3, 4]
        with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                    SimpleNamespace(boundary_predicates=predicates)}):
            obs, reward, success, infos = env.step(actions)
        self.assertEqual(sum(isinstance(x, tuple) for x in env.calls), 1)
        np.testing.assert_array_equal(env.calls[1][1], actions)
        np.testing.assert_array_equal(reward, [1, 2, 3, 4])
        np.testing.assert_array_equal(success, [False, False, True, False])
        self.assertEqual([i['env_id'] for i in infos], [0, 1, 2, 3])
        self.assertTrue(all(s.timestep == 1 and s.cur_time == .05 for s in env.shells))
        self.assertIsNone(env._snapshot_cache)

    def test_invalid_action_cannot_advance_any_world(self):
        env = self.environment()
        for actions in (np.zeros((3, 7)), np.full((4, 7), np.nan)):
            with self.assertRaises(ValueError):
                env.step(actions)
        self.assertFalse(env.calls)
        self.assertFalse(env._failed)

    def test_failed_batch_is_quarantined(self):
        env = self.environment()
        def fail():
            raise ValueError('actual stage error')
        env._check = fail
        with self.assertRaisesRegex(ValueError, 'stage error'):
            with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                        SimpleNamespace(boundary_predicates=predicates)}):
                env.step(np.zeros((4, 7)))
        with self.assertRaisesRegex(RuntimeError, 'rebuild'):
            env.step(np.zeros((4, 7)))

    def test_missing_world_commit_cannot_return_successful_observations(self):
        env = self.environment()
        def incomplete():
            env._integrations[:3] += env.substeps
        env._check = incomplete
        with patch.dict(sys.modules, {'libero.gpu.host.adapters.observations.boundary':
                                    SimpleNamespace(boundary_predicates=predicates)}):
            with self.assertRaisesRegex(RuntimeError, 'commit every'):
                env.step(np.zeros((4, 7)))
        self.assertTrue(env._failed)

    def test_reset_only_changes_selected_lifecycle_and_rng_is_reproducible(self):
        env = self.environment()
        env._nq = env._nv = 1
        env.initial_states = np.arange(30).reshape(10, 3).astype(float)
        writes = []
        env._restore = lambda rows, ids, reset: writes.append((rows.copy(), ids, reset))
        for shell in env.shells:
            shell.timestep, shell.cur_time, shell.done = 8, .4, True
        env.seed(9)
        env.reset(id=[3, 1])
        first = writes[-1][0].copy()
        env.seed(9)
        env.reset(id=[3, 1])
        np.testing.assert_array_equal(writes[-1][0], first)
        self.assertEqual(writes[-1][1], (3, 1))
        self.assertTrue(writes[-1][2])
        self.assertEqual([s.timestep for s in env.shells], [8, 0, 8, 0])
        with self.assertRaises(ValueError):
            env.set_init_state(np.zeros((2, 3)), id=[1, 1])

    def test_close_releases_shared_owner_once_and_rejects_future_use(self):
        env = self.environment()
        env._facades = []
        env.observer = SimpleNamespace(close=lambda: env.calls.append('observer-close'))
        env._physics.close = lambda: env.calls.append('physics-close')
        env._status = SimpleNamespace(release=lambda: env.calls.append('status-release'))
        env.close()
        env.close()
        self.assertEqual(env.calls, ['observer-close', 'physics-close', 'status-release'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            env.seed(0)


if __name__ == '__main__':
    unittest.main()
