"""Local vector-protocol qualification. No CUDA, renderer, native compilation."""
from pathlib import Path
import json
import multiprocessing
import os
import random
import sys
import subprocess
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1] / 'libero/libero/envs'
sys.path.insert(0, str(SOURCE))
import venv
import gym
import numpy as np


class FakeEnv:
    def __init__(self, index=0, modern=False, parent=None, fail=None):
        if fail == 'init':
            raise ValueError('constructor failure must be visible')
        self.index = index
        self.modern = modern
        self.parent = parent
        self.fail = fail
        self.owner = os.getpid()
        self.value = 0
        self.unwrapped = self
        self.observation_space = gym.spaces.Box(-100, 100, shape=(2,), dtype=np.float64)

    def reset(self, **kwargs):
        self.value = kwargs.get('seed', 0)
        obs = np.array([self.index, self.value], dtype=np.float64)
        return (obs, {'reset': True}) if self.modern else obs

    def step(self, action):
        if self.fail == 'step':
            raise ValueError('step failure must be visible')
        self.value += float(action[0])
        obs = np.array([self.index, self.value], dtype=np.float64)
        info = {'owner': self.owner, 'parent': self.parent}
        if self.modern:
            return obs, self.value, self.value > 5, False, info
        return obs, self.value, self.value > 5, info

    def seed(self, seed):
        self.value = seed
        return [seed]

    def set_init_state(self, state):
        self.value = float(state)
        return np.array([self.index, self.value], dtype=np.float64)

    def get_sim_state(self):
        return np.array([self.index, self.value], dtype=np.float64)

    def get_segmentation_of_interest(self, image):
        return image + self.index

    def check_success(self):
        return self.value > 5

    def render(self, **kwargs):
        return np.full((2, 2, 3), self.index, dtype=np.uint8)

    def close(self):
        pass


def must_fail(fn, text):
    try:
        fn()
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('Missing worker failure')


def exercise(cls, modern=False):
    parent = os.getpid()
    env = cls([lambda i=i: FakeEnv(i, modern=modern, parent=parent) for i in range(2)])
    try:
        assert len(env) == 2
        reset = env.reset(seed=2)
        obs = reset[0] if modern else reset
        np.testing.assert_array_equal(obs, [[0, 2], [1, 2]])
        if modern:
            assert reset[1] == [{'reset': True}] * 2
        assert env.seed(7) == [[7], [8]]
        restored = env.set_init_state([3, 4])
        np.testing.assert_array_equal(restored, [[0, 3], [1, 4]])
        np.testing.assert_array_equal(env.set_init_state([9], id=[1]), [[1, 9]])
        env.set_env_attr('value', 1, id=0)
        assert env.get_env_attr('value') == [1, 9]
        result = env.step(np.array([[2], [3]], dtype=np.float64))
        assert len(result) == (5 if modern else 4)
        np.testing.assert_array_equal(result[0], [[0, 3], [1, 12]])
        np.testing.assert_array_equal(result[1], [3, 12])
        assert [i['env_id'] for i in result[-1]] == [0, 1]
        owners = [i['owner'] for i in result[-1]]
        if cls is venv.SubprocVectorEnv:
            assert parent not in owners and len(set(owners)) == 2
            assert all(w.process._start_method == 'subprocess' for w in env.workers)
        else:
            assert owners == [parent, parent]
        assert env.check_success() == [False, True]
        np.testing.assert_array_equal(env.get_sim_state(), result[0])
        assert len(env.render()) == 2
        np.testing.assert_array_equal(env.get_segmentation_of_interest(np.zeros((2, 2))), [[0, 0], [1, 1]])
    finally:
        env.close()
    if cls is venv.SubprocVectorEnv:
        assert all(not w.process.is_alive() for w in env.workers)


def test_vector_protocol():
    start = time.perf_counter()
    # Deliberately choose fork globally: the wrapper must still spawn locally.
    multiprocessing.set_start_method('fork', force=True)
    for cls in (venv.DummyVectorEnv, venv.SubprocVectorEnv):
        for modern in (False, True):
            exercise(cls, modern)
    env = venv.SubprocVectorEnv([lambda i=i: FakeEnv(i) for i in range(2)], wait_num=1)
    try:
        env.reset()
        env.step(np.ones((2, 1)))
        while env.waiting_id:
            env.step(None)
        assert sorted(env.ready_id) == [0, 1]
    finally:
        env.close()
    before = {p.pid for p in multiprocessing.active_children()}
    must_fail(lambda: venv.SubprocVectorEnv([lambda: FakeEnv(), lambda: FakeEnv(fail='init')]), 'constructor failure')
    assert {p.pid for p in multiprocessing.active_children()} == before
    env = venv.SubprocVectorEnv([lambda: FakeEnv(fail='step')])
    must_fail(lambda: env.step(np.zeros((1, 1))), 'step failure')
    env.close()
    assert not env.workers[0].process.is_alive()
    # Exercise the existing internal shared-memory worker without a parent env.
    worker = venv.SubprocEnvWorker(lambda: FakeEnv(index=1), share_memory=True)
    try:
        np.testing.assert_array_equal(worker.reset(seed=4), [1, 4])
        np.testing.assert_array_equal(worker.set_init_state(9), [1, 9])
        retained = worker.step(np.array([1]))[0]
        np.testing.assert_array_equal(retained, [1, 10])
    finally:
        worker.close()
    del worker
    np.testing.assert_array_equal(retained, [1, 10])
    assert not multiprocessing.active_children()
    result = {'passed': True, 'cases': ['legacy and modern Dummy/Subproc API', 'lambda cloudpickle', 'subset restoration', 'attributes', 'seed', 'render', 'segmentation', 'owner PID isolation', 'global fork override', 'async', 'constructor errors and partial cleanup', 'step errors', 'shared-memory reset and restore'], 'wall_seconds': time.perf_counter() - start, 'gpu_tested': False, 'whole_application_tested': False}
    print(json.dumps(result))


def test_top_level_factory_without_main_guard():
    """The original top-level lambda call must not import/re-execute its script."""
    with tempfile.TemporaryDirectory(prefix='libero-vector-api-') as temporary:
        directory = Path(temporary)
        marker = directory/'parent-runs.txt'
        script = directory/'original_api.py'
        script.write_text(
            'import os, sys\n'
            'from pathlib import Path\n'
            'import numpy as np\n'
            f'sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n'
            'from test_gpu_vector_protocol import FakeEnv, venv\n'
            f'marker=Path({str(marker)!r})\n'
            'with marker.open("a") as stream: stream.write(str(os.getpid())+"\\n")\n'
            'env=venv.SubprocVectorEnv([lambda: FakeEnv()])\n'
            'try:\n'
            '    np.testing.assert_array_equal(env.reset(seed=3), [[0, 3]])\n'
            '    result=env.step(np.array([[2]]))\n'
            '    np.testing.assert_array_equal(result[0], [[0, 5]])\n'
            '    assert result[-1][0]["owner"] != os.getpid()\n'
            'finally: env.close()\n'
            'assert marker.read_text().splitlines()==[str(os.getpid())]\n'
        )
        completed=subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr


class CountingEnv(FakeEnv):
    def __init__(self, index, marker, fail=None):
        super().__init__(index=index, fail=fail)
        self.marker = marker
        self.instance_token = id(self)
        self.storage = np.zeros(7)
        self.rng = np.random.RandomState(index)
        with Path(marker).open('a') as stream:
            stream.write(f'init {index} {os.getpid()}\n')

    def close(self):
        with Path(self.marker).open('a') as stream:
            stream.write(f'close {self.index} {os.getpid()}\n')


def test_shared_worker_group_twenty_slots():
    with tempfile.TemporaryDirectory(prefix='libero-vector-group-') as temporary:
        marker = str(Path(temporary)/'owners.txt')
        env = venv.SubprocVectorEnv([lambda i=i: CountingEnv(i, marker) for i in range(20)])
        process = env.workers[0].process
        try:
            assert all(worker.process is process for worker in env.workers)
            assert len(set(env.get_env_attr('owner'))) == 1
            assert env.get_env_attr('owner')[0] != os.getpid()
            assert len(set(env.get_env_attr('instance_token'))) == 20
            obs = env.reset(seed=3)
            np.testing.assert_array_equal(obs, [[i, 3] for i in range(20)])
            assert env.seed(11) == [[11+i] for i in range(20)]
            restored = env.set_init_state(np.arange(20))
            np.testing.assert_array_equal(restored, [[i, i] for i in range(20)])
            env.set_env_attr('value', 99, id=8)
            assert env.get_env_attr('value', id=[7,8,9]) == [7., 99, 9.]
            result = env.step(np.ones((20, 1)))
            assert [info['env_id'] for info in result[-1]] == list(range(20))
            np.testing.assert_array_equal(result[0][[7,8,9]], [[7,8], [8,100], [9,10]])
            env.workers[0].close()
            assert process.is_alive()
            np.testing.assert_array_equal(env.reset(id=[19], seed=6), [[19,6]])
        finally:
            env.close()
        assert not process.is_alive()
        records = Path(marker).read_text().splitlines()
        assert len(records) == 40
        assert sorted(int(line.split()[1]) for line in records if line.startswith('init ')) == list(range(20))
        assert sorted(int(line.split()[1]) for line in records if line.startswith('close ')) == list(range(20))


def test_group_constructor_failure_cleanup_and_slot_failure_isolation():
    with tempfile.TemporaryDirectory(prefix='libero-vector-failed-') as temporary:
        marker = str(Path(temporary)/'owners.txt')
        must_fail(lambda: venv.SubprocVectorEnv([lambda i=i: CountingEnv(i, marker, fail='init' if i==2 else None) for i in range(4)]), 'constructor failure')
        records = Path(marker).read_text().splitlines()
        assert sorted(line.split()[0:2] for line in records) == [['close','0'],['close','1'],['init','0'],['init','1']]
        env = venv.SubprocVectorEnv([lambda: FakeEnv(fail='step'), lambda: FakeEnv(index=1), lambda: FakeEnv(index=2)])
        process = env.workers[0].process
        try:
            must_fail(lambda: env.step(np.ones((1,1)), id=0), 'step failure')
            np.testing.assert_array_equal(env.reset(id=[1], seed=4), [[1,4]])
            np.testing.assert_array_equal(env.step(np.ones((1,1)), id=1)[0], [[1,5]])
            assert process.is_alive()
        finally:
            env.close()
        assert not process.is_alive()


def test_shared_group_buffers_and_async_protocol():
    factories = [lambda i=i: FakeEnv(i) for i in range(2)]
    group = venv._SubprocWorkerGroup(factories, share_memory=True)
    workers = [group.worker(i, factory) for i, factory in enumerate(factories)]
    try:
        for i, worker in enumerate(workers):
            np.testing.assert_array_equal(worker.reset(seed=3), [i,3])
            np.testing.assert_array_equal(worker.set_init_state(9), [i,9])
        retained = workers[1].step(np.array([1]))[0]
    finally:
        for worker in workers:
            worker.close()
    np.testing.assert_array_equal(retained, [1,10])
    assert not group.process.is_alive()
    env = venv.SubprocVectorEnv(factories + [lambda: FakeEnv(2)], wait_num=1, timeout=.5)
    try:
        env.reset()
        initial = env.step(np.ones((3,1)))
        seen = [info['env_id'] for info in initial[-1]]
        while env.waiting_id:
            seen.extend(info['env_id'] for info in env.step(None)[-1])
        assert sorted(seen) == [0,1,2]
    finally:
        env.close()


class GlobalRngEnv(FakeEnv):
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        return np.array([np.random.random(), random.random()]), 0., False, {}


def test_group_preserves_per_world_global_rng():
    env = venv.SubprocVectorEnv([lambda: GlobalRngEnv() for _ in range(3)])
    try:
        env.seed(5)
        n = [np.random.RandomState(i) for i in range(5,8)]
        p = [random.Random(i) for i in range(5,8)]
        for _ in range(3):
            expected = [[n[i].random(), p[i].random()] for i in range(3)]
            np.testing.assert_array_equal(env.step(np.zeros((3,1)))[0], expected)
        env.seed([8,8,8])
        result = env.step(np.zeros((3,1)))[0]
        np.testing.assert_array_equal(result[0], result[1])
    finally:
        env.close()


if __name__ == '__main__':
    test_vector_protocol()
    test_top_level_factory_without_main_guard()
    test_shared_worker_group_twenty_slots()
    test_group_constructor_failure_cleanup_and_slot_failure_isolation()
    test_shared_group_buffers_and_async_protocol()
    test_group_preserves_per_world_global_rng()
