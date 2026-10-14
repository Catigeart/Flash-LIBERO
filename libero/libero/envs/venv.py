import cloudpickle
import gym
import numpy as np
import warnings
import time
import traceback
import os
import random
import subprocess
import sys
from pathlib import Path

from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import connection, get_context, resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, List, Optional, Tuple, Union


gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)

# Environments create their own CUDA and renderer owners in the child.
# Never inherit a parent CUDA context or execute the parent main module.
_SUBPROC_CONTEXT = get_context("spawn")
_WORKER_READY = "libero.worker.ready"
_WORKER_FAILURE = "libero.worker.failure"



def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


################################################################################
#
# Workers
#
################################################################################


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            Tuple[np.ndarray, dict],
            np.ndarray,
        ]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """Send action signal to low-level worker.

        When action is None, it indicates sending "reset" signal; otherwise
        it indicates "step" signal. The paired return value from "recv"
        function is determined by such kind of different signal.
        """
        if hasattr(self, "send_action"):
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """Receive result from low-level worker.

        If the last "send" function sends a NULL action, it only returns a
        single observation; otherwise it returns a tuple of (obs, rew, done,
        info) or (obs, rew, terminated, truncated, info), based on whether
        the environment is using the old step API or the new one.
        """
        if hasattr(self, "get_result"):
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Perform one timestep of the environment's dynamic.

        "send" and "recv" are coupled in sync simulation, so users only call
        "step" function. But they can be called separately in async
        simulation, i.e. someone calls "send" first, and calls "recv" later.
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


class _IndependentProcess:
    """Process-compatible lifecycle over a fresh interpreter, without spawn's
    re-import of the user script. Lambdas and local factories use cloudpickle.
    """
    _start_method = "subprocess"

    def __init__(self, child_remote, env_fn, share_memory):
        grouped = isinstance(child_remote, (tuple, list))
        self._child_remotes = tuple(child_remote) if grouped else (child_remote,)
        self._factories = tuple(env_fn) if grouped else (env_fn,)
        self._share_memory = share_memory
        self._process = None

    @property
    def pid(self):
        return None if self._process is None else self._process.pid

    @property
    def exitcode(self):
        return None if self._process is None else self._process.poll()

    def start(self):
        if os.name != "posix":
            raise RuntimeError("GPU subprocess environments require Linux or WSL")
        if self._process is not None:
            raise RuntimeError("environment worker already started")
        payload = cloudpickle.dumps((__name__, list(sys.path), cloudpickle.dumps(self._factories), self._share_memory))
        launcher = Path(__file__).with_name("_vector_worker.py")
        fds = tuple(remote.fileno() for remote in self._child_remotes)
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(launcher), ",".join(map(str, fds))],
            stdin=subprocess.PIPE, pass_fds=fds, close_fds=True,
        )
        try:
            self._process.stdin.write(payload)
            self._process.stdin.close()
        except BaseException:
            self._process.terminate()
            self._process.wait(timeout=5)
            raise
        finally:
            self._factories = None

    def is_alive(self):
        return self._process is not None and self._process.poll() is None

    def join(self, timeout=None):
        if self._process is not None:
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    def terminate(self):
        if self.is_alive():
            self._process.terminate()

    def kill(self):
        if self.is_alive():
            self._process.kill()



class _SharedObservation(np.ndarray):
    """Keep the local buffer owner alive for a returned shared-memory view."""

    def __array_finalize__(self, source):
        self._storage_owner = getattr(source, "_storage_owner", None)


class ShArray:
    """Observation buffer attachable after a spawned child creates its env."""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.dtype = np.dtype(dtype)
        self.shape = shape
        size = int(np.prod(shape)) * self.dtype.itemsize
        self.arr = SharedMemory(create=True, size=max(1, size))
        self._owns_segment = True
        self._unlinked = False

    def __getstate__(self):
        return self.arr.name, self.dtype, self.shape

    def __setstate__(self, state):
        name, self.dtype, self.shape = state
        self.arr = SharedMemory(name=name)
        # This is a separately launched interpreter with its own tracker.
        # Only the allocating parent may unlink the segment.
        resource_tracker.unregister(self.arr._name, "shared_memory")
        self._owns_segment = False
        self._unlinked = False

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        np.copyto(self.get(), ndarray)

    def get(self) -> np.ndarray:
        observation = np.ndarray(self.shape, dtype=self.dtype, buffer=self.arr.buf).view(_SharedObservation)
        observation._storage_owner = self
        return observation

    def close(self) -> None:
        if self._owns_segment:
            if not self._unlinked:
                self.arr.unlink()
                self._unlinked = True
            # Returned views retain this wrapper, so the map must live until
            # the final view/worker drops its owner (the original Array contract).
        else:
            self.arr.close()


def _close_buf(buffer) -> None:
    if isinstance(buffer, ShArray):
        buffer.close()
    elif isinstance(buffer, dict):
        for child in buffer.values():
            _close_buf(child)
    elif isinstance(buffer, tuple):
        for child in buffer:
            _close_buf(child)


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)  # type: ignore


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    share_memory: bool = False,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    if parent is not None:
        parent.close()
    env = None
    obs_bufs = None
    try:
        env = env_fn_wrapper.data()
        # This handshake also reports constructor failures before any request.
        p.send((_WORKER_READY, env.observation_space if share_memory else None))
        if share_memory:
            obs_bufs = p.recv()
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                result = env.close()
                env = None
                p.send(result)
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
                p.send(None)
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send(obs)
            else:
                p.close()
                raise NotImplementedError
    except BaseException as exc:
        try:
            p.send((_WORKER_FAILURE, type(exc).__name__, str(exc), traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        _close_buf(obs_bufs)
        p.close()


def _group_worker(remotes, factories, share_memory=False):
    """One fresh interpreter/context, with independently constructed env slots.

    Each pipe retains the original request/response and async wait protocol.
    A constructor failure releases the complete partially built group.
    A later slot failure is reported only to that slot and releases its env.
    """
    environments = []
    rng_states = []
    buffers = [None] * len(remotes)

    def activate_rng(index):
        np.random.set_state(rng_states[index][0])
        random.setstate(rng_states[index][1])

    def capture_rng(index):
        rng_states[index] = (np.random.get_state(), random.getstate())

    def encode(obs, buffer):
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for value, target in zip(obs, buffer):
                encode(value, target)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for key in obs:
                encode(obs[key], buffer[key])

    def error(remote, exc):
        try:
            remote.send((_WORKER_FAILURE, type(exc).__name__, str(exc), traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass

    def release(index):
        if environments[index] is not None:
            env, environments[index] = environments[index], None
            activate_rng(index)
            try:
                env.close()
            except BaseException:
                pass
            finally:
                capture_rng(index)
        _close_buf(buffers[index])
        buffers[index] = None
        remotes[index].close()

    try:
        try:
            for factory in factories:
                # Match independent fresh interpreters before any explicit seed.
                np.random.seed(None)
                random.seed(None)
                environments.append(factory())
                rng_states.append((np.random.get_state(), random.getstate()))
            # Publish readiness only after every owner was created successfully.
            for remote, env in zip(remotes, environments):
                remote.send((_WORKER_READY, env.observation_space if share_memory else None))
            if share_memory:
                for index, remote in enumerate(remotes):
                    buffers[index] = remote.recv()
        except BaseException as exc:
            # Finish owner cleanup before the first startup failure reaches the
            # parent; it may release/terminate the failed group immediately.
            for index, env in enumerate(environments):
                environments[index] = None
                activate_rng(index)
                try:
                    env.close()
                except BaseException:
                    pass
                finally:
                    capture_rng(index)
            for remote in remotes:
                error(remote, exc)
            return
        active = {remote: index for index, remote in enumerate(remotes)}
        while active:
            for remote in connection.wait(list(active)):
                index = active[remote]
                env, buffer = environments[index], buffers[index]
                try:
                    cmd, data = remote.recv()
                except EOFError:
                    release(index)
                    del active[remote]
                    continue
                activate_rng(index)
                try:
                    if cmd == "step":
                        result = env.step(data)
                        if buffer is not None:
                            encode(result[0], buffer)
                            result = (None, *result[1:])
                        remote.send(result)
                    elif cmd == "reset":
                        result = env.reset(**data)
                        with_info = isinstance(result, (tuple, list)) and len(result) == 2 and isinstance(result[1], dict)
                        obs = result[0] if with_info else result
                        if buffer is not None:
                            encode(obs, buffer)
                            obs = None
                        remote.send((obs, result[1]) if with_info else obs)
                    elif cmd == "close":
                        result = env.close()
                        environments[index] = None
                        remote.send(result)
                        release(index)
                        del active[remote]
                    elif cmd == "render":
                        remote.send(env.render(**data) if hasattr(env, "render") else None)
                    elif cmd == "seed":
                        if hasattr(env, "seed"):
                            remote.send(env.seed(data))
                        else:
                            env.reset(seed=data)
                            remote.send(None)
                    elif cmd == "getattr":
                        remote.send(getattr(env, data) if hasattr(env, data) else None)
                    elif cmd == "setattr":
                        setattr(env.unwrapped, data["key"], data["value"])
                        remote.send(None)
                    elif cmd == "check_success":
                        remote.send(env.check_success())
                    elif cmd == "get_segmentation_of_interest":
                        remote.send(env.get_segmentation_of_interest(data))
                    elif cmd == "get_sim_state":
                        remote.send(env.get_sim_state())
                    elif cmd == "set_init_state":
                        obs = env.set_init_state(data)
                        if buffer is not None:
                            encode(obs, buffer)
                            obs = None
                        remote.send(obs)
                    else:
                        raise NotImplementedError(f"Unknown LIBERO worker command: {cmd}")
                except BaseException as exc:
                    error(remote, exc)
                    release(index)
                    del active[remote]
                finally:
                    capture_rng(index)
    finally:
        for index in range(len(environments)):
            release(index)
        for remote in remotes:
            remote.close()


class _SubprocWorkerGroup:
    """Parent lifecycle for slots sharing a single fresh interpreter/context."""
    def __init__(self, factories, share_memory=False):
        self.share_memory = share_memory
        pairs = [_SUBPROC_CONTEXT.Pipe() for _ in factories]
        self.parent_remotes = [pair[0] for pair in pairs]
        self.child_remotes = [pair[1] for pair in pairs]
        self.buffers = [None] * len(pairs)
        self._live = set(range(len(pairs)))
        self.process = _IndependentProcess(self.child_remotes, factories, share_memory)
        try:
            self.process.start()
            for remote in self.child_remotes:
                remote.close()
            for index, remote in enumerate(self.parent_remotes):
                try:
                    response = remote.recv()
                except (EOFError, ConnectionResetError) as exc:
                    raise RuntimeError(f"LIBERO environment group exited during initialization (pid={self.process.pid})") from exc
                if isinstance(response, tuple) and len(response) == 4 and response[0] == _WORKER_FAILURE:
                    _, name, message, remote_traceback = response
                    raise RuntimeError(f"LIBERO environment group {name}: {message}\n{remote_traceback}")
                ready, space = response
                if ready != _WORKER_READY:
                    raise RuntimeError("Invalid LIBERO group initialization response")
                if share_memory:
                    self.buffers[index] = _setup_buf(space)
                    remote.send(self.buffers[index])
        except BaseException:
            self.close()
            raise

    def release(self, index):
        self._live.discard(index)
        _close_buf(self.buffers[index])
        self.parent_remotes[index].close()
        if not self._live:
            self.process.join(timeout=5)
            self.close()

    def close(self):
        # Closing parent endpoints lets the broker close every remaining owner.
        for remote in self.parent_remotes:
            remote.close()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        for remote in self.parent_remotes + self.child_remotes:
            remote.close()
        for buffer in self.buffers:
            _close_buf(buffer)
        self._live.clear()

    def worker(self, index, factory):
        result = SubprocEnvWorker.__new__(SubprocEnvWorker)
        result.parent_remote = self.parent_remotes[index]
        result.child_remote = self.child_remotes[index]
        result.share_memory = self.share_memory
        result.buffer = self.buffers[index]
        result.process = self.process
        result._group = self
        result._group_index = index
        EnvWorker.__init__(result, factory)
        return result


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()

    def check_success(self):
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self._group = None
        self.parent_remote, self.child_remote = _SUBPROC_CONTEXT.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        self.process = _IndependentProcess(self.child_remote, env_fn, share_memory)
        try:
            self.process.start()
            self.child_remote.close()
            ready, obs_space = self._recv()
            if ready != _WORKER_READY:
                raise RuntimeError("Invalid LIBERO worker initialization response")
            if share_memory:
                self.buffer = _setup_buf(obs_space)
                self.parent_remote.send(self.buffer)
        except BaseException:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=5)
            _close_buf(self.buffer)
            self.parent_remote.close()
            self.child_remote.close()
            raise
        super().__init__(env_fn)

    def _recv(self) -> Any:
        try:
            result = self.parent_remote.recv()
        except (EOFError, ConnectionResetError) as exc:
            raise RuntimeError(
                f"LIBERO environment worker exited unexpectedly (pid={self.process.pid})"
            ) from exc
        if (isinstance(result, tuple) and len(result) == 4
                and isinstance(result[0], str) and result[0] == _WORKER_FAILURE):
            _, name, message, remote_traceback = result
            raise RuntimeError(
                f"LIBERO environment worker {name}: {message}\n{remote_traceback}"
            )
        return result

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self._recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])
        self._recv()

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        result = self._recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self._recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self._recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self._recv()

    def close_env(self) -> None:
        try:
            if self.process.is_alive():
                self.parent_remote.send(["close", None])
                if self.parent_remote.poll(30):
                    self._recv()
                if self._group is None:
                    self.process.join(timeout=5)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            if self._group is not None:
                self._group.release(self._group_index)
            else:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=5)
                _close_buf(self.buffer)
                self.parent_remote.close()

    def check_success(self):
        self.parent_remote.send(["check_success", None])
        return self._recv()

    def get_segmentation_of_interest(self, segmentation_image):
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self._recv()

    def get_sim_state(self):
        self.parent_remote.send(["get_sim_state", None])
        return self._recv()

    def set_init_state(self, init_state):
        self.parent_remote.send(["set_init_state", init_state])
        obs = self._recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    """Base class for vectorized environments.

    Usage:
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments

    .. warning::

        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::

            def seed(self, seed):
                np.random.seed(seed)

        Otherwise, the outputs of these envs may be the same with each other.

    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the i-th env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param int wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param float timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = []
        try:
            for fn in env_fns:
                self.workers.append(worker_fn(fn))
        except BaseException:
            for worker in self.workers:
                try:
                    worker.close()
                except BaseException:
                    pass
            raise
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, **kwargs)
        ret_list = [self.workers[i].recv() for i in id]

        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )
        if reset_returns_info:
            obs_list = [r[0] for r in ret_list]
        else:
            obs_list = ret_list

        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        if reset_returns_info:
            infos = [r[1] for r in ret_list]
            return obs, infos  # type: ignore
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            or:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            The case distinction is made based on whether the underlying environment
            uses the old step API (first case) or the new step API (second case).

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        return_lists = tuple(zip(*result))
        obs_list = return_lists[0]
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # type: ignore

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        error = None
        for w in self.workers:
            try:
                w.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        self.is_closed = True
        if error is not None:
            raise error


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        # Small world counts retain the faster safe independent-process route.
        # Larger groups share a process/context to avoid repeated GPU contexts;
        # this is a resource-compatibility selection, not a throughput claim.
        if len(env_fns) <= 2:
            def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
                return SubprocEnvWorker(fn, share_memory=False)
            super().__init__(env_fns, worker_fn, **kwargs)
            return

        # Every factory is constructed once, in one fresh process. The public
        # per-slot pipes and wait semantics remain unchanged; GPU owners remain
        # independent while sharing the process CUDA context and loaded modules.
        group = _SubprocWorkerGroup(env_fns)
        indices = iter(range(len(env_fns)))

        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return group.worker(next(indices), fn)

        try:
            super().__init__(env_fns, worker_fn, **kwargs)
        except BaseException:
            group.close()
            raise

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs
