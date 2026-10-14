"""Shared-driver vector environment; never spawns CPU physics workers."""
import numpy as np
from .facade import UnsupportedGpuOperation


class GpuVectorEnv:
    """A driver batches worlds, internally grouping compatible model families.

    Results are returned in requested ID order. Capacity failures must be explicit;
    this wrapper never truncates contacts or substitutes sequential CPU env steps.
    The driver owns allocations/render resources and must close idempotently.
    """
    physics_backend = "gpu"

    def __init__(self, driver, num_envs):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if getattr(driver, "physics_backend", None) != "gpu":
            raise UnsupportedGpuOperation("shared vector driver must use GPU physics")
        for name in ("step_worlds", "reset_worlds", "set_init_states", "close"):
            if not callable(getattr(driver, name, None)):
                raise UnsupportedGpuOperation(f"vector driver lacks {name}")
        self.driver, self.num_envs, self._closed = driver, num_envs, False
        self.env_num = num_envs
        self.is_async = False

    def __len__(self): return self.num_envs

    def _ids(self, ids):
        if self._closed:
            raise RuntimeError("GPU vector environment is closed")
        if isinstance(ids, (int, np.integer)) and not isinstance(ids, bool):
            ids = [ids]
        if isinstance(ids, bool):
            raise ValueError("invalid world IDs")
        ids = list(range(self.num_envs)) if ids is None else list(ids)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("invalid world IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate world IDs")
        return tuple(int(i) for i in ids)

    def step(self, actions, id=None):
        ids = self._ids(id)
        actions = list(actions)
        if len(actions) != len(ids):
            raise ValueError("one action is required per selected world")
        return self.driver.step_worlds(ids, actions)

    def reset(self, id=None):
        return self.driver.reset_worlds(self._ids(id))

    def set_init_state(self, states, id=None):
        ids = self._ids(id)
        states = list(states)
        if len(states) != len(ids):
            raise ValueError("one initial state is required per selected world")
        return self.driver.set_init_states(ids, states)

    def _dispatch(self, operation, *args, **kwargs):
        self._ids(None)
        method = getattr(self.driver, operation, None)
        if not callable(method):
            raise UnsupportedGpuOperation(f"shared vector driver lacks {operation}")
        return method(*args, **kwargs)

    def seed(self, seed=None):
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
            seeds = [int(seed) + i for i in range(self.num_envs)]
        else:
            if isinstance(seed, bool): raise ValueError("invalid seed")
            seeds = list(seed)
            if len(seeds) != self.num_envs: raise ValueError("one seed is required per world")
        return self._dispatch("seed_worlds", tuple(range(self.num_envs)), seeds)

    def check_success(self):
        return self._dispatch("check_success_worlds", tuple(range(self.num_envs)))

    def get_sim_state(self):
        return self._dispatch("get_states", tuple(range(self.num_envs)))

    def get_env_attr(self, key, id=None):
        return self._dispatch("get_env_attrs", self._ids(id), key)

    def set_env_attr(self, key, value, id=None):
        return self._dispatch("set_env_attrs", self._ids(id), key, value)

    def render(self, **kwargs):
        return self._dispatch("render_worlds", tuple(range(self.num_envs)), **kwargs)

    def get_segmentation_of_interest(self, images):
        images = list(images)
        if len(images) != self.num_envs: raise ValueError("one image is required per world")
        return self._dispatch("segmentation_worlds", tuple(range(self.num_envs)), images)

    def close(self):
        if not self._closed:
            self.driver.close()
            self._closed = True

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
