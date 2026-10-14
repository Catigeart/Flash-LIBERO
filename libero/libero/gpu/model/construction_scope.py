"""Cold, task-local constructor routing without CPU physics fallback.

Raw original classes remain in Task/robosuite registries. A public allocation
returns a subtype with GPU lifecycle methods and a constructor that builds the
original model-only shell once. Custom __new__ runs once on the public object;
its ordinary instance state is handed to the separate cold shell.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from inspect import signature
from threading import RLock

_MODEL_ONLY_TASK = ContextVar("libero_gpu_model_only_task", default=None)
_PROXY_TYPES = {}
_PROXY_LOCK = RLock()


def is_model_only_construction(task_class=None):
    target = _MODEL_ONLY_TASK.get()
    return target is not None and (task_class is None or target[0] is task_class)


@contextmanager
def model_only_construction(task_class, *, allow_allocation=False):
    token = _MODEL_ONLY_TASK.set((task_class, bool(allow_allocation)))
    try:
        yield
    finally:
        _MODEL_ONLY_TASK.reset(token)


def consume_model_only_allocation(task_class):
    # Only the explicit builder allocation bypasses GPU routing. A task created
    # by user callbacks during cold __init__ must not silently become CPU physics.
    target = _MODEL_ONLY_TASK.get()
    if target is None or target[0] is not task_class or not target[1]: return False
    _MODEL_ONLY_TASK.set((task_class, False))
    return True


def _constructor_packet(task_class, instance, args, kwargs):
    kwargs = dict(kwargs)
    backend = kwargs.pop("physics_backend", "gpu")
    if backend != "gpu":
        raise ValueError("public GPU proxy constructor requires physics_backend=gpu")
    if "_gpu_constructor_call" in kwargs:
        raise ValueError("reserved internal GPU constructor argument")
    bound = signature(task_class.__init__).bind(instance, *args, **kwargs)
    filename = bound.arguments.get("bddl_file_name")
    if filename is None:
        # User task subclasses can rename the first normal path parameter.
        if args:
            filename = args[0]
        else:
            for name, parameter in signature(task_class.__init__).parameters.items():
                if name != "self" and name in bound.arguments and parameter.kind.name not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                    filename = bound.arguments[name]
                    break
    if filename is None:
        raise TypeError("LIBERO GPU task constructor requires bddl_file_name")
    return filename, {"args": tuple(args), "kwargs": kwargs,
                      "preallocated_state": dict(instance.__dict__)}


def allocate_gpu_task_proxy(task_class, *, driver_class=None, creator=None):
    """Allocate a subtype; its GPU __init__ runs after custom public __new__."""
    if driver_class is None:
        from ..host.adapters.environment.task_driver import GpuTaskEnvironment
        driver_class = GpuTaskEnvironment
    if creator is None:
        from ..host.adapters.environment.factory import create_gpu_environment
        creator = create_gpu_environment
    key = task_class, driver_class, creator
    with _PROXY_LOCK:
        proxy_type = _PROXY_TYPES.get(key)
        if proxy_type is None:
            # Bypass EnvMeta registration for the implementation subtype. The
            # original registered class and registry keys remain unchanged.
            base_meta = type(task_class)
            proxy_meta = type("_GpuTaskConstructorMeta", (base_meta,), {
                "__new__": lambda meta, name, bases, namespace: type.__new__(meta, name, bases, namespace),
                "__init__": type.__init__,
            })

            def gpu_init(self, *args, **kwargs):
                filename, packet = _constructor_packet(task_class, self, args, kwargs)
                env = creator(task_class.__name__.lower(), filename, _gpu_constructor_call=packet)
                if not isinstance(env, driver_class) or getattr(env, "physics_backend", None) != "gpu":
                    close = getattr(env, "close", None)
                    if callable(close): close()
                    raise RuntimeError("public task constructor did not supply the actual GPU lifecycle")
                try:
                    # No second physics initialization: move the actual driver
                    # state, then rebind methods/owner callbacks to this object.
                    self.__dict__.update(env.__dict__)
                    self._install_shell()
                except BaseException:
                    env.close()
                    raise
                env.components = None
                env._closed = True

            proxy_type = proxy_meta(task_class.__name__, (driver_class, task_class), {
                "__module__": task_class.__module__, "__qualname__": task_class.__qualname__,
                "__init__": gpu_init,
                "__doc__": task_class.__doc__,
            })
            _PROXY_TYPES[key] = proxy_type
    return object.__new__(proxy_type)
