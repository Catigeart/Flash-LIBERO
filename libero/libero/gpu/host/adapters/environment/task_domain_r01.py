"""Receipt-bound owner action guard; unchanged diagnostic physics delegation."""
from .task_cleanup_r02 import GpuTaskEnvironment as _CleanupTask

class GpuTaskEnvironment(_CleanupTask):
    def __init__(self, components, **kwargs):
        binding = getattr(components.observation_fields, "__self__", None)
        prepared = getattr(binding, "prepared", None)
        self._domain_admission_owner = getattr(prepared, "domain_admission", None)
        super().__init__(components, **kwargs)

    def step(self, action):
        from libero.gpu.host.runtime.production.owner_action import validate_supported_owner_action
        with self._lock:
            self._open()
            # Validation precedes original mutation context: invalid external
            # inputs cannot quarantine a healthy owner or enqueue GPU writes.
            validate_supported_owner_action(self, action)
            return super().step(action)
