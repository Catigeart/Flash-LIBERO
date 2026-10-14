"""Bind prepared resident physics to observation sampling without host reads.

The caller creates the observer from the prepared runtime's exact native model
and source-qualified model-only metadata. This module does not construct an env
or perform a physics forward to obtain metadata or observations.
"""
from types import MappingProxyType


class FixedRuntimeObservationBinding:
    def __init__(self, prepared, observer, *, expected_observe=None):
        if observer.model is not prepared.model_pack.model:
            raise ValueError("observer must use the exact prepared native model")
        runtime = prepared.runtime
        if observer.nworld != runtime.valid.size:
            raise ValueError("observer and runtime world counts differ")
        self.prepared = prepared
        self.observer = observer
        self.runtime = runtime
        self.stream = prepared.stream
        if runtime.stream is not self.stream:
            raise ValueError("prepared/runtime streams differ")
        self.expected_observe = expected_observe
        self._forward = runtime.forward
        self._euler = runtime.euler
        position = prepared.dynamics.stages.position
        # Retain flattened views once. No allocation, download, or dynamic model
        # reconstruction takes place in the resident sampling callback.
        sources = {
            "qpos": runtime.euler.qpos, "qvel": runtime.euler.qvel,
            "body_xpos": position.body.xpos, "body_xquat": position.body.xquat,
            "body_xmat": position.body.xmat,
            "geom_xpos": position.frames.geom_xpos,
            "geom_xmat": position.frames.geom_xmat,
            "site_xpos": position.frames.site_xpos,
            "site_xmat": position.frames.site_xmat,
            "subtree_com": position.mass.subtree_com,
        }
        self._position = position
        self._owners = sources
        self._tokens = {name: self._token(value) for name, value in sources.items()}
        self._mask = runtime.valid
        self._fields = MappingProxyType({name: value if value.ndim == 1 else value.flatten()
                                         for name, value in sources.items()})
        self.observation_hook = self._observe_substep
        self.scene_hook = self._scene_forward
        self.installed = False

    @staticmethod
    def _token(value):
        return (getattr(value,"ptr",None), getattr(value,"dtype",None),
                tuple(getattr(value,"shape",())), getattr(value,"device",None), value.ndim)

    def validate(self):
        p, r = self.prepared, self.runtime
        if (p.runtime is not r or p.stream is not self.stream or r.stream is not self.stream
                or r.forward is not self._forward or r.euler is not self._euler
                or p.dynamics.stages.position is not self._position
                or p.model_pack.model is not self.observer.model or r.valid is not self._mask):
            raise ValueError("observation runtime owner replaced")
        pos = self._position
        current = {"qpos":r.euler.qpos,"qvel":r.euler.qvel,
                   "body_xpos":pos.body.xpos,"body_xquat":pos.body.xquat,"body_xmat":pos.body.xmat,
                   "geom_xpos":pos.frames.geom_xpos,"geom_xmat":pos.frames.geom_xmat,
                   "site_xpos":pos.frames.site_xpos,"site_xmat":pos.frames.site_xmat,
                   "subtree_com":pos.mass.subtree_com}
        for name,value in current.items():
            if value is not self._owners[name] or self._token(value) != self._tokens[name]:
                raise ValueError("observation source owner replaced: "+name)
        if self.installed and (r.observe is not self.observation_hook or
                (self.observer.scene_transforms is not None and self._forward.scene_transform is not self.scene_hook)):
            raise ValueError("observation callback replaced")

    def fields(self):
        self.validate()
        return self._fields

    def _observe_substep(self):
        # Euler's actual publication mask holds every clock/cache on failed
        # worlds, while successful worlds read new qpos and old forward geometry.
        self.validate()
        self.observer.sample(self._fields, world_mask=self.runtime.valid,
                             stream=self.stream)

    def _scene_forward(self, mask, stream):
        self.validate()
        if stream is not self.stream:
            raise ValueError("foreign scene transform stream")
        transforms = self.observer.scene_transforms
        if transforms is not None:
            transforms.update(self._fields["body_xpos"], self._fields["body_xmat"],
                              mask, stream)

    def install(self):
        self.validate()
        if self.installed:
            return self
        if self.runtime.observe is not self.expected_observe:
            raise ValueError("existing observation hook must be explicitly authorized for replacement")
        forward = self.runtime.forward
        if self.observer.scene_transforms is not None:
            if forward.scene_transform is not None:
                raise ValueError("prepared runtime already has a scene transform hook")
            forward.scene_transform = self.scene_hook
            if "scene_transform_hook" in forward.scope_gaps:
                forward.scope_gaps.remove("scene_transform_hook")
        self.runtime.observe = self.observation_hook
        self.installed = True
        return self