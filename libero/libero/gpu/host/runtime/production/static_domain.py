"""Immutable byte assets for the opt-in static-domain ABI.

Old public resident arrays are deliberately not admitted here: pointer/epoch
identity cannot prove absence of arbitrary external writes. This is a private
asset/descriptor prototype, to be consumed only by a sealed allocation owner.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class StaticLease:
    domain: object
    generation: int
    model_identity: str
    schema: tuple

@dataclass(frozen=True,init=False)
class ImmutableStaticDomain:
    _assets: tuple
    _schema: tuple
    _model: str
    _generation: int
    _closed: bool
    def __init__(self, model_identity, fields):
        if not isinstance(model_identity, str) or not model_identity:
            raise ValueError("actual model identity required")
        # Only independent immutable bytes are accepted. Callers cannot retain
        # a writable alias of these exact assets.
        if any(type(v) is not bytes for v in fields.values()):
            raise ValueError("mutable resident storage is not an immutable static domain")
        assets=tuple(sorted(fields.items()))
        object.__setattr__(self,"_assets",assets)
        object.__setattr__(self,"_schema",tuple((k,len(v)) for k,v in assets))
        object.__setattr__(self,"_model",model_identity)
        object.__setattr__(self,"_generation",0)
        object.__setattr__(self,"_closed",False)

    def lease(self):
        if self._closed:
            raise ValueError("static domain revoked")
        return StaticLease(self, self._generation, self._model,
                           self._schema)

    def assets(self, lease):
        self.assert_current(lease)
        return self._assets

    def assert_current(self, lease):
        if (self._closed or lease.domain is not self or
                lease.generation != self._generation or lease.model_identity != self._model or
                lease.schema is not self._schema):
            raise ValueError("foreign or revoked static lease")

    def revoke(self):
        object.__setattr__(self,"_closed",True)
        object.__setattr__(self,"_generation",self._generation+1)
