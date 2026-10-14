"""Opt-in copy-before-overwrite witness ABI; existing eager arrays are unchanged.

This owner only works when every writer participates in before_write under the
same execution lock. A generation identity is never a substitute for operands.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class WitnessRequest:
    owner: object
    generation: int
    ordinal: int

class RetainedWitness:
    def __init__(self, owner, generation, fields, *, controlled_writers=False):
        if not controlled_writers:
            raise ValueError("retained witness requires a complete controlled writer domain")
        self.owner, self.generation = owner, generation
        self.fields = dict(fields)
        self._ordinal = -1
        self._request = None
        self._pending = {}
        self._payload = {}

    def begin(self, ordinal, *, eager=()):
        eager=tuple(eager)
        if type(ordinal) is not int or ordinal <= self._ordinal:
            raise ValueError("witness request must advance")
        if self._request is not None:
            raise ValueError("previous witness request must be explicitly discarded")
        if set(eager) - self.fields.keys():
            raise ValueError("foreign eager witness field")
        self._ordinal = ordinal
        self._request = WitnessRequest(self.owner, self.generation, ordinal)
        self._pending = dict(self.fields)
        self._payload = {}
        self._copy(eager)
        return self._request

    def _require(self, request):
        if (request is not self._request or request.owner is not self.owner or
                request.generation != self.generation):
            raise ValueError("foreign or overwritten witness request")

    def _copy(self, names):
        for name in names:
            if name in self._pending:
                # copy() is bound at cold registration to the actual source.
                payload=self._pending[name]()
                self._payload[name]=payload
                del self._pending[name]

    def before_write(self, names):
        names=tuple(names)
        if set(names) - self.fields.keys():
            raise ValueError("writer outside registered witness domain")
        if self._request is not None:
            self._copy(names)

    def materialize(self, request):
        self._require(request)
        self._copy(tuple(self._pending))
        return dict(self._payload)

    def discard(self, request):
        self._require(request)
        self._pending.clear()
        self._payload.clear()
        self._request = None

    def revoke(self):
        if self._request is not None:
            self._copy(tuple(self._pending))
        self.generation += 1
        self._request = None
