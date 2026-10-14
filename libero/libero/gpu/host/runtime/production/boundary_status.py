"""Pure typed boundary packet validation, shared with controlled completion scopes."""
from dataclasses import dataclass
from numbers import Integral

FIELDS = ("request", "world", "runtime_error", "foundation_error", "newton_error",
          "forward_error", "model", "expected_model", "model_ready", "state",
          "expected_state", "forward", "expected_forward", "n_attempt", "n_written",
          "expected_newton", "integrations", "step_valid", "forward_valid", "n_valid")
WIDTH = len(FIELDS)

@dataclass(frozen=True)
class BoundaryRecord:
    owner: object
    stream: object
    request: int
    model_identity: str
    rows: tuple

def validate_boundary_packet(packet, *, owner, stream, request, model_identity):
    if type(request) is not int or not 0 <= request < 2**63:
        raise ValueError("boundary request outside int64 domain")
    rows=[]
    for row in packet:
        if any(isinstance(v,bool) or not isinstance(v,Integral) or not -2**63<=v<2**63 for v in row):
            raise ValueError("boundary packet must contain exact int64 values")
        rows.append(tuple(int(v) for v in row))
    rows=tuple(rows)
    if not rows or any(len(row) != WIDTH for row in rows):
        raise ValueError("boundary packet extent differs")
    if any(row[0] != request or row[1] != w for w, row in enumerate(rows)):
        raise ValueError("foreign or stale boundary request")
    # Preserve the existing four-source priority across ALL worlds. Do not
    # stop at world0, since foundation/newton errors may be in world1 only.
    if any(row[2] or row[3] for row in rows):
        raise ValueError("actual GPU source/stage first error")
    if any(row[4] for row in rows):
        raise ValueError("actual Newton error")
    if any(row[5] for row in rows):
        raise ValueError("actual complete-forward publication error")
    return BoundaryRecord(owner, stream, request, model_identity, rows)
