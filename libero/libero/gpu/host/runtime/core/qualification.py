"""Cold loading of an explicitly pinned independent acceptance receipt."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .contracts import ContractError
from .registry import AcceptanceRecord, CapabilityRegistry


def load_qualified_registry(path, expected_sha256, *, required_scope=None):
    """Load authored independent records, rechecking reports and actual sources.

    Unlike ordinary from_file, this is an explicit qualification operation: the
    caller must pin the reviewed receipt SHA. Cached admitted flags are ignored.
    It does not authorize fields, models or scopes outside each exact record.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ContractError('qualified receipt content differs from trusted pin')
    document = json.loads(raw)
    registry = CapabilityRegistry.from_document(document)
    for row in document.get('entries', [document]):
        candidate = registry.candidates[row['candidate']['candidate_id']]
        candidate.verify_source_files()
        payloads = row.get('acceptance_records')
        if payloads is None:
            payload = row.get('acceptance')
            payloads = [] if payload is None else [payload]
        if not payloads:
            raise ContractError('explicit independent acceptance records required')
        for payload in payloads:
            try:
                record = AcceptanceRecord(**payload)
            except TypeError as exc:
                raise ContractError('acceptance record schema mismatch') from exc
            if record.candidate_id != candidate.candidate_id:
                raise ContractError('acceptance candidate identity differs from receipt')
            if required_scope is not None and record.execution_scope != required_scope:
                raise ContractError('qualification scope differs from requested scope')
            registry.record_acceptance(record)
    return registry
