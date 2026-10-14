"""Read-only integration candidate preparation. Never grants execution permission.

This module has no GPU, native-model, runtime registry or package side effects.
An integrated independent receipt must be authored separately after actual review.
"""
import hashlib
import json
from pathlib import Path

class PreparationError(ValueError):
    pass

def read_pinned_json(record):
    raw=Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
        raise PreparationError("evidence changed from explicit source pin")
    return json.loads(raw)

def prepare_capture(record, tasks):
    """Attribute an actual execution, retaining each actual model generation.

    Metadata completed/accepted flags never become a production grant. BDDL byte
    identity is required; model dimensions and names do not substitute for it.
    Actual source pins are listed by epoch; drift from current files is reported,
    never silently rebound and never erases historical execution.
    """
    report=read_pinned_json(record)
    bddl=hashlib.sha256(Path(report["bddl"]).read_bytes()).hexdigest()
    matches=[t for t in tasks if t["bddl_sha256"]==bddl]
    if len(matches)!=1:
        raise PreparationError("actual BDDL must identify exactly one registered task")
    task=matches[0]
    stages=[]; sources={}
    for stage in report["stages"]:
        e=stage["evidence"]
        if e.get("physics_backend")!="gpu":
            raise PreparationError("capture is not an actual GPU boundary")
        for rec in e.get("source_records",[]):
            key=(rec["path"],rec["sha256"])
            if key not in sources:
                p=Path(rec["path"])
                current=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
                sources[key]={**rec,"current_sha256":current,"current_matches_captured_epoch":current==rec["sha256"]}
        stages.append({"stage":stage["stage"],"model_sha256":e["model_sha256"],
                       "model_identity":e["model_identity"],
                       "model_epoch":e.get("device_model_epoch"),
                       "independent_model_expected":e.get("independent_model_expected"),
                       "model_ready":e.get("device_model_ready"),
                       "state_epoch":e.get("state_epoch"),
                       "independent_state_expected":e.get("independent_state_expected"),
                       "state_status":e.get("state_status"),
                       "forward_status":e.get("forward_status"),
                       "first_error":e.get("first_error"),
                       "integration_count":e.get("actual_integration_count")})
    return {"schema":"actual-integration-candidate-preparation-v1", "actual_task":task,
            "actual_report":record,"reported_completed":report.get("completed"),
            "reported_cpu_physics_calls":report.get("cpu_physics_calls"),
            "captured_stages":stages,"captured_source_epochs":list(sources.values()),
            "production_admission":False,"independent_integrated_review_required":True,
            "permission_scope":"preparation only; actual outputs/held/lineage remain independently reviewed"}
