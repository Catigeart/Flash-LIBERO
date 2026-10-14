"""Preparation for source-backed GPU backend domains, never a trace allow-list.

Acceptance must cover the algorithm/source-supported input domain plus real
integrated representative evidence. Mere component/unit flags cannot activate.
"""
import hashlib
import json
import math
from pathlib import Path
from .admission import QualificationError, _json, candidate_id

def controller_action_box(metadata):
    robots=metadata["robot_metadata"]
    if len(robots)!=1:raise QualificationError("single actual robot domain required")
    robot=robots[0];config=robot["controller_config"]
    if (config["type"]!="OSC_POSE" or config["impedance_mode"]!="fixed" or
        config["interpolation"] is not None or robot["gripper"]["dof"]!=1):
        raise QualificationError("controller feature domain unsupported")
    def six(x):
        if isinstance(x,(int,float)) and not isinstance(x,bool):return [float(x)]*6
        if isinstance(x,list) and len(x)==6:return list(map(float,x))
        raise QualificationError("original OSC input limits malformed")
    lo=six(config["input_min"])+[-1.0];hi=six(config["input_max"])+[1.0]
    if any(not math.isfinite(a) or not math.isfinite(b) or a>=b for a,b in zip(lo,hi)):
        raise QualificationError("finite ordered original action limits required")
    return {"dimension":7,"lower":lo,"upper":hi,"outside_box":"not part of this declared supported domain; original controller clipping semantics are unchanged"}

def check_new_legal_action(action,box):
    """Accept EVERY new finite action in the original declared box, no replay IDs."""
    if hasattr(action,"tolist"):action=action.tolist()
    if not isinstance(action,(list,tuple)) or len(action)!=box["dimension"]:
        raise QualificationError("action dimension outside controller domain")
    values=[]
    for x,lo,hi in zip(action,box["lower"],box["upper"]):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:
            raise QualificationError("action outside supported finite box")
        values.append(float(x))
    return values

def check_model_readset(actual,baseline):
    """Exact static model intake. Future lawful generations need a source bridge.

    A different digest is not classified as corruption; it explicitly needs the
    already-owned actual typed-row/immutable-field generation proof. This
    function never ignores hash changes or guesses equivalence from dimensions.
    """
    for field in ("dimensions","options","topology_sha256","static_arrays_sha256","model_sha256","name_maps","robot_metadata"):
        if actual[field]!=baseline[field]:
            raise QualificationError("model domain differs: "+field+"; lawful generation requires explicit typed source bridge")

def load_supported_domain(receipt_record):
    receipt=_json(receipt_record)
    if receipt.get("schema")!="qualified-gpu-supported-domain-v1":
        raise QualificationError("source-supported backend-domain receipt required; finite replay/unit/math is insufficient")
    c=receipt["candidate"]
    if receipt.get("candidate_id")!=candidate_id(c):
        raise QualificationError("receipt candidate closure identity differs")
    review=_json(receipt["independent_integrated_review"])
    if (review.get("qualification_kind")!="integrated-gpu-supported-domain" or
        review.get("verdict")!="PASS_SCOPED" or review.get("candidate_id")!=candidate_id(c) or
        review.get("accepted_domain")!=c["domain"] or review.get("reviewer")==c["author"] or
        not review.get("reviewer")):
        raise QualificationError("independent supported-domain acceptance absent")
    required={"immutable-model-controller-features","controller-new-legal-action-domain",
        "live-producer-native-feedback","source-legal-stop-residual-quality",
        "actual-current-epoch-write-held-error","original-observation-boundary"}
    if not required.issubset(set(review.get("accepted_closure_sections",[]))):
        raise QualificationError("source/domain closure incomplete")
    # Actual representative proof and source-supported domain are both needed;
    # neither every possible action/state enumeration nor global mathematical
    # forward-error theorem is imposed here.
    if not review.get("actual_integrated_evidence") or not review.get("source_domain_evidence"):
        raise QualificationError("actual integrated and source domain evidence required")
    executions=[]
    for rec in review["actual_integrated_evidence"]+review["source_domain_evidence"]:
        doc=_json(rec)
        if "stages" in doc and "bddl" in doc:
            if doc.get("completed") is not True or doc.get("cpu_physics_calls")!=[]:
                raise QualificationError("actual integrated execution incomplete or CPU physics observed")
            if hashlib.sha256(Path(doc["bddl"]).read_bytes()).hexdigest()!=c["domain"]["task"]["bddl_sha256"]:
                raise QualificationError("actual task differs from accepted supported domain")
            executions.append(doc)
    if not executions:
        raise QualificationError("completed actual integrated GPU execution absent")
    for rec in c["source_files"]:
        if hashlib.sha256(Path(rec["path"]).read_bytes()).hexdigest()!=rec["sha256"]:
            raise QualificationError("supported implementation source epoch changed")
    return c
