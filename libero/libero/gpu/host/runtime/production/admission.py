"""Cold admission for a finite independently reviewed GPU case.

No GPU imports and no registration side effects. This is deliberately a bounded
case API; it cannot authorize arbitrary reset, state, task or policy actions.
Original diagnostic factories remain unchanged and directly usable.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct

class QualificationError(ValueError):
    pass

def _json(record):
    raw=Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record["sha256"]:
        raise QualificationError("qualification artifact differs from trusted pin")
    return json.loads(raw)

def candidate_id(candidate):
    return hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def action_digest(action):
    if hasattr(action,"tolist"): action=action.tolist()
    if not isinstance(action,(list,tuple)) or not action:
        raise QualificationError("flat nonempty action required")
    values=[]
    for x in action:
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
            raise QualificationError("finite scalar action required")
        values.append(float(x))
    return hashlib.sha256(struct.pack("<Q",len(values))+struct.pack("<"+"d"*len(values),*values)).hexdigest()

def load_qualified_case(receipt_record):
    """Validate an explicitly trusted INTEGRATED report; unit/math cannot grant.

    Pending candidate documents and observed execution/protocol booleans never
    become permission. The independent report binds the complete candidate,
    exact finite domain and implementation epoch, rather than child licence union.
    """
    receipt=_json(receipt_record)
    if receipt.get("schema")!="qualified-gpu-finite-case-receipt-v1":
        raise QualificationError("integrated finite-case receipt required")
    candidate=receipt["candidate"]
    cid=candidate_id(candidate)
    if receipt.get("candidate_id")!=cid:
        raise QualificationError("candidate closure identity differs")
    review=_json(receipt["independent_integrated_review"])
    if (review.get("qualification_kind")!="integrated-gpu-finite-case" or
        review.get("verdict")!="PASS_SCOPED" or
        review.get("candidate_id")!=cid or
        review.get("accepted_domain")!=candidate["domain"] or
        not review.get("reviewer") or review["reviewer"]==candidate["author"]):
        raise QualificationError("independent exact integrated scope is absent")
    # These are the reviewer-declared actual closure sections, not guarantees
    # derived from metadata or a forced rerun of all internal arithmetic.
    required={"source-model-inputs","controller-feedback-native-fidelity",
              "legal-solver-quality","epoch-write-error-publication","observations-reset"}
    if not required.issubset(set(review.get("accepted_closure_sections",[]))):
        raise QualificationError("independent integrated closure remains incomplete")
    if not review.get("actual_evidence"):
        raise QualificationError("actual integrated execution evidence is absent")
    for rec in review["actual_evidence"]: _json(rec)
    actual=_json(candidate["actual_report"])
    if not actual.get("completed") or actual.get("cpu_physics_calls")!=[]:
        raise QualificationError("actual integrated GPU execution is incomplete")
    for rec in candidate["source_files"]:
        raw=Path(rec["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=rec["sha256"]:
            raise QualificationError("qualified implementation epoch changed")
    return candidate

class FiniteCaseGuard:
    """Operation scope checks independent from GPU execution."""
    def __init__(self,domain):
        self.domain=domain;self.position=0;self.ready=False;self.failed=False
    def check_reset(self,seed):
        if self.failed or self.ready or self.position:
            raise QualificationError("only the reviewed first seeded reset is supported")
        if type(seed) is not int or seed!=self.domain["seed"]:
            raise QualificationError("seed outside reviewed finite case")
    def accept_reset(self,evidence):
        self._evidence(evidence);self.ready=True
    def check_action(self,action):
        if self.failed or not self.ready:
            raise QualificationError("reviewed seeded reset required before public step")
        expected=self.domain["action_digests"]
        if self.position>=len(expected) or action_digest(action)!=expected[self.position]:
            raise QualificationError("action outside independently reviewed finite trace")
    def accept_action(self,evidence):
        self._evidence(evidence);self.position+=1
    def _evidence(self,e):
        if (e.get("physics_backend")!="gpu" or
            e.get("model_sha256")!=self.domain["reset_model_sha256"] or
            e.get("model_identity")!=self.domain["reset_model_identity"]):
            self.failed=True;raise QualificationError("actual model differs from reviewed reset generation")
        for actual,expected in (("device_model_epoch","independent_model_expected"),
                                ("state_epoch","independent_state_expected"),
                                ("actual_forward_epoch","independent_forward_expected"),
                                ("actual_newton_attempt","independent_newton_expected"),
                                ("actual_newton_written","independent_newton_expected")):
            if actual not in e or expected not in e or e[actual]!=e[expected]:
                self.failed=True;raise QualificationError("actual publication is not independently current: "+actual)
        for field in ("first_error","foundation_first_error"):
            if not e.get(field) or any(e[field]):
                self.failed=True;raise QualificationError("actual GPU stage error: "+field)
        if e.get("device_model_ready")!=[1]:
            self.failed=True;raise QualificationError("actual model owner is not ready")

class QualifiedFiniteCase:
    """Private diagnostic constructor, then qualified reset and finite trace.

    Arbitrary sim/controllers/state mutators are not exported. This preserves
    original seed/reset ordering and does not reseed a second reset silently.
    """
    physics_backend="gpu"
    def __init__(self,private_env,candidate):
        self._env=private_env;self._candidate=candidate
        self._guard=FiniteCaseGuard(candidate["domain"]);self._closed=False
    def _mutation(self,fn):
        try:return fn()
        except BaseException:
            self._guard.failed=True
            # Preserve the first exception; existing owner cleanup handles the
            # arena. No unknown mutation can continue as a qualified state.
            try:self.close()
            except BaseException:pass
            raise
    def reset(self):
        d=self._candidate["domain"];self._guard.check_reset(d["seed"])
        def run():
            self._env.seed(d["seed"])
            result=self._env.reset()
            self._guard.accept_reset(self._env.gpu_physics_evidence())
            return result
        return self._mutation(run)
    def step(self,action):
        self._guard.check_action(action)
        def run():
            result=self._env.step(action)
            self._guard.accept_action(self._env.gpu_physics_evidence())
            return result
        return self._mutation(run)
    def get_sim_state(self):
        if not self._guard.ready or self._guard.failed:raise QualificationError("qualified reset unavailable")
        return self._env.get_sim_state()
    def gpu_physics_evidence(self):
        if not self._guard.ready or self._guard.failed:raise QualificationError("qualified reset unavailable")
        result=dict(self._env.gpu_physics_evidence())
        result["finite_public_admission"]={"candidate_id":candidate_id(self._candidate),
              "domain":self._candidate["domain"],"completed_actions":self._guard.position}
        # Internal mode/production_numeric_acceptance diagnostic fields stay
        # honest; the separately trusted integrated receipt grants this domain.
        return result
    @property
    def action_spec(self):return self._env.action_spec
    def close(self):
        if not self._closed:
            self._env.close();self._closed=True

def create_qualified_case(receipt_record,*,bddl_file_name,environment_options):
    """New bounded API. It does not install/alter the global ControlEnv route."""
    candidate=load_qualified_case(receipt_record);d=candidate["domain"]
    if hashlib.sha256(Path(bddl_file_name).read_bytes()).hexdigest()!=d["bddl_sha256"]:
        raise QualificationError("task BDDL outside qualified case")
    if environment_options!=candidate["environment_options"]:
        raise QualificationError("environment/render options outside qualified case")
    target=candidate["private_environment_factory"]
    module,name=target.split(":",1)
    factory=getattr(importlib.import_module(module),name)
    env=factory(bddl_file_name=bddl_file_name,**environment_options)
    return QualifiedFiniteCase(env,candidate)
