"""Cold typed domain admission hooks for a combined qualified GPU owner.

These hooks read existing decoder packets and two small native arrays. They
neither compile/save models nor allocate physics. A live publication check is
protocol validation, never an invented per-run canonical residual proof.
"""
from dataclasses import dataclass
import hashlib
import json
from .admission import QualificationError, _json
from .domain import load_supported_domain, controller_action_box, check_new_legal_action
from .generation_domain import check_typed_source_generation
from .immutable_json import freeze_json, thaw_json


def _sealed(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

@dataclass(frozen=True)
class DomainLease:
    candidate_identity: str
    model_sha256: str
    parameters_sha256: str
    action_box_json: str
    generation_proof_json: str
    def __post_init__(self):
        parsed=None;ready=False
        if type(self.action_box_json) is str:
            try:
                box=json.loads(self.action_box_json)
                # Cache only the action schema produced by the cold admission.
                # Other JSON values retain original list/scalar/error behavior.
                dimension=box.get('dimension') if isinstance(box,dict) else None
                lower=box.get('lower') if isinstance(box,dict) else None
                upper=box.get('upper') if isinstance(box,dict) else None
                ready=(type(dimension) is int and dimension>=0 and
                    isinstance(lower,list) and isinstance(upper,list) and
                    len(lower)==dimension and len(upper)==dimension and
                    all(type(v) in (int,float) for v in lower+upper))
                if ready:parsed=freeze_json(box)
            except (TypeError,ValueError):
                # Baseline construction/serialization do not parse or reject.
                # The original json.loads exception remains at first use.
                pass
        object.__setattr__(self,'_parsed_action_box',parsed)
        object.__setattr__(self,'_parsed_action_box_ready',ready)
        object.__setattr__(self,'_parsed_action_box_json',self.action_box_json)

    def __getstate__(self):
        # Preserve the original public five-field dataclass/pickle schema.
        # Parsed MappingProxyType values are private cache, never wire state.
        return dict(candidate_identity=self.candidate_identity,model_sha256=self.model_sha256,
            parameters_sha256=self.parameters_sha256,action_box_json=self.action_box_json,
            generation_proof_json=self.generation_proof_json)

    def __reduce__(self):
        state=self.__getstate__()
        return (type(self),tuple(state[k] for k in ('candidate_identity','model_sha256',
            'parameters_sha256','action_box_json','generation_proof_json')))

    def __setstate__(self,state):
        # Baseline pickles contain these five strings without a parsed cache.
        names=('candidate_identity','model_sha256','parameters_sha256',
            'action_box_json','generation_proof_json')
        if not isinstance(state,dict) or set(state)!=set(names):
            raise ValueError('domain lease serialized state differs')
        for name in names:object.__setattr__(self,name,state[name])
        self.__post_init__()

    def action_limits(self):
        if (self._parsed_action_box_ready and type(self.action_box_json) is str and
                self.action_box_json==self._parsed_action_box_json):
            return self._parsed_action_box
        # Forced replacement of a frozen field was parsed afresh by baseline.
        # It cannot reuse stale cached authority; preserve that strict fallback.
        return json.loads(self.action_box_json)

    @property
    def action_box(self):return thaw_json(self.action_limits())

class SupportedDomainAdmission:
    """Receipt-qualified admission owned by one combined runtime factory.

    No public mutable-model handle or trace allow-list is introduced. Every
    fresh prepare, including cold replacement, must obtain a fresh lease from
    actual built metadata/arrays before allocation, then check actual device
    publication before installing components. Unknown failures quarantine the
    arena through its existing owner; these hooks never clear device errors.
    """
    def __init__(self,receipt_record):
        candidate=load_supported_domain(receipt_record)
        domain=candidate['domain']
        packet=_json(domain['baseline_native_packet'])
        self._candidate_json=_sealed(candidate)
        self._baseline_json=_sealed(packet)
        self._policy_json=_sealed(domain['generation_policy'])
        self._factory_entry=domain['combined_factory_entry']
        self._candidate_identity=hashlib.sha256(self._candidate_json.encode()).hexdigest()
        policy=json.loads(self._policy_json)
        if policy.get('source_quaternion_norm_atol')!=1e-12:
            raise QualificationError('original source quaternion guard required')
        if policy.get('parameter_producer')!='original-task-placement-source':
            raise QualificationError('original source parameter producer required')

    def admission_domain_for_built(self,built,*,factory_entry):
        if factory_entry!=self._factory_entry:
            raise QualificationError('combined implementation entry differs from receipt')
        md=built.metadata
        packet=json.loads(self._baseline_json)
        policy=json.loads(self._policy_json)
        values={name:getattr(built.native_model,name) for name in policy['physics_mutable_fields']}
        # The sealed source rule is bound to this task's actual named shell IDs,
        # rather than a caller supplied boolean saying that an upload is legal.
        ids=sorted(set(int(v) for v in built.domain_shell.obj_body_id.values()))
        if ids!=sorted(policy['source_body_ids']):
            raise QualificationError('actual original placement body roots differ')
        domain=json.loads(self._candidate_json)['domain']
        if hashlib.sha256(__import__('pathlib').Path(md['bddl_file_name']).read_bytes()).hexdigest()!=domain['task']['bddl_sha256']:
            raise QualificationError('actual task source differs')
        proof=check_typed_source_generation(packet['metadata'],md,packet['parameter_values'],values,policy)
        box=controller_action_box(md)
        if box!=controller_action_box(packet['metadata']):
            raise QualificationError('original controller action limits changed')
        # Host sampler history is compatibility provenance, not physics identity.
        # Its implementation source is already in the receipt's source closure.
        history=md.get('source_property_sampler_history')
        if history is not None:
            if (history.get('schema')!='source-property-sampler-history-rebind-v1' or
                history.get('old_native_owner_retained') is not False or
                history.get('RNG_draws_performed')!=0 or
                history.get('current_count')!=len(history.get('descriptors',[]))):
                raise QualificationError('sampler history rebind protocol differs')
        parameter_sha=hashlib.sha256(_sealed(proof['actual_typed_parameter_rows']).encode()).hexdigest()
        return DomainLease(self._candidate_identity,md['model_sha256'],parameter_sha,_sealed(box),_sealed(proof))

    def assert_domain_publication(self,lease,witness,*,expected_model_epoch):
        """Consume the owner's actual synchronized device witness after upload.

        The caller is the source-pinned combined owner. Host ModelGeneration
        alone cannot serve as witness. Numerical quality remains the separate
        source-domain certificate or actual canonical residual evidence named
        in the integrated receipt; finite/error-zero is not that evidence.
        """
        if not isinstance(lease,DomainLease) or lease.candidate_identity!=self._candidate_identity:
            raise QualificationError('foreign or stale domain lease')
        if type(expected_model_epoch) is not int or expected_model_epoch<0:
            raise QualificationError('expected model epoch malformed')
        if any(type(witness.get(k)) is not int for k in ('device_model_epoch','expected_model_epoch','model_ready','first_error')):
            raise QualificationError('actual device witness must contain synchronized integer values')
        if witness.get('model_sha256')!=lease.model_sha256:
            raise QualificationError('published native identity differs')
        if (witness.get('device_model_epoch')!=expected_model_epoch or
            witness.get('expected_model_epoch')!=expected_model_epoch or
            witness.get('model_ready')!=1):
            raise QualificationError('actual model upload publication is not current')
        # Existing cold owner propagates H2D failures as exceptions and never
        # publishes ready. Successful return is explicit source-branch evidence,
        # not a fabricated independent GPU upload-status array.
        if witness.get('first_error')!=0 or witness.get('upload_completed') is not True:
            raise QualificationError('actual cold upload did not complete; owner must quarantine')
        if 'upload_error' in witness and (type(witness['upload_error']) is not int or witness['upload_error']!=0):
            raise QualificationError('actual optional upload status reports failure')
        return {'model_sha256':lease.model_sha256,'model_epoch':expected_model_epoch,
                'protocol_current':True,'per_run_residual_proof':False}

    def check_action(self,lease,action):
        if not isinstance(lease,DomainLease) or lease.candidate_identity!=self._candidate_identity:
            raise QualificationError('foreign domain lease')
        return check_new_legal_action(action,lease.action_limits())
