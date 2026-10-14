"""Production entry to the accepted ordered collision chain.

Preparation may inspect host model metadata. Launch never downloads geometry,
contacts, counts or first-error values. The source owner must publish a live
F epoch-qualified active mask; replacing source allocations requires rebuilding
this adapter and any downstream captured graph.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

_CHAIN_MAX_BYTES = 6 * 1024**3
# Actual qualified fixed-model chain settings. Dimensions/route qualification
# still belongs to the caller; these numbers alone are not a support proof.
FIXED_ACCEPTED_CAPACITIES = {
    'slots': 36315, 'req_slots': 36315, 'call_capacity': 15633,
    'contact_capacity': 1900, 'pair_capacity': 711, 'stage_slots': 711,
}


def _integer(name, value, minimum=1, maximum=2**31 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(name + ' is outside the supported range')
    return result


@dataclass(frozen=True)
class CollisionCapacityPlan:
    nworld: int
    nbody: int
    ngeom: int
    slots: int
    call_capacity: int
    pair_capacity: int
    stage_slots: int
    req_slots: int
    contact_capacity: int
    preflight_device_bytes: int
    max_device_bytes: int
    reserved_other_bytes: int
    total_device_budget_bytes: int

    def workspace_kwargs(self):
        return {name: getattr(self, name) for name in (
            'slots', 'call_capacity', 'pair_capacity', 'stage_slots',
            'req_slots', 'contact_capacity', 'max_device_bytes')}

    def manifest(self):
        return dict(asdict(self),
            schema='collision-runtime-capacity-plan-v1',
            scope='accepted chain conservative allocation admission; not measured peak VRAM',
            overflow_policy='device error, no contact or request truncation')


def plan_capacity(static, nworld, *, total_device_budget_bytes,
                  reserved_other_bytes, slots=None, call_capacity=None,
                  pair_capacity=None, stage_slots=None, req_slots=None,
                  contact_capacity=None):
    """Budget C before allocation alongside D/S/N/Euler/render/context.

    The reservation is caller supplied; this function must not guess that all
    remaining VRAM belongs to C. Capacity reductions are explicit, never based
    on an observed contact maximum masquerading as a guaranteed upper bound.
    """
    nworld = _integer('nworld', nworld, maximum=128)
    nbody, ngeom = len(static['body_geomnum']), len(static['geom_size'])
    _integer('nbody', nbody)
    _integer('ngeom', ngeom)
    total = _integer('total_device_budget_bytes', total_device_budget_bytes,
                     maximum=2**63 - 1)
    reserve = _integer('reserved_other_bytes', reserved_other_bytes,
                       minimum=0, maximum=2**63 - 1)
    budget = min(_CHAIN_MAX_BYTES, total - reserve)
    if budget <= 0:
        raise ValueError('no collision budget remains after other reservations')
    slots = _integer('slots', ngeom * 4 + 16 if slots is None else slots)
    calls = _integer('call_capacity', slots if call_capacity is None else call_capacity)
    pairs = _integer('pair_capacity', nbody * (nbody - 1) // 2 + 8
                     if pair_capacity is None else pair_capacity)
    stages = _integer('stage_slots', pairs if stage_slots is None else stage_slots)
    requests = _integer('req_slots', slots if req_slots is None else req_slots)
    contacts = _integer('contact_capacity', 512 if contact_capacity is None else contact_capacity)
    if calls > slots:
        raise ValueError('call_capacity exceeds slots')
    lengths = (nworld * calls * 50 * 26, nworld * slots * 12,
               nworld * contacts * 65, nworld * requests * 4,
               nworld * stages * 6)
    if max(lengths) >= 2**31:
        raise ValueError('int32 device offset overflow')
    estimated = nworld * (calls * 50 * 2048 + slots * 512 +
        contacts * 4096 + requests * 2048 + stages * 512 +
        nbody * 512 + ngeom * 1024)
    if estimated > budget:
        raise ValueError('collision preflight exceeds reserved device budget')
    return CollisionCapacityPlan(nworld, nbody, ngeom, slots, calls, pairs,
        stages, requests, contacts, estimated, budget, reserve, total)



def plan_model_capacity(static, nworld, *, total_device_budget_bytes,
                        reserved_other_bytes):
    """Source-filter candidate capacities for new models, not old grants.

    Requests cover every unordered geom pair plus explicit records. Calls cover
    the source-filter potential including explicit pairs, never an observed
    contact count. Contacts use the original model's positive nconmax; if the
    model has an unbounded sentinel, explicit caller policy is required.
    """
    from .routes import analyze_model_potential
    analysis = analyze_model_potential(static, count_only=True)
    ngeom, nbody = len(static['geom_size']), len(static['body_geomnum'])
    npair = len(static['pair_geom1'])
    nconmax = int(static['nconmax'])
    if nconmax < 0:
        raise ValueError('unbounded model contact capacity requires explicit policy')
    potential = sum(route['pair_count'] for route in analysis['eligible_routes'])
    requests = max(1, ngeom * (ngeom - 1) // 2 + npair + 16)
    calls = max(1, potential + 16)
    stages = max(1, nbody * (nbody - 1) // 2 + npair + 8)
    return plan_capacity(static, nworld,
        total_device_budget_bytes=total_device_budget_bytes,
        reserved_other_bytes=reserved_other_bytes,
        slots=requests, req_slots=requests, call_capacity=calls,
        pair_capacity=stages, stage_slots=stages,
        contact_capacity=max(1, nconmax))
class CollisionRuntimeAdapter:
    """One retained model/workspace/source graph, reusable across all substeps.

    Device error and contact semantics are exactly those of the accepted chain.
    Explicit boundary reset never clears contact buffers, scratch or ncon here.
    """
    def __init__(self, static, plan, *, source_owner, source_paths, stream,
                 device='cuda:0', borrow_detector_raw=True, capsule_source_root=None):
        if not isinstance(plan, CollisionCapacityPlan):
            raise TypeError('CollisionCapacityPlan required')
        # Recompute against the actual model to reject a stale foreign plan.
        actual = plan_capacity(static, plan.nworld,
            total_device_budget_bytes=plan.total_device_budget_bytes,
            reserved_other_bytes=plan.reserved_other_bytes,
            **{name: getattr(plan, name) for name in (
                'slots', 'call_capacity', 'pair_capacity', 'stage_slots',
                'req_slots', 'contact_capacity')})
        if actual != plan:
            raise ValueError('capacity plan does not match the actual model')
        from .chain import CollisionChainWorkspace
        self.plan, self.stream = plan, stream
        self.workspace = CollisionChainWorkspace(static, plan.nworld,
            device=device, borrow_detector_raw=borrow_detector_raw,
            **plan.workspace_kwargs())
        self.candidate_manifest = None
        if capsule_source_root is not None:
            from scripts.gpu_c_runtime.native_capsule_extension import prepare_workspace_extension
            self.candidate_manifest = prepare_workspace_extension(self.workspace, capsule_source_root)
        self.workspace.bind_device_inputs(source_owner, source_paths, stream=stream)
        self.contacts = self.workspace.contact_field_views()
        self.source_owner = source_owner
        self.source_paths = dict(source_paths)

    @property
    def ncon(self):
        return self.workspace.finalize.ncon

    @property
    def contact_f(self):
        return self.workspace.finalize.contact_f

    @property
    def contact_i(self):
        return self.workspace.finalize.contact_i

    @property
    def effective_active(self):
        return self.workspace.effective_active

    @property
    def first_error(self):
        return {'stage': self.workspace.fe_stage, 'code': self.workspace.fe_code,
                'item': self.workspace.fe_item}

    def launch(self, stream=None):
        self.workspace.launch(stream=self.stream if stream is None else stream)

    def reset_errors(self, mask):
        # Only the parent lifecycle may request this boundary operation.
        self.workspace.reset_errors(mask, stream=self.stream)

    def capture_graph(self):
        return self.workspace.capture_graph()

    def launch_graph(self, graph):
        self.workspace.launch_graph(graph)

    def resident_metadata(self):
        return self.workspace.resident_metadata()

    def manifest(self):
        # Host allocation metadata only; no production physics output reads.
        return {'schema': 'collision-runtime-adapter-v1',
                'capacity': self.plan.manifest() if self.plan is not None else None,
                'candidate_binding': self.candidate_manifest,
                'binding': self.workspace.validate_device_bindings(self.stream),
                'failure_contract': 'sticky first cause; failed ncon=0; contact bytes held; scratch follows stage-specific source masks',
                'contacts_layout': 'contact_f AoS stride=65; contact_i AoS stride=5; field views are strided'}

    @classmethod
    def from_workspace(cls, workspace, *, stream, source_owner, source_paths):
        """Wrap a cold prepared workspace without duplicating its allocations."""
        if workspace._device_input_binding is None:
            workspace.bind_device_inputs(source_owner, source_paths, stream=stream)
        else:
            binding = workspace._device_input_binding
            if binding.owner is not source_owner or binding.paths != source_paths:
                raise ValueError('workspace is already bound to another source graph')
            workspace.validate_device_bindings(stream)
        result = cls.__new__(cls)
        result.workspace, result.stream, result.plan = workspace, stream, None
        result.source_owner, result.source_paths = source_owner, dict(source_paths)
        result.contacts = workspace.contact_field_views()
        result.candidate_manifest = getattr(workspace, '_capsule_candidate_binding', None)
        return result

    def enqueue(self, mask=None, stream=None):
        if mask is not None:
            binding = self.workspace._device_input_binding
            if mask is not binding.source['active'] and mask is not self.workspace.active:
                raise ValueError('enqueue mask must be the retained F active allocation')
        self.launch(stream)

    def contact_field_views(self):
        return self.workspace.contact_field_views()

    def outputs(self):
        return {'ncon': self.ncon, 'contact_f': self.contact_f,
                'contact_i': self.contact_i, 'effective_active': self.effective_active,
                'first_error': self.first_error, 'fields': self.contacts.fields,
                'field_owner': self.contacts, 'status': self.workspace.finalize.status}





