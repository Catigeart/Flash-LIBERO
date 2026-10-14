"""Static potential route table and detector support ledger for the P06-C chain.

The 8x8 present table is the boolean image of the frozen MuJoCo 2.3.7
mjCOLLISIONFUNC table. The live library symbol is read by
``local_cases.static_from_model`` at model-load time and compared against
DEFAULT_PRESENT; this module itself does not open the library (R01-26).
Type order is the original enum: PLANE=0 HFIELD=1 SPHERE=2 CAPSULE=3
ELLIPSOID=4 CYLINDER=5 BOX=6 MESH=7.

Static potential (what the table allows) is kept separate from observed /
local / not_observed runtime evidence; this module never records runtime
observation as static capability.
"""
from __future__ import annotations

import numpy as np

# Original enum order; names follow the engine_collision_driver.c table comment.
GEOM_NAMES = ('PLANE', 'HFIELD', 'SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH')
GEOM_PLANE, GEOM_HFIELD, GEOM_SPHERE, GEOM_CAPSULE = 0, 1, 2, 3
GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH = 4, 5, 6, 7

# Boolean image of the frozen mjCOLLISIONFUNC table (function != NULL).
# Identical to libero.gpu.reference.collision.collision_dispatch.kernel.DEFAULT_PRESENT; both are
# checked against the live library symbol at prepare time.
DEFAULT_PRESENT = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int32)

# Original function name per present slot (driver table). Absent slots are
# empty strings and must take the dispatch empty-slot path, never a call.
_ROW_FUNCS = (
    # row PLANE
    ('', '', 'mjc_PlaneSphere', 'mjc_PlaneCapsule', 'mjc_PlaneConvex',
     'mjc_PlaneCylinder', 'mjc_PlaneBox', 'mjc_PlaneConvex'),
    # row HFIELD
    ('', '', 'mjc_ConvexHField', 'mjc_ConvexHField', 'mjc_ConvexHField',
     'mjc_ConvexHField', 'mjc_ConvexHField', 'mjc_ConvexHField'),
    # row SPHERE
    ('', '', 'mjc_SphereSphere', 'mjc_SphereCapsule', 'mjc_Convex',
     'mjc_SphereCylinder', 'mjc_SphereBox', 'mjc_Convex'),
    # row CAPSULE
    ('', '', '', 'mjc_CapsuleCapsule', 'mjc_Convex', 'mjc_Convex',
     'mjc_CapsuleBox', 'mjc_Convex'),
    # row ELLIPSOID
    ('', '', '', '', 'mjc_Convex', 'mjc_Convex', 'mjc_Convex', 'mjc_Convex'),
    # row CYLINDER
    ('', '', '', '', '', 'mjc_Convex', 'mjc_Convex', 'mjc_Convex'),
    # row BOX
    ('', '', '', '', '', '', 'mjc_BoxBox', 'mjc_Convex'),
    # row MESH
    ('', '', '', '', '', '', '', 'mjc_Convex'),
)
ORIGINAL_FUNC = {}
for _r, _row in enumerate(_ROW_FUNCS):
    for _c, _fn in enumerate(_row):
        if DEFAULT_PRESENT[_r, _c]:
            assert _fn, (_r, _c)
            ORIGINAL_FUNC[(_r, _c)] = _fn
        else:
            assert _fn == '', (_r, _c)

# Detector implementation registry status vocabulary (frozen):
#   warp_impl            - this candidate ships a Warp implementation
#   native_not_implemented - pinned algorithm lives in a native-CUDA module;
#                          the toolchain is available on this host (nvcc 13.1
#                          V13.1.115, gcc 15.2.0) but the route is not yet
#                          implemented in this candidate; route must raise
#                          UNSUPPORTED_ROUTE, never a clean zero set
#   unsupported_gate     - deliberately outside this candidate's support scope
STATUS_WARP_IMPL = 'warp_impl'
STATUS_NATIVE_BLOCKED = 'native_not_implemented'
STATUS_GATE = 'unsupported_gate'
STATUS_ABSENT = 'absent_table_slot'

# Registry for routes this candidate implements in Warp. Keys are
# (type1, type2) with type1 <= type2 in the original enum order.
WARP_DETECTORS = {
    (GEOM_PLANE, GEOM_SPHERE): 'plane_sphere',
    (GEOM_PLANE, GEOM_CAPSULE): 'plane_capsule',
    (GEOM_PLANE, GEOM_ELLIPSOID): 'plane_ellipsoid',
    (GEOM_PLANE, GEOM_CYLINDER): 'plane_cylinder',
    (GEOM_PLANE, GEOM_BOX): 'plane_box',
    (GEOM_PLANE, GEOM_MESH): 'plane_mesh',
    (GEOM_SPHERE, GEOM_SPHERE): 'sphere_sphere',
    (GEOM_SPHERE, GEOM_CAPSULE): 'sphere_capsule',
    (GEOM_SPHERE, GEOM_CYLINDER): 'sphere_cylinder',
    (GEOM_SPHERE, GEOM_BOX): 'sphere_box',
    (GEOM_CAPSULE, GEOM_CAPSULE): 'capsule_capsule',

}

# Full pinned-source bodies compiled for CUDA, connected through device buffers.
NATIVE_DETECTORS = {
    (GEOM_CYLINDER, GEOM_CYLINDER): 'convex_mpr',
    (GEOM_CYLINDER, GEOM_BOX): 'convex_mpr',
    (GEOM_CYLINDER, GEOM_MESH): 'convex_mpr',
    (GEOM_BOX, GEOM_BOX): 'boxbox_native',
    (GEOM_BOX, GEOM_MESH): 'convex_mpr',
    (GEOM_MESH, GEOM_MESH): 'convex_mpr',
}

# Routes not yet implemented or out of scope. The toolchain is present
# (nvcc 13.1 V13.1.115, gcc 15.2.0) — the reason records the actual state.
NATIVE_BLOCKED = {
    (GEOM_SPHERE, GEOM_ELLIPSOID): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_ELLIPSOID, GEOM_CYLINDER): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_ELLIPSOID, GEOM_ELLIPSOID): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_ELLIPSOID, GEOM_BOX): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_ELLIPSOID, GEOM_MESH): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_CAPSULE, GEOM_CYLINDER): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_CAPSULE, GEOM_MESH): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_CAPSULE, GEOM_ELLIPSOID): 'mjc_Convex MPR route not required by fixed model',
    (GEOM_CAPSULE, GEOM_BOX): 'mjc_CapsuleBox not required by fixed model',
    (GEOM_HFIELD, GEOM_SPHERE): 'mjc_ConvexHField: hfield support not implemented',
    (GEOM_HFIELD, GEOM_CAPSULE): 'mjc_ConvexHField: hfield support not implemented',
    (GEOM_HFIELD, GEOM_ELLIPSOID): 'mjc_ConvexHField: hfield support not implemented',
    (GEOM_HFIELD, GEOM_CYLINDER): 'mjc_ConvexHField: hfield support not implemented',
    (GEOM_HFIELD, GEOM_BOX): 'mjc_ConvexHField: hfield support not implemented',
    (GEOM_HFIELD, GEOM_MESH): 'mjc_ConvexHField: hfield support not implemented',
}


def route_key(type1: int, type2: int) -> tuple[int, int]:
    if type1 > type2:
        type1, type2 = type2, type1
    return int(type1), int(type2)


def detector_status(type1: int, type2: int) -> tuple[str, str]:
    """Return (status, registry_key_or_reason) for one potential route."""
    key = route_key(type1, type2)
    if not DEFAULT_PRESENT[key]:
        return STATUS_ABSENT, 'mjCOLLISIONFUNC slot is NULL; dispatch empty-slot path'
    if key in NATIVE_DETECTORS:
        return 'native_cuda_impl', NATIVE_DETECTORS[key]
    if key in WARP_DETECTORS:
        return STATUS_WARP_IMPL, WARP_DETECTORS[key]
    if key in NATIVE_BLOCKED:
        return STATUS_NATIVE_BLOCKED, NATIVE_BLOCKED[key]
    return STATUS_GATE, ORIGINAL_FUNC[key] + ' not in candidate support scope'


def static_potential_routes() -> list[tuple[int, int]]:
    """All present routes of the frozen table (static potential)."""
    out = []
    for r in range(8):
        for c in range(r, 8):
            if DEFAULT_PRESENT[r, c]:
                out.append((r, c))
    return out


def model_potential_routes(geom_types) -> list[tuple[int, int]]:
    """Routes whose both endpoint types appear in the model geom set.

    Model presence is static structure, not observation: a route is listed
    even if no recorded state ever reaches it.
    """
    present = sorted({int(t) for t in np.asarray(geom_types).ravel()})
    out = []
    for i, t1 in enumerate(present):
        for t2 in present[i:]:
            key = route_key(t1, t2)
            if DEFAULT_PRESENT[key]:
                out.append(key)
    return out


def analyze_model_potential(static: dict, *, count_only: bool = False) -> dict:
    """Pose-independent source filters; never infer support from observed states.

    engine_collision_driver.c:390-405, 433-453, 556-571, 879-898.
    Explicit pairs bypass body/exclude/affinity filters. Dynamic pairs retain
    every pose-dependent broadphase/OBB/sphere possibility. ``count_only``
    returns per-route pair_count rather than materializing the pair ledger;
    the default public ledger and its explicit ordinal order are unchanged.
    """
    import hashlib
    from collections import Counter
    required = ('geom_type', 'geom_bodyid', 'geom_contype', 'geom_conaffinity',
                'body_weldid', 'body_parentid', 'exclude_signature',
                'pair_geom1', 'pair_geom2')
    arrays = {n: np.asarray(static[n]) for n in required}
    for name, arr in arrays.items():
        if arr.dtype != np.int32 or arr.ndim != 1:
            raise ValueError('static potential schema: ' + name)
    ng, nb = len(arrays['geom_type']), len(arrays['body_weldid'])
    if np.any((arrays['geom_type'] < 0) | (arrays['geom_type'] > 7)) or \
            np.any((arrays['geom_bodyid'] < 0) | (arrays['geom_bodyid'] >= nb)):
        raise ValueError('static potential geom/type/body bounds')
    mode = int(static['collision_mode'])
    if mode not in (0, 1, 2):
        raise ValueError('static potential collision mode')
    disabled = bool(static['contact_disabled'] or static['constraint_disabled'] or
                    int(static['nconmax']) == 0 or nb < 2)
    exclude = set(map(int, arrays['exclude_signature']))
    admitted, rejected = {}, Counter()
    # Suppression is unordered, while the explicit call ledger below retains
    # each original ordinal, including repeated and reversed pairs.
    explicit_keys = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in
                     zip(arrays['pair_geom1'], arrays['pair_geom2'])} if mode == 0 else set()
    def admit(g1, g2, kind, ipair=-1):
        key = route_key(arrays['geom_type'][g1], arrays['geom_type'][g2])
        if not DEFAULT_PRESENT[key]:
            rejected['absent_dispatch'] += 1
            return
        if count_only:
            admitted[key] = admitted.get(key, 0) + 1
        else:
            admitted.setdefault(key, []).append(
                {'geom1': g1, 'geom2': g2, 'source': kind, 'ipair': ipair})
    if not disabled:
        if mode in (0, 1):
            for ip, (g1, g2) in enumerate(zip(arrays['pair_geom1'], arrays['pair_geom2'])):
                if not 0 <= g1 < ng or not 0 <= g2 < ng:
                    raise ValueError('static potential explicit pair bounds')
                admit(int(g1), int(g2), 'predefined', ip)
        if mode != 1:
            for g1 in range(ng):
                b1 = int(arrays['geom_bodyid'][g1])
                for g2 in range(g1 + 1, ng):
                    b2 = int(arrays['geom_bodyid'][g2])
                    w1, w2 = int(arrays['body_weldid'][b1]), int(arrays['body_weldid'][b2])
                    p1 = int(arrays['body_weldid'][arrays['body_parentid'][w1]])
                    p2 = int(arrays['body_weldid'][arrays['body_parentid'][w2]])
                    if w1 == w2:
                        rejected['same_weld'] += 1
                        continue
                    if not static['filterparent_disabled'] and w1 != 0 and w2 != 0 and \
                            (w1 == p2 or w2 == p1):
                        rejected['weld_parent'] += 1
                        continue
                    low, high = sorted((b1, b2))
                    if ((low + 1) << 16) + high + 1 in exclude:
                        rejected['exclude_body_signature'] += 1
                        continue
                    if not ((int(arrays['geom_contype'][g1]) & int(arrays['geom_conaffinity'][g2])) |
                            (int(arrays['geom_contype'][g2]) & int(arrays['geom_conaffinity'][g1]))):
                        rejected['affinity_mask'] += 1
                        continue
                    if mode == 0 and (g1, g2) in explicit_keys:
                        rejected['explicit_pair_dynamic_suppression'] += 1
                        continue
                    admit(g1, g2, 'dynamic_pose_potential')
    bindings = {n: {'shape': list(a.shape), 'dtype': str(a.dtype),
                    'sha256': hashlib.sha256(a.tobytes()).hexdigest()} for n, a in arrays.items()}
    return {'schema': 'p06c-model-potential-v2',
            'driver_source_sha256': 'cd120732b9eea5c26c4db5dbfd8def2435c0a4c2e521dc1715264dd2b1258cd8',
            'analysis_level': 'source_static_filters_no_pose_observation',
            'callback_contactfilter': 'not admitted; original fixed-model callback must be NULL',
            'gates': {n: int(static[n]) for n in (
                'collision_mode', 'contact_disabled', 'constraint_disabled',
                'nconmax', 'filterparent_disabled')},
            'static_array_bindings': bindings,
            'type_upper_bound_routes': [list(k) for k in model_potential_routes(arrays['geom_type'])],
            'rejection_counts': dict(sorted(rejected.items())),
            'eligible_routes': [dict(type1=k[0], type2=k[1], **(
                {'pair_count': v} if count_only else {'pairs': v}))
                                for k, v in sorted(admitted.items())]}


def build_support_matrix(geom_types=None, observed=None, local=None) -> dict:
    """Build the support/observation ledger.

    observed/local map route keys (as tuples) to evidence counts from paired
    or local runs; missing keys mean not_observed / not_local. This function
    never invents counts.
    """
    observed = observed or {}
    local = local or {}
    rows = []
    analysis = analyze_model_potential(geom_types) if isinstance(geom_types, dict) else None
    model_routes = {route_key(a['type1'], a['type2']) for a in analysis['eligible_routes']} if analysis else (set(model_potential_routes(geom_types)) if geom_types is not None else set())
    for key in static_potential_routes():
        status, impl = detector_status(*key)
        obs = int(observed.get(key, 0))
        loc = int(local.get(key, 0))
        rows.append({
            'type1': key[0], 'type2': key[1],
            'type1_name': GEOM_NAMES[key[0]], 'type2_name': GEOM_NAMES[key[1]],
            'original_function': ORIGINAL_FUNC[key],
            'detector_status': status,
            'implementation': impl,
            'in_fixed_model_potential': key in model_routes,
            'observation': 'observed' if obs else 'not_observed',
            'observed_count': obs,
            'local_state': 'local' if loc else 'not_local',
            'local_count': loc,
        })
    return {
        'schema': 'phase06-c-support-matrix-v2',
        'model_potential_analysis': analysis,
        'analysis_level': analysis['analysis_level'] if analysis else 'type_upper_bound_only',
        'geom_type_order': list(GEOM_NAMES),
        'present_table': DEFAULT_PRESENT.tolist(),
        'static_potential_count': len(static_potential_routes()),
        'fixed_model_potential_count': sum(1 for r in rows if r['in_fixed_model_potential']),
        'native_cuda_impl_count': sum(1 for r in rows if r['detector_status'] == 'native_cuda_impl'),
        'warp_impl_count': sum(1 for r in rows if r['detector_status'] == STATUS_WARP_IMPL),
        'native_blocked_count': sum(1 for r in rows if r['detector_status'] == STATUS_NATIVE_BLOCKED),
        'gate_count': sum(1 for r in rows if r['detector_status'] == STATUS_GATE),
        'routes': rows,
    }
