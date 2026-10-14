"""Source-derived local paired models for the P06-C chain.

Local XML models exercise the device chain against same-call original
MuJoCo mj_collision detector hooks, all 15 final contact fields and counters.
Cylinder/box/mesh native routes are implemented; capsule-box is an explicit
unsupported counterexample. This local capture is not the full REF corpus.
"""
from __future__ import annotations

import numpy as np

# Local model: world plane + boxes/spheres/capsules/cylinders on free bodies.
# Single-geom bodies exercise the flat invocation path; a two-geom body
# exercises the tree path (route=1 with contactcompare segment sort).
BASE_XML = '''<mujoco model="p06c_local">
<option collision="all"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" contype="1" conaffinity="1"/>
  <body name="b_sph" pos="0 0 1"><freejoint/>
    <geom name="sph" type="sphere" size="0.3" contype="1" conaffinity="1"/>
  </body>
  <body name="b_box" pos="0.4 0 0.7"><freejoint/>
    <geom name="box" type="box" size="0.2 0.2 0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="b_cyl" pos="-0.5 0 0.9"><freejoint/>
    <geom name="cyl" type="cylinder" size="0.15 0.25" contype="1" conaffinity="1"/>
  </body>
  <body name="b_cap" pos="0 0.5 1.2"><freejoint/>
    <geom name="cap" type="capsule" size="0.1 0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="b_two" pos="0.2 -0.4 1.5"><freejoint/>
    <geom name="t1" type="sphere" size="0.15" contype="1" conaffinity="1"/>
    <geom name="t2" type="box" size="0.1 0.1 0.1" pos="0 0 0.3" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# Unsupported-route model: capsule-box pair must take the explicit gate
# (mjc_CapsuleBox is not implemented).
GATE_XML = '''<mujoco model="p06c_gate">
<option collision="all"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
  <body name="b1" pos="0 0 0.5"><freejoint/>
    <geom name="g1" type="capsule" size="0.15 0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="b2" pos="0.15 0 0.9"><freejoint/>
    <geom name="g2" type="box" size="0.2 0.2 0.2" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# Zero-return route: sphere-sphere called but disjoint (and mask-reject cases).
ZERO_XML = '''<mujoco model="p06c_zero">
<option collision="all"/>
<worldbody>
  <body name="z1" pos="0 0 0"><freejoint/>
    <geom name="s1" type="sphere" size="0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="z2" pos="5 0 0"><freejoint/>
    <geom name="s2" type="sphere" size="0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="z3" pos="0 0 0.25"><freejoint/>
    <geom name="s3" type="sphere" size="0.2" contype="0" conaffinity="0"/>
  </body>
</worldbody>
</mujoco>'''

# Plane-route coverage: floor under box/cylinder/capsule/sphere/mesh-shape.
# Poses are set below (cases list) so each pair produces known contacts.
PLANES_XML = '''<mujoco model="p06c_planes">
<option collision="all"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
  <body name="pb" pos="0 0 0"><freejoint/>
    <geom name="pbox" type="box" size="0.2 0.2 0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="pc" pos="1 0 0"><freejoint/>
    <geom name="pcyl" type="cylinder" size="0.15 0.25" contype="1" conaffinity="1"/>
  </body>
  <body name="pk" pos="2 0 0"><freejoint/>
    <geom name="pcap" type="capsule" size="0.1 0.2" contype="1" conaffinity="1"/>
  </body>
  <body name="ps" pos="3 0 0"><freejoint/>
    <geom name="psph" type="sphere" size="0.18" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# Multi-contact ordering: three bodies whose floor contacts must follow
# body-pair signature order and plane-detector corner order.
ORDER_XML = '''<mujoco model="p06c_order">
<option collision="all"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
  <body name="o1" pos="0.3 0 0.12"><freejoint/>
    <geom name="o1g" type="box" size="0.15 0.15 0.15" contype="1" conaffinity="1"/>
  </body>
  <body name="o2" pos="-0.4 0 0.1"><freejoint/>
    <geom name="o2g" type="box" size="0.15 0.15 0.15" contype="1" conaffinity="1"/>
  </body>
  <body name="o3" pos="0 0.35 0.08"><freejoint/>
    <geom name="o3g" type="cylinder" size="0.12 0.12" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# Tree-segment sorting: a 2-geom body whose tree path yields two calls whose
# contactcompare order differs from invocation order when types invert.
TREE_XML = '''<mujoco model="p06c_tree">
<option collision="all"/>
<worldbody>
  <body name="tb" pos="0 0 0"><freejoint/>
    <geom name="tg1" type="box" size="0.1 0.1 0.1" pos="0 0 0" contype="1" conaffinity="1"/>
    <geom name="tg2" type="sphere" size="0.1" pos="0 0.28 0" contype="1" conaffinity="1"/>
  </body>
  <body name="tr" pos="0 0.02 0.14"><freejoint/>
    <geom name="trg" type="sphere" size="0.12" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# Predefined <pair> route: two overlapping spheres with an explicit pair.
PAIR_XML = '''<mujoco model="p06c_pair">
<option collision="all"/>
<worldbody>
  <body name="pa" pos="0 0 0"><freejoint/>
    <geom name="ga" type="sphere" size="0.3" contype="1" conaffinity="1"/>
  </body>
  <body name="pb" pos="0.35 0 0"><freejoint/>
    <geom name="gb" type="sphere" size="0.3" contype="1" conaffinity="1"/>
  </body>
</worldbody>
<contact>
  <pair geom1="ga" geom2="gb" margin="0.02"/>
</contact>
</mujoco>'''

# Mesh plane route: icosahedron above a plane exercises plane_convex mesh
# branch (local/global hull ids, maxplanemesh bound).
MESH_XML = '''<mujoco model="p06c_mesh">
<option collision="all"/>
<asset>
  <mesh name="ico" vertex="0 0 1  0.943 0 -0.333  -0.471 0.816 -0.333  -0.471 -0.816 -0.333  0.471 0.816 0.333  -0.471 0.816 0.333  -0.943 0 0.333  -0.471 -0.816 0.333  0.471 -0.816 0.333  0.471 0.816 -0.333  0 -0.1 -1"/>
</asset>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
  <body name="mb" pos="0 0 0.85"><freejoint/>
    <geom name="mg" type="mesh" mesh="ico" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

# C-P06 detector route coverage: cylinder-cylinder and box-box
CYL_XML = '''<mujoco model="p06c_cyl">
<option collision="all"/>
<worldbody>
  <body name="c1" pos="0 0 0"><freejoint/>
    <geom name="g1" type="cylinder" size="0.3 0.3" contype="1" conaffinity="1"/>
  </body>
  <body name="c2" pos="0.4 0 0"><freejoint/>
    <geom name="g2" type="cylinder" size="0.3 0.3" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

BOXBOX_XML = '''<mujoco model="p06c_boxbox">
<option collision="all"/>
<worldbody>
  <body name="b1" pos="0 0 0"><freejoint/>
    <geom name="g1" type="box" size="0.3 0.3 0.3" contype="1" conaffinity="1"/>
  </body>
  <body name="b2" pos="0.4 0 0"><freejoint/>
    <geom name="g2" type="box" size="0.3 0.3 0.3" contype="1" conaffinity="1"/>
  </body>
</worldbody>
</mujoco>'''

CASES = [
    ('base', BASE_XML, False),
    ('gate', GATE_XML, True),
    ('zero', ZERO_XML, False),
    ('planes', PLANES_XML, False),
    ('order', ORDER_XML, False),
    ('tree', TREE_XML, False),
    ('pair', PAIR_XML, False),
    ('mesh', MESH_XML, False),
    ('cyl_cyl', CYL_XML, False),
    ('boxbox', BOXBOX_XML, False),
]


def static_from_model(m) -> dict:
    """Extract the chain's static model arrays from a MuJoCo MjModel."""
    import mujoco
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    ngeom = int(m.ngeom)
    nbody = int(m.nbody)
    npair = int(m.npair)
    # dispatch table from the live library (frozen 2.3.7 image)
    import ctypes as ct
    lib = ct.CDLL(str(mujoco.__path__[0]) + '/libmujoco.so.2.3.7')
    table = (ct.c_void_p * 64).in_dll(lib, 'mjCOLLISIONFUNC')
    present = np.array([[int(bool(table[8 * r + c])) for c in range(8)] for r in range(8)], dtype=np.int32)
    # mesh data
    mesh_vert = np.ascontiguousarray(m.mesh_vert, dtype=np.float32).ravel() if m.nmesh else np.zeros(0, np.float32)
    mesh_vertadr = i32(m.mesh_vertadr) if m.nmesh else np.zeros(0, np.int32)
    mesh_vertnum = i32(m.mesh_vertnum) if m.nmesh else np.zeros(0, np.int32)
    mesh_graphadr = i32(m.mesh_graphadr) if m.nmesh else np.zeros(0, np.int32)
    mesh_graph = i32(m.mesh_graph) if m.nmesh else np.zeros(0, np.int32)
    # Body trees occupy the first sum(body_bvhnum) BVH nodes (mesh trees follow).
    body_nodes = int(np.sum(m.body_bvhnum))
    return {
        'body_geomadr': i32(m.body_geomadr), 'body_geomnum': i32(m.body_geomnum),
        'body_weldid': i32(m.body_weldid), 'body_parentid': i32(m.body_parentid),
        'body_bvhadr': i32(m.body_bvhadr), 'body_bvhnum': i32(m.body_bvhnum),
        'geom_bodyid': i32(m.geom_bodyid), 'geom_type': i32(m.geom_type),
        'geom_contype': i32(m.geom_contype), 'geom_conaffinity': i32(m.geom_conaffinity),
        'geom_size': f64(m.geom_size.reshape(ngeom, 3)),
        'geom_dataid': i32(m.geom_dataid),
        'geom_rbound': f64(m.geom_rbound), 'geom_margin': f64(m.geom_margin),
        'geom_gap': f64(m.geom_gap), 'geom_priority': i32(m.geom_priority),
        'geom_condim': i32(m.geom_condim),
        'geom_friction': f64(m.geom_friction.reshape(ngeom, 3)),
        'geom_solmix': f64(m.geom_solmix),
        'geom_solref': f64(m.geom_solref.reshape(ngeom, 2)),
        'geom_solimp': f64(m.geom_solimp.reshape(ngeom, 5)),
        'geom_aabb': f64(m.geom_aabb.reshape(ngeom, 6)),
        'bvh_aabb': f64(m.bvh_aabb.reshape(-1, 6)[:body_nodes]),
        'bvh_child': i32(m.bvh_child.reshape(-1, 2)[:body_nodes]),
        'bvh_geomid': i32(m.bvh_geomid[:body_nodes]),
        'pair_geom1': i32(m.pair_geom1) if npair else np.zeros(0, np.int32),
        'pair_geom2': i32(m.pair_geom2) if npair else np.zeros(0, np.int32),
        'pair_dim': i32(m.pair_dim) if npair else np.zeros(0, np.int32),
        'pair_margin': f64(m.pair_margin) if npair else np.zeros(0, np.float64),
        'pair_gap': f64(m.pair_gap) if npair else np.zeros(0, np.float64),
        'pair_friction': f64(m.pair_friction.reshape(npair, 5)) if npair else np.zeros((0, 5), np.float64),
        'pair_solref': f64(m.pair_solref.reshape(npair, 2)) if npair else np.zeros((0, 2), np.float64),
        'pair_solreffriction': f64(m.pair_solreffriction.reshape(npair, 2)) if npair else np.zeros((0, 2), np.float64),
        'pair_solimp': f64(m.pair_solimp.reshape(npair, 5)) if npair else np.zeros((0, 5), np.float64),
        'pair_signature': i32(m.pair_signature) if npair else np.zeros(0, np.int32),
        'exclude_signature': i32(m.exclude_signature) if m.nexclude else np.zeros(0, np.int32),
        'override': i32([int(bool(m.opt.enableflags & 1))]),
        'o_margin': f64([m.opt.o_margin]),
        'o_solref': f64(m.opt.o_solref),
        'o_solimp': f64(m.opt.o_solimp),
        'dispatch_present': present,
        'mesh_vert': mesh_vert, 'mesh_vertadr': mesh_vertadr,
        'mesh_vertnum': mesh_vertnum, 'mesh_graphadr': mesh_graphadr,
        'mesh_graph': mesh_graph,
        'filterparent_disabled': int(bool(m.opt.disableflags & 512)),
        'midphase_disabled': int(bool(m.opt.disableflags & 8192)),
        'contact_disabled': int(bool(m.opt.disableflags & 16)),
        'constraint_disabled': int(bool(m.opt.disableflags & 1)),
        'nconmax': int(m.nconmax) if hasattr(m, 'nconmax') else 5000,
        'collision_mode': int(m.opt.collision),
        'mpr_iterations': int(m.opt.mpr_iterations),
        'mpr_tolerance': float(m.opt.mpr_tolerance),
        'multiccd': int(bool(m.opt.enableflags & 16)),
    }


def poses_from_model(m, d):
    """Current geom/body poses as chain input arrays.

    Returns (xpos, xmat, active, xipos, ximat) — body inertial poses included
    for the C04 tree OBB tests (R01-23).
    """
    xpos = np.ascontiguousarray(d.geom_xpos.reshape(1, m.ngeom, 3), dtype=np.float64)
    xmat = np.ascontiguousarray(d.geom_xmat.reshape(1, m.ngeom, 9), dtype=np.float64)
    xipos = np.ascontiguousarray(d.xipos.reshape(1, m.nbody, 3), dtype=np.float64)
    ximat = np.ascontiguousarray(d.ximat.reshape(1, m.nbody, 9), dtype=np.float64)
    return xpos, xmat, np.array([1], np.int32), xipos, ximat


def cpu_reference(m, d):
    from scripts.gpu_phase06_collision.local_capture import cpu_capture
    return cpu_capture(m, d)

# Additional legal original-driver zero-return partial-frame witnesses.
BOXBOX_ZERO_FRAME_XML = '<mujoco><option collision="predefined"/><contact><pair geom1="a" geom2="b" margin="0"/></contact><worldbody>\n<body><freejoint/><geom name="a" type="box" size=".3 .27 .2"/></body>\n<body><freejoint/><geom name="b" type="box" size=".21 .36 .16"/></body>\n</worldbody></mujoco>'
BOXBOX_ZERO_FRAME_QPOS_HEX = ('000000000000000000000000000000000000000000000000046ffce4980fdd3f7b3b846df99ad43fecd5dcff698bc83ff5080a15c9dde93fdc76791ce684d1bfe6c414f66dc0e33fb8606c25cbe8cf3f1930381ee5cde3bf2216d3e6af3ca4bf31c035b480d2e63f4d43c53ba2e9d4bf', '00000000000000000000000000000000000000000000000059bf1869482283bf6baa2fd0c35fddbfdce6864b3c01e13f705fabc44ac8e63f743c8109fb64d4bf9bbe999104b3d2bff3f40f5f1472e03f2b9bf5cf9160d2bf010482244d5388bf1e29ae59b0cfd33fd1bebb8c5601ed3f')
