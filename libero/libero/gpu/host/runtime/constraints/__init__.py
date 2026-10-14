"""P06-S integrated constraint position / impedance / reference chain.

Source baseline: MuJoCo 2.3.7 ``mj_makeConstraint`` (equality, friction, limit,
contact), ``mj_diagApprox``, ``mj_makeImpedance``, ``mj_referenceConstraint``
(engine_core_constraint.c SHA256
1717235915250442ba48b10f22d20771aecd28c66f412bb41237c6a585dbff03), on the
frozen task MJB (SHA256
3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42).

This package owns only the integrated assembly and stage entry points
``launch_position`` / ``launch_reference`` plus the error/capacity protocol and
the Newton input descriptor.  Segment arithmetic reuses the already accepted
phase-05 producers (constraint_scalar S02, constraint_contact S01,
constraint_impedance S03) as read-only kernels; nothing in those packages is
modified.  Equality/tendon row counts are structurally zero for the pinned
model and are statically proven (see identity.py); models that would need
them are rejected, never silently dropped.

Production launches never consume host-provided prediag, route, clamped
parameters, R/D/KBIP, aref or CPU oracle intermediates: prediag is recomputed
by a device kernel inside ``launch_position`` and ``launch_reference`` reads
the resident qvel buffer under an explicit epoch tag.
"""

SCHEMA_VERSION = 'phase06-constraints-v1'
