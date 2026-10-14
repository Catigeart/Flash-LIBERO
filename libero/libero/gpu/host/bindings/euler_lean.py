"""Explicit E01-only fast plan. Cold installation and public default stay off.

Compatible diagnostics continue to use the original Euler binding. This plan
retains the original gate, publication, warnings, used_acc and held semantics.
E02--E09 switches and their metadata/private caches are absent from this ABI.
"""
from pathlib import Path
import re
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.bindings.euler import _defines, _joint_values, _values, stage

HEADER=GPU_ROOT/"cuda/integration/euler_lean.cuh"
_BINDINGS=dict(qp="qpos",qv="qvel",qm="qM",smooth="qfrc_smooth",
    constraint="qfrc_constraint",acc="qacc",damping="dof_damping",madr="dof_Madr",
    parent="dof_parentid",count="mass_count",ma="mass_active",damp="damped",
    ms="mass_status",wc="mass_warning_count",wl="mass_warning_last",jt="jnt_type",
    padr="jnt_qposadr",vadr="jnt_dofadr",used="used_acc",nj="njnt",qh="qH",
    qhi="qHDiagInv",offsets="joint_type_offsets",ids="joint_type_indices")


def prepare_lean(e,root,*,runtime=None,extra_flags=()):
    """Create an opt-in plan; never replaces an existing callback/default.

    Runtime tape already supplies the original gate in its surrounding plan.
    The local no-tape qualification replays gate in this plan. Installation
    requires a separately qualified exact source/model/profile receipt.
    """
    values=_joint_values(e,_values(e))
    definitions=dict(_defines(e,"fast"))
    definitions["CF_EULER_LEAN_JOINT_PARALLEL"]=int(values["joint_quat"] is not None)
    if values["joint_quat"] is None:
        values["joint_quat"]=wp.zeros(1,dtype=wp.int32,device=e.device)
    text=HEADER.read_text()
    declaration=re.search(r"void evaluate_plain\(int w,(.*?)\) \{",text,re.S).group(1)
    parts=[p.strip().rsplit(None,1) for p in declaration.split(",")]
    parameters=[(dtype+("*" if name.startswith("*") else ""),name.lstrip("*")) for dtype,name in parts]
    stages=[("cf_euler_lean::evaluate_plain",[values[_BINDINGS.get(name,name)] for _,name in parameters],
        "block",[dtype for dtype,_ in parameters])]
    if runtime is not None:
        values.update(intent=runtime.intent,state_epoch=runtime.state_epoch,
            forward_valid=runtime.forward_valid,forward_written=runtime.forward_written,
            forward_error=runtime.forward_error,first_error=runtime.first_error,
            written=runtime.written,valid=runtime.valid,expected=runtime.expected_state_epoch)
        if runtime.integration_tape is None:
            stages.insert(0,stage("gate",values))
        stages.append(stage("publish",values))
        if runtime.expected_state_epoch is not None:
            stages.append(stage("expected",values))
        if hasattr(runtime,"integration_count"):
            values["integration_count"]=runtime.integration_count
            stages.append(stage("count",values))
    plan=NativePlan("euler_lean_e01",[HEADER],stages,e.capacity,Path(root),defines=definitions,
        extra_flags=extra_flags)
    plan.euler_candidate_stage_parameters=tuple(name for _,name in parameters)
    plan.euler_candidate_stages=tuple(stages)
    plan.euler_candidate_sources=(HEADER,)
    plan.manifest.update(experimental=True,optimization_id="E01",default_enabled=False,
        diagnostic_abi="fast-public-usedacc-status-only-e01",other_optimization_switches=[])
    return plan
