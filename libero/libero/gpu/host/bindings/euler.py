"""Handwritten full Euler, sparse damped solve, and runtime publication."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.assembly.pipeline.euler import _values
from libero.gpu.host.runtime.step.shape_runtime import _token
HEADER=(GPU_ROOT / 'cuda/integration/euler.cuh')

# This is an ABI field list, not an arithmetic translation.
FIELDS={
 'prepare':'active qpos qvel time qM qfrc_smooth qfrc_constraint qacc dof_damping dof_Madr mass_qm mass_rhs mass_count mass_active damped status dt nq nv nm',
 'mass':'mass_qm mass_rhs mass_count mass_active dof_Madr dof_parentid mass_ld mass_inv mass_sqrt_inv mass_solution mass_pivot mass_status mass_warning_count mass_warning_last nv nm',
 'replay':'active damped mass_status mass_rhs mass_ld dof_Madr dof_parentid dof_simplenum zero_work zero_skip nv nm',
 'integrate':'active qpos qvel time qacc mass_solution mass_status damped jnt_type jnt_qposadr jnt_dofadr scratch_pos scratch_vel used_acc scratch_time axis angle increment pre_norm axis_fallback zero_angle quat_fallback status dt nq nv njnt nquat',
 'commit':'active qpos qvel time qH qHDiagInv mass_ld mass_inv mass_qm mass_rhs mass_sqrt_inv mass_solution mass_pivot mass_warning_count mass_warning_last damped scratch_pos scratch_vel used_acc scratch_time axis angle increment pre_norm axis_fallback zero_angle quat_fallback zero_work zero_skip status nq nv nm nquat',
 'gate':'intent state_epoch forward_valid forward_written forward_error first_error qpos qvel qacc active nq nv',
 'publish':'active status state_epoch written valid first_error forward_valid',
 'expected':'intent expected',
 'count':'valid integration_count',
 'shared':'active qpos qvel time qM qfrc_smooth qfrc_constraint qacc dof_damping dof_Madr mass_qm mass_rhs mass_count mass_active damped status dt nq nv nm dof_parentid mass_ld mass_inv mass_sqrt_inv mass_solution mass_pivot mass_status mass_warning_count mass_warning_last dof_simplenum zero_work zero_skip jnt_type jnt_qposadr jnt_dofadr scratch_pos scratch_vel used_acc scratch_time axis angle increment pre_norm axis_fallback zero_angle quat_fallback njnt nquat qH qHDiagInv qacc_stride',
}

def stage(name,values):
    collective = name in ('prepare', 'gate')
    function = name+'_block' if collective else name
    return ('cf_euler::'+function,[values[k] for k in FIELDS[name].split()],
            'block' if collective else 'lane0')

def _defines(e,profile):
    if profile not in ('compatible','fast'):raise ValueError('Euler dataflow profile')
    return dict(CF_EULER_NV=e.nv,CF_EULER_NQ=e.nq,CF_EULER_NM=e.nM,
                CF_EULER_QUAT_STORAGE=max(1,e.nquat),CF_EULER_DIAGNOSTICS=int(profile=='compatible'))

def _shared_bytes(e):
    quats=max(1,e.nquat)
    raw=8*(2*e.nM+13*e.nv+e.nq+1+12*quats)+4*(8+2*e.nv+3*quats)
    return (raw+7)//8*8 + 128  # integrate_block's one integer per warp

def _shared_stage(values):
    arguments = [values[k] for k in FIELDS['shared'].split()]
    if values.get('joint_quat') is not None:
        arguments.extend(values[k] for k in ('joint_quat', 'joint_type_offsets', 'joint_type_indices'))
    return ('cf_euler::evaluate_shared', arguments, 'block')

def _joint_values(e, values):
    from libero.gpu.model.topology import prepare_joint_topology
    topology = prepare_joint_topology(e.jnt_type.numpy(), e.jnt_qposadr.numpy(),
        e.jnt_dofadr.numpy(), e.device, nq=e.nq, nv=e.nv)
    e._cuda_joint_topology = topology
    proof = topology.proof
    parallel = (proof['joint_qpos_disjoint'] and proof['joint_dof_disjoint'] and
                proof['njnt'] == e.njnt and proof['joint_quat_count'] == e.nquat)
    values['joint_quat'] = topology.arrays['joint_quat_index'] if parallel else None
    values['joint_type_offsets'] = topology.arrays['joint_type_offsets']
    values['joint_type_indices'] = topology.arrays['joint_type_indices']
    return values

def _stage_barriers(stages):
    # Lane zero is sequential with itself. Every transition involving a block
    # stage retains its RAW/WAR/WAW synchronization, including event snapshots.
    return [i+1<len(stages) and (s[2]=='block' or stages[i+1][2]=='block')
            for i,s in enumerate(stages)]

def prepare_workspace(e,root,profile='compatible'):
    values=_joint_values(e,_values(e))
    values['qacc_stride']=e.nv
    stages=[_shared_stage(values)]
    return NativePlan('full_euler_shared',[HEADER],stages,e.capacity,root,defines=_defines(e,profile),stage_barriers=_stage_barriers(stages))

def install(bundle,root,*,e01_selection=None,selected_selection=None):
    r=getattr(bundle,'runtime',None)
    if r is None and hasattr(bundle,'euler'):r=bundle
    e=getattr(bundle,'euler',None) or (r.euler if r is not None else None)
    if e is None:return {'euler':'not present'}
    profile=getattr(bundle,'dataflow_profile','compatible')
    workspace=prepare_workspace(e,root,profile)
    e._cuda_euler_plan=workspace
    def launch(self,stream=None):self._cuda_euler_plan.launch(stream)
    e.launch=types.MethodType(launch,e)
    if r is None:return {'euler':workspace.manifest}
    values=_joint_values(e,_values(e))
    values['qacc_stride']=e.nv
    values.update(intent=r.intent,state_epoch=r.state_epoch,forward_valid=r.forward_valid,
                  forward_written=r.forward_written,forward_error=r.forward_error,first_error=r.first_error,
                  written=r.written,valid=r.valid,expected=r.expected_state_epoch)
    counted=hasattr(r,'integration_count')
    if counted:values['integration_count']=r.integration_count
    gate=NativePlan('integration_gate',[HEADER],[stage('gate',values)],e.capacity,root)
    stages=([stage('gate',values)] if r.integration_tape is None else [])+[_shared_stage(values),stage('publish',values)]
    if r.expected_state_epoch is not None:stages.append(stage('expected',values))
    if counted:stages.append(stage('count',values))
    event=hasattr(r,'event_after')
    event_plans=()
    sources=[HEADER]
    definitions=_defines(e,profile)
    e01_selected=False
    retained_selected=False
    retained_plan=None
    e01_core_barriers=None
    if e01_selection is not None:
        from .euler_lean_selection import use_e01
        e01_selected=use_e01(e01_selection,e,r,profile)
        if e01_selected:
            from .euler_lean import prepare_lean
            lean=prepare_lean(e,root,runtime=r)
            stages=list(lean.euler_candidate_stages)
            sources=list(lean.euler_candidate_sources)
            definitions['CF_EULER_LEAN_JOINT_PARALLEL']=1
            e01_core_barriers=list(lean.manifest['stage_barriers'])
    if selected_selection is not None:
        from .euler_selected_selection import RECIPES,use_selected
        from .euler_candidates import EulerCandidateOptions,_prepare_candidate
        recipe=getattr(selected_selection,'recipe',None)
        choices=EulerCandidateOptions(**RECIPES.get(recipe,{}))
        retained_selected=use_selected(selected_selection,e,r,profile,choices)
        if retained_selected:
            retained_plan=_prepare_candidate(e,root,choices,runtime=r)
            stages=list(retained_plan.euler_candidate_stages)
            sources=list(retained_plan.euler_candidate_sources)
            definitions.update(choices.defines(),CF_EC_JOINT_PARALLEL=1)
            e01_core_barriers=list(retained_plan.manifest['stage_barriers'])
            e01_selected=False
    fused_events=False
    if event:
        from .copies import CopyPlan
        before=CopyPlan([(r.event_sources[name],target) for name,target in r.event_after.items()],root,r.event_slot,compile_plan=False)
        warnings=CopyPlan([(r.recovery.warning_number.reshape((-1,)),r.event_warnings),
                           (r.recovery.warning_lastinfo.reshape((-1,)),r.event_lastinfo)],root,r.event_slot,compile_plan=False)
        after=CopyPlan([(r.event_sources[name],target) for name,target in r.event_after_euler.items()],root,r.event_slot,compile_plan=False)
        event_plans=(before,warnings,after)
        # An optional input tape retains its separate pre-consumer sampling
        # boundary. Keep event copies in their original wrapper in that mode.
        fused_events=r.integration_tape is None and not any(p.fallback for p in event_plans)
        if fused_events:
            if any(p.nworld!=e.capacity for p in event_plans):raise ValueError('Euler event world count differs')
            sources.extend(dict.fromkeys(p.source for p in event_plans))
            stages=before.stages+warnings.stages+stages+after.stages
    runtime_barriers=_stage_barriers(stages)
    if e01_core_barriers is not None:
        core_start=len(before.stages)+len(warnings.stages) if fused_events else 0
        runtime_barriers[core_start:core_start+len(e01_core_barriers)]=e01_core_barriers
    runtime=NativePlan('runtime_euler_shared',[*sources],stages,e.capacity,root,defines=definitions,stage_barriers=runtime_barriers)
    if retained_plan is not None:
        runtime.euler_candidate_values=retained_plan.euler_candidate_values
        runtime.euler_candidate_options=retained_plan.euler_candidate_options
        runtime.invalidate_cache=retained_plan.invalidate_cache
    def integrate(self):
        if counted and (self.integration_count is not self._count_ref or _token(self.integration_count)!=self._count_token):raise ValueError('actual integration count owner changed')
        if event and fused_events:
            for plan in event_plans:plan.validate_bindings()
            if self._event_owner_tokens()!=self._event_tokens or self.forward.n.final_event is not self._event_solver:raise ValueError('event retained owner drift')
        if event and not fused_events:
            self._copy_event(self.event_after)
            self._fusion_warning_copy.launch(self.stream)
        if self.integration_tape is not None:
            gate.launch(self.stream);self.integration_tape.capture()
        runtime.launch(self.stream)
        if event and not fused_events:self._copy_event(self.event_after_euler)
    r._integrate=types.MethodType(integrate,r)
    r._cuda_euler_plan=runtime
    r._cuda_euler_gate_plan=gate
    r._cuda_euler_event_copies=event_plans
    r._cuda_euler_shared_profile=profile
    return {'euler':workspace.manifest,'runtime_euler':runtime.manifest,'one_kernel':True,'fma':True,
            'joint_topology_sha256': e._cuda_joint_topology.source_sha256,
            'joint_parallel_disjoint': values['joint_quat'] is not None,
            'joint_mapping': 'stable cold type buckets; original quaternion diagnostic indices',
            'dataflow_profile':profile,'shared_work':True,'phase_event_copies_in_kernel':fused_events,
            'shared_work_bytes_per_block':_shared_bytes(e) if not (e01_selected or retained_selected) else None,
            'e01_lean_selected':e01_selected,
            'e01_local_receipt':'E01/attempt-03' if e01_selected else None,
            'retained_euler_selected':retained_selected,
            'retained_euler_recipe':selected_selection.recipe if retained_selected else None,
            'commit_mapping':'block flat copy; vectors then time then damped qH; publication after barrier',
            'optional_numerical_diagnostics':profile=='compatible',
            'held_public_fields':['qM','qacc','qfrc_smooth','qfrc_constraint','ctrl','qacc_warmstart','sensordata']}
