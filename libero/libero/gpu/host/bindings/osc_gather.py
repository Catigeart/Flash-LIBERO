"""Handwritten native-count OSC gather, including producer gates and publication."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
import warp as wp
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics import runtime_adapter as r

HEADER = (GPU_ROOT / 'cuda/control/osc_gather.cuh')

def _body_dofs(j):
    """Cold CSR from the admitted immutable ancestor owner, shared by worlds."""
    from libero.gpu.model.topology import prepare_ancestor_topology
    topology=getattr(j, '_cuda_osc_body_topology', None)
    if topology is None:
        topology=prepare_ancestor_topology(j.ancestor.numpy(), device=j.ancestor.device)
        j._cuda_osc_body_topology=topology
    return [topology.arrays['body_dof_offsets'],topology.arrays['body_dof_indices']]

def stages(j,*,dataflow_profile="compatible"):
    dofs=_body_dofs(j)
    if dataflow_profile=="fast":
        return [("cf_osc_gather::local_selected",[j.cdof,j.subtree_com,j.site_pos,j.active,
            j.ancestor,j.root,j.body,j.nv,j.nbody,j.jacp,j.jacr,j.jac_status,j.adapter_active,
            j.qpos,j.qvel,j.site_mat,j.qM,j.qfrc_bias,j.qi,j.vi,j.sparse,j.nq,j.nM,
            j.ee_pos,j.ee_mat,j.xvelp,j.xvelr,j.joint_pos,j.joint_vel,j.J,j.M,j.bias,
            j.adapter_status,j.status]+dofs,"block")]
    """Jacobian workspace stages; direct arrays retain native selectors."""
    return [
        ('cf_osc_gather::jacobian', [j.cdof,j.subtree_com,j.site_pos,j.active,
            j.ancestor,j.root,j.body,j.nv,j.nbody,j.jacp,j.jacr,j.jac_status]+dofs, 'block'),
        ('cf_osc_gather::route', [j.active,j.jac_status,j.adapter_active], 'lane0'),
        ('cf_osc_gather::selected', [j.qpos,j.qvel,j.site_pos,j.site_mat,j.jacp,j.jacr,
            j.qM,j.qfrc_bias,j.adapter_active,j.qi,j.vi,j.sparse,j.nq,j.nv,j.nM,
            j.ee_pos,j.ee_mat,j.xvelp,j.xvelr,j.joint_pos,j.joint_vel,j.J,j.M,j.bias,
            j.adapter_status], 'block'),
        ('cf_osc_gather::status', [j.jac_status,j.adapter_status,j.status], 'lane0')]

def install(adapter, root,*,dataflow_profile="compatible"):
    """Replace generic adapter gather as one kernel, plus stream mask copy.

    A private immutable mask owner keeps arbitrary admitted call-time masks
    graph-safe; copying occurs on the original stream before the kernel.
    Fixed-model D07 has a different selector schema and is explicitly rejected.
    """
    if not hasattr(adapter, 'jacobian'):
        raise TypeError('handwritten OSC gather requires native-count generic dynamics')
    j=adapter.jacobian
    adapter.validate();j.validate_bindings()
    mask=wp.zeros(adapter.capacity,dtype=wp.int32,device=adapter.device)
    chain=[('cf_osc_gather::gate',[mask,adapter.first_error,adapter.state_epoch,
        adapter.source_state_epoch,adapter.forward_epoch,adapter.valid['bias'],
        adapter.written['bias'],adapter.gate_control], 'lane0'),
        ('cf_osc_gather::site',[adapter.position.frames.site_xpos,
        adapter.position.frames.site_xmat,adapter.gate_control,adapter.site_position,
        adapter.site_rotation,adapter.nsite,int(adapter.robot['eef_site_id'])], 'lane0')]
    chain+=stages(j,dataflow_profile=dataflow_profile)
    chain.append(('cf_osc_gather::publish_velocity',[adapter.gate_control,j.status,
        j.xvelp,j.xvelr,adapter.control.phys_ee_velocity,adapter.first_error], 'lane0'))
    plan=NativePlan('osc_gather',[HEADER],chain,adapter.capacity,root,defines={'OSC_NV':j.nv})
    adapter._cuda_osc_gather_plan=plan
    bias_mask=wp.zeros(adapter.capacity,dtype=wp.int32,device=adapter.device)
    bias_plan=NativePlan('osc_live_bias',[HEADER],[('cf_osc_gather::live_bias',[
        bias_mask,adapter.first_error,adapter.state_epoch,adapter.source_state_epoch,
        adapter.forward_epoch,adapter.valid['bias'],adapter.written['bias'],
        adapter.public['qfrc_bias'],adapter.control.phys_bias,adapter._arm_indices,
        adapter.controller_parent_valid,adapter.controller_parent_written,adapter.nv],
        'lane0')],adapter.capacity,root)
    adapter._cuda_osc_bias_plan=bias_plan
    def gather(self, requested, stream=None):
        self.validate(stream);self.jacobian.validate_bindings()
        r._array(requested,(self.capacity,),wp.int32,self.device)
        wp.copy(mask,requested,stream=self.stream)
        self._cuda_osc_gather_plan.launch(self.stream)
    def bound(self, mask, stream=None):
        return gather(self,mask,stream)
    adapter.gather_controller_inputs=types.MethodType(bound,adapter)
    def live_bias(self,mask,stream=None):
        self.validate(stream);r._array(mask,(self.capacity,),wp.int32,self.device)
        wp.copy(bias_mask,mask,stream=self.stream)
        self._cuda_osc_bias_plan.launch(self.stream)
    adapter.gather_live_bias=types.MethodType(live_bias,adapter)
    return {'gather':plan.manifest,'live_bias':bias_plan.manifest}

def install_jacobian(workspace, root):
    """Optional standalone use with exactly the original workspace contract."""
    workspace.validate_bindings()
    plan=NativePlan('osc_jacobian',[HEADER],stages(workspace),workspace.capacity,root)
    workspace._cuda_jacobian_plan=plan
    def launch(self,stream=None):
        self.validate_bindings();self._cuda_jacobian_plan.launch(stream)
    workspace.launch=types.MethodType(launch,workspace)
    return plan.manifest

def _validate_main():
    """Bounded same-input comparison against original real generic adapter."""
    import argparse,json,os,sys
    import numpy as np
    import mujoco
    from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
    require_canonical_gpu_namespace()
    from libero.gpu.model.factory import BuiltTaskModel
    from libero.gpu.host.runtime.step.shape_factory import _native_intake_package, default_integration_state, actual_decoder_record
    from libero.gpu.host.assembly.pipeline.factory import prepare_fused_physics
    scripts=(REPO_ROOT / 'scripts/gpu_fusion')
    sys.path.insert(0,str(scripts))
    from validate_dynamics_r03 import MODEL_ROOT,arrays
    from cuda_compare import compare
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--cache')
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'work'/'tmp').mkdir(parents=True);os.chdir(out/'work')
    wp.config.kernel_cache_dir=args.cache or str(out/'warp-cache')
    os.environ['P06_STEP_COLD_COPY']='cn'
    metadata=json.loads((MODEL_ROOT/'actual-native-metadata.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(MODEL_ROOT/'actual-model.mjb'))
    built=BuiltTaskModel(model,'',None,metadata)
    built.metadata['decoder_record']=actual_decoder_record(built)
    spec,package=_native_intake_package(built)
    bundle=prepare_fused_physics(built,state=default_integration_state(model),
        model_spec=spec,constraint_package=package,mode='diagnostic',with_controller=True)
    report={'passed':False,'cases':[]}
    try:
        bundle.controller.bootstrap_from_current_physics(bundle.runtime.intent)
        d=bundle.dynamics;refs=arrays(d)
        d.qvel.assign(np.tile(np.linspace(-.002,.001,d.nv),(d.capacity,1)))
        old={name:getattr(d,name) for name in ('gather_controller_inputs','gather_live_bias')}
        manifests=install(d,out/'native-build')
        requested=wp.array(np.ones(d.capacity,np.int32),dtype=wp.int32,device=d.device)
        wp.synchronize_stream(d.stream)
        clean={k:a.numpy() for k,a in refs.items()}
        for mode in ('normal','inactive','stale'):
            for k,a in refs.items():a.assign(clean[k])
            requested.assign(np.full(d.capacity,0 if mode=='inactive' else 1,np.int32))
            if mode=='stale':d.source_state_epoch.assign(d.state_epoch.numpy()-1)
            # Exercise velocity and force gather with nonzero held publication.
            d.control.phys_ee_velocity.assign(np.full((d.capacity,6),91.))
            d.control.phys_bias.assign(np.full((d.capacity,7),92.))
            d.controller_parent_written.assign(np.full(d.capacity,89,np.int64))
            for name in old:
                wp.synchronize_stream(d.stream);before={k:a.numpy() for k,a in refs.items()}
                old[name](mask=requested);wp.synchronize_stream(d.stream)
                baseline={k:a.numpy() for k,a in refs.items()}
                for k,a in refs.items():a.assign(before[k])
                getattr(d,name)(mask=requested);wp.synchronize_stream(d.stream)
                candidate={k:a.numpy() for k,a in refs.items()}
                verdicts={k:compare(baseline[k],candidate[k]) for k in baseline}
                failed={k:v for k,v in verdicts.items() if not v['passed']}
                if mode=='normal' and name=='gather_controller_inputs':
                    assert np.all(d.gate_control.numpy()==1) and np.all(d.jacobian.status.numpy()==0)
                if mode in ('inactive','stale'):
                    assert np.array_equal(candidate['root.control.phys_ee_velocity'],before['root.control.phys_ee_velocity'])
                    assert np.array_equal(candidate['root.control.phys_bias'],before['root.control.phys_bias'])
                case={'mode':mode,'method':name,'compared':len(refs),'passed':not failed,
                    'max_abs':max(v.get('max_abs',0.) for v in verdicts.values()),'failures':failed}
                report['cases'].append(case);print(json.dumps(case),flush=True)
                np.savez_compressed(out/(mode+'-'+name+'-baseline.npz'),**baseline)
                np.savez_compressed(out/(mode+'-'+name+'-candidate.npz'),**candidate)
                if failed:raise AssertionError((mode,name,list(failed)))
                for k,a in refs.items():a.assign(baseline[k])
        report.update(passed=True,manifests=manifests,nv=d.nv,nq=d.nq,nworld=d.capacity)
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        (out/'REPORT.json').write_text(json.dumps(report,indent=2));bundle.close()

if __name__=='__main__':_validate_main()
