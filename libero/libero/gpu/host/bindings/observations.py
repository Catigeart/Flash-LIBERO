"""Direct handwritten CUDA observable sampling from resident world-major fields.

The math/clock implementation lives in observations.cuh. This file cold-binds raw
pointers and static descriptors; it does not translate Python arithmetic to CUDA.
Compatible mode keeps public staging diagnostics in the same kernel; fast mode
leaves these unused diagnostics untouched. Retained sample/cache semantics match.
"""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
from types import MethodType
import numpy as np
import warp as wp
from libero.gpu.host.build.native import NativePlan

RAW_FIELDS = ('qpos','qvel','body_xpos','body_xquat','site_xpos')
SCENE_FIELDS = ('xpos','xmat','geom_xpos','geom_xmat','site_xpos','site_xmat',
                'cam_xpos','cam_xmat','light_xpos','light_xdir','subtree_com',
                'site_rgba','geom_rgba')

def _element_runs(dag, strides, packoffsets):
    """Cold contiguous source/output runs, ordered by the accepted DAG layers.

    Each six-int record is (element_begin, sensor, field, source, output, count).
    Trig sources use cache addresses; raw sources never cross a resident field.
    No floating-point mirror or hot allocation is introduced.
    """
    ns=len(dag.sensor_order); cursor=0; arrays=[]
    for count in (ns,ns,ns,ns,dag.nindex,ns,ns,ns,3*ns):
        arrays.append(dag.schema[cursor:cursor+count]); cursor+=count
    op, offsets, sizes, ixoffs, indices, d0, _, _, _=arrays
    # Shifted self-overlap needs the original forward read/write sequence.
    for sensor,kind in enumerate(op):
        if kind in (1,2) and d0[sensor]!=offsets[sensor] and max(d0[sensor],offsets[sensor])<min(d0[sensor]+sizes[sensor],offsets[sensor]+sizes[sensor]):
            return None
    runs=[]; layer_runs=[0]; layer_elements=[0]; element=0
    for level in range(dag.nlayers):
        for at in range(dag.layer_offsets[level],dag.layer_offsets[level+1]):
            sensor=dag.sensor_order[at]; kind=op[sensor]
            if kind>2: continue
            j=0
            while j<sizes[sensor]:
                absolute=indices[ixoffs[sensor]+j] if kind==0 else d0[sensor]+j
                field=0
                if kind==0:
                    while field<4 and absolute>=packoffsets[field+1]: field+=1
                    source=absolute-packoffsets[field]
                else: field=-kind; source=absolute
                count=1
                while j+count<sizes[sensor]:
                    nxt=indices[ixoffs[sensor]+j+count] if kind==0 else d0[sensor]+j+count
                    if nxt!=absolute+count: break
                    if kind==0 and source+count>=strides[field]: break
                    count+=1
                runs.extend((element,sensor,field,source,offsets[sensor]+j,count))
                element+=count; j+=count
                if element>2147483647 or len(runs)>2147483647: return None
        layer_runs.append(len(runs)//6); layer_elements.append(element)
    return tuple(layer_runs),tuple(layer_elements),tuple(runs)

class DirectSampler:
    def __init__(self,observer,fields,build_root,*,dataflow_profile='compatible'):
        if dataflow_profile not in ('compatible','fast'):
            raise ValueError('unknown observation dataflow profile')
        self.observer=observer; self.profile=dataflow_profile
        self.nworld=observer.nworld
        self.baseline_sample=observer.sample
        fields=observer._flat(fields)
        self._validate(fields,observer.lowdim.all_worlds,observer.metadata['model_timestep'])
        low=observer.lowdim
        def ai(v):return wp.array(v,dtype=wp.int32,device=low.device)
        strides=[observer.raw_packer.layout[n][1] for n in RAW_FIELDS]
        packoffsets=[observer.raw_packer.layout[n][0] for n in RAW_FIELDS]
        self.strides=ai(strides);self.packoffsets=ai(packoffsets)
        # Original index metadata remains authoritative, including dynamic edits.
        meta=list(low.meta)
        args=[len(observer.plan.specs),observer.plan.width,low.source_stride,
              *meta,self.strides,self.packoffsets,
              *[fields[n] for n in RAW_FIELDS],observer.raw_packer.buffer,
              int(self.profile=='compatible'),low.cache,low.observed,low.sidequat,
              low.elapsed,low.sampled,low.present,low.all_worlds,
              float(observer.metadata['model_timestep']),0,low.events]
        self.low_field_start=16; self.low_mask=len(args)-4
        self.low_dt=len(args)-3;self.low_force=len(args)-2
        stages=[('libero_observe::lowdim',args,'lane0')]
        # Only disjoint resident owners use element-level DAG parallelism. Exact
        # structural metadata is checked on device each sample; dynamic edits
        # automatically execute the original ordered sampler in the same kernel.
        from libero.gpu.model.observation_dag import prepare_observation_dag
        targets=(low.cache,low.observed,low.sidequat,low.elapsed,low.sampled,
                 low.present,low.events,observer.raw_packer.buffer)
        aliases=any(self._overlap(a,b) for i,a in enumerate(targets)
                    for b in targets[:i]+tuple(low.meta))
        self.observation_dag=None;self.observation_dag_arrays=None
        self.element_run_arrays=None;self.element_run_bytes=0
        if not aliases:
            self.observation_dag,self.observation_dag_arrays=prepare_observation_dag(
                low,observer.plan.width,low.source_stride)
            if self.observation_dag_arrays is not None:
                schedule=_element_runs(self.observation_dag,strides,packoffsets)
                if schedule is not None:
                    run_offsets,element_offsets,runs=schedule
                    self.element_run_arrays=(ai(run_offsets),ai(element_offsets),ai(runs),ai(strides+packoffsets))
                    self.element_run_bytes=4*(len(run_offsets)+len(element_offsets)+len(runs)+10)
                    stages=[('libero_observe::lowdim_dag',args+list(self.observation_dag_arrays)+list(self.element_run_arrays),'block')]
        self.transform_stages=[]
        self.scene_stage=None
        # New depth cache semantics require capture of its RGB producer event.
        # Keep all built-in lowdim handwritten math, and use schema-v2 device
        # scene capture only for this additional supported feature path.
        self.scene_v2=bool(observer.scene_cache is not None and any(s["name"].endswith("_depth") for s in observer.plan.image_specs))
        self.scene_field_start=5
        if observer.scene_cache is not None:
            scene=observer.scene_cache
            trans=observer.scene_transforms
            if not hasattr(trans,'_scene_cache'):
                from .position_scene import install_scene
                install_scene(trans,fields['body_xpos'],fields['body_xmat'],Path(build_root)/'scene-cache')
            for label,count,static,outputs in (
                ('camera',trans.ncam,trans.camera_static,(trans.cam_xpos,trans.cam_xmat)),
                ('light',trans.nlight,trans.light_static,(trans.light_xpos,trans.light_xdir))):
                if count:
                    self.transform_stages.append((len(stages),label))
                    stages.append(('libero_scene::'+label+'_cached',
                         [trans.nbody,count,fields['body_xpos'],fields['body_xmat'],
                          low.all_worlds,*static,*outputs,*trans._scene_cache[label]],'block'))
            fields=dict(fields)
            fields.update({n:getattr(observer.scene_transforms,n) for n in
                          ('cam_xpos','cam_xmat','light_xpos','light_xdir')})
            fields.update(observer.visual_sources)
            if not self.scene_v2:
                self.scene_offsets=ai([scene.layout[n][0] for n in SCENE_FIELDS])
                self.scene_lengths=ai([int(np.prod(scene.layout[n][1])) for n in SCENE_FIELDS])
                args=[scene.ncamera,scene.stride,self.scene_offsets,self.scene_lengths,
                      *[fields[n] for n in SCENE_FIELDS],observer.scene_packer.buffer,
                      int(self.profile=='compatible'),scene.scenes,scene.periods,
                      scene.enabled,scene.elapsed,scene.sampled,scene.generation,
                      scene.mask,float(observer.metadata['model_timestep']),0]
                self.scene_field_start=4
                self.scene_mask=len(args)-3;self.scene_dt=len(args)-2;self.scene_force=len(args)-1
                self.scene_stage=len(stages)
                stages.append(('libero_observe::scenes',args,'block'))
        self.plan=NativePlan('direct_observations',[(GPU_ROOT / 'cuda/observations/observations.cuh'),(GPU_ROOT / 'cuda/observations/position_scene.cuh')],
                             stages,self.nworld,build_root,device=low.device)
        self.owner_arrays=tuple(low.meta)+(low.cache,low.observed,low.sidequat,low.elapsed,
                             low.sampled,low.present,low.events,observer.raw_packer.buffer)
        if observer.scene_cache is not None:
            sc=observer.scene_cache
            self.owner_arrays+=tuple(getattr(sc,n) for n in
                                     ('periods','enabled','elapsed','sampled','generation','scenes','parent_generation','captured'))
            self.owner_arrays+=(observer.scene_packer.buffer,)
        self.manifest=dict(self.plan.manifest,dataflow_profile=self.profile,
            scope='ordered lowdim plus retained scene, direct resident fields',
            scene_capture=('schema-v2 GPU producer-event capture' if self.scene_v2 else 'handwritten native scene'),
            raw_packing_launches=0,scene_packing_launches=0,
            staging_diagnostics=('schema-v2 scene materialized; lowdim fast staging untouched' if self.scene_v2 and self.profile=='fast' else 'materialized' if self.profile=='compatible' else 'not updated'))
        self.manifest['lowdim_schedule']=dict(
            entry=stages[0][0],layers=self.observation_dag.nlayers if self.element_run_arrays else 0,
            exact_schema_guard=bool(self.element_run_arrays),
            element_runs=bool(self.element_run_arrays),element_run_bytes=self.element_run_bytes,
            fallback=('element mapping ineligible: shifted self-overlap or int32 span' if self.observation_dag_arrays and not self.element_run_arrays else self.observation_dag.reason) if self.observation_dag else 'aliased resident owners')
    def _validate(self,fields,mask,timestep):
        ob=self.observer
        if ob.closed or not ob.valid:raise RuntimeError('observation adapter is closed or quarantined')
        if not np.isfinite(timestep) or timestep<=0:raise ValueError('sampling timestep must be positive finite')
        ob.raw_packer.validate(fields)
        if mask.dtype!=wp.int32 or mask.shape!=(ob.nworld,) or mask.device!=ob.lowdim.observed.device:
            raise ValueError('world mask must be same-device int32 vector')
        if ob.scene_cache is not None:
            lengths={'body_xmat':ob.model.nbody*9,'geom_xpos':ob.model.ngeom*3,
                     'geom_xmat':ob.model.ngeom*9,'site_xmat':ob.model.nsite*9,
                     'subtree_com':ob.model.nbody*3}
            for name,length in lengths.items():
                f=fields[name]
                if f.dtype!=wp.float64 or f.ndim!=1 or f.size!=length*ob.nworld or not f.is_contiguous or f.device!=ob.lowdim.observed.device:
                    raise ValueError('invalid camera scene field: '+name)
        for n in RAW_FIELDS:
            if not fields[n].is_contiguous:raise ValueError('contiguous observation fields required')
    @staticmethod
    def _overlap(a,b):
        return a.size and b.size and a.ptr < b.ptr+b.size*wp.types.type_size_in_bytes(b.dtype) and b.ptr < a.ptr+a.size*wp.types.type_size_in_bytes(a.dtype)

    def sample(self,raw_fields,timestep=None,force=False,world_mask=None,stream=None):
        ob=self.observer;fields=ob._flat(raw_fields)
        dt=ob.metadata['model_timestep'] if timestep is None else timestep
        mask=ob.lowdim.all_worlds if world_mask is None else world_mask
        self._validate(fields,mask,dt)
        low=ob.lowdim
        current=tuple(low.meta)+(low.cache,low.observed,low.sidequat,low.elapsed,
                               low.sampled,low.present,low.events,ob.raw_packer.buffer)
        if ob.scene_cache is not None:
            sc=ob.scene_cache
            current+=tuple(getattr(sc,n) for n in
                           ('periods','enabled','elapsed','sampled','generation','scenes','parent_generation','captured'))
            current+=(ob.scene_packer.buffer,)
        # The accepted public API allows shifted/cross-field aliasing. Such
        # publications need the original pack-before-sample snapshot semantics;
        # ordinary resident owners use the direct path.
        if any(a is not b for a,b in zip(current,self.owner_arrays)) or any(
                self._overlap(source,target) for source in fields.values()
                for target in current):
            return self.baseline_sample(raw_fields,dt,force,world_mask,stream)
        # Validate every caller field before launching anything. New Euler qpos/
        # qvel and old-forward geometry are consumed exactly as supplied.
        if ob.scene_cache is not None:
            trans=ob.scene_transforms
            outputs=[trans.cam_xpos,trans.cam_xmat,trans.light_xpos,trans.light_xdir]
            reads=[fields['body_xpos'],fields['body_xmat'],*trans.camera_static,*trans.light_static]
            # Packed-scene aliases require the original ordered field snapshots,
            # not a parallel copy from pointers that another field may overwrite.
            if any(self._overlap(a,b) for a in outputs+list(ob.visual_sources.values())
                   for b in current):
                return self.baseline_sample(raw_fields,dt,force,world_mask,stream)
            if any(self._overlap(a,b) for a in reads for b in outputs) or any(
                    self._overlap(outputs[i],outputs[j]) for i in range(4) for j in range(i)):
                return self.baseline_sample(raw_fields,dt,force,world_mask,stream)
            for stage,label in self.transform_stages:
                descriptor=(*getattr(trans,label+'_static'),*(outputs[:2] if label=='camera' else outputs[2:]))
                self.plan.set_pointer(stage,2,fields['body_xpos'],stream)
                self.plan.set_pointer(stage,3,fields['body_xmat'],stream)
                self.plan.set_pointer(stage,4,trans.all_worlds if world_mask is None else mask,stream)
                for j,value in enumerate(descriptor):self.plan.set_pointer(stage,5+j,value,stream)
            fields.update(dict(zip(('cam_xpos','cam_xmat','light_xpos','light_xdir'),outputs)))
            fields.update(ob.visual_sources)
        for j,n in enumerate(RAW_FIELDS):self.plan.set_pointer(0,self.low_field_start+j,fields[n],stream)
        self.plan.set_pointer(0,self.low_mask,mask,stream)
        self.plan.set_scalar(0,self.low_dt,float(dt),stream);self.plan.set_scalar(0,self.low_force,int(force),stream)
        if ob.scene_cache is not None and not self.scene_v2:
            for j,n in enumerate(SCENE_FIELDS):self.plan.set_pointer(self.scene_stage,self.scene_field_start+j,fields[n],stream)
            self.plan.set_pointer(self.scene_stage,self.scene_mask,ob.scene_cache.mask if world_mask is None else mask,stream)
            self.plan.set_scalar(self.scene_stage,self.scene_dt,float(dt),stream);self.plan.set_scalar(self.scene_stage,self.scene_force,int(force),stream)
        ob.valid=False
        self.plan.launch(stream=stream)
        if ob.scene_cache is not None and self.scene_v2:
            source=ob.scene_packer.copy_from(fields,stream=stream)
            ob.scene_cache.update(source,dt,force,world_mask,stream)
        ob.lowdim.last_stream=stream or wp.get_stream(ob.device)
        if ob.scene_cache is not None:ob.scene_cache.last_stream=ob.lowdim.last_stream
        ob.valid=True
        ob._after_gpu_sample(dt,force,world_mask,stream)

def install(observer,fields,build_root,*,dataflow_profile='compatible'):
    existing=getattr(observer,'_cuda_direct_sampler',None)
    if existing is not None:
        if existing.profile!=dataflow_profile:raise ValueError('observer profile already installed')
        return existing.manifest
    sampler=DirectSampler(observer,fields,build_root,dataflow_profile=dataflow_profile)
    def sample(self,*args,**kwargs):return sampler.sample(*args,**kwargs)
    observer.sample=MethodType(sample,observer)
    observer._cuda_direct_sampler=sampler
    return sampler.manifest

def install_environment(environment,build_root,*,dataflow_profile='compatible'):
    components=getattr(environment,'components',environment)
    observer=components.observations
    existing=getattr(components,'cuda_observations_manifest',None)
    sampler=getattr(observer,'_cuda_direct_sampler',None)
    if existing is not None and sampler is not None:
        if sampler.profile!=dataflow_profile:raise ValueError('observer profile already installed')
        return dict(existing,installed=True)
    manifest=install(observer,components.observation_fields(),build_root,
                     dataflow_profile=dataflow_profile)
    components.cuda_observations_manifest=manifest
    binding=getattr(components.observation_fields,'__self__',None)
    bundle=getattr(binding,'prepared',None)
    if bundle is not None:
        plan=observer._cuda_direct_sampler.plan
        if hasattr(bundle,'cuda_native_plans') and plan not in bundle.cuda_native_plans:
            bundle.cuda_native_plans=(*bundle.cuda_native_plans,plan)
        if hasattr(bundle,'cuda_manifests'):bundle.cuda_manifests.append(manifest)
        if hasattr(bundle,'cuda_selection_manifest'):bundle.cuda_selection_manifest['observations']=manifest
    return dict(manifest,installed=True)
