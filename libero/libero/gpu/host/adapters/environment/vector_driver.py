"""Multiple simultaneously resident, independently owned real GPU task worlds.

Host dispatch order follows LIBERO's synchronous vector API. Every world retains
its own CUDA arena, model parameters, RNG, controller and observation clocks.
This is not a one-slot capacity workaround and does not claim fused execution.
"""
from threading import RLock
from types import ModuleType
import numpy as np
from .facade import UnsupportedGpuOperation
from .vector import GpuVectorEnv
from .ownership_spans import first_cross_world_overlap


def _stack(values):
    try:
        return np.stack(values)
    except ValueError:
        result=np.empty(len(values),dtype=object)
        result[:]=values
        return result


def _task(env):
    return getattr(env,'env',env)


def _array_region(array,wp):
    if not int(array.size):return int(array.ptr),0
    itemsize=int(wp.types.type_size_in_bytes(array.dtype))
    low=high=0
    for dimension,stride in zip(array.shape,array.strides):
        offset=(int(dimension)-1)*int(stride)
        low+=min(0,offset);high+=max(0,offset)
    return int(array.ptr)+low,high-low+itemsize


def resident_ownership(env):
    task=_task(env)
    components=task.components
    components.validate()
    boundary=task.sim._runtime
    if boundary._closed:raise RuntimeError('vector resident owner closed')
    if not boundary.stream.device.is_cuda:
        raise UnsupportedGpuOperation('vector worlds require actual CUDA owners')
    fields=[]
    for name,descriptor in boundary.fields.items():
        array=descriptor.array
        pointer,size=_array_region(array,boundary.wp)
        fields.append({'name':name,'pointer':pointer,'bytes':size,
                       'device':str(array.device)})
    binding=getattr(components.observation_fields,'__self__',None)
    prepared=getattr(binding,'prepared',None)
    seen=set()
    def walk(value,path,depth):
        if id(value) in seen:return
        seen.add(id(value))
        if hasattr(value,'ptr') and hasattr(value,'dtype') and hasattr(value,'device') and hasattr(value,'size'):
            if not value.device.is_cuda:return
            pointer,size=_array_region(value,boundary.wp)
            fields.append({'name':path,'pointer':pointer,'bytes':size,
                           'device':str(value.device)})
            return
        if not depth:return
        if isinstance(value,dict):items=value.items()
        elif isinstance(value,(tuple,list)):items=enumerate(value)
        elif hasattr(value,'__dict__') and not isinstance(value,(ModuleType,type)) and not callable(value) and not type(value).__module__.startswith(('warp.','numpy.','mujoco.')):
            items=vars(value).items()
        else:return
        for name,child in items:walk(child,f'{path}.{name}',depth-1)
    walk(prepared,'prepared',6)
    walk(components.controller,'controller',4)
    walk(components.observations,'observations',4)
    return {'allocation_id':boundary.allocation_id,'model_owner':id(task.built.native_model),
            'controller_owner':id(components.controller),'observer_owner':id(components.observations),
            'stream_device':str(boundary.stream.device),'fields':fields}


def resident_allocation_evidence(env):
    """Actual typed-array byte union, explicitly not a complete allocator bound."""
    ownership=resident_ownership(env)
    intervals=sorted((field['pointer'],field['pointer']+field['bytes'])
                     for field in ownership['fields'] if field['bytes'])
    merged=[]
    for start,stop in intervals:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],stop)
        else:merged.append([start,stop])
    device=_task(env).components.stream.device
    return {'ownership':ownership,'enumerated_typed_array_bytes':sum(stop-start for start,stop in merged),
            'enumerated_array_regions':len(intervals),'unique_byte_regions':len(merged),
            'actual_device_free_bytes':int(device.free_memory),'actual_device_total_bytes':int(device.total_memory),
            'coverage':'reachable prepared/controller/observation Warp arrays and facade roots; external native allocations, CUDA module/context and allocator alignment/scratch require independent allowances',
            'complete_allocation_bound':False}

class IndependentResidentVectorDriver:
    physics_backend='gpu'

    def __init__(self,environments):
        self.environments=tuple(environments)
        self._lock=RLock();self._closed=False;self._closing=False;self._failed=set();self._close_done=set();self._last_live_ownership={}
        if not self.environments:raise ValueError('at least one resident world required')
        if len({id(e) for e in self.environments})!=len(self.environments):
            raise ValueError('each world requires a different environment')
        for env in self.environments:
            if getattr(env,'physics_backend',None)!='gpu':
                raise UnsupportedGpuOperation('CPU world cannot enter GPU vector driver')
        if type(self) is IndependentResidentVectorDriver and len(self.environments)>1:
            self.ownership=self._span_ownership
        self.ownership()

    @classmethod
    def build(cls,factories):
        """Allocate all worlds together; release every earlier arena on failure."""
        environments=[]
        try:
            for factory in factories:environments.append(factory())
            return cls(environments)
        except BaseException as exc:
            errors=[]
            for env in reversed(environments):
                try:env.close()
                except BaseException as error:errors.append(repr(error))
            if errors:exc.gpu_cleanup_errors=errors
            raise

    def _span_ownership(self):
        records=[]
        for index,env in enumerate(self.environments):
            task=_task(env);components=getattr(task,'components',None)
            boundary=getattr(getattr(task,'sim',None),'_runtime',None)
            released=bool(getattr(components,'released',False) or getattr(boundary,'_closed',False))
            if index in self._failed and released:
                records.append({'world':index,'released_failed_world':True,'last_live_owner':self._last_live_ownership.get(index)})
                continue
            record=resident_ownership(env);record['world']=index
            records.append(record);self._last_live_ownership[index]=record
        live=[record for record in records if not record.get('released_failed_world',False)]
        for key in ('allocation_id','model_owner','controller_owner','observer_owner'):
            if len({r[key] for r in live})!=len(live):
                raise UnsupportedGpuOperation(f'vector worlds share mutable owner: {key}')
        if live and len({r['stream_device'] for r in live})!=1:
            raise UnsupportedGpuOperation('this vector driver requires one GPU device')
        # W1 keeps its full owner walk without entering the span sweep.
        if len(live)>1:
            conflict=first_cross_world_overlap(live)
            if conflict is not None:
                x,y=conflict
                raise UnsupportedGpuOperation(f'vector device owners overlap: {x["name"]}/{y["name"]}')
        return records

    def ownership(self):
        records=[]
        for index,env in enumerate(self.environments):
            task=_task(env);components=getattr(task,'components',None)
            boundary=getattr(getattr(task,'sim',None),'_runtime',None)
            released=bool(getattr(components,'released',False) or getattr(boundary,'_closed',False))
            if index in self._failed and released:
                records.append({'world':index,'released_failed_world':True,'last_live_owner':self._last_live_ownership.get(index)})
                continue
            record=resident_ownership(env);record['world']=index
            records.append(record);self._last_live_ownership[index]=record
        live=[record for record in records if not record.get('released_failed_world',False)]
        for key in ('allocation_id','model_owner','controller_owner','observer_owner'):
            if len({r[key] for r in live})!=len(live):
                raise UnsupportedGpuOperation(f'vector worlds share mutable owner: {key}')
        if live and len({r['stream_device'] for r in live})!=1:
            raise UnsupportedGpuOperation('this vector driver requires one GPU device')
        for index,a in enumerate(live):
            for b in live[index+1:]:
                for x in a['fields']:
                    for y in b['fields']:
                        if x['bytes'] and y['bytes'] and max(x['pointer'],y['pointer']) < min(x['pointer']+x['bytes'],y['pointer']+y['bytes']):
                            raise UnsupportedGpuOperation(f'vector device owners overlap: {x["name"]}/{y["name"]}')
        return records

    def _ids(self,ids):
        if self._closed or self._closing:raise RuntimeError('GPU vector driver closed or closing')
        ids=tuple(ids)
        if len(set(ids))!=len(ids) or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(self.environments) for i in ids):
            raise ValueError('invalid or duplicate world IDs')
        if self._failed.intersection(ids):raise RuntimeError('selected GPU world failed; close and rebuild required')
        return ids

    def _dispatch(self,ids,operation,args=None):
        with self._lock:
            ids=self._ids(ids)
            if args is not None and len(args)!=len(ids):raise ValueError('one argument per selected world required')
            self.ownership()
            results=[]
            for offset,index in enumerate(ids):
                try:
                    method=getattr(self.environments[index],operation)
                    result=method() if args is None else method(args[offset])
                    results.append(result)
                except BaseException:
                    self._failed.add(index)
                    raise
            self.ownership()
            return results

    def step_worlds(self,ids,actions):
        with self._lock:
            ids=self._ids(ids);actions=list(actions)
            if len(actions)!=len(ids):raise ValueError('one action per selected world required')
            # Validate the complete batch before mutating its first world.
            for index,action in zip(ids,actions):
                dim=_task(self.environments[index]).action_dim
                value=np.asarray(action,dtype=np.float64)
                if value.shape not in ((dim,), (7,) if dim==4 else (dim,)) or not np.isfinite(value).all():
                    raise ValueError('invalid vector policy action')
            results=self._dispatch(ids,'step',actions)
            if not results:return tuple(np.empty(0,dtype=object) for _ in range(4))
            packed=[]
            for index,result in zip(ids,results):
                if len(result)!=4:raise UnsupportedGpuOperation('expected original LIBERO four-field step result')
                obs,reward,done,info=result
                info=dict(info);info['env_id']=index
                packed.append((obs,reward,done,info))
            return tuple(_stack(list(values)) for values in zip(*packed))

    def reset_worlds(self,ids):return _stack(self._dispatch(ids,'reset'))
    def set_init_states(self,ids,states):
        with self._lock:
            ids=self._ids(ids);states=list(states)
            if len(states)!=len(ids):raise ValueError('one initial state per selected world required')
            for index,state in zip(ids,states):_task(self.environments[index]).sim.validate_flat_state(state)
            return _stack(self._dispatch(ids,'set_init_state',states))
    def seed_worlds(self,ids,seeds):
        with self._lock:
            ids=self._ids(ids);seeds=list(seeds)
            if len(seeds)!=len(ids):raise ValueError('one seed per selected world required')
            for seed in seeds:
                if seed is not None:np.random.RandomState(seed)
            return self._dispatch(ids,'seed',seeds)
    def check_success_worlds(self,ids):return self._dispatch(ids,'check_success')
    def get_states(self,ids):return self._dispatch(ids,'get_sim_state')
    def get_env_attrs(self,ids,key):
        with self._lock:return [getattr(self.environments[i],key) for i in self._ids(ids)]
    def set_env_attrs(self,ids,key,value):
        raise UnsupportedGpuOperation('unrestricted vector attribute writes have no GPU mutation contract')
    def render_worlds(self,ids,**kwargs):
        with self._lock:
            ids=self._ids(ids);self.ownership()
            methods=[getattr(self.environments[i],'render',None) for i in ids]
            if any(not callable(method) for method in methods):raise UnsupportedGpuOperation('world render interface unavailable')
            results=[]
            for index,method in zip(ids,methods):
                try:results.append(method(**kwargs))
                except BaseException:
                    self._failed.add(index)
                    raise
            self.ownership()
            return results
    def segmentation_worlds(self,ids,images):return self._dispatch(ids,'get_segmentation_of_interest',list(images))
    def close(self):
        with self._lock:
            if self._closed:return
            self._closing=True
            errors=[]
            for index,env in enumerate(self.environments):
                if index in self._close_done:continue
                try:env.close();self._close_done.add(index)
                except BaseException as exc:errors.append(exc)
            if errors:
                error=RuntimeError('one or more GPU vector owners failed to close; retry close')
                error.gpu_cleanup_errors=[repr(e) for e in errors]
                raise error from errors[0]
            self._closed=True


def create_resident_vector(factories):
    driver=IndependentResidentVectorDriver.build(factories)
    return GpuVectorEnv(driver,len(driver.environments))
