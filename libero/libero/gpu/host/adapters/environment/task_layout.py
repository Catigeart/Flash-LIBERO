"""Cold immutable task descriptors and ordered routing; no GPU-owner grant.

Descriptor bytes exist independently of mutable model/controller/observer
objects. Production consumers still validate their live clocks and owners. Runs
describe adjacent equal recipes and never reorder host predicates or operations.
"""
from dataclasses import dataclass
import hashlib
import json
import struct
import numpy as np
from types import SimpleNamespace


def task_batch_key(built, *, clock, controller_abi, observation_abi,
                   execution_policy, lifecycle_abi):
    """Canonical facts only; matching keys never grant a shared GPU owner."""
    clock=tuple(clock)
    if len(clock)!=4 or any(not np.isfinite(x) or x<=0 for x in clock[:3]):
        raise ValueError('actual (control,outer,native,substeps) clock descriptor required')
    if isinstance(clock[3],bool) or not isinstance(clock[3],(int,np.integer)) or clock[3]<0:
        raise ValueError('integral actual substep count required')
    for value in (controller_abi,observation_abi,execution_policy,lifecycle_abi):
        if not isinstance(value,str) or not value:raise ValueError('complete batch semantic identity required')
    descriptor={'model':built.metadata,'clock':clock,'controller_abi':controller_abi,
                'observation_abi':observation_abi,'execution_policy':execution_policy,
                'lifecycle_abi':lifecycle_abi}
    return hashlib.sha256(json.dumps(descriptor,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def ordered_task_groups(keys):
    groups={}
    for index,key in enumerate(keys):groups.setdefault(key,[]).append(index)
    return tuple(tuple(group) for group in groups.values())


def selected_worlds(ids,nworld):
    ids=tuple(ids)
    if (any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<nworld for i in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('invalid or duplicate reset world IDs')
    return tuple(int(i) for i in ids)


def freeze_model_metadata(value):
    """Exact CPU facts, including ndarray bits; no live-owner permission."""
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise TypeError('object arrays are not closed model facts')
        return ['ndarray',value.dtype.str,list(value.shape),list(value.strides),
                np.ascontiguousarray(value).tobytes().hex()]
    if isinstance(value,np.generic):
        if value.dtype.hasobject:raise TypeError('object scalars are not closed model facts')
        return ['numpy_scalar',value.dtype.str,value.tobytes().hex()]
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise TypeError('model fact keys must be strings')
        return ['mapping',[[k,freeze_model_metadata(value[k])] for k in sorted(value)]]
    if isinstance(value,(list,tuple)):
        return ['tuple' if isinstance(value,tuple) else 'list',[freeze_model_metadata(v) for v in value]]
    if type(value) is float:return ['float64',struct.pack('<d',value).hex()]
    if value is None or type(value) in (str,int,bool):return [type(value).__name__,value]
    raise TypeError('unsupported live object in closed model facts')


@dataclass(frozen=True)
class FrozenTaskDescriptor:
    payload: bytes
    key: str

    @classmethod
    def create(cls,metadata,*,clock,controller_abi,observation_abi,execution_policy,lifecycle_abi):
        metadata=freeze_model_metadata(metadata)
        key=task_batch_key(SimpleNamespace(metadata=metadata),clock=clock,
            controller_abi=controller_abi,observation_abi=observation_abi,
            execution_policy=execution_policy,lifecycle_abi=lifecycle_abi)
        payload=json.dumps({'model':metadata,'clock':tuple(clock),'controller_abi':controller_abi,
            'observation_abi':observation_abi,'execution_policy':execution_policy,'lifecycle_abi':lifecycle_abi},
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return cls(payload,key)

    def __post_init__(self):
        if not isinstance(self.payload,bytes) or hashlib.sha256(self.payload).hexdigest()!=self.key:
            raise ValueError('exact immutable task descriptor required')
        value=json.loads(self.payload)
        expected=task_batch_key(SimpleNamespace(metadata=value['model']),clock=value['clock'],
            controller_abi=value['controller_abi'],observation_abi=value['observation_abi'],
            execution_policy=value['execution_policy'],lifecycle_abi=value['lifecycle_abi'])
        if expected!=self.key:raise ValueError('canonical complete task descriptor required')


class TaskBatchLayout:
    def __init__(self,descriptors):
        descriptors=tuple(descriptors)
        if not descriptors or any(type(d) is not FrozenTaskDescriptor for d in descriptors):
            raise TypeError('complete frozen task descriptors required')
        self._keys=tuple(d.key for d in descriptors)
        self.groups=ordered_task_groups(self._keys)

    def selected_runs(self,ids):
        ids=selected_worlds(ids,len(self._keys));runs=[]
        for world in ids:
            key=self._keys[world]
            if runs and runs[-1][0]==key:runs[-1][1].append(world)
            else:runs.append((key,[world]))
        return tuple((key,tuple(worlds)) for key,worlds in runs)

    def apply_ordered(self,ids,values,*,validate_values,apply_world):
        """Independent host transaction; arbitrary first failure stays online.

        Whole-batch value validation precedes mutation, as in the vector API.
        Every selected world finishes its actual callback before the next starts.
        Adjacent recipe runs never authorize submit-all or skipping observers.
        """
        if not callable(validate_values) or not callable(apply_world):
            raise TypeError('actual value validator/world operation required')
        ids=selected_worlds(ids,len(self._keys));values=tuple(values)
        if len(ids)!=len(values):raise ValueError('one value per selected world required')
        validate_values(ids,values)
        output=[]
        for world,value in zip(ids,values):output.append(apply_world(world,value))
        return output
