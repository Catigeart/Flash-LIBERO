"""Preserve original same-task property-sampler history across fresh GPU arenas.
Host-only logical descriptors; no RNG draw, physics, or previous model retained.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import inspect
import numpy as np

class SamplerHistoryRejected(ValueError):pass

def _classes():
 from libero.libero.envs.regions.object_property_sampler import OpenCloseSampler,TurnOnOffSampler
 return (OpenCloseSampler,TurnOnOffSampler)

def _value(x):
 if isinstance(x,np.ndarray):
  if x.dtype.hasobject or not np.issubdtype(x.dtype,np.number):raise SamplerHistoryRejected('non-numeric sampler range')
  return {'kind':'array','dtype':x.dtype.str,'shape':list(x.shape),'bytes_hex':np.ascontiguousarray(x).tobytes().hex()}
 if isinstance(x,np.generic):
  return {'kind':'numpy-scalar','dtype':x.dtype.str,'bytes_hex':x.tobytes().hex()}
 if type(x) is float:return {'kind':'float','hex':x.hex()}
 if type(x) is int:return {'kind':'int','value':x}
 if isinstance(x,(list,tuple)):return {'kind':'tuple' if isinstance(x,tuple) else 'list','values':[_value(v) for v in x]}
 raise SamplerHistoryRejected('unknown sampler range representation')

def _restore(v):
 k=v['kind']
 if k=='float':return float.fromhex(v['hex'])
 if k=='int':return v['value']
 if k in ('array','numpy-scalar'):
  a=np.frombuffer(bytes.fromhex(v['bytes_hex']),dtype=np.dtype(v['dtype'])).copy()
  return a[0] if k=='numpy-scalar' else a.reshape(v['shape'])
 if k in ('tuple','list'):
  a=[_restore(x) for x in v['values']];return tuple(a) if k=='tuple' else a
 raise SamplerHistoryRejected('unknown descriptor kind')

def _bindings(built,name):
 shell=built.domain_shell
 if name not in shell.object_states_dict:raise SamplerHistoryRejected('property target is absent from current task states')
 obj=shell.get_object(name);names=list(obj.joints)
 if not names:raise SamplerHistoryRejected('property target has no current joint')
 nm=built.metadata['name_maps']['joint'];sl=built.metadata['joint_qpos_slices'];model=built.native_model
 result=[]
 for n in names:
  if n not in nm or n not in sl:raise SamplerHistoryRejected('property joint missing from current compiled model')
  i=int(nm[n]);start,end=map(int,sl[n]);kind=int(model.jnt_type[i])
  if int(model.joint(n).id)!=i:raise SamplerHistoryRejected('current native named joint differs from metadata')
  if int(model.jnt_qposadr[i])!=start or end-start!=1 or kind not in (2,3):raise SamplerHistoryRejected('property joint is not an actual scalar slide/hinge binding')
  result.append({'name':n,'joint_type':kind,'joint_id':i,'qpos_slice':[start,end]})
 return result

def describe_property_samplers(built):
 result=[]
 for sampler in built.domain_shell.object_property_initializers:
  if type(sampler) not in _classes():raise SamplerHistoryRejected('unqualified property sampler class')
  if set(vars(sampler))!={'name','state_type','joint_ranges','mujoco_objects'}:raise SamplerHistoryRejected('unknown sampler state fields')
  if sampler.mujoco_objects:raise SamplerHistoryRejected('sampler carries object/model references')
  ranges=sampler.joint_ranges
  a=np.asarray(ranges)
  if a.shape!=(2,) or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() or a[0]>a[1]:raise SamplerHistoryRejected('invalid source property ranges')
  cls=type(sampler);path=Path(inspect.getfile(cls)).resolve()
  allowed=('open','close') if cls is _classes()[0] else ('turnon','turnoff')
  if sampler.state_type not in allowed:raise SamplerHistoryRejected('invalid source property mode')
  result.append({'class_name':cls.__name__,'source_path':str(path),'source_sha256':sha256(path.read_bytes()).hexdigest(),'name':sampler.name,'state_type':sampler.state_type,'joint_ranges':_value(ranges),'current_bindings':_bindings(built,sampler.name)})
 return result

def inherit_property_sampler_history(built,previous,*,xml_override):
 """Preflight/rebind before previous runtime release; preserve duplicate order.
 Original CPU hard _load_model appends fresh source recipes. XML override does
 not _load_model, so it keeps prior history without adding fresh recipes.
 """
 old=describe_property_samplers(previous.built);fresh=describe_property_samplers(built)
 task=lambda b:(b.metadata['bddl_file_name'],b.metadata['problem_name'])
 if task(built)!=task(previous.built):raise SamplerHistoryRejected('cannot transfer logical sampler history between different tasks')
 source_by_class={c.__name__:c for c in _classes()};clones=[];rebound=[]
 for d in old:
  cls=source_by_class.get(d['class_name']);path=Path(inspect.getfile(cls)).resolve()
  if str(path)!=d['source_path'] or sha256(path.read_bytes()).hexdigest()!=d['source_sha256']:raise SamplerHistoryRejected('property sampler decoder source changed')
  bind=_bindings(built,d['name'])
  if [(x['name'],x['joint_type']) for x in bind]!=[(x['name'],x['joint_type']) for x in d['current_bindings']]:raise SamplerHistoryRejected('property target changed named joint semantics')
  clone=cls(name=d['name'],state_type=d['state_type'],joint_ranges=_restore(d['joint_ranges']))
  clones.append(clone);new=deepcopy(d);new['current_bindings']=bind;rebound.append(new)
 # No sampled value, seed-specific skip, old object/model owner or RNG is copied.
 if not xml_override:clones.extend(built.domain_shell.object_property_initializers);rebound.extend(fresh)
 built.domain_shell.object_property_initializers=clones
 return {'schema':'source-property-sampler-history-rebind-v1','scope':'host logical original-CPU lifecycle compatibility; no GPU numeric grant','transition':'xml-keep' if xml_override else 'hard-source-append','old_count':len(old),'fresh_source_count':len(fresh),'current_count':len(clones),'descriptors':rebound,'old_native_owner_retained':False,'RNG_draws_performed':0}
def wrap_runtime_factory(runtime_factory):
 """Compose this host fix with any actual owner; no global registry mutation."""
 if not callable(runtime_factory):raise TypeError('actual runtime factory required')
 if not callable(getattr(runtime_factory,'rebuild_components',None)):
  raise SamplerHistoryRejected('history compatibility requires an explicit physical rebuild owner')
 if getattr(runtime_factory,'source_property_history_compatible',False):
  raise SamplerHistoryRejected('sampler history wrapper already applied')
 def create(built):return runtime_factory(built)
 def rebuild_components(built,previous):
  proof=inherit_property_sampler_history(built,previous,xml_override=bool(built.metadata['xml_overridden']))
  built.metadata['source_property_sampler_history']=proof
  return runtime_factory.rebuild_components(built,previous)
 create.rebuild_components=rebuild_components
 create.source_property_history_compatible=True
 return create