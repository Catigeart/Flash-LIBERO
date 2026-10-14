"""C-owned resident D four-leaf consumer; actual validation recorded separately.
The C expected epoch is an independent captured callsite input, never D output.
"""
import numpy as np
import warp as wp
from libero.gpu.host.runtime.collision.shared import array_token, _allocation, DeviceBindingError
@wp.kernel(enable_backward=False)
def _consumer_gate(source_active:wp.array(dtype=wp.int32),valid:wp.array(dtype=wp.int32),first_error:wp.array(dtype=wp.int32),source_epoch:wp.array(dtype=wp.int64),written_epoch:wp.array(dtype=wp.int64),expected_epoch:wp.array(dtype=wp.int64),active:wp.array(dtype=wp.int32)):
 w=wp.tid();active[w]=0
 if source_active[w]==0:return
 # Illegal active2 forces the existing C first-error/input refusal path;
 # rejection is never disguised as a successful inactive world.
 if source_active[w]!=1 or valid[w]!=1 or first_error[w]!=0 or expected_epoch[w]<wp.int64(1) or source_epoch[w]!=expected_epoch[w] or written_epoch[w]!=expected_epoch[w]:
  active[w]=2;return
 active[w]=1
@wp.kernel(enable_backward=False)
def _contact_publish(active:wp.array(dtype=wp.int32),fe_stage:wp.array(dtype=wp.int32),fe_code:wp.array(dtype=wp.int32),expected_epoch:wp.array(dtype=wp.int64),ready:wp.array(dtype=wp.int32),written_epoch:wp.array(dtype=wp.int64)):
 w=wp.tid()
 ready[w]=0
 # This enqueue follows the complete real C chain on the same stream.
 # A clock is published only after an actually enabled, error-free producer.
 if active[w]==1 and fe_stage[w]==0 and fe_code[w]==0:
  written_epoch[w]=expected_epoch[w]
  ready[w]=1
def _source_active(producer):
 # Missing uploaded intent is legal only for the exact legacy producer source.
 import sys,hashlib
 from pathlib import Path
 if hasattr(producer,'input_active'):return producer.input_active,'uploaded_input_active'
 cls=type(producer); module=sys.modules.get('libero.gpu.host.runtime.dynamics.paired_pose')
 if module is None or cls is not getattr(module,'PairedPoseProducer',None) or cls.__module__!='libero.gpu.host.runtime.dynamics.paired_pose':
  raise DeviceBindingError('missing root activity on unqualified producer class')
 source=Path(module.__file__).resolve()
 if hashlib.sha256(source.read_bytes()).hexdigest()!='04aeb48abeae47a97a11436e2da0650a28301a71c08ed77c20c48466e6a490d4':
  raise DeviceBindingError('missing root activity on nonlegacy producer source')
 return producer.body.active,'legacy_exact_83e2_body_active'
class ResidentPoseConsumer:
 def __init__(self,ws,producer,callsite_epochs,*,stream):
  N=ws.nworld
  if producer.capacity!=N or producer.frames.nbody!=ws.nbody or producer.frames.ngeom!=ws.ngeom:raise DeviceBindingError('actual D/C model shape mismatch')
  if not isinstance(stream,wp.Stream) or stream is not producer.stream or stream.device!=wp.get_device(ws.device):raise DeviceBindingError('same actual producer stream required')
  if type(callsite_epochs) is not np.ndarray or callsite_epochs.dtype!=np.dtype('int64') or callsite_epochs.shape!=(N,) or not callsite_epochs.flags.c_contiguous or np.any(callsite_epochs<1):raise DeviceBindingError('independent exact C callsite epoch schema')
  self.producer=producer;self.ws=ws;self.stream=stream
  self.callsite_epochs=np.array(callsite_epochs,copy=True,order='C');self.callsite_epochs.flags.writeable=False
  with wp.ScopedStream(stream):
   self.expected_epoch=wp.array(self.callsite_epochs,dtype=wp.int64,device=ws.device)
   self.active=wp.zeros(N,dtype=wp.int32,device=ws.device)
   self.contact_ready=wp.zeros(N,dtype=wp.int32,device=ws.device)
   self.contact_written_epoch=wp.zeros(N,dtype=wp.int64,device=ws.device)
  # D source025 separates immutable uploaded root intent from its working gate.
  # Original83e2 has only body.active; retain that exact legacy source scope.
  source_active,self.source_active_kind=_source_active(producer)
  fields={'source_active':source_active,'valid':producer.valid,'first_error':producer.first_error,'source_epoch':producer.source_epoch,'written_epoch':producer.written_epoch,'expected_epoch':self.expected_epoch,'active':self.active,'contact_ready':self.contact_ready,'contact_written_epoch':self.contact_written_epoch,'completion_fe_stage':self.ws.fe_stage,'completion_fe_code':self.ws.fe_code}
  self.fields=fields;self.tokens={};self.owners=[]
  for key,a in fields.items():
   dtype=wp.int64 if key in ('source_epoch','written_epoch','expected_epoch','contact_written_epoch') else wp.int32
   if a.dtype!=dtype or a.shape!=(N,) or a.device!=stream.device or not a.is_contiguous:raise DeviceBindingError('actual guard dtype/shape/device '+key)
   root,chain=_allocation(a);width=8 if dtype==wp.int64 else 4
   if int(a.ptr)<int(root.ptr) or int(a.ptr)+N*width>int(root.ptr)+int(root.capacity):raise DeviceBindingError('actual guard root/span '+key)
   self.tokens[key]=array_token(a);self.owners.extend(chain)
  self.manifest=ws.bind_device_inputs(self,{'xpos':'producer.frames.geom_xpos','xmat':'producer.frames.geom_xmat','xipos':'producer.frames.xipos','ximat':'producer.frames.ximat','active':'active'},stream=stream)
 def launch(self,stream):
  if stream is not self.stream:raise DeviceBindingError('consumer stream identity changed')
  if self.producer.stream is not stream:raise DeviceBindingError('producer stream owner changed')
  source_active,kind=_source_active(self.producer)
  if kind!=self.source_active_kind:raise DeviceBindingError('producer source activity qualification changed')
  actual={'source_active':source_active,'valid':self.producer.valid,'first_error':self.producer.first_error,'source_epoch':self.producer.source_epoch,'written_epoch':self.producer.written_epoch,'expected_epoch':self.expected_epoch,'active':self.active,'contact_ready':self.contact_ready,'contact_written_epoch':self.contact_written_epoch,'completion_fe_stage':self.ws.fe_stage,'completion_fe_code':self.ws.fe_code}
  for key,a in actual.items():
   if array_token(a)!=self.tokens[key] or a is not self.fields[key]:raise DeviceBindingError('producer/consumer guard allocation changed '+key)
  self.ws.validate_device_bindings(stream)
  wp.launch(_consumer_gate,dim=self.ws.nworld,inputs=[actual[k] for k in ('source_active','valid','first_error','source_epoch','written_epoch','expected_epoch','active')],device=self.ws.device,stream=stream)
  self.ws.launch(stream=stream)
  wp.launch(_contact_publish,dim=self.ws.nworld,inputs=[self.active,self.ws.fe_stage,self.ws.fe_code,self.expected_epoch,self.contact_ready,self.contact_written_epoch],device=self.ws.device,stream=stream)
