"""Pure complete native-actuation fixed-input raw replay; no CUDA/native import."""
import json,pathlib
import numpy as np
from typed_compare import verify_index,compare_arrays,hash_file
from ptx_audit import verify_module_set,ACTUATION_MODULES
from f_owner_binding import TR_PRIVATE,AF_PRIVATE,OBSERVER_PRIVATE,private_schema
from actuation_source import CAPS,MODES,NAMES,FIELDS,FLOAT_FIELDS,INT_FIELDS,SHAPES,INPUT_FIELDS,STATIC_FIELDS,key
def binding(v):
 p=pathlib.Path(v["path"])
 if not p.is_file() or p.stat().st_size!=v["bytes"] or hash_file(p)!=v["sha256"]:raise ValueError("binding: "+str(p))
def private_manifest(profile):
 n=profile["N"];manifest=profile["actual_F_owner_binding"];roots=manifest["roots"]
 if set(roots)!={"tr","af","observer","call"} or any(type(x)!=int or x<=0 for x in roots.values()) or len(set(roots.values()))!=4:raise ValueError("complete actual source/observer/consumer root owners")
 stream=manifest["stream"]
 if len(stream)!=3 or type(stream[0])!=int or stream[0]<=0 or stream[1]!="cuda:0" or type(stream[2])!=int or stream[2]<0:raise ValueError("actual private consumer stream identity")
 if manifest["native_pointers"]!={} or manifest["consumer_streams"]!=[{"root":"call","path":"stream","native_handle_path":None,"parent_ids":[roots["call"]]}]:raise ValueError("complete standalone consumer stream binding")
 fields=manifest["fields"];specs=private_schema(n);spans=[]
 logical=physical=0
 for name,spec in specs.items():
  b=fields[name];t=b["actual"];a=b["allocation"];d=b["descriptor"];root=a[0]
  if len(t)!=8 or t!=root or t[0]<=0 or t[1]<=0 or t[1]% (4 if spec["dtype"]=="int32" else 8) or t[2]!=spec["shape"] or t[3]!=spec["strides"] or t[4]!=spec["dtype"] or t[5]!="cuda:0" or type(t[6])!=int or t[6]<spec["logical_bytes"] or t[7] is not True:raise ValueError("complete canonical private typed root token: "+name)
  if len(a)!=5 or any(type(x)!=int or x<=0 for x in a[1:4]) or a[1]!=a[3] or a[4]!=[t[0]] or b["view_ids"]!=[t[0]] or b["parent_ids"]!=[roots[spec["owner"]]] or b["alias_of"] is not None:raise ValueError("complete canonical private allocator/deleter/keeper lineage: "+name)
  want={"name":name,"dtype":spec["dtype"],"shape":spec["shape"],"owner":spec["owner"],"lifetime":spec["lifetime"],"epoch":spec["epoch"],"writers":spec["writers"],"valid_write":spec["valid_write"],"read_domain":spec["read_domain"],"retain":spec["retain"],"offset":0,"sentinel_unread":False,"access":"read_write","logical_elements":spec["logical_bytes"]//(4 if spec["dtype"]=="int32" else 8),"physical_shape":spec["shape"],"strides_bytes":spec["strides"],"allocation_bytes":spec["logical_bytes"],"layout":"C contiguous, world major"}
  if d!=want:raise ValueError("complete private source descriptor/write/hold contract: "+name)
  spans.append((t[1],t[1]+t[6],name));logical+=spec["logical_bytes"];physical+=t[6]
 spans.sort()
 for left,right in zip(spans,spans[1:]):
  if left[1]>right[0]:raise ValueError("undeclared canonical private allocation overlap")
 duplicate_read_bytes=0
 for field,source in (("length","tr.length"),("moment","tr.moment"),("velocity","tr.velocity"),("ctrl","af.ctrl"),("active","af.active"),("vel_status","tr.velocity_status")):
  alias=fields["call."+field];canonical=fields[source];want=dict(canonical["descriptor"])
  want.update(name="call."+field,access="read",read_domain="actual force kernel argument")
  if alias["descriptor"]!=want or alias["actual"]!=canonical["actual"] or alias["allocation"]!=canonical["allocation"] or alias["view_ids"]!=canonical["view_ids"] or alias["parent_ids"]!=[roots["call"]] or alias["alias_of"]!=source:raise ValueError("complete actual consumer alias descriptor and retained owner")
  duplicate_read_bytes+=want["allocation_bytes"]
 return {"N":n,"canonical_arrays":len(specs),"explicit_read_aliases":6,"logical_private_bytes":logical,"actual_root_capacity_bytes":physical,"duplicate_read_alias_bytes":duplicate_read_bytes,"full_production_epoch_pending":True}
def check(run,freeze_sha256):
 run=pathlib.Path(run)
 if hash_file(run/"SCIENCE_FREEZE.json")!=freeze_sha256:raise ValueError("external freeze")
 f=json.loads((run/"SCIENCE_FREEZE.json").read_text())
 if f["schema"]!="p06d-native-actuation-pre-output-v1" or f["budget_uses_expected_or_GPU"]:raise ValueError("source freeze")
 want=[{"N":n,"case":c,"mode":m} for n in CAPS for c in NAMES for m in MODES]
 if f["cases"]!=want:raise ValueError("original complete fixed inventory")
 for v in f["source_records"]+f["numeric_amendments"]:binding(v)
 expected=verify_index(run/"EXPECTED.npz",f["expected"]);inputs=verify_index(run/"INPUTS.npz",f["inputs"])
 budget=verify_index(run/"BUDGETS.npz",f["budgets"]);masks=verify_index(run/"MASKS.npz",f["masks"])
 static=verify_index(run/"STATIC_NATIVE_MODEL.npz",f["static"])
 if set(static)!=set(STATIC_FIELDS):raise ValueError("static schema")
 names={key(n,c,m,x) for n in CAPS for c in NAMES for m in MODES for x in FIELDS}
 if set(expected)!=names:raise ValueError("complete typed output schema")
 if set(inputs)!={key(n,c,m,x) for n in CAPS for c in NAMES for m in MODES for x in INPUT_FIELDS}:raise ValueError("input schema")
 bounds={}
 for n in CAPS:
  for c in NAMES:
   for m in MODES:
    for x in FIELDS:
     k=key(n,c,m,x);v=expected[k]
     if v.dtype!=np.dtype("<f8" if x in FLOAT_FIELDS else "<i4") or v.shape!=(n,)+SHAPES[x]:raise ValueError("dtype/shape:"+k)
     if x in FLOAT_FIELDS:bounds[k]=(budget[k+"__lo"],budget[k+"__hi"])
 if set(budget)!={k+"__"+s for k in bounds for s in ("lo","hi")} or set(masks)!=set(bounds):raise ValueError("budget/mask schema")
 if not compare_arrays(expected,expected,bounds=bounds,write_masks=masks)["ok"]:raise ValueError("native source-budget mismatch")
 idx=json.loads((run/"SCIENTIFIC_RAW_INDEX.json").read_text())
 if idx["freeze_sha256"]!=freeze_sha256:raise ValueError("GPU freeze")
 binding(idx["first_gpu_marker"]);marker=json.loads(pathlib.Path(idx["first_gpu_marker"]["path"]).read_text())
 if marker["freeze_sha256"]!=freeze_sha256 or marker["expected_sha256"]!=f["expected"]["artifact"]["sha256"]:raise ValueError("first GPU")
 for v in idx["compile_cache"]+idx["runtime_records"]:binding(v)
 ptx_claims=verify_module_set(idx["compile_cache"],idx["actual_ptx_premises"]["relevant_ptx"],idx["runtime_records"],ACTUATION_MODULES)
 profiles=idx["actual_bindings"]
 if [(v["N"],v["case"]) for v in profiles]!=[(n,c) for n in CAPS for c in NAMES]:raise ValueError("complete actual binding profiles")
 mapping={x:"tr."+x for x in ("length","moment","velocity","position_status","velocity_status")}
 mapping.update({x:("observer." if x.startswith("consumer_") else "af.")+x for x in FIELDS if x not in mapping})
 canonical={"tr."+x for x in TR_PRIVATE}|{"af."+x for x in AF_PRIVATE}|{"observer."+x for x in OBSERVER_PRIVATE}
 aliases={"call."+x for x in ("length","moment","velocity","ctrl","active","vel_status")}
 for profile in profiles:
  n=profile["N"];manifest=profile["actual_F_owner_binding"];bf=manifest["fields"]
  private_manifest(profile)
  if set(profile["fields"])!=set(FIELDS) or set(bf)!=canonical|aliases:raise ValueError("complete private/actual consumer owner manifest")
  if not profile["original_production_epochs_pending"]:raise ValueError("standalone unit cannot claim full production epochs")
  for field,v in profile["fields"].items():
   shape=(n,)+SHAPES[field];itemsize=8 if field in FLOAT_FIELDS else 4
   stride=itemsize;strides=[]
   for extent in reversed(shape):strides.append(stride);stride*=extent
   if v["dtype"]!=("<class 'warp._src.types.float64'>" if itemsize==8 else "<class 'warp._src.types.int32'>") or v["shape"]!=list(shape) or v["strides"]!=list(reversed(strides)) or v["device"]!="cuda:0" or v["ptr"]<=0 or v["ptr"]%itemsize or v["capacity_bytes"]<stride:raise ValueError("actual typed view extent: "+field)
   bind=bf[mapping[field]];token=bind["actual"];allocation=bind["allocation"];d=bind["descriptor"]
   if (token[0]!=v["actual_array_id"] or token[1]!=v["ptr"] or token[2]!=v["shape"] or token[3]!=v["strides"] or token[4]!=("float64" if itemsize==8 else "int32") or token[5]!=v["device"] or not token[7] or bind["alias_of"] is not None):raise ValueError("actual canonical owner/view token: "+field)
   root=allocation[0]
   if not allocation[1] or not allocation[2] or allocation[1]!=allocation[3] or d["offset"]<0 or token[1]!=root[1]+d["offset"] or d["offset"]+stride>root[6]:raise ValueError("actual allocator/deleter/root extent: "+field)
  for i,(arg,source) in enumerate((("length","tr.length"),("moment","tr.moment"),("velocity","tr.velocity"),("ctrl","af.ctrl"),("active","af.active"),("vel_status","tr.velocity_status"))):
   a,b=bf["call."+arg],bf[source]
   if a["alias_of"]!=source or a["actual"]!=b["actual"] or a["allocation"]!=b["allocation"] or profile["source_force_launch_arg_ptrs"][i]!=a["actual"][1]:raise ValueError("actual source/consumer allocation alias")
 negatives=idx["actual_owner_negative_controls"]
 if {(x["N"],x["label"]) for x in negatives}!={(n,label) for n in CAPS for label in ("same_pointer_replaced_consumer_view","same_pointer_replaced_workspace_owner","replaced_actual_consumer_stream")} or len(negatives)!=12 or not all(x["rejected"] for x in negatives):raise ValueError("actual owner/view/stream negative coverage")
 reset_expected=verify_index(run/"RESET_INVALID_SOURCE_EXPECTED.npz",f["reset_invalid_source_expected"])
 reset_raw=verify_index(pathlib.Path(idx["reset_invalid_actual"]["artifact"]["path"]),idx["reset_invalid_actual"])
 reset_keys={f"N{n}__{x}" for n in CAPS for x in ("warning_number","warning_lastinfo","warning_reset_status")}
 if set(reset_expected)!=reset_keys or any(v.dtype!=np.int32 or v.shape!=(int(k.split("__")[0][1:]),) for k,v in reset_expected.items()) or not compare_arrays(reset_expected,reset_raw)["ok"]:raise ValueError("actual invalid reset failed source hold/status contract")
 required={"EAGER_ALL","GRAPH_ALL","REPLAY_ALL","SIGNEDZERO","LAWFUL_1ULP"}|{"DIRTY_"+x for x in FIELDS}|{"DIRTY_HELD_warning_number","DIRTY_HELD_warning_lastinfo"}
 if set(idx["artifacts"])!=required:raise ValueError("complete dirty packets")
 verdicts={}
 for label,item in idx["artifacts"].items():
  raw=verify_index(pathlib.Path(item["artifact"]["path"]),item)
  verdicts[label]=compare_arrays(expected,raw,bounds=bounds,write_masks=masks)
 # Actual numeric positives must change the device payload at all four N;
 # diagnostic variants must retain every other actual baseline represented word.
 baseline=verify_index(pathlib.Path(idx["artifacts"]["EAGER_ALL"]["artifact"]["path"]),idx["artifacts"]["EAGER_ALL"])
 for label in ("SIGNEDZERO","LAWFUL_1ULP"):
  variant=verify_index(pathlib.Path(idx["artifacts"][label]["artifact"]["path"]),idx["artifacts"][label])
  targets={}
  for n in CAPS:
   plan=f["test_positive_control_plan_input_only"][str(n)]
   field=plan["zero_field"] if label=="SIGNEDZERO" else plan["nonzero_field"]
   index=plan["zero_flat_index"] if label=="SIGNEDZERO" else plan["nonzero_flat_index"]
   k=key(n,"original_split_seed4404","EAGER",field);targets[k]=index
   x,y=baseline[k].reshape(-1)[index],variant[k].reshape(-1)[index]
   if x.tobytes()==y.tobytes():raise ValueError("unmutated actual positive payload")
   if label=="SIGNEDZERO":
    if not (x==0. and y==0.):raise ValueError("signedzero positive domain")
   elif x==y or float(y-x).hex()!=plan["delta_hex"] or y!=np.nextafter(x,np.float64(np.inf)):raise ValueError("actual frozen lawful one-ULP nonzero delta")
  for k,x in baseline.items():
   y=variant[k]
   if k in targets:
    xx=x.copy().reshape(-1);yy=y.reshape(-1);xx[targets[k]]=yy[targets[k]]
    if xx.tobytes()!=yy.tobytes():raise ValueError("extra diagnostic positive mutation")
   elif x.tobytes()!=y.tobytes():raise ValueError("extra diagnostic positive packet mutation")
 positive=all(verdicts[x]["ok"] for x in ("EAGER_ALL","GRAPH_ALL","REPLAY_ALL","SIGNEDZERO","LAWFUL_1ULP"))
 dirty=all(not verdicts["DIRTY_"+x]["ok"] and any(z.get("field","").endswith("__"+x) for z in verdicts["DIRTY_"+x]["failures"]) for x in FIELDS)
 held=all(not verdicts["DIRTY_HELD_"+x]["ok"] and any(z.get("field","").endswith("__"+x) and "__nan_then_inactive__REPLAY__" in z.get("field","") for z in verdicts["DIRTY_HELD_"+x]["failures"]) for x in ("warning_number","warning_lastinfo"))
 return {"schema":"p06d-native-actuation-scientific-result-v1","ok":positive and dirty and held,"positive_controls":positive,"all_output_device_dirty_rejected":dirty,"held_resident_device_dirty_rejected":held,"complete_private_physical_owner_manifest_checked":True,"private_byte_capacities":[private_manifest(p) for p in profiles if p["case"]=="original_split_seed4404"],"actual_owner_view_stream_negatives_rejected":True,"invalid_device_reset_holds_warning_and_returns_error":True,"cases":len(want),"typed_outputs":len(expected),"verdicts":verdicts,"old_candidate_repro":idx["old_candidate_repro"],"full_D_ready":False,"accepted":False,"independently_checked_actual_ptx":ptx_claims,"scope":"native transmission/force and warning fixed-input source unit; original whole producer/13section/fullerror scopes remain mandatory"}
