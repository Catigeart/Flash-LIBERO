"""Consume the exact F ae1 authentic owner/view API without changing F sources.

This standalone native-entry control has no full D forward/control epoch ABI.
The descriptor states that pending premise explicitly; no ordinal is promoted
into a production epoch. Source/API acceptance and whole shared-v1 remain open.
"""
from __future__ import annotations
import hashlib,json,pathlib,subprocess,sys
F_ROOT=pathlib.Path("/srv/libero-split-receive-20261004-r01/worktrees/p06-f-supplement-codex-execution-g01")
F_HEAD="ae1e0eb7770cf019576ac2232affdaa2b2a951e2"
F_TREE="4586e228ad76dd46848f296009e5d0e2eeeadae2"
PINS={"__init__.py":"289af6547b89163b27073747d099d9f15ff9032f3f999ecb80a43c2153b7b56c",
      "schema.py":"41b5c5c3b7e987296359673ff5ac7b6c2b18c3946d71edfacc332941df8d2b72",
      "binding.py":"54ff0f7c2f0eb6a809d4f1c7674fb46eaea7f5c1f22a63349d048ccbe53a4853"}
TR_PRIVATE=("qadr","dadr","gear","qpos","qvel","position_active","velocity_active","length","moment","velocity","position_status","velocity_status")
AF_PRIVATE=("qadr","gain","bias0","bias1","bias2","ctrl_lo","ctrl_hi","force_lo","force_hi","force_limited","bias_type","ctrl","active","force","force_pre","ctrl_used","ctrl_route","force_route","qfrc_actuator","status","warn_badctrl_count","warn_badctrl_lastinfo","warning_number","warning_lastinfo","warning_reset_mask","warning_reset_status")
OBSERVER_PRIVATE=("consumer_length","consumer_moment","consumer_velocity","consumer_ctrl","consumer_active","consumer_vel_status")
def contract(owner,field):
    if owner=="observer":return ("test_device_observer_copy",),"D2D copy of actual force launch argument immediately before consumer","last observed exact argument until next consumer","source-unit observer lifetime"
    if owner=="tr":
        if field in ("qadr","dadr","gear"):return ("original_model_constructor",),"constructor only, actual fixed model typed source","all launches hold static fields","workspace lifetime"
        if field in ("qpos","qvel","position_active","velocity_active"):return ("source_unit_input_delivery",),"pre-frozen typed input delivery at explicit boundary","until next input delivery","input lifetime"
        if field in ("length","moment","position_status"):return ("transmission_position",),"unconditional zero/status clear, then active==1 finite fixed topology source writes","no output retention across a called position kernel","position-stage lifetime"
        return ("transmission_velocity","test_velocity_gate_injection") if field=="velocity_status" else ("transmission_velocity",),"unconditional zero/status clear, active==1 producer status0 finite qvel then RN source reduction; named test may inject velocity_status3","no output retention across a called velocity kernel","velocity-stage lifetime"
    if field in ("warning_number","warning_lastinfo"):return ("force_stage.BADCTRL","reset_warning_at_boundary"),"first BADCTRL after valid enabled clamped ctrl increments and sets firstinfo; binary device reset==1 clears","inactive/healthy/disabled/upstream-error/reset0 or invalid-reset holds","original warning sub-state lifetime; full native reset integration pending"
    if field=="warning_reset_mask":return ("source_reset_input_delivery",),"explicit reset boundary mask D2D copy","until next reset boundary","reset-boundary mask lifetime"
    if field=="warning_reset_status":return ("reset_warning_at_boundary",),"reset kernel always sets status0, invalid binary mask sets1 without warning writes","outside explicit reset boundary retains status","reset-boundary diagnostic lifetime"
    if field in ("ctrl","active"):return ("source_unit_input_delivery",),"pre-frozen typed source input delivery at explicit boundary","until next input delivery","input lifetime"
    if field in ("force","force_pre","ctrl_used","ctrl_route","force_route","qfrc_actuator","status","warn_badctrl_count","warn_badctrl_lastinfo"):return ("force_stage",),"unconditional force/qfrc/diagnostic clear; active and velocity gates; disable; clamp; firstBADCTRL; fixedgain/affinebias/forceclip/ordered moment transpose","per-call outputs cleared; per-call warning delta distinct from persistent state","force-stage lifetime"
    return ("original_model_constructor",),"constructor only, actual original fixed model typed source","all launches hold static fields","workspace lifetime"
PENDING_EPOCH="pending original forward/control epoch; standalone native source-entry unit"
CANONICAL_READ_DOMAIN="actual source-entry consumer; whole D epoch gate pending"
def private_schema(n):
    """Input-independent complete standalone allocator/shape/write contract.

    This schema is not a whole D shared-v1 descriptor or a production epoch.
    """
    fields={}
    for owner,names in (("tr",TR_PRIVATE),("af",AF_PRIVATE),("observer",OBSERVER_PRIVATE)):
        for field in names:
            integer=(field in ("qadr","dadr","position_active","velocity_active",
                "position_status","velocity_status","force_limited","bias_type",
                "active","ctrl_route","force_route","status","warn_badctrl_count",
                "warn_badctrl_lastinfo","warning_number","warning_lastinfo",
                "warning_reset_mask","warning_reset_status","consumer_active",
                "consumer_vel_status"))
            static=(owner=="tr" and field in ("qadr","dadr","gear") or
                    owner=="af" and field in ("qadr","gain","bias0","bias1","bias2",
                       "ctrl_lo","ctrl_hi","force_lo","force_hi","force_limited","bias_type"))
            if static:shape=(9,)
            elif field=="qpos":shape=(n,48)
            elif field in ("qvel","qfrc_actuator"):shape=(n,43)
            elif "moment" in field:shape=(n,9,43)
            elif field in ("length","velocity","ctrl","force","force_pre","ctrl_used",
                           "ctrl_route","force_route","consumer_length","consumer_velocity",
                           "consumer_ctrl"):shape=(n,9)
            else:shape=(n,)
            dtype="int32" if integer else "float64";stride=4 if integer else 8
            strides=[]
            for extent in reversed(shape):strides.append(stride);stride*=extent
            writers,domain,retain,life=contract(owner,field)
            fields[owner+"."+field]={"shape":list(shape),"dtype":dtype,
                "strides":list(reversed(strides)),"logical_bytes":stride,"owner":owner,
                "writers":list(writers),"valid_write":domain,"retain":retain,
                "lifetime":life,"epoch":PENDING_EPOCH,"read_domain":CANONICAL_READ_DOMAIN}
    return fields
def rec(p):
    b=p.read_bytes();return {"path":str(p.resolve()),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def prepare(run):
    if subprocess.check_output(["git","-C",str(F_ROOT),"rev-parse",F_HEAD+"^{tree}"],text=True).strip()!=F_TREE:raise ValueError("F exact immutable tree")
    dest=run/"dependency-source-snapshot/phase06_foundation";dest.mkdir(parents=True,exist_ok=False)
    records=[]
    for name,sha in PINS.items():
        rel="libero/libero/gpu/phase06_foundation/"+name
        b=subprocess.check_output(["git","-C",str(F_ROOT),"show",F_HEAD+":"+rel])
        if hashlib.sha256(b).hexdigest()!=sha:raise ValueError("F source pin: "+name)
        p=dest/name
        with p.open("xb") as f:f.write(b)
        p.chmod(0o444)
        records.append({"git_source":{"head":F_HEAD,"tree":F_TREE,"relative_path":rel,"blob":subprocess.check_output(["git","-C",str(F_ROOT),"rev-parse",F_HEAD+":"+rel],text=True).strip()},"actual_import_source":rec(p)})
    p=run/"F_OWNER_SOURCE_FREEZE.json"
    with p.open("x") as f:json.dump({"schema":"p06d-exact-F-api-source-consumption-v1","accepted":False,"shared_v1_final":False,"candidate_head":F_HEAD,"candidate_tree":F_TREE,"sources":records},f,indent=2,sort_keys=True);f.write("\n")
    return rec(p)
def load(run):
    p=run/"F_OWNER_SOURCE_FREEZE.json";j=json.loads(p.read_text())
    for r in j["sources"]:
        v=r["actual_import_source"]
        if rec(pathlib.Path(v["path"]))!=v:raise ValueError("F imported source drift")
    sys.path.insert(0,str(run/"dependency-source-snapshot"))
    from phase06_foundation.binding import BindingRegistry,allocation_origin,dtype_name
    from phase06_foundation.schema import FieldDescriptor
    return BindingRegistry,allocation_origin,dtype_name,FieldDescriptor

def bind(run,tr,af,arr,stream):
    import warp as wp
    from types import SimpleNamespace
    Registry,origin,dtype,Descriptor=load(run)
    observer=SimpleNamespace(**{f:a for f,a in arr.items() if f.startswith("consumer_")})
    call=SimpleNamespace(length=tr.length,moment=tr.moment,velocity=tr.velocity,
                        ctrl=af.ctrl,active=af.active,vel_status=tr.velocity_status,stream=stream)
    reg=Registry({"tr":tr,"af":af,"observer":observer,"call":call},stream)
    descriptors={}
    pending_epoch=PENDING_EPOCH
    for owner,root in (("tr",tr),("af",af),("observer",observer)):
        required={"tr":set(TR_PRIVATE),"af":set(AF_PRIVATE),"observer":set(OBSERVER_PRIVATE)}[owner]
        if {k for k,v in vars(root).items() if isinstance(v,wp.array)}!=required:raise ValueError("complete actual private allocation inventory")
        for field,a in vars(root).items():
            if not isinstance(a,wp.array):continue
            allocation,chain=origin(a);name=owner+"."+field
            writers,domain,retain,life=contract(owner,field)
            d=Descriptor(name=name,dtype=dtype(a.dtype),shape=tuple(a.shape),owner=owner,lifetime=life,
                         epoch=pending_epoch,writers=writers,valid_write=domain,
                         read_domain=CANONICAL_READ_DOMAIN,retain=retain,
                         offset=int(a.ptr)-int(allocation.ptr))
            reg.add(d,owner,field,allocation);descriptors[name]=(d,allocation)
    for field,canonical in (("length","tr.length"),("moment","tr.moment"),("velocity","tr.velocity"),
                            ("ctrl","af.ctrl"),("active","af.active"),("vel_status","tr.velocity_status")):
        d,allocation=descriptors[canonical]
        alias=Descriptor(name="call."+field,dtype=d.dtype,shape=d.shape,owner=d.owner,lifetime=d.lifetime,
                         epoch=d.epoch,writers=d.writers,valid_write=d.valid_write,read_domain="actual force kernel argument",retain=d.retain,offset=d.offset,access="read")
        reg.add(alias,"call",field,allocation,alias_of=canonical)
    reg.bind_stream("call","stream");reg.seal()
    return reg,call
