"""Actual typed-row source checks for lawful runtime model generations.

The independent domain receipt must authorize these parameter rules. This file
only validates their actual preconditions; it grants no numerical licence.
"""
import hashlib
import math
import struct
from .admission import QualificationError

_PHYSICS_FIELDS={"body_pos":3,"body_quat":4}

def _rows(name,values,inventory):
    if hasattr(values,"tolist"):values=values.tolist()
    header=inventory[name]
    width=_PHYSICS_FIELDS[name]
    if header["dtype"]!="<f8" or len(header["shape"])!=2 or header["shape"][1]!=width:
        raise QualificationError("actual FP64 source parameter header differs")
    if not isinstance(values,list) or len(values)!=header["shape"][0]:
        raise QualificationError("actual parameter row extent differs")
    out=[]
    for row in values:
        if not isinstance(row,(list,tuple)) or len(row)!=width:
            raise QualificationError("actual parameter row width differs")
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row):
            raise QualificationError("actual source parameter must be finite")
        out.append(tuple(float(v) for v in row))
    raw=b"".join(struct.pack("<"+"d"*width,*row) for row in out)
    encoded=name.encode()+b"\0"+b"<f8"+b"\0"+str(tuple(header["shape"])).encode()+b"\0"+raw
    return out,raw,hashlib.sha256(encoded).hexdigest()

def check_typed_source_generation(base_md,current_md,base_values,current_values,policy):
    """Compare real metadata AND actual row bytes, never hash-ignore or sameNV.

    Metadata is the already computed actual native decoder packet, so unchanged
    textures/full MJB need not be saved or rehashed again at each reset.
    Parameter arrays are actual readonly compiler/arena owner values, not a
    fabricated boolean proof. Physics generation and visual epoch stay separate.
    """
    mutable=set(policy["physics_mutable_fields"])
    if not mutable or mutable-set(_PHYSICS_FIELDS):
        raise QualificationError("source physics parameter rule unsupported")
    for field in ("dimensions","options","topology_sha256","name_maps","robot_metadata","stat","static_array_inventory"):
        if base_md[field]!=current_md[field]:
            raise QualificationError("immutable native feature changed: "+field)
    base_hashes=base_md["array_sha256"];now_hashes=current_md["array_sha256"]
    if set(base_hashes)!=set(now_hashes):raise QualificationError("native array inventory differs")
    changed={name for name in base_hashes if base_hashes[name]!=now_hashes[name]}
    if changed-mutable:raise QualificationError("immutable actual field changed: "+str(sorted(changed-mutable)))
    if changed and base_md["model_sha256"]==current_md["model_sha256"]:
        raise QualificationError("changed actual typed parameters retain stale binary identity")
    proof=[]
    allowed=set(policy["source_body_ids"])
    if not allowed or any(type(i) is not int or not 0<=i<base_md["dimensions"]["nbody"] for i in allowed):
        raise QualificationError("source body-ID policy malformed")
    for name in sorted(mutable):
        b,br,bhash=_rows(name,base_values[name],base_md["static_array_inventory"])
        c,cr,chash=_rows(name,current_values[name],current_md["static_array_inventory"])
        if bhash!=base_hashes[name] or chash!=now_hashes[name]:
            raise QualificationError("actual parameter values do not match native descriptor: "+name)
        width=_PHYSICS_FIELDS[name]
        # Pack actual FP64 words rather than numeric ==, preserving signed-zero.
        ids=[i for i,(x,y) in enumerate(zip(b,c)) if struct.pack("<"+"d"*width,*x)!=struct.pack("<"+"d"*width,*y)]
        if set(ids)-allowed:raise QualificationError("source-foreign body row changed")
        if name=="body_quat" and any(abs(sum(v*v for v in row)-1.0)>policy["source_quaternion_norm_atol"] for row in c):
            raise QualificationError("actual source body quaternion is not unit normalized")
        proof.append({"field":name,"actual_changed_body_ids":ids,"base_field_sha256":bhash,"current_field_sha256":chash})
    if not changed and base_md["model_sha256"]!=current_md["model_sha256"]:
        # Identical public typed metadata is not a licence for unexplained blob
        # differences; cross-compile semantic certificates are a separate scope.
        raise QualificationError("unexplained native binary identity change")
    return {"base_model_sha256":base_md["model_sha256"],"current_model_sha256":current_md["model_sha256"],
            "actual_typed_parameter_rows":proof,"immutable_native_fields_equal":True,
            "source_policy":policy,"numerical_permission_granted":False}
