"""Read-only source-typed scientific gate; never imports CUDA/native oracle.

A full-D request remains blocked until every original unit and dependency has
actual typed evidence. Hashes/report labels alone never produce a science pass.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import numpy as np
from typed_compare import compare_arrays, verify_index, hash_file
from ptx_audit import verify_module_set,ACCELERATION_MODULES
from source_budget import CAPS, NAMES, FIELDS, FLOAT_FIELDS, INT_FIELDS, INPUT_FIELDS, key

FULL_REQUIRED = (
    "original13sections", "qpos_root_nonzero_connected_producers",
    "original_controller_native_cache_and_live_bias",
    "actual_sm120_controller49FMA_math", "actual_native_dynamic_callsite_FEnv",
    "original_shared_v1_live_owner_ABI", "wholelife_paired_REF",
    "constraints_postconstraint_sensor_refresh_CSN", "original_native_errors",
)
UNIT = "source_acceleration_complete_fixed_inputs"
SCICAPS = CAPS + (5,)
FLOAT_SHAPES = {"smooth":(43,), "qfrc_smooth":(43,), "qacc_smooth":(43,),
                "ld":(153,), "inv":(43,), "sqrt_inv":(43,)}


def rec(path):
    p = pathlib.Path(path)
    return {"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":hash_file(p)}


def verify_record(item):
    p = pathlib.Path(item["path"])
    if not p.is_file() or p.stat().st_size != item["bytes"] or hash_file(p) != item["sha256"]:
        raise ValueError("source/evidence binding: "+str(p))


def policy(freeze, expected, budgets, inputs):
    """Only this complete native-source unit has an executable policy yet.

    Missing source math for other units is a mandatory incomplete result,
    not a broad tolerance or a developer label silently treated as proof.
    """
    if freeze["unit"] != UNIT:
        raise ValueError("no executable source policy for requested unit")
    cases = [(n,name) for n in SCICAPS for name in NAMES]
    if [(c["N"],c["name"]) for c in freeze["cases"]] != cases:
        raise ValueError("original fixed-input case/capacity inventory mismatch")
    input_names = {"STATIC__"+f for f in ("body_parentid","body_rootid","body_dofadr","body_dofnum","dof_parentid","dof_Madr")} | {key(n,c,f) for n,c in cases for f in INPUT_FIELDS}
    if set(inputs) != input_names:
        raise ValueError("complete original source typed input schema required")
    names = {key(n,c,f) for n,c in cases for f in FIELDS}
    if set(expected) != names:
        raise ValueError("complete thirteen-output schema required")
    wanted = {key(n,c,f+"__"+s) for n,c in cases for f in FLOAT_FIELDS for s in ("lo","hi")}
    wanted |= {key(n,c,f) for n,c in cases for f in ("residual_diag","residual_bound","residual_mask")}
    if set(budgets) != wanted:
        raise ValueError("complete per-field source budget schema required")
    bounds, masks = {}, {}
    for n,c in cases:
        diag=budgets[key(n,c,"residual_diag")]
        bound=budgets[key(n,c,"residual_bound")]
        mask=budgets[key(n,c,"residual_mask")]
        if (diag.dtype!=np.float64 or diag.shape!=(n,43) or
            not np.isfinite(diag).all() or np.any(diag<1e-15) or
            bound.dtype!=np.float64 or bound.shape!=(n,) or
            not np.isfinite(bound).all() or np.any(bound<0) or
            mask.dtype!=np.uint8 or mask.shape!=(n,) or np.any(mask>1)):
            raise ValueError("pre-output residual certificate schema/domain")
        for f in FIELDS:
            name = key(n,c,f)
            shape = (n,)+FLOAT_SHAPES[f] if f in FLOAT_FIELDS else (n,43) if f == "pivot" else (n,)
            dtype = np.dtype("<f8") if f in FLOAT_FIELDS else np.dtype("<i4")
            if expected[name].shape != shape or expected[name].dtype != dtype:
                raise ValueError("original source typed output schema: "+name)
            if f in FLOAT_FIELDS:
                bounds[name] = (budgets[name+"__lo"],budgets[name+"__hi"])
                masks[name] = np.ones(shape,np.uint8)
    return bounds,masks


def check(run, *, freeze_sha256, require_full=False):
    run = pathlib.Path(run)
    freeze_path = run/"SCIENCE_FREEZE.json"
    if hash_file(freeze_path) != freeze_sha256:
        raise ValueError("externally bound pre-GPU freeze SHA mismatch")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("unit")=="original_native_transmission_actuation":
        from actuation_gate import check as actuation_check
        verdict=actuation_check(run,freeze_sha256=freeze_sha256)
        verdict.update(unit=freeze["unit"],scope_pass=verdict["ok"],fields=verdict["typed_outputs"],
                       full_missing=list(FULL_REQUIRED),require_full=require_full)
        verdict["ok"]=verdict["scope_pass"] and not require_full
        return verdict

    if freeze["schema"] != "p06d-pre-output-source-typed-v2":
        raise ValueError("missing pre-output source-typed schema")
    for item in freeze["source_records"] + freeze["numeric_amendments"]:
        verify_record(item)
    if freeze["budget_construction"]["gpu_answer_used"] or freeze["budget_construction"]["cpu_expected_used"]:
        raise ValueError("source budget must depend only on frozen input/domain")
    expected = verify_index(run/"EXPECTED.npz",freeze["expected"])
    inputs = verify_index(run/"INPUTS.npz",freeze["inputs"])
    budgets = verify_index(run/"BUDGETS.npz",freeze["budgets"])
    for _ in inputs:  # Identity only; never regenerate native truth or solve.
        pass
    bounds,masks = policy(freeze,expected,budgets,inputs)
    # Budget/native-truth agreement is checked before any raw output.
    if not compare_arrays(expected,expected,bounds=bounds,write_masks=masks)["ok"]:
        raise ValueError("invalid original native truth/source budget")
    idx = json.loads((run/"SCIENTIFIC_RAW_INDEX.json").read_text())
    if idx["freeze_sha256"] != freeze_sha256:
        raise ValueError("actual GPU launch consumed a different pre-output freeze")
    verify_record(idx["first_gpu_marker"])
    marker = json.loads(pathlib.Path(idx["first_gpu_marker"]["path"]).read_text())
    if marker["freeze_sha256"] != freeze_sha256 or marker["expected_sha256"] != freeze["expected"]["artifact"]["sha256"]:
        raise ValueError("pre-first-GPU actual event binding")
    for item in idx["compile_cache"] + idx["runtime_records"]:
        verify_record(item)
    # Complete actual-runtime/cache binding and all predicates are inspected.
    ptx_claims=verify_module_set(idx["compile_cache"],idx["actual_ptx_premises"]["relevant_ptx"],
                                 idx["runtime_records"],ACCELERATION_MODULES)
    # Check actual allocation/view metadata separately from numerical values.
    profiles=idx["actual_bindings"]
    if [v["N"] for v in profiles] != list(SCICAPS):
        raise ValueError("actual capacity binding profile coverage")
    for profile in profiles:
        n=profile["N"]
        if set(profile["fields"]) != set(FIELDS):
            raise ValueError("complete actual output view inventory")
        ranges=[]
        for f,view in profile["fields"].items():
            shape=(n,)+FLOAT_SHAPES[f] if f in FLOAT_FIELDS else (n,43) if f=="pivot" else (n,)
            itemsize=8 if f in FLOAT_FIELDS else 4
            extent=int(np.prod(shape))*itemsize
            strides=[];step=itemsize
            for d in reversed(shape):
                strides.append(step);step*=d
            if (view["shape"] != list(shape) or view["strides"] != list(reversed(strides)) or
                view["dtype"] != ("<class 'warp._src.types.float64'>" if f in FLOAT_FIELDS else "<class 'warp._src.types.int32'>") or
                view["device"] != "cuda:0" or view["ptr"] <= 0 or
                view["ptr"] % itemsize or view["capacity_bytes"] < extent):
                raise ValueError("actual typed view/capacity/range differs: "+f)
            ranges.append((view["ptr"],view["ptr"]+extent,f))
        for a,b in zip(sorted(ranges),sorted(ranges)[1:]):
            if a[1]>b[0]:
                raise ValueError("unexpected actual output allocation alias: "+a[2]+"/"+b[2])

    artifacts = idx["artifacts"]
    required = {"EAGER","GRAPH","SIGNEDZERO","LAWFUL_1ULP"} | {"DIRTY_"+f for f in FIELDS}
    if set(artifacts) != required:
        raise ValueError("complete positive and per-output actual dirty raw required")
    verdicts = {}
    for name,item in artifacts.items():
        raw = verify_index(pathlib.Path(item["artifact"]["path"]),item)
        verdict=compare_arrays(expected,raw,bounds=bounds,write_masks=masks)
        residual_failures=[]
        for case in freeze["cases"]:
            n,c=case["N"],case["name"]
            diag=budgets[key(n,c,"residual_diag")]
            allowance=budgets[key(n,c,"residual_bound")]
            mask=budgets[key(n,c,"residual_mask")].astype(bool)
            x=raw[key(n,c,"qacc_smooth")]
            b=raw[key(n,c,"qfrc_smooth")]
            if x.dtype!=np.float64 or b.dtype!=np.float64 or x.shape!=(n,43) or b.shape!=(n,43):
                continue  # The independent schema assertion already fails.
            residual=np.max(np.abs(diag*x-b),axis=1)
            bad=mask & (~np.isfinite(residual) | (residual>allowance))
            if np.any(bad):
                w=int(np.flatnonzero(bad)[0])
                residual_failures.append({"case":c,"N":n,"world":w,
                    "residual":float(residual[w]).hex(),"frozen_source_bound":float(allowance[w]).hex()})
        verdict["residual_failures"]=residual_failures
        verdict["ok"] &= not residual_failures
        verdicts[name]=verdict
    baseline=verify_index(pathlib.Path(artifacts["GRAPH"]["artifact"]["path"]),artifacts["GRAPH"])
    plans=freeze["test_positive_control_plan_input_only"]
    if set(plans)!=set(map(str,SCICAPS)):raise ValueError("complete pre-output positive stimulus plan")
    for label in ("SIGNEDZERO","LAWFUL_1ULP"):
        variant=verify_index(pathlib.Path(artifacts[label]["artifact"]["path"]),artifacts[label])
        targets={}
        for n in SCICAPS:
            plan=plans[str(n)]
            want={"zero_case":"signed_zero","zero_field":"smooth","zero_flat_index":0,
                "nonzero_case":"synthetic_repair25_original","nonzero_field":"qacc_smooth",
                "nonzero_flat_index":5,"delta_hex":"0x1.0000000000000p-3"}
            if plan!=want:raise ValueError("input-independent original positive plan")
            prefix="zero" if label=="SIGNEDZERO" else "nonzero"
            k=key(n,plan[prefix+"_case"],plan[prefix+"_field"]);i=plan[prefix+"_flat_index"]
            targets[k]=i;x=baseline[k].reshape(-1)[i];y=variant[k].reshape(-1)[i]
            if x.tobytes()==y.tobytes():raise ValueError("unmutated actual acceleration positive payload")
            if label=="SIGNEDZERO":
                if not (x==0. and y==0.):raise ValueError("actual acceleration signed-zero domain")
            elif x==y or float(y-x).hex()!=plan["delta_hex"] or y!=np.nextafter(x,np.float64(np.inf)):raise ValueError("actual frozen acceleration one-ULP stimulus")
        for k,x in baseline.items():
            y=variant[k]
            if k in targets:
                original=x.copy().reshape(-1);changed=y.reshape(-1)
                original[targets[k]]=changed[targets[k]]
                if original.tobytes()!=changed.tobytes():raise ValueError("extra acceleration positive word mutation")
            elif x.tobytes()!=y.tobytes():raise ValueError("extra acceleration positive packet mutation")
    positive = all(verdicts[n]["ok"] for n in ("EAGER","GRAPH","SIGNEDZERO","LAWFUL_1ULP"))
    negative = all(not verdicts["DIRTY_"+f]["ok"] for f in FIELDS)
    for f in FIELDS:
        target = "DIRTY_"+f
        if not any(v.get("field","").endswith("__"+f) for v in verdicts[target]["failures"]):
            negative = False
    scope_pass = positive and negative
    full_missing = list(FULL_REQUIRED)
    return {"schema":"p06d-raw-scientific-verdict-v2","unit":freeze["unit"],
            "scope_pass":scope_pass,"positive_controls":positive,"actual_device_dirty_rejected":negative,
            "fields":len(expected),"cases":len(freeze["cases"]),"verdicts":verdicts,
            "independently_checked_actual_ptx":ptx_claims,
            "freeze":rec(freeze_path),"raw_index":rec(run/"SCIENTIFIC_RAW_INDEX.json"),
            "require_full":require_full,"full_missing":full_missing,
            "full_D_ready":False,"accepted":False,
            "ok":scope_pass and not require_full,
            "claim":"developer self-test of the stated complete source unit; independent review required"}
