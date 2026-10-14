"""New original-source pre-output science unit, original cases retained.

CPU produces all truth, inputs and independent field budgets before first GPU
output. GPU never loads EXPECTED/BUDGETS values into device arrays. Readonly
runs load raw and frozen typed data only; no native oracle, solve or CUDA.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
import numpy as np
from runtime_audit import (ROOT, RECEIVE, SOURCE_PINS, MJB_SHA, digest, record,
                           write_json, save_npz, runtime_identity, utc)
from typed_compare import verify_index, compare_arrays
from source_budget import (CAPS, NAMES, FIELDS, FLOAT_FIELDS, INT_FIELDS,
                           INPUT_FIELDS, key, derive, residual_prepare)
from scientific_gate import SCICAPS, UNIT, check

STATIC = ("body_parentid","body_rootid","body_dofadr","body_dofnum",
          "dof_parentid","dof_Madr")
G = RECEIVE/"audit/codex-goal-20261005-g01"


def cases(model):
    old_rng = np.random.default_rng(4401)
    original_rn = []
    for i in range(4):
        row = {}
        for f in ("passive","bias","applied","actuator"):
            row[f] = old_rng.normal(size=(1,43)) * 10.**old_rng.integers(-3,6)
        original_rn.append(row)
    original_rn[0]["passive"][0,0] = 1.+2.**52
    original_rn[0]["bias"][0,0] = 2.**52
    original_rn[0]["applied"][0,0] = -1.
    original_rn[0]["actuator"][0,0] = .5
    rng = np.random.default_rng(4401)
    for n in SCICAPS:
        for ci,name in enumerate(NAMES):
            p = {"active":np.ones(n,np.int32),
                 **{f:np.zeros((n,43)) for f in ("passive","bias","applied","actuator")},
                 "xfrc":np.zeros((n,38,6)),"xipos":np.zeros((n,38,3)),
                 "subtree_com":np.zeros((n,38,3)),"cdof":np.zeros((n,43,6)),
                 "qm":np.zeros((n,153))}
            p["qm"][:,model.dof_Madr] = 2.+.01*np.arange(43)
            if ci < 6:
                values=(1e-18,np.nextafter(1e-15,-np.inf),1e-15,
                        np.nextafter(1e-15,np.inf),0.,-1e-18)
                p["qm"][:,int(model.dof_Madr[5])] = values[ci]
                p["passive"][:,5] = 1.
            elif name=="three_rn_original":
                for f,scale in zip(("passive","bias","applied","actuator"),(1e3,1e2,1.,10.)):
                    p[f][:] = rng.normal(size=(n,43))*scale
                p["passive"][:,0]=1.+2.**52;p["bias"][:,0]=2.**52
                p["applied"][:,0]=-1.;p["actuator"][:,0]=.5
            elif name=="all_37_bodies_force_torque":
                for f in ("passive","bias","applied","actuator"):
                    p[f][:] = rng.normal(size=(n,43))
                p["xfrc"][:,1:]=rng.normal(size=(n,37,6))*.3
                p["xipos"][:]=rng.normal(size=(n,38,3))*.1
                p["subtree_com"][:]=rng.normal(size=(n,38,3))*.01
                p["cdof"][:]=rng.normal(size=(n,43,6))
            elif name=="source_body_association":
                p["passive"][:,0]=1.;p["cdof"][:,0,3]=1.
                for b,v in zip((1,2,3),(2.**53,-2.**53,1.)):
                    p["xfrc"][:,b,0]=v
            elif name=="signed_zero":
                for f in ("passive","applied","actuator","xfrc"):p[f][:]=-0.
            elif name.startswith("original_rn_case"):
                old=original_rn[int(name[-1])]
                for f in old:p[f][:]=old[f]
            elif name=="original_xfrc_seed4402":
                old=np.random.default_rng(4402)
                for f in ("passive","bias","applied","actuator"):
                    p[f][:]=old.normal(size=(1,43))
                for b in (1,7,20):p["xfrc"][:,b,:]=old.normal(size=6)*1e3
                p["xipos"][:]=old.normal(size=(1,38,3))*.1
                p["subtree_com"][:]=old.normal(size=(1,38,3))*.01
                p["cdof"][:]=old.normal(size=(1,43,6))
            elif name in ("original_capacity_seed","original_capacity_changed"):
                old=np.random.default_rng(4401+100+n)
                if n>1:p["active"][-1]=0
                for f in ("passive","bias","applied","actuator"):
                    p[f][:]=old.normal(size=(n,43))
                p["xipos"][:]=old.normal(size=(n,38,3))*.01
                p["subtree_com"][:]=old.normal(size=(n,38,3))*.001
                p["cdof"][:]=old.normal(size=(n,43,6))
                p["qm"][:,model.dof_Madr]=3.+.01*np.arange(43)
                if name.endswith("changed"):p["passive"]+=1.
            elif name in ("original_error_worlds","original_error_relaunch"):
                # Original five-world row pattern preserved intact at N5,
                # plus every required N. No active7/NaN/repair row is deleted.
                p["qm"][:,model.dof_Madr]=2.
                for w in range(n):
                    route=w%5
                    if route==0:p["passive"][w,0]=1.
                    elif route==1:p["active"][w]=0
                    elif route==2:p["active"][w]=7
                    elif route==3:p["passive"][w,0]=np.nan
                    else:p["qm"][w,model.dof_Madr[5]]=1e-18
            elif name=="original_healthy_reinput":
                p["qm"][:,model.dof_Madr]=2.
            else:raise AssertionError(name)
            yield n,name,p


def positive_plan():
    # Test-only stimuli fixed before observing any native/GPU answer. The
    # existing input-only source bounds must already support them; no budget
    # is widened if a legal positive fails the later physical/raw replay.
    return {str(n):{"zero_case":"signed_zero","zero_field":"smooth","zero_flat_index":0,
        "nonzero_case":"synthetic_repair25_original","nonzero_field":"qacc_smooth",
        "nonzero_flat_index":5,"delta_hex":"0x1.0000000000000p-3"}
        for n in SCICAPS}


def preserve_native_failure(run,reason,n,name,w,payload,static,pair,integer,expect,native=None,truth_gate=None):
    directory=run/"CPU_ORIGINAL_FAILURES"/("N%d-%s-w%d"%(n,name,w))
    directory.mkdir(parents=True,exist_ok=False)
    inputs={**{"INPUT__"+k:np.array(v,copy=True) for k,v in payload.items()},**{"STATIC__"+k:np.array(v,copy=True) for k,v in static.items()}}
    derived={"EXPECTED_PARTIAL__"+k:np.array(v,copy=True) for k,v in expect.items()}
    for k,(lo,hi) in pair.items():derived[k+"__lo"]=np.array(lo,copy=True);derived[k+"__hi"]=np.array(hi,copy=True)
    derived.update({"SOURCE_INTEGER__"+k:np.array(v,copy=True) for k,v in integer.items()})
    actual={}
    if native is not None:
        for key in ("qM","qLD","qLDiagInv","qLDiagSqrtInv","qfrc_smooth","qacc_smooth"):
            actual[key]=np.array(getattr(native,key),copy=True)
        actual["warning_number"]=np.array([int(x.number) for x in native.warning],np.int32)
        actual["warning_lastinfo"]=np.array([int(x.lastinfo) for x in native.warning],np.int32)
    artifacts={"input":save_npz(directory/"INPUTS.npz",inputs),"prederived_source_and_partial_cpu":save_npz(directory/"SOURCE_BUDGET_AND_PARTIAL_EXPECTED.npz",derived),"actual_native":save_npz(directory/"ACTUAL_NATIVE.npz",actual)}
    write_json(directory/"FAILURE.json",{"reason":reason,"N":n,"name":name,"world":w,"artifacts":artifacts,"truth_gate":truth_gate,"expected_is_partial":True,"original_failure_retained":True,"accepted":False})

def cpu(run):
    sys.path[:0]=[str(ROOT/"libero"),str(ROOT)]
    import mujoco
    from run_author import load_pack
    pack=load_pack();m=pack.model
    runtime_identity(pathlib.Path.cwd()/"RUNTIME_BEFORE.json",require_mujoco=True)
    static={f:np.asarray(getattr(m,f),np.int32).copy() for f in STATIC}
    inputs={"STATIC__"+f:v for f,v in static.items()}
    expected,budgets,native_invalid,inventory={},{},{},[]
    plan=positive_plan()
    for n,name,payload in cases(m):
        for f,v in payload.items():inputs[key(n,name,f)]=v
        pair,integer=derive(payload,static)  # Before observing the native answer.
        expect={f:integer[f].copy() for f in INT_FIELDS}
        expect.update({f:np.zeros_like(pair[f][0]) for f in FLOAT_FIELDS})
        for w in range(n):
            if payload["active"][w]!=1:continue
            d=mujoco.MjData(m)
            for f,df in (("qm","qM"),("passive","qfrc_passive"),("bias","qfrc_bias"),
                         ("applied","qfrc_applied"),("actuator","qfrc_actuator"),
                         ("xfrc","xfrc_applied"),("xipos","xipos"),
                         ("subtree_com","subtree_com"),("cdof","cdof")):
                getattr(d,df)[:]=payload[f][w]
            try:
                mujoco.mj_factorM(m,d);mujoco.mj_fwdAcceleration(m,d)
            except Exception as error:
                preserve_native_failure(run,"native call exception: "+repr(error),n,name,w,payload,static,pair,integer,expect,d)
                raise
            valid=all(np.isfinite(payload[f][w]).all() for f in ("passive","bias","applied","actuator"))
            if valid:
                expect["smooth"][w]=(payload["passive"][w]-payload["bias"][w])+payload["applied"][w]+payload["actuator"][w]
                expect["qfrc_smooth"][w]=d.qfrc_smooth
                expect["qacc_smooth"][w]=d.qacc_smooth
            else:
                native_invalid[key(n,name,"native_nonfinite_qfrc")+"__w%d"%w]=np.array(d.qfrc_smooth)
                native_invalid[key(n,name,"native_nonfinite_qacc")+"__w%d"%w]=np.array(d.qacc_smooth)
            expect["ld"][w]=d.qLD;expect["inv"][w]=d.qLDiagInv
            expect["sqrt_inv"][w]=d.qLDiagSqrtInv
            warn=d.warning[int(mujoco.mjtWarning.mjWARN_INERTIA)]
            if int(warn.number)!=int(integer["warning_count"][w]):
                preserve_native_failure(run,"native warning sequence differs from pre-input source derivation",n,name,w,payload,static,pair,integer,expect,d)
                raise AssertionError("native warning sequence differs from pre-input source derivation")
            if warn.number and int(warn.lastinfo)!=int(integer["warning_last"][w]):
                preserve_native_failure(run,"native warning lastinfo source order",n,name,w,payload,static,pair,integer,expect,d)
                raise AssertionError("native warning lastinfo source order")
        for f,v in expect.items():expected[key(n,name,f)]=v
        local_bounds={}
        for f,(lo,hi) in pair.items():
            budgets[key(n,name,f+"__lo")]=lo
            budgets[key(n,name,f+"__hi")]=hi
            local_bounds[f]=(lo,hi)
        diag,residual,mask=residual_prepare(payload,static,pair,integer)
        budgets[key(n,name,"residual_diag")]=diag
        budgets[key(n,name,"residual_bound")]=residual
        budgets[key(n,name,"residual_mask")]=mask
        truth_gate=compare_arrays(expect,expect,bounds=local_bounds)
        if not truth_gate["ok"]:
            preserve_native_failure(run,"original native truth unsupported",n,name,-1,payload,static,pair,integer,expect,None,truth_gate)
            raise ValueError("original native truth unsupported: "+repr(truth_gate))
        inventory.append({"N":n,"name":name,"worlds":n,
                          "original_native_factorM_fwdAcceleration":True,
                          "invalid_wrapper_rules_distinct_from_native_NaN":True})
        print(json.dumps({"preparing":name,"N":n}),flush=True)
    inp=save_npz(run/"INPUTS.npz",inputs)
    exp=save_npz(run/"EXPECTED.npz",expected)
    bud=save_npz(run/"BUDGETS.npz",budgets)
    invalid=save_npz(run/"NATIVE_INVALID_ORIGINAL.npz",native_invalid)
    source_files=[pathlib.Path(__file__),pathlib.Path(__file__).with_name("source_budget.py"),
                  pathlib.Path(__file__).with_name("scientific_gate.py"),
                  pathlib.Path(__file__).with_name("typed_compare.py"),
                  pathlib.Path(__file__).with_name("scientific_faults.py"),
                  pathlib.Path(__file__).with_name("ptx_audit.py"),
                  pathlib.Path(__file__).with_name("compiler_audit.py"),
                  pathlib.Path(__file__).with_name("runtime_audit.py"),
                  pathlib.Path(__file__).with_name("ptx_gate_controls.py"),
                  ROOT/"scripts/gpu_smooth_acceleration_chain/budget.py",
                  ROOT/"libero/libero/gpu/host/runtime/dynamics/acceleration.py",
                  ROOT/"libero/libero/gpu/reference/dynamics/external_force_projection/kernel.py",
                  ROOT/"libero/libero/gpu/reference/position/mass_solve/kernel.py",
                  ROOT/"doc/phase_05/smooth_acceleration_chain/NUMERICS.md",
                  ROOT/"doc/phase_05/external_force_projection/NUMERICS.md",
                  ROOT/"scripts/gpu_smooth_acceleration_chain/run_all.py"]
    for sha in SOURCE_PINS.values():source_files.append(RECEIVE/"received-r01/evidence/objects"/sha)
    amendments=[G/"planner"/n for n in (
        "ACCEPTANCE_AMENDMENT_ZERO_ABSOLUTE_ERROR_001.json",
        "ACCEPTANCE_AMENDMENT_WITHIN_PRECISION_002.json",
        "ACCEPTANCE_PRECISION_CONFIRMATION_003.json",
        "ACCEPTANCE_RULES_RESTORATION_ADDENDUM_004.txt")]
    runtime_identity(pathlib.Path.cwd()/"RUNTIME_AFTER.json",require_mujoco=True)
    write_json(run/"SCIENCE_FREEZE.json",{
        "schema":"p06d-pre-output-source-typed-v2","unit":UNIT,"utc":utc(),
        "inputs":inp,"expected":exp,"budgets":bud,"original_native_invalid":invalid,
        "source_records":[record(p) for p in source_files],
        "numeric_amendments":[record(p) for p in amendments],"cases":inventory,
        "test_positive_control_plan_input_only":plan,
        "mjb_sha256":MJB_SHA,"package_sha256":"e79f0232f98aea00a626102840d6484ada4dc5e5f3245d01724edd0cf7fb8a53",
        "budget_construction":{"gpu_answer_used":False,"cpu_expected_used":False,
            "graph":"entry exact rational three RN -> outward source ordered each body force/torque -> independently strict floor sparse factor/solve",
            "scope":"fixed-source entry inputs; qpos-root neighborhoods/upstream producer errors still mandatory"},
        "full_D_ready":False,"original_other_units_pending":True})
    print("All original source-unit inputs, CPU truth and field budgets frozen before GPU.",flush=True)
    return 0


def gpu(run):
    freeze_path=run/"SCIENCE_FREEZE.json";freeze_sha=digest(freeze_path)
    freeze=json.loads(freeze_path.read_text())
    inputs=verify_index(run/"INPUTS.npz",freeze["inputs"])
    # Bind expected bytes without opening their values in the GPU process.
    if digest(run/"EXPECTED.npz")!=freeze["expected"]["artifact"]["sha256"]:raise ValueError("pre-GPU expected SHA")
    for x in freeze["source_records"]:
        if digest(x["path"])!=x["sha256"]:raise ValueError("frozen source drift")
    write_json(run/"FIRST_GPU_EVENT.json",{"utc":utc(),"freeze_sha256":freeze_sha,
        "expected_sha256":freeze["expected"]["artifact"]["sha256"],
        "before_Warp_import_or_CUDA_output":True})
    import warp as wp
    wp.config.kernel_cache_dir=str(pathlib.Path.cwd()/"warp-cache")
    from run_author import load_pack
    from libero.gpu.host.runtime.dynamics.acceleration import SmoothXfrcSolveWorkspace, StaticTopology
    from scientific_faults import dirty_f64,dirty_i32,flip_signbit_u64,add_f64
    pack=load_pack();m=pack.model
    topology=StaticTopology({f:np.asarray(getattr(m,f),np.int32)
                            for f in STATIC if f!="dof_Madr"})
    raw={label:{} for label in ("EAGER","GRAPH","SIGNEDZERO","LAWFUL_1ULP")}
    raw.update({"DIRTY_"+f:{} for f in FIELDS})
    allocation_records=[]
    for n in SCICAPS:
        acc=SmoothXfrcSolveWorkspace(topology,pack.solve_model(),n,"cuda:0")
        arrays={"smooth":acc.smooth,"qfrc_smooth":acc.qfrc,"qacc_smooth":acc.qacc_smooth,
                "status":acc.status,"entry_status":acc.entry_status,"d02_status":acc.d02.status,
                "solve_status":acc.solver.status,"warning_count":acc.solver.warning_count,
                "warning_last":acc.solver.warning_last,"ld":acc.solver.ld,
                "inv":acc.solver.inv,"sqrt_inv":acc.solver.sqrt_inv,"pivot":acc.solver.pivot}
        allocation_records.append({"N":n,"fields":{f:{"ptr":int(a.ptr),
            "shape":list(a.shape),"strides":list(a.strides),"dtype":str(a.dtype),
            "device":str(a.device),"capacity_bytes":int(a.capacity)} for f,a in arrays.items()}})
        for name in NAMES:
            payload={f:inputs[key(n,name,f)] for f in INPUT_FIELDS}
            acc.upload(**payload);acc.launch()
            snap=lambda:{f:np.array(a.numpy(),copy=True) for f,a in arrays.items()}
            eager=snap()
            for f,v in eager.items():raw["EAGER"][key(n,name,f)]=v
            wp.capture_begin();acc.launch();graph=wp.capture_end()
            # No-op graphs must not retain the legal output by accident.
            acc.qacc_smooth.fill_(42.);wp.capture_launch(graph)
            got=snap()
            for label in raw:
                if label=="EAGER":continue
                for f,v in got.items():raw[label][key(n,name,f)]=v.copy()
            if name=="signed_zero":
                a=acc.smooth;backup=wp.clone(a)
                wp.launch(flip_signbit_u64,dim=1,inputs=[a.view(wp.uint64).flatten(),freeze["test_positive_control_plan_input_only"][str(n)]["zero_flat_index"]],device="cuda:0")
                out=snap()
                for f,v in out.items():raw["SIGNEDZERO"][key(n,name,f)]=v
                wp.copy(a,backup)
            if name=="synthetic_repair25_original":
                a=acc.qacc_smooth;backup=wp.clone(a)
                plan=freeze["test_positive_control_plan_input_only"][str(n)]
                wp.launch(add_f64,dim=1,inputs=[a.flatten(),plan["nonzero_flat_index"],float.fromhex(plan["delta_hex"])],device="cuda:0")
                out=snap()
                for f,v in out.items():raw["LAWFUL_1ULP"][key(n,name,f)]=v
                wp.copy(a,backup)
                for f,a in arrays.items():
                    backup=wp.clone(a)
                    kernel=dirty_f64 if f in FLOAT_FIELDS else dirty_i32
                    wp.launch(kernel,dim=1,inputs=[a.flatten(),5 if f in ("smooth","qfrc_smooth","qacc_smooth") else 0,1],
                              device="cuda:0")
                    out=snap()
                    for ff,v in out.items():raw["DIRTY_"+f][key(n,name,ff)]=v
                    wp.copy(a,backup)  # earlier actual device backup; no CPU answer.
            print(json.dumps({"actual_gpu":name,"N":n}),flush=True)
    artifacts={label:save_npz(run/(label+".npz"),data) for label,data in raw.items()}
    runtime=runtime_identity(pathlib.Path.cwd()/"RUNTIME.json",require_mujoco=True)
    caches=[record(p) for p in sorted((pathlib.Path.cwd()/"warp-cache").rglob("*")) if p.is_file()]
    ptx=[pathlib.Path(x["path"]) for x in caches if x["path"].endswith(".ptx")]
    relevant=[p for p in ptx if "scientific_faults" not in str(p)]
    from ptx_audit import parse_ptx,verify_module_set,ACCELERATION_MODULES
    instructions={str(p):[r["opcode"] for r in parse_ptx(p.read_text())] for p in relevant}
    verify_module_set(caches,[record(p) for p in relevant],[record(pathlib.Path.cwd()/"RUNTIME.json")],ACCELERATION_MODULES)
    strict=bool(relevant) and all("sm_120" in p.read_text() for p in relevant) and not any(
        ".f64" in op and (op.startswith(("fma.","mad.")) or ".ftz" in op or ".approx" in op)
        for ops in instructions.values() for op in ops)
    write_json(run/"SCIENTIFIC_RAW_INDEX.json",{"freeze_sha256":freeze_sha,
        "artifacts":artifacts,"first_gpu_marker":record(run/"FIRST_GPU_EVENT.json"),
        "compile_cache":caches,"runtime_records":[record(pathlib.Path.cwd()/"RUNTIME.json")],
        "actual_bindings":allocation_records,"actual_ptx_premises":{
            "strict_rn_f64_no_FMA_FTZ_approx":strict,
            "relevant_ptx":[record(p) for p in relevant],
            "controller49FMA_scope_not_inherited":True},
        "device_dirty":"actual device kernels, snapshots before restoring actual GPU D2D backup; no CUDA clean rerun used as dirty packet",
        "signedzero_and_lawful_ulp":"diagnostic numerical comparator controls; no unauthorized production write claim"})
    return 0


def readonly(run):
    runtime_identity(pathlib.Path.cwd()/"RUNTIME.json",forbid_cuda=True)
    verdict=check(run,freeze_sha256=digest(run/"SCIENCE_FREEZE.json"))
    write_json(pathlib.Path.cwd()/"RESULT.json",verdict)
    print(json.dumps({k:v for k,v in verdict.items() if k!="verdicts"}))
    return 0 if verdict["ok"] else 1


def unit_tests(run):
    sys.path[:0]=[str(ROOT/"libero"),str(ROOT)]
    import unittest
    names=["tests.gpu_phase06_dynamics."+n for n in
           ("test_source_numeric_gate","test_fixed_model_rules","test_rn_nodes",
            "test_d02_seed_order","test_control_clock")]
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromNames(names))
    runtime_identity(pathlib.Path.cwd()/"RUNTIME.json",require_mujoco="mujoco" in sys.modules)
    write_json(pathlib.Path.cwd()/"RESULT.json",
               {"ok":result.wasSuccessful(),"tests_run":result.testsRun,
                "failures":len(result.failures),"errors":len(result.errors),
                "full_D_ready":False})
    return 0 if result.wasSuccessful() else 1


def ptx_controls(run):
    from ptx_gate_controls import run_controls
    return run_controls(run)

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",required=True)
    p.add_argument("--run",type=pathlib.Path,required=True);a=p.parse_args()
    return {"cpu":cpu,"gpu":gpu,"readonly":readonly,"unit-tests":unit_tests,"ptx-controls":ptx_controls}[a.mode](a.run)
if __name__=="__main__":sys.exit(main())
