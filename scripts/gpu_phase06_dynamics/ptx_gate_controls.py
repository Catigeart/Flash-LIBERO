"""Fresh no-CUDA/no-oracle G04-F02 metadata-only gate counterexamples.

Every actual typed raw and pre-output CPU truth remains unchanged. Synthetic
PTX is a validator diagnostic and is never described as physical device math.
"""
import copy,json,pathlib,os
from runtime_audit import record,write_json,runtime_identity,digest
from scientific_gate import check
from ptx_audit import inspect_ptx

def run_controls(run):
    runtime_identity(pathlib.Path.cwd()/"RUNTIME_BEFORE.json",forbid_cuda=True)
    idx=json.loads((run/"SCIENTIFIC_RAW_INDEX.json").read_text());fh=digest(run/"SCIENCE_FREEZE.json")
    positive=check(run,freeze_sha256=fh)
    if not positive["ok"]:raise ValueError("actual complete legal source-unit gate failed")
    def case(label,mutate):
        d=pathlib.Path.cwd()/label;d.mkdir(exist_ok=False)
        for p in [run/"SCIENCE_FREEZE.json",run/"INPUTS.npz",run/"EXPECTED.npz",run/"BUDGETS.npz",run/"MASKS.npz",run/"STATIC_NATIVE_MODEL.npz",run/"RESET_INVALID_SOURCE_EXPECTED.npz"]:
            if p.is_file():os.link(p,d/p.name)
        altered=copy.deepcopy(idx);mutate(altered,d)
        write_json(d/"SCIENTIFIC_RAW_INDEX.json",altered)
        try:
            z=check(d,freeze_sha256=fh);reason="returned verdict";rejected=not z["ok"]
        except ValueError as e:reason=str(e);rejected=True
        write_json(d/"RESULT.json",{"rejected":rejected,"reason":reason,"same_pre_output_freeze_sha256":fh,
                    "actual_device_raw_unchanged":altered["artifacts"]==idx["artifacts"],"compiled_device_math_failure_claim":False})
        if not rejected:raise ValueError("gate metadata negative accepted: "+label)
        return record(d/"RESULT.json")
    rows=[]
    rows.append(case("missing_actual_modules",lambda j,d:j["actual_ptx_premises"].update(relevant_ptx=j["actual_ptx_premises"]["relevant_ptx"][:1])))
    rows.append(case("duplicate_relevant",lambda j,d:j["actual_ptx_premises"]["relevant_ptx"].append(copy.deepcopy(j["actual_ptx_premises"]["relevant_ptx"][0]))))
    rows.append(case("duplicate_compile_cache",lambda j,d:j["compile_cache"].append(copy.deepcopy(j["compile_cache"][0]))))
    def foreign(j,d):
        p=d/"foreign_predicated_FMA.ptx";p.write_text(".version 8.8\n.target sm_120\n@%p fma.rn.f64 %fd1, %fd2, %fd3, %fd4;\n")
        j["actual_ptx_premises"]["relevant_ptx"]=[record(p)]
    rows.append(case("foreign_predicated_FMA",foreign))
    def missing_cache(j,d):
        victim=j["actual_ptx_premises"]["relevant_ptx"][0]["path"]
        j["compile_cache"]=[v for v in j["compile_cache"] if v["path"]!=victim]
    rows.append(case("missing_actual_cache_record",missing_cache))
    direct=[]
    for label,prefix in (("unpredicated",""),("predicated","@%p "),
                         ("negated_predicate","@!%p "),
                         ("second_statement","mov.f64 %fd0, %fd1; @%p "),
                         ("inline_braced_body","{ @!%p "),
                         ("comment_then_predicate","/* token boundary */ @%p "),
                         ("multiple_labels","label0: label1: @!%p "),
                         ("split_predicate","@%p\n")):
        p=pathlib.Path.cwd()/(label+"_FMA_metadata_only.ptx")
        p.write_text(".version 8.8\n.target sm_120\n"+prefix+
                     "fma.rn.f64 %fd1, %fd2, %fd3, %fd4;"+
                     (" }" if label=="inline_braced_body" else "")+"\n")
        try:inspect_ptx(record(p));raise RuntimeError("predicated opcode parser accepted FMA")
        except ValueError as e:direct.append({"file":record(p),"rejected":True,"reason":str(e)})
    runtime_identity(pathlib.Path.cwd()/"RUNTIME_AFTER.json",forbid_cuda=True)
    result={"schema":"p06d-G04-F02-fresh-negative-regression-v1","ok":True,"complete_actual_physical_positive_pass":positive["scope_pass"],
            "actual_typed_fields":positive["fields"],"metadata_only_negative_results":rows,"direct_predicate_parser_controls":direct,
            "no_new_CUDA_or_native_oracle":True,"full_D_ready":False,"accepted":False}
    write_json(pathlib.Path.cwd()/"RESULT.json",result);return 0
