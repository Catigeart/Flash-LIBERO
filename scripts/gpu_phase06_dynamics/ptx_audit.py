"""Complete source-unit PTX cache membership and all-predicate instruction audit.

Pure source/evidence reader: imports neither Warp nor native oracle libraries.
"""
from __future__ import annotations
import hashlib,pathlib,re,json
ACCELERATION_MODULES=("libero.gpu.host.runtime.dynamics.acceleration", "p06d_accepted_d02_external_force", "p06d_accepted_n06_mass_solve")
ACTUATION_MODULES=("libero.gpu.host.runtime.dynamics.actuation_split", "libero.gpu.host.runtime.dynamics.transmission")
FAULT_MODULE="scientific_faults"
_LABEL=re.compile(r"^\s*(?:[A-Za-z_$][\w.$]*):\s*")
_PREDICATE_ONLY=re.compile(r"^\s*(@!?%[A-Za-z_$][\w.$]*)\s*$")
_OPCODE=re.compile(r"^\s*(?:(@!?%[A-Za-z_$][\w.$]*)\s+)?([A-Za-z][A-Za-z0-9_.]*)\b")
def parse_ptx(text):
    # Preserve source line numbers, and keep comments as token separators.
    clean=re.sub(r"/\*.*?\*/",lambda m:" "+"\n"*m.group(0).count("\n"),
                 text,flags=re.S)
    rows=[];pending_predicate=None
    for number,line in enumerate(clean.splitlines(),1):
        line=line.split("//",1)[0]
        for statement,piece in enumerate(re.split(r"[;{}]",line)):
            # Several labels can designate the same following instruction.
            while (label:=_LABEL.match(piece)):piece=piece[label.end():]
            only=_PREDICATE_ONLY.match(piece)
            if only:
                if pending_predicate is not None:raise ValueError("two pending PTX predicates")
                pending_predicate=only.group(1);continue
            m=_OPCODE.match(piece)
            if m:
                if pending_predicate and m.group(1):raise ValueError("duplicate PTX instruction predicate")
                rows.append({"line":number,"statement":statement,
                             "predicate":m.group(1) or pending_predicate,"opcode":m.group(2)})
                pending_predicate=None
    if pending_predicate is not None:raise ValueError("dangling PTX instruction predicate")
    return rows

def verify_record(v):
    p=pathlib.Path(v["path"])
    if not p.is_file() or p.stat().st_size!=v["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest()!=v["sha256"]:
        raise ValueError("PTX byte/source binding: "+str(p))
    return p.resolve()
def identity(v):return (str(pathlib.Path(v["path"]).resolve()),v["bytes"],v["sha256"])

def inspect_ptx(v):
    p=verify_record(v);text=p.read_text();rows=parse_ptx(text)
    targets=re.findall(r"(?m)^\s*\.target\s+([^\n]+)",text)
    versions=re.findall(r"(?m)^\s*\.version\s+([^\n]+)",text)
    if targets!=["sm_120"] or versions!=["8.8"]:raise ValueError("actual source-unit architecture/PTX version: "+str(p))
    bad=[r for r in rows if ".f64" in r["opcode"] and (r["opcode"].startswith(("fma.","mad.")) or ".ftz" in r["opcode"] or ".approx" in r["opcode"])]
    if bad:raise ValueError("actual PTX violates source primitive premises: "+repr(bad[:8]))
    return {"path":str(p),"sha256":v["sha256"],"target":targets[0],"version":versions[0],
            "f64_instruction_count":sum(".f64" in r["opcode"] for r in rows),
            "predicated_instruction_count":sum(r["predicate"] is not None for r in rows),
            "all_instruction_count":len(rows)}

def verify_module_set(cache,relevant,runtimes,required):
    if len(cache)!=len({identity(v)[0] for v in cache}):raise ValueError("duplicate actual compile-cache path")
    if len(relevant)!=len({identity(v)[0] for v in relevant}):raise ValueError("duplicate relevant module")
    for v in cache:verify_record(v)
    actual=[]
    for v in runtimes:
        verify_record(v);j=json.loads(pathlib.Path(v["path"]).read_text())
        if j.get("schema")=="p06d-runtime-passive-v1" and not j.get("cuda_forbidden",False):actual.append(j)
    if len(actual)!=1:raise ValueError("one actual GPU runtime required")
    runtime=actual[0];root=pathlib.Path(runtime["actual_cwd"]).resolve()/"warp-cache"
    for v in cache:
        if not pathlib.Path(v["path"]).resolve().is_relative_to(root):raise ValueError("foreign compile-cache path")
    emitted=runtime["actual_machine_and_compile"]["actual_emitted_code"]
    if len(emitted)!=len({identity(v)[0] for v in emitted}):
        raise ValueError("duplicate actual runtime emitted-code record")
    for v in emitted:verify_record(v)
    wanted={identity(v) for v in emitted}
    observed={identity(v) for v in cache if pathlib.Path(v["path"]).suffix in (".cu",".ptx",".meta")}
    if wanted!=observed:raise ValueError("compile-cache differs from actual runtime emitted inventory")
    physical={str(p.resolve()) for p in root.rglob("*.ptx") if p.is_file()}
    ptx={identity(v)[0]:v for v in cache if v["path"].endswith(".ptx")}
    if set(ptx)!=physical:raise ValueError("incomplete physical PTX cache inventory")
    selected={identity(v) for v in relevant}
    if not selected.issubset({identity(v) for v in ptx.values()}):raise ValueError("foreign relevant PTX absent from actual cache")
    modules={}
    for v in ptx.values():
        filename=pathlib.Path(v["path"]).name
        names=[m for m in tuple(required)+(FAULT_MODULE,) if re.fullmatch("wp_"+re.escape(m)+r"_[0-9a-f]{7}\.sm120\.ptx",filename)]
        if len(names)!=1:raise ValueError("unknown actual source-unit PTX module: "+filename)
        m=names[0]
        if m in modules:raise ValueError("duplicate actual math module: "+m)
        modules[m]=v
    if set(modules)!=set(required)|{FAULT_MODULE}:raise ValueError("complete actually consumed production/test PTX module set required")
    must={identity(modules[m]) for m in required}
    if selected!=must:raise ValueError("missing/extra complete actual consumed math module set")
    return [dict(inspect_ptx(modules[m]),module=m) for m in required]
