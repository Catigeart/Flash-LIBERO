"""Passive source/library/FEnv identities, typed artifact records, no CUDA import."""
from __future__ import annotations
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import sys
import numpy as np
from compiler_audit import actual_code

RECEIVE = pathlib.Path("/srv/libero-split-receive-20261004-r01")
ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIGINAL_LIB = RECEIVE / "audit/codex-goal-20261005-g01/runtime/original-native-deps-r01/lib"
PINS = {
    "ld-linux-x86-64.so.2": "9eb34cb2da3ae2a9398cc09b3cd2d069563ec40d9858cb711af15cd23fa80abf",
    "libm.so.6": "00830df310aac5023e10bd7b715149ab8ddc682277fd8ae3539a7a44fcdd6c38",
    "libc.so.6": "c53819710b163d3f1d2541778590d58d3ef31cb0ed75adcbe059faac68c1e72d",
    "libmujoco.so.2.3.7": "a82e1280ec6a72a309135202aff5f7a2ee71fc1eb69c719ac9bbf30ce7b9ef0e",
}
SOURCE_PINS = {
    "engine_core_smooth.c": "2b34a3efe31691ac93e7a9cc632d5340258290a533954be98d46248257ea9c0b",
    "engine_forward.c": "d8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5",
}
MJB_SHA = "3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42"

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def record(path):
    p = pathlib.Path(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": digest(p)}

def write_json(path, obj):
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    with pathlib.Path(path).open("x") as f:
        f.write(data)

def typed_index(arrays):
    return {k: {"dtype": v.dtype.str, "shape": list(v.shape),
                "bytes": v.nbytes, "sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest()}
            for k,v in sorted(arrays.items())}

def save_npz(path, arrays):
    p = pathlib.Path(path)
    with p.open("xb") as f:
        np.savez(f, **arrays)
    return {"artifact": record(p), "fields": typed_index(arrays)}

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}

def runtime_identity(path, require_mujoco=False, forbid_cuda=False):
    cwd=pathlib.Path.cwd().resolve()
    if pathlib.Path(os.environ.get("PWD","")).resolve()!=cwd:
        raise RuntimeError("actual cwd and PWD differ")
    private_names=("LIBERO_CONFIG_PATH","TMPDIR","TMP","TEMP","XDG_CONFIG_HOME",
                   "XDG_CACHE_HOME","XDG_DATA_HOME","XDG_STATE_HOME","XDG_RUNTIME_DIR",
                   "CUDA_CACHE_PATH","WARP_CACHE_PATH","WARP_KERNEL_CACHE_DIR","NUMBA_CACHE_DIR","HF_HOME")
    private={}
    for name in private_names:
        if name not in os.environ: raise RuntimeError("private runtime variable missing: "+name)
        resolved=pathlib.Path(os.environ[name]).resolve()
        if not resolved.is_relative_to(cwd) or resolved==cwd:
            raise RuntimeError("runtime path outside exclusive cwd: "+name)
        private[name]=str(resolved)
    if (cwd/"xdg-runtime").stat().st_mode & 0o077:
        raise RuntimeError("XDG runtime directory must have private permissions")
    maps = pathlib.Path("/proc/self/maps").read_text()
    loaded = []
    for name, sha in PINS.items():
        paths = sorted(set(line.split()[-1] for line in maps.splitlines()
                           if "/" in line and pathlib.Path(line.split()[-1]).name == name))
        if name == "libmujoco.so.2.3.7" and not require_mujoco:
            continue
        if len(paths) != 1 or digest(paths[0]) != sha:
            raise RuntimeError("actual original library identity mismatch: " + name + ": " + repr(paths))
        loaded.append(record(paths[0]))
    if forbid_cuda:
        if any(k == "warp" or k.startswith(("warp.", "mujoco")) for k in sys.modules):
            raise RuntimeError("readonly replay imported CUDA/oracle module")
        if any(s in maps for s in ("libcuda.so", "libcudart.so", "libmujoco")):
            raise RuntimeError("readonly replay loaded CUDA/oracle library")
    helper = ctypes.CDLL(os.environ["P06D_FENV_HELPER"])
    helper.p06d_fegetround.restype = ctypes.c_int
    helper.p06d_mxcsr.restype = ctypes.c_uint
    helper.p06d_x87cw.restype = ctypes.c_uint
    fenv = {"fegetround": helper.p06d_fegetround(),
            "mxcsr": helper.p06d_mxcsr(), "x87_control_word": helper.p06d_x87cw()}
    if fenv["fegetround"] != 0 or fenv["mxcsr"] & ((1 << 15) | (1 << 6) | (3 << 13)) or fenv["x87_control_word"] & (3 << 10):
        raise RuntimeError("FEnv requires RN, gradual underflow: " + repr(fenv))
    prefixes = ("OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_", "NUMBA_", "XDG_", "CUDA_", "WARP_", "PYTHON", "LIBERO_", "TMP", "P06D_", "LD_LIBRARY", "MUJOCO")
    env = {k:v for k,v in os.environ.items() if k.startswith(prefixes)}
    repo_modules = {}
    for name, module in list(sys.modules.items()):
        f = getattr(module,"__file__",None)
        if f and pathlib.Path(f).is_file() and str(pathlib.Path(f).resolve()).startswith(str(ROOT)+"/"):
            repo_modules[name] = record(f)
    machine = actual_code(maps, os.getcwd())
    out = {"schema": "p06d-runtime-passive-v1", "utc": utc(), "argv": sys.argv,
           "actual_cwd": os.getcwd(), "actual_PWD":os.environ["PWD"], "private_runtime_paths":private,
           "actual_nice":os.getpriority(os.PRIO_PROCESS,0),
           "actual_ionice":__import__("subprocess").check_output(["ionice","-p",str(os.getpid())],text=True).strip(), "python": sys.version, "executable": sys.executable,
           "actual_backend": "unknown", "actual_native_libraries": loaded, "fenv": fenv,
           "actual_machine_and_compile": machine,
           "fenv_helper": record(os.environ["P06D_FENV_HELPER"]),
           "environment": env, "repo_modules": repo_modules,
           "cuda_forbidden": forbid_cuda, "source": SOURCE_PINS}
    maps_path = pathlib.Path(path).with_suffix(".maps.txt")
    with maps_path.open("x") as f:
        f.write(maps)
    out["maps"] = record(maps_path)
    write_json(path, out)
    return out
