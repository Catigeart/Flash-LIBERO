"""Verification-only frozen S07 CUDA fault; production search PTX is unchanged.

The frozen kernel.cu owns a separate kernel after the production device prefix.
Compile its exact 86-byte body into a separate private NVRTC module, retaining
PTX/module/function/context/argument lifetimes and every complete compile log.
This is the chosen receiver NVRTC route, not a claim about an nvcc host build.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import uuid

FAULT_UNIT_SHA256 = "96541997ca43f0a9e6708bceeb3fa1587da5182be0bdcbdf5eeb0d419a2a53e8"
FAULT_UNIT_BYTES = 86
FAULT_SYMBOL = b"_Z18s07_corrupt_kernelPd"

def _write(path, data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

class S07FaultModule:
    """Private, retained verification kernel compiled from authentic source."""
    def __init__(self, bridge):
        from .s07_bridge import device_prefix, sha256_bytes
        self.bridge = bridge
        raw, _, _ = device_prefix(bridge.kernel_source)
        start = raw.index(b"__global__ void s07_corrupt_kernel")
        end = raw.index(b'extern "C" int s07_corrupt_eval_cost', start)
        self.source = raw[start:end]
        if len(self.source) != FAULT_UNIT_BYTES or sha256_bytes(self.source) != FAULT_UNIT_SHA256:
            raise RuntimeError("frozen verification-only CUDA fault body drift")
        self.root = bridge.build_dir / ("verification-fault-" + uuid.uuid4().hex)
        self.root.mkdir(exist_ok=False)
        _write(self.root / "original_fault_kernel.cu", self.source)
        self.record = {
            "scope": "verification-only exact original fault body, separate module",
            "source_owner": str(bridge.kernel_source),
            "source_owner_sha256": sha256_bytes(raw),
            "source_byte_range": [start, end],
            "fault_unit_sha256": FAULT_UNIT_SHA256,
            "fault_unit_bytes": FAULT_UNIT_BYTES,
            "production_search_ptx_sha256": bridge.ptx_sha256,
            "production_search_ptx_bytes": bridge.ptx_bytes,
            "nvrtc_path": bridge.nvrtc_path,
            "arch": bridge.arch,
            "accepted": False,
        }
        self._module = self._function = self._ptx_buffer = None
        self._context = self._argument = self._kernel_args = None
        try:
            self._compile()
            self._load()
        except BaseException as e:
            self.record["failure"] = {"type": type(e).__name__, "message": str(e)}
            _write(self.root / "compile_load_failure.json",
                   (json.dumps(self.record, indent=2)+"\n").encode())
            raise
        _write(self.root / "compile_load_record.json",
               (json.dumps(self.record, indent=2)+"\n").encode())

    def _compile(self):
        from .s07_bridge import FIXED_OPTIONS
        nvrtc = ct.CDLL(self.bridge.nvrtc_path)
        for name, args in (
            ("nvrtcCreateProgram", [ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,
                                    ct.c_int,ct.POINTER(ct.c_char_p),ct.POINTER(ct.c_char_p)]),
            ("nvrtcCompileProgram", [ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)]),
            ("nvrtcGetProgramLogSize", [ct.c_void_p,ct.POINTER(ct.c_size_t)]),
            ("nvrtcGetProgramLog", [ct.c_void_p,ct.c_char_p]),
            ("nvrtcGetPTXSize", [ct.c_void_p,ct.POINTER(ct.c_size_t)]),
            ("nvrtcGetPTX", [ct.c_void_p,ct.c_char_p]),
            ("nvrtcDestroyProgram", [ct.POINTER(ct.c_void_p)]),
        ):
            fun = getattr(nvrtc,name);fun.argtypes=args;fun.restype=ct.c_int
        prog = ct.c_void_p()
        create_rc = nvrtc.nvrtcCreateProgram(ct.byref(prog),self.source,
                                            b"original_fault_kernel.cu",0,None,None)
        self.record["nvrtc_create_return"] = create_rc
        if create_rc: raise RuntimeError(("nvrtcCreateProgram fault",create_rc))
        compile_failed = False
        try:
            options = [("--gpu-architecture="+self.bridge.arch).encode()] + list(FIXED_OPTIONS)
            self.record["options"] = [x.decode() for x in options]
            opts = (ct.c_char_p*len(options))(*options)
            compile_rc = nvrtc.nvrtcCompileProgram(prog,len(options),opts)
            self.record["nvrtc_compile_return"] = compile_rc
            size = ct.c_size_t()
            log_size_rc = nvrtc.nvrtcGetProgramLogSize(prog,ct.byref(size))
            self.record["nvrtc_log_size_return"] = log_size_rc
            if log_size_rc: raise RuntimeError(("nvrtcGetProgramLogSize fault",log_size_rc))
            log = ct.create_string_buffer(max(size.value,1))
            log_rc = nvrtc.nvrtcGetProgramLog(prog,log)
            self.record["nvrtc_log_return"] = log_rc
            if log_rc: raise RuntimeError(("nvrtcGetProgramLog fault",log_rc))
            complete_log = log.raw[:size.value]
            _write(self.root/"complete_nvrtc_log.raw",complete_log)
            self.record["complete_log"] = {"bytes":len(complete_log),
                        "sha256":hashlib.sha256(complete_log).hexdigest()}
            if compile_rc: raise RuntimeError(("NVRTC original fault compilation",compile_rc,
                                                "complete log retained",str(self.root)))
            size = ct.c_size_t()
            rc = nvrtc.nvrtcGetPTXSize(prog,ct.byref(size))
            if rc or size.value<2: raise RuntimeError(("nvrtcGetPTXSize fault",rc,size.value))
            buf = ct.create_string_buffer(size.value)
            rc = nvrtc.nvrtcGetPTX(prog,buf)
            if rc: raise RuntimeError(("nvrtcGetPTX fault",rc))
            if buf.raw[size.value-1:] != b"\x00": raise RuntimeError("fault PTX missing NUL")
            self.ptx = buf.raw[:-1]
            _write(self.root/"original_fault_kernel.ptx",self.ptx)
            self.record["fault_ptx"] = {"bytes":len(self.ptx),
                            "sha256":hashlib.sha256(self.ptx).hexdigest()}
        except BaseException as failure:
            compile_failed = True
            self.record["original_compile_failure"] = {"type":type(failure).__name__,"message":str(failure)}
            raise
        finally:
            destroy_rc = nvrtc.nvrtcDestroyProgram(ct.byref(prog))
            self.record["nvrtc_destroy_return"] = destroy_rc
            if destroy_rc and not compile_failed: raise RuntimeError(("nvrtcDestroyProgram fault",destroy_rc))

    def _load(self):
        cuda = self.bridge._driver()
        self._context = ct.c_void_p()
        rc = cuda.cuCtxGetCurrent(ct.byref(self._context))
        if rc or not self._context.value: raise RuntimeError(("fault context",rc))
        self._module = ct.c_void_p()
        self._ptx_buffer = ct.create_string_buffer(self.ptx)
        rc = cuda.cuModuleLoadData(ct.byref(self._module),ct.cast(self._ptx_buffer,ct.c_void_p))
        if rc: raise RuntimeError(("fault cuModuleLoadData",rc))
        self._function = ct.c_void_p()
        rc = cuda.cuModuleGetFunction(ct.byref(self._function),self._module,FAULT_SYMBOL)
        if rc: raise RuntimeError(("fault cuModuleGetFunction",rc))
        self.record["native_context"] = self._context.value
        self.record["native_module"] = self._module.value
        self.record["native_function"] = self._function.value
        self.record["symbol"] = FAULT_SYMBOL.decode()

    def launch(self, eval_trace_ptr, stream_ptr):
        if type(eval_trace_ptr) is not int or type(stream_ptr) is not int:
            raise TypeError("verification requires actual integer CUDA buffer and stream pointers")
        if not 0<eval_trace_ptr<2**64 or not 0<=stream_ptr<2**64: return -1
        context = ct.c_void_p()
        cuda = self.bridge._driver()
        rc = cuda.cuCtxGetCurrent(ct.byref(context))
        if rc or context.value != self._context.value:
            raise RuntimeError(("fault original CUDA context changed",rc))
        self._argument = ct.c_void_p(eval_trace_ptr)
        self._kernel_args = (ct.c_void_p*1)(ct.addressof(self._argument))
        return int(cuda.cuLaunchKernel(self._function,1,1,1,1,1,1,0,
                   ct.c_void_p(stream_ptr),self._kernel_args,None))
