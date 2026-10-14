"""Preallocated native CUDA workspace for the SAT-exit BoxBox edge tail."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import numpy as np

H2D, D2H = 1, 2
OUTPUTS = {'count': ('i', 1), 'raw': ('d', 500), 'state': ('i', 1),
           'stage_i': ('i', 64), 'stage_d': ('d', 320),
           'trace_i': ('i', 128), 'trace_d': ('d', 256),
           'frame_written': ('i', 1), 'frame0': ('d', 6)}
INPUTS = {'pair_f': ('d', 31), 'sat_f': ('d', 45), 'sat_i': ('i', 6),
          'active': ('i', 1), 'counts': ('i', 1)}


class CudaError(RuntimeError):
    pass


class BoxEdgeContactWorkspace:
    """N worlds by P ordered pairs; launch only enqueues one CUDA kernel.

    All device pointers are public through ``buffers`` for a later device
    adapter. ``upload`` validates original shape/dtype before any reshape.
    ``download`` is a diagnostic boundary and never called by ``launch``.
    """
    def __init__(self, nworld: int, pairs: int, library: str | Path,
                 stream: int = 0):
        if nworld < 1 or pairs < 1:
            raise ValueError('N and P must be positive')
        self.N, self.P, self.stream = nworld, pairs, stream
        self.owned_stream=None
        self.graph=None
        self.graph_exec=None
        self.cudart = ct.CDLL('libcudart.so')
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.cudart.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        self.cudart.cudaMalloc.restype = ct.c_int
        self.cudart.cudaFree.argtypes = [ct.c_void_p]
        self.cudart.cudaFree.restype = ct.c_int
        self.cudart.cudaMemcpy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int]
        self.cudart.cudaMemcpy.restype = ct.c_int
        self.cudart.cudaDeviceSynchronize.restype = ct.c_int
        self.lib.launch_box_edge.argtypes = [ct.c_void_p]*5 + [ct.c_int]*2 + [ct.c_void_p]*9 + [ct.c_int, ct.c_void_p]
        self.lib.launch_box_edge.restype = ct.c_int
        self.buffers = {}
        self.spec = {**INPUTS, **OUTPUTS}
        try:
            for name, (code, width) in self.spec.items():
                length = nworld if name in ('active', 'counts') else nworld*pairs*width
                size = length*(8 if code == 'd' else 4)
                ptr = ct.c_void_p()
                self._check(self.cudart.cudaMalloc(ct.byref(ptr), size), 'cudaMalloc '+name)
                self.buffers[name] = (ptr, length, code)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(code, label):
        if code:
            raise CudaError(f'{label}: CUDA error {code}')

    def close(self):
        if getattr(self,'graph_exec',None):self.cudart.cudaGraphExecDestroy(self.graph_exec);self.graph_exec=None
        if getattr(self,'graph',None):self.cudart.cudaGraphDestroy(self.graph);self.graph=None
        if getattr(self,'owned_stream',None):self.cudart.cudaStreamDestroy(self.owned_stream);self.owned_stream=None
        for ptr, _, _ in self.buffers.values():
            self.cudart.cudaFree(ptr)
        self.buffers.clear()

    def __del__(self):
        if hasattr(self, 'buffers'):
            self.close()

    def upload(self, pair_f, sat_f, sat_i, active, counts):
        values = locals()
        expected = {'pair_f': (self.N,self.P,31), 'sat_f': (self.N,self.P,45),
                    'sat_i': (self.N,self.P,6), 'active': (self.N,), 'counts': (self.N,)}
        arrays = {}
        for name, shape in expected.items():
            value = values[name]
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype('float64' if INPUTS[name][0]=='d' else 'int32'):
                raise ValueError(f'{name}: expected {shape} {INPUTS[name][0]} with original dtype')
            arrays[name] = value
        if np.any((active != 0) & (active != 1)):
            raise ValueError('active must be 0 or 1')
        for w in range(self.N):
            if active[w] == 0 or not 0 <= counts[w] <= self.P:
                continue
            used = slice(0, int(counts[w]))
            si = sat_i[w,used]
            if np.any((si[:,4]<0)|(si[:,4]>2)|(si[:,5]!=0)):
                raise ValueError('invalid SAT route/status')
            edge=si[:,4]==2
            if np.any((si[edge,0]<12)|(si[edge,0]>20)|
                      (si[edge,1]<0)|(si[edge,1]>7)|
                      (si[edge,2]<0)|(si[edge,2]>7)|
                      (si[edge,3]<0)|(si[edge,3]>1)):
                raise ValueError('invalid edge SAT tuple')
            if not np.all(np.isfinite(pair_f[w,used])) or not np.all(np.isfinite(sat_f[w,used][edge])):
                raise ValueError('nonfinite active input')
        for name in INPUTS:
            array=np.ascontiguousarray(arrays[name])
            ptr,length,code=self.buffers[name]
            self._check(self.cudart.cudaMemcpy(ptr,ct.c_void_p(array.ctypes.data),array.nbytes,H2D),'upload '+name)

    def launch(self, *, wrong_d0: bool = False):
        b=self.buffers
        names=['pair_f','sat_f','sat_i','active','counts']
        outputs=list(OUTPUTS)
        args=[b[name][0] for name in names]+[self.N,self.P]+[b[name][0] for name in outputs]+[int(wrong_d0),ct.c_void_p(self.stream)]
        self._check(self.lib.launch_box_edge(*args),'launch_box_edge')

    def synchronize(self):
        self._check(self.cudart.cudaDeviceSynchronize(),'cudaDeviceSynchronize')

    def capture(self):
        """Capture one production launch on an owned nonblocking CUDA stream."""
        if self.graph_exec is not None:raise RuntimeError('already captured')
        self.cudart.cudaStreamCreateWithFlags.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint]
        self.cudart.cudaStreamBeginCapture.argtypes=[ct.c_void_p,ct.c_int]
        self.cudart.cudaStreamEndCapture.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
        self.cudart.cudaGraphInstantiate.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_size_t]
        self.cudart.cudaGraphLaunch.argtypes=[ct.c_void_p,ct.c_void_p]
        self.cudart.cudaGraphExecDestroy.argtypes=[ct.c_void_p]
        self.cudart.cudaGraphDestroy.argtypes=[ct.c_void_p]
        self.cudart.cudaStreamDestroy.argtypes=[ct.c_void_p]
        stream=ct.c_void_p()
        self._check(self.cudart.cudaStreamCreateWithFlags(ct.byref(stream),1),'cudaStreamCreateWithFlags')
        self.owned_stream=stream;self.stream=stream.value
        self._check(self.cudart.cudaStreamBeginCapture(stream,0),'cudaStreamBeginCapture')
        self.launch()
        graph=ct.c_void_p()
        self._check(self.cudart.cudaStreamEndCapture(stream,ct.byref(graph)),'cudaStreamEndCapture')
        self.graph=graph
        graph_exec=ct.c_void_p()
        self._check(self.cudart.cudaGraphInstantiate(ct.byref(graph_exec),graph,None,None,0),'cudaGraphInstantiate')
        self.graph_exec=graph_exec

    def graph_launch(self):
        if self.graph_exec is None:raise RuntimeError('capture first')
        self._check(self.cudart.cudaGraphLaunch(self.graph_exec,ct.c_void_p(self.stream)),'cudaGraphLaunch')

    def download(self):
        self.synchronize()
        result={}
        for name,(code,width) in OUTPUTS.items():
            ptr,length,_=self.buffers[name]
            arr=np.empty(length,np.float64 if code=='d' else np.int32)
            self._check(self.cudart.cudaMemcpy(ct.c_void_p(arr.ctypes.data),ptr,arr.nbytes,D2H),'download '+name)
            shape=(self.N,self.P) if width==1 else (self.N,self.P,width)
            result[name]=arr.reshape(shape)
        result['raw']=result['raw'].reshape(self.N,self.P,50,10)
        return result
