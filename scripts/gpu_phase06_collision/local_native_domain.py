"""CPU-only control/write witness paired with the actual original callback.

The arithmetic is the byte-pinned generated original body with device qualifiers
removed, plus write-mask instrumentation. This helper never runs in the DUT or
readonly checker. Actual original mjContact before/after bytes remain the oracle.
"""
from __future__ import annotations
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import ctypes as ct
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT = (REPO_ROOT / '.')
MODULE = (GPU_ROOT / 'host/runtime/collision')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

from scripts.gpu_phase06_collision.private_runtime import child_env

def projection():
    from scripts.gpu_phase06_collision.derive_native import derive
    generated, _ = derive()
    source = ((GPU_ROOT / 'cuda/collision/native/native_source.cuh')).read_text()
    if source != generated:
        raise ValueError('original-source derivation drift before CPU witness')
    value = source.replace('#include <cuda_runtime.h>',
        '#define __device__\n#define __forceinline__ inline')
    value = value.replace('struct mjModel {', 'struct mjContact;\nstruct mjModel {', 1)
    value = value.replace('int nvert,ngraph,nmesh,watchdog;',
        'int nvert,ngraph,nmesh,watchdog;\n int* adapter_write; const mjContact* adapter_base;', 1)
    marker = 'struct mjContact { double dist,pos[3],frame[9]; };'
    value = value.replace(marker, marker + r'''
static void p06c_mark(const mjModel* m,const mjContact* con,int first,int count) {
 ptrdiff_t row=con-m->adapter_base;
 if(row<0 || row>=50) { *m->adapter_error=8; return; }
 for(int k=first;k<first+count;k++)m->adapter_write[10*row+k]=1;
}
''', 1)
    replacements = [
        ('// fill in contact data', '// fill in contact data\n    p06c_mark(m,con,0,10);', 1),
        ('*M->adapter_frame0=1;', '*M->adapter_frame0=1; p06c_mark(M,con,4,6);', 2),
        ('con[i].dist = ', 'p06c_mark(M,con+i,0,10);\n    con[i].dist = ', 2),
    ]
    for old, new, count in replacements:
        if value.count(old) != count:
            raise ValueError('CPU source write instrumentation match count: '+old)
        value = value.replace(old, new)
    diff = ''.join(difflib.unified_diff(source.splitlines(True),value.splitlines(True),
        fromfile='exact-device-source-body', tofile='cpu-control-write-witness'))
    return value, diff

class SourceDomain:
    def __init__(self, m):
        source = (REPO_ROOT / 'scripts/gpu_phase06_collision/local_native_domain.cpp')
        header, diff = projection()
        cc = shutil.which('gcc')
        if not cc:
            raise RuntimeError('CPU witness compiler unavailable')
        flags = ['-x','c++','-std=c++17','-shared','-fPIC','-O2',
                 '-ffp-contract=off','-fno-fast-math','-frounding-math',
                 '-fno-builtin','-fno-exceptions','-fno-rtti']
        inputs = {'header_sha256':hashlib.sha256(header.encode()).hexdigest(),
                  'source_sha256':sha(source),'compiler_sha256':sha(cc),'flags':flags}
        key = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        root = Path(os.environ['P06C_NATIVE_BUILD_ROOT']) / ('capture-domain-'+key[:20])
        artifact = root / 'domain.so'
        if not root.exists():
            root.mkdir()
            (root/'cpu_native_source.h').open('x').write(header)
            (root/'SOURCE_PROJECTION.diff').open('x').write(diff)
            shutil.copyfile(source,root/source.name)
            argv = [cc,*flags,str(root/source.name),'-o',str(artifact),'-lm']
            env = child_env(root)
            # Compiler tools use their pinned normal loader; the scientific
            # process itself and the loaded witness use the original prefix.
            env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'
            result = subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True)
            (root/'stdout').open('x').write(result.stdout)
            (root/'stderr').open('x').write(result.stderr)
            (root/'BUILD.json').open('x').write(json.dumps({
                'key':key,**inputs,'argv':argv,'cwd':str(root),
                'env':{k:env[k] for k in sorted(env) if k in {
                    'PWD','TMP','TMPDIR','TEMP','LIBERO_CONFIG_PATH',
                    'LD_LIBRARY_PATH','CUDA_CACHE_PATH','WP_CACHE_DIR',
                    'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'} or k.startswith('XDG_')},
                'exit_code':result.returncode,
                'artifact_sha256':sha(artifact) if artifact.is_file() else None,
                'projection_diff_sha256':sha(root/'SOURCE_PROJECTION.diff'),
            },sort_keys=True,indent=2)+'\n')
        record=json.loads((root/'BUILD.json').read_text())
        if record['exit_code']!=0 or record['artifact_sha256']!=sha(artifact) or \
                sha(root/'cpu_native_source.h')!=inputs['header_sha256'] or \
                sha(root/source.name)!=inputs['source_sha256']:
            raise ValueError('CPU source witness compilation/identity failure: '+str(root))
        self.lib=ct.CDLL(str(artifact))
        self.fn=self.lib.p06c_source_domain
        self.fn.argtypes=([ct.c_void_p]*11+
            [ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_double]+[ct.c_int]*4+
            [ct.c_void_p]*6)
        self.fn.restype=ct.c_int
        self.data=[np.ascontiguousarray(getattr(m,k),dtype=d).ravel() for k,d in (
            ('geom_size',np.float64),('geom_rbound',np.float64),
            ('geom_type',np.int32),('geom_dataid',np.int32),
            ('mesh_vert',np.float32),('mesh_vertadr',np.int32),
            ('mesh_vertnum',np.int32),('mesh_graphadr',np.int32),('mesh_graph',np.int32))]
        self.m=m
        self.identity={'build':record,'build_path':str(root),'private_contact_bytes':104,
            'actual_original_contact_bytes':544,'source_header_sha256':sha((GPU_ROOT / 'cuda/collision/native/native_source.cuh')),
            'library_path':str(artifact),'library_sha256':sha(artifact),
            'function_address':ct.cast(self.fn,ct.c_void_p).value,
            'options':{'mpr_iterations':int(m.opt.mpr_iterations),
                       'mpr_tolerance_hex':np.float64(m.opt.mpr_tolerance).tobytes().hex(),
                       'multiccd':int(bool(m.opt.enableflags & 16))},
            'scope':'CPU-only original body control/write witness checked against actual same-call original bytes',
            'DUT_fallback':False}
    def observe(self,d,g1,g2,margin,before,after,count):
        pos=np.array(d.geom_xpos,copy=True,order='C',dtype=np.float64).ravel()
        mat=np.array(d.geom_xmat,copy=True,order='C',dtype=np.float64).ravel()
        orig_pos=pos.tobytes();orig_mat=mat.tobytes()
        result=np.zeros((50,10),np.uint64);diag=np.zeros(16,np.int32)
        flag=np.zeros(1,np.int32);mask=np.zeros((50,10),np.int32);error=np.zeros(1,np.int32)
        arrays=[*self.data,pos,mat]
        # 9 static pointers and 2 poses, then the original argument scalars.
        n=self.fn(*[ct.c_void_p(a.ctypes.data) for a in arrays],
            g1,g2,margin,int(self.m.opt.mpr_iterations),float(self.m.opt.mpr_tolerance),
            int(bool(self.m.opt.enableflags & 16)),int(self.m.nmesh),
            int(self.m.mesh_vert.shape[0]),int(len(self.m.mesh_graph)),
            *[ct.c_void_p(a.ctypes.data) for a in (before,result,diag,flag,mask,error)])
        if error[0]!=0 or n!=count or not 0<=n<=50 or pos.tobytes()!=orig_pos or mat.tobytes()!=orig_mat:
            raise ValueError('CPU source control witness/original count/frame disagreement')
        if not np.all((mask==0)|(mask==1)):
            raise ValueError('CPU source write mask is not boolean')
        if result[mask==1].tobytes()!=after[mask==1].tobytes():
            mismatch=np.argwhere((result!=after)&(mask==1))[0].tolist()
            raise ValueError('original computed geometry differs from CPU source witness at '+str(mismatch))
        if before[mask==0].tobytes()!=after[mask==0].tobytes() or \
                before[mask==0].tobytes()!=result[mask==0].tobytes():
            raise ValueError('original/source witness held raw bytes changed')
        return {'diag':diag,'framewritten':flag,'count':np.asarray([n],np.int32),
                'error':error,'write_mask':mask,'source_after_bits':result}
