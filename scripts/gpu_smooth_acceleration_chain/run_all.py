"""D09 staged CPU-boundary CUDA replay, independent checks, and trace entry."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import budget

REPO = (REPO_ROOT / '.')
DOC = (REPO_ROOT / 'doc/phase_05/smooth_acceleration_chain')
REQ = (REPO_ROOT / 'doc/phase_05/smooth_acceleration_chain/INPUT_REQUIREMENTS.json')
DEFAULT_REFERENCE = (REPO_ROOT / 'doc/results/phase_05_smooth_acceleration_chain/cpu_20260930_02')
SOURCE_STATE = Path('/rjs2/lhw/LIBERO/doc/results/phase_03_sources/states_20260930_01')
LOCK = Path('/rjs2/lhw/tmp/libero-phase2-compute.lock')
GPU_PYTHON = Path('/rjs2/lhw/tmp/libero-phase1-gpu-venv/bin/python')
CPU_PYTHON = Path('/rjs2/lhw/tmp/libero-perf-venv/bin/python')
_WORKSPACE_CACHE = None
FIELDS = ('passive','bias','applied','actuator','qm','xfrc','smooth','acc','ld','inv','sqrt_inv')
OUT = {'smooth':np.float64,'acc':np.float64,'ld':np.float64,'inv':np.float64,
       'sqrt_inv':np.float64,'solution':np.float64,'pivot':np.int32,
       'status':np.int32,'warning_count':np.int32,'warning_last':np.int32}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_class():
    global _WORKSPACE_CACHE
    if _WORKSPACE_CACHE is not None:return _WORKSPACE_CACHE
    path=(GPU_ROOT / 'reference/dynamics/smooth_acceleration_chain/kernel.py')
    spec=importlib.util.spec_from_file_location('d09_smooth_acceleration_kernel',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    _WORKSPACE_CACHE=(mod.SmoothAccelerationWorkspace,mod.validate_host_payload)
    return _WORKSPACE_CACHE


def exact_zero(a):
    return np.array_equal(np.asarray(a).view(np.uint8),np.zeros_like(a).view(np.uint8))


def load_reference(ref=DEFAULT_REFERENCE):
    req=json.loads(REQ.read_text())
    if sha((REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/derive_cpu.py'))!=req['cpu_reference']['derive_script_sha256']:
        raise AssertionError('CPU bridge script SHA')
    for path,digest in req['source_sha256'].items():
        if sha(path)!=digest:
            raise AssertionError(('source SHA drift',path))
    if sha(ref/'manifest.json') != req['cpu_reference']['manifest_sha256']:
        raise AssertionError('reference manifest SHA')
    manifest=json.loads((ref/'manifest.json').read_text())
    if manifest['real_count']!=1498 or manifest['local_count']!=7:
        raise AssertionError('reference census')
    for file,digest in manifest['files'].items():
        if sha(ref/file)!=digest:
            raise AssertionError(('reference file SHA',file))
    ids=json.loads((ref/'identities.json').read_text())
    if sha(ref/'identities.json')!=req['identity_file_sha256']:
        raise AssertionError('identity file SHA')
    expected=[]
    for entry in ('reset','set_init_state'):
        original=json.loads((SOURCE_STATE/(entry+'.json')).read_text())
        for j,e in enumerate(original):
            one=ids[len(expected)]
            if (any(one.get(k)!=v for k,v in e.items()) or one['record_id']!=f'{entry}:{j}'
                    or one['source_file_sha256']!=req['source_sha256'][str(SOURCE_STATE/(entry+'.npz'))]
                    or one['original_boundary']!='outer_call_exit'
                    or one['derived_boundary']!='mj_forward_scratch_same_state'
                    or one['invocation']!='offline_scratch_fwdAcceleration'):
                raise AssertionError(('source identity mismatch',entry,j))
            expected.append(one['record_id'])
    if len(ids)!=1498 or len(set(expected))!=1498:
        raise AssertionError('required identity count/duplicates')
    rows={}
    for group,n in (('real',1498),('local',7)):
        with np.load(ref/(group+'.npz'),allow_pickle=False) as z:
            d={k:np.array(z[k],copy=True) for k in z.files}
        if set(d)!=set(FIELDS):
            raise TypeError(('reference fields',group))
        shapes={k:(n,43) for k in ('passive','bias','applied','actuator','smooth','acc','inv','sqrt_inv')}
        shapes.update(qm=(n,153),xfrc=(n,38,6),ld=(n,153))
        for k,shape in shapes.items():
            if d[k].shape!=shape or d[k].dtype!=np.float64:
                raise TypeError(('reference schema',group,k,d[k].shape,str(d[k].dtype)))
        if np.any(d['xfrc']!=0) or not np.isfinite(d['qm']).all():
            raise ValueError(('reference domain',group))
        rows[group]=d
    with np.load(ref/'topology.npz',allow_pickle=False) as z:
        topology={k:np.array(z[k],copy=True) for k in z.files}
    for k in ('dof_Madr','dof_parentid','dof_simplenum'):
        if topology[k].shape!=(43,) or topology[k].dtype!=np.int32:
            raise TypeError(('topology schema',k))
    return rows,topology,ids,manifest


def interval_for(rows,topology):
    force=budget.force_bounds(rows['passive'],rows['bias'],rows['applied'],rows['actuator'])
    n=len(rows['qm'])
    rhs=np.zeros((n,4,43),np.float64)
    # The solve certificate starts at the independently rounded entry force.
    # CPU smooth is checked below as an oracle, never used as the bound input.
    rhs[:,0,:]=force[0]
    rhs_interval=[]
    for side in force:
        a=np.zeros_like(rhs);a[:,0,:]=side;rhs_interval.append(a)
    pivot=np.zeros((n,43),np.int32)
    if n==7:
        # Native local warning chronology is frozen in local_specs.json.
        pivot[3,0]=1
    source={'qm':rows['qm'],'rhs':rhs,'count':np.ones(n,np.int32),'pivot':pivot}
    interval=budget.make(source,topology,tuple(rhs_interval))
    if not np.array_equal(force[0].view(np.uint64),rows['smooth'].view(np.uint64)):
        first=np.argwhere(force[0].view(np.uint64)!=rows['smooth'].view(np.uint64))[0].tolist()
        raise AssertionError(('original CPU source RN force bit difference',first))
    for k,src in (('ld','ld'),('inv','inv'),('sqrt_inv','sqrt_inv'),('solution','acc')):
        pair=interval[k]
        if k=='solution':pair=(pair[0][:,0],pair[1][:,0])
        if not budget.contains(pair,rows[src]):
            bad=np.argwhere((rows[src]<pair[0])|(rows[src]>pair[1]))
            raise AssertionError(('original CPU outside interval',k,bad[:1].tolist()))
    return {'force':force,'factor':interval,'pivot':pivot}


def model(topology):
    return SimpleNamespace(nq=48,nv=43,nM=153,nbody=38,nmocap=0,**topology)


def payload(rows,ids,cap):
    if len(ids)==0 or len(ids)>cap or len(set(map(int,ids)))!=len(ids) or np.any(ids<0) or np.any(ids>=len(rows['qm'])):
        raise ValueError('identity selection')
    p={k:np.zeros((cap,)+(rows[k].shape[1:]),np.float64) for k in ('passive','bias','applied','actuator','qm','xfrc')}
    for k in p:p[k][:len(ids)]=rows[k][ids]
    p['active']=np.zeros(cap,np.int32);p['active'][:len(ids)]=1
    return p


def upload(ws,p):
    import warp as wp
    _,validate_host_payload=workspace_class()
    validate_host_payload(p,ws.capacity,ws.nv,ws.nM,ws.nbody)
    for k in ('passive','bias','applied','actuator','qm','xfrc','active'):
        wp.copy(getattr(ws,k),wp.array(p[k],device=ws.device))
    wp.copy(ws.count,wp.array(np.ones(ws.capacity,np.int32),device=ws.device))


def candidate(ws):
    return {k:(ws.mass.solution.numpy() if k=='solution' else getattr(ws,k).numpy()) for k in OUT}


def check_preserved(ws,p,topology):
    for k in ('passive','bias','applied','actuator','qm','xfrc','active'):
        actual=getattr(ws,k).numpy()
        if not np.array_equal(actual.view(np.uint8),p[k].view(np.uint8)):
            raise AssertionError(('must_preserve input',k))
    if not np.array_equal(ws.count.numpy(),np.ones(ws.capacity,np.int32)):
        raise AssertionError('must_preserve count')
    for k in ('dof_Madr','dof_parentid','dof_simplenum'):
        if not np.array_equal(getattr(ws.mass,k).numpy(),topology[k]):
            raise AssertionError(('must_preserve topology',k))


def out_shape(k,cap):
    if k=='ld':return (cap,153)
    if k=='solution':return (cap,4,43)
    if k in ('smooth','acc','inv','sqrt_inv','pivot'):return (cap,43)
    return (cap,)


def first_interval_error(pair,actual):
    bad=np.argwhere(~np.isfinite(actual)|(actual<pair[0])|(actual>pair[1]))
    return bad[0].tolist() if len(bad) else None


def check(got,rows,cert,ids,active,kind):
    if not isinstance(active,np.ndarray) or active.ndim!=1 or active.dtype!=np.int32:
        raise TypeError('active schema')
    cap=len(active)
    if set(got)!=set(OUT):
        raise TypeError(('candidate fields',set(got)^set(OUT)))
    for k,dtype in OUT.items():
        if got[k].shape!=out_shape(k,cap) or got[k].dtype!=dtype:
            raise TypeError(('candidate schema',k,got[k].shape,str(got[k].dtype)))
    if (kind not in ('real','local') or not isinstance(ids,np.ndarray) or
            ids.dtype!=np.int32 or ids.ndim!=1 or len(ids)==0 or len(ids)>cap
            or len(set(map(int,ids)))!=len(ids) or np.any(ids<0)
            or np.any(ids>=len(rows['qm']))):
        raise ValueError('identity selection')
    if not np.array_equal(active[:len(ids)],np.ones(len(ids),np.int32)) or np.any(active[len(ids):]):
        raise AssertionError('active routing')
    expected=np.where(cert['pivot'][ids].sum(axis=1)>0,5,0).astype(np.int32)
    if not np.array_equal(got['status'][:len(ids)],expected) or np.any(got['status'][len(ids):]!=0):
        raise AssertionError(('status first difference',int(np.flatnonzero(got['status'][:len(ids)]!=expected)[0]) if np.any(got['status'][:len(ids)]!=expected) else len(ids)))
    if not np.array_equal(got['pivot'][:len(ids)],cert['pivot'][ids]):
        raise AssertionError(('pivot first difference',np.argwhere(got['pivot'][:len(ids)]!=cert['pivot'][ids])[0].tolist()))
    warn_count=cert['pivot'][ids].sum(axis=1).astype(np.int32)
    warn_last=np.where(warn_count>0,0,-1).astype(np.int32)
    if not np.array_equal(got['warning_count'][:len(ids)],warn_count) or not np.array_equal(got['warning_last'][:len(ids)],warn_last):
        raise AssertionError(('warning event routing','discrete',kind))
    for k in ('smooth','acc','ld','inv','sqrt_inv','solution'):
        if not exact_zero(got[k][len(ids):]):
            raise AssertionError(('inactive structural +0',k))
    if not exact_zero(got['pivot'][len(ids):]) or not exact_zero(got['warning_count'][len(ids):]) or np.any(got['warning_last'][len(ids):]!=-1):
        raise AssertionError('inactive discrete output')
    # Dependency order after schema, identity and discrete routes:
    # force -> factor -> solve -> output wiring.
    force_bits=got['smooth'][:len(ids)].view(np.uint64)
    reference_bits=rows['smooth'][ids].view(np.uint64)
    if not np.array_equal(force_bits,reference_bits):
        first=np.argwhere(force_bits!=reference_bits)[0].tolist()
        raise AssertionError(('source-order force bit difference','force',kind,int(ids[first[0]]),first))
    checks=(('force','smooth',cert['force']),
            ('factor','ld',cert['factor']['ld']),
            ('factor','inv',cert['factor']['inv']),
            ('factor','sqrt_inv',cert['factor']['sqrt_inv']),
            ('solve','solution',cert['factor']['solution']),
            ('output','acc',tuple(x[:,0] for x in cert['factor']['solution'])))
    for stage,k,pair in checks:
        if k=='solution':
            pair=(pair[0][ids,0],pair[1][ids,0]);actual=got[k][:len(ids),0]
        else:
            pair=(pair[0][ids],pair[1][ids]);actual=got[k][:len(ids)]
        first=first_interval_error(pair,actual)
        if first is not None:
            raise AssertionError(('GPU outside source interval',stage,kind,k,int(ids[first[0]]),first))
        if k=='ld':
            madr=cert['factor']['madr'];parent=cert['factor']['parent']
            leaf=np.ones(43,bool);leaf[parent[parent>=0]]=False
            for slot,rid in enumerate(ids):
                for coord in np.flatnonzero(leaf&(cert['pivot'][rid]==0)):
                    address=int(madr[coord])
                    if got['ld'][slot,address].view(np.uint64)!=rows['qm'][rid,address].view(np.uint64):
                        raise AssertionError(('exact leaf qM copy','factor',kind,int(rid),'ld',[slot,address]))
        if k=='solution' and not exact_zero(got[k][:len(ids),1:]):
            bad=np.argwhere(got[k][:len(ids),1:].view(np.uint64)!=0)[0].tolist()
            raise AssertionError(('unused RHS structural +0','solve',kind,int(ids[bad[0]]),'solution',bad))
    if not np.array_equal(got['acc'][:len(ids)].view(np.uint64),got['solution'][:len(ids),0].view(np.uint64)):
        bad=np.argwhere(got['acc'][:len(ids)].view(np.uint64)!=got['solution'][:len(ids),0].view(np.uint64))[0].tolist()
        raise AssertionError(('qacc single-RHS wiring','output',kind,int(ids[bad[0]]),'acc',bad))


def dense(qm,madr,parent):
    M=np.zeros((43,43),np.float64)
    for k in range(43):
        a=int(madr[k]);j=k
        while j>=0:
            M[k,j]=qm[a];M[j,k]=qm[a]
            a+=1;j=int(parent[j])
    return M


def residual_check(rows,cert,got,ids,topology):
    u=2.0**-53;gamma=43*u/(1-43*u)
    largest=0.0;largest_bound=0.0;repaired=0
    for slot,rid in enumerate(ids):
        qm=rows['qm'][rid].copy()
        if cert['pivot'][rid].any():
            repaired+=1
            offdiag=np.ones(153,bool);offdiag[topology['dof_Madr']]=False
            if np.any(qm[offdiag]!=0) or np.flatnonzero(cert['pivot'][rid]).tolist()!=[0]:
                raise AssertionError(('unverified repaired matrix',int(rid)))
            qm[int(topology['dof_Madr'][0])]=1e-15
        M=dense(qm,topology['dof_Madr'],topology['dof_parentid'])
        x=got['acc'][slot];b=got['smooth'][slot]
        residual=M@x-b
        normM=np.max(np.sum(np.abs(M),axis=1));normx=np.max(np.abs(x));normb=np.max(np.abs(b))
        denom=normM*normx+normb
        xlo=cert['factor']['solution'][0][rid,0];xhi=cert['factor']['solution'][1][rid,0]
        blo=cert['force'][0][rid];bhi=cert['force'][1][rid]
        xmid=xlo/2+xhi/2;bmid=blo/2+bhi/2
        radiusx=np.max((xhi-xlo)/2);radiusb=np.max((bhi-blo)/2)
        maxnormx=np.max(np.maximum(np.abs(xlo),np.abs(xhi)))
        # The envelope is fixed by source inputs and the propagated intervals.
        # It must not expand when an output is corrupted.
        bound=(np.max(np.abs(M@xmid-bmid))+normM*radiusx+radiusb+
               gamma*(normM*(np.max(np.abs(xmid))+maxnormx)+np.max(np.abs(bmid))+radiusb))
        actual=np.max(np.abs(residual))
        if not np.isfinite(bound) or not np.isfinite(actual) or actual>bound:
            raise AssertionError(('residual exceeds derived bound',int(rid),actual,bound))
        largest=max(largest,actual/denom if denom else 0.0)
        largest_bound=max(largest_bound,bound/denom if denom else 0.0)
    return {'records':len(ids),'repaired_modified_M':repaired,
            'max_normalized_residual':largest,'max_derived_bound':largest_bound}


def gpu_stage(ref,out):
    import warp as wp
    SmoothAccelerationWorkspace,_=workspace_class()
    rows,topology,identities,manifest=load_reference(ref)
    cert={k:interval_for(v,topology) for k,v in rows.items()}
    wp.init();out.mkdir(parents=True)
    before=source_hashes()
    model_arg=model(topology)
    produced={k:np.empty(out_shape(k,1498),dtype=dtype) for k,dtype in OUT.items()}
    batch_count=0
    real_residual={'records':0,'repaired_modified_M':0,
                   'max_normalized_residual':0.0,'max_derived_bound':0.0}
    for start in range(0,1498,128):
        ids=np.arange(start,min(start+128,1498),dtype=np.int32)
        ws=SmoothAccelerationWorkspace(model_arg,128)
        p=payload(rows['real'],ids,128);upload(ws,p)
        ws.launch();wp.synchronize();got=candidate(ws)
        check(got,rows['real'],cert['real'],ids,p['active'],'real')
        check_preserved(ws,p,topology)
        one_residual=residual_check(rows['real'],cert['real'],got,ids,topology)
        real_residual['records']+=one_residual['records']
        real_residual['repaired_modified_M']+=one_residual['repaired_modified_M']
        for k in ('max_normalized_residual','max_derived_bound'):
            real_residual[k]=max(real_residual[k],one_residual[k])
        for k in OUT:produced[k][ids]=got[k][:len(ids)]
        batch_count+=1
    np.savez_compressed(out/'gpu_real.npz',**produced)
    ids=np.arange(7,dtype=np.int32)
    ws=SmoothAccelerationWorkspace(model_arg,7)
    p=payload(rows['local'],ids,7);upload(ws,p);ws.launch();wp.synchronize();local_got=candidate(ws)
    check(local_got,rows['local'],cert['local'],ids,p['active'],'local')
    check_preserved(ws,p,topology)
    local_residual=residual_check(rows['local'],cert['local'],local_got,ids,topology)
    np.savez_compressed(out/'gpu_local.npz',**local_got)
    graph_checks={}
    for cap in (1,4,20,128):
        ws=SmoothAccelerationWorkspace(model_arg,cap)
        ids=np.arange(cap,dtype=np.int32)
        p=payload(rows['real'],ids,cap);upload(ws,p)
        ws.launch();wp.synchronize();check(candidate(ws),rows['real'],cert['real'],ids,p['active'],'real')
        check_preserved(ws,p,topology)
        wp.capture_begin(device=ws.device);ws.launch();graph=wp.capture_end(device=ws.device)
        graph_checks[str(cap)]=0
        for turn,start in enumerate((0,137,1498-cap,13)):
            n=cap-1 if cap>1 and turn%2 else cap
            ids=np.arange(start,start+n,dtype=np.int32)
            p=payload(rows['real'],ids,cap)
            upload(ws,p)
            wp.capture_launch(graph);wp.synchronize()
            got=candidate(ws)
            check(got,rows['real'],cert['real'],ids,p['active'],'real')
            check_preserved(ws,p,topology)
            residual_check(rows['real'],cert['real'],got,ids,topology)
            graph_checks[str(cap)]+=1
        if np.array_equal(rows['real']['qm'][0],rows['real']['qm'][137]):
            raise AssertionError('Graph qM update control not distinct')
    after=source_hashes()
    if before!=after:raise AssertionError('source drift during GPU stage')
    (out/'source_before.json').write_text(json.dumps(before,indent=2)+'\n')
    (out/'source_after.json').write_text(json.dumps(after,indent=2)+'\n')
    summary={'real_required':1498,'real_gpu_checked':1498,'local_gpu_checked':7,
             'real_batches':batch_count,'graph_checks':graph_checks,
             'real_residual':real_residual,'local_residual':local_residual,
             'cpu_reference_manifest_sha256':sha(ref/'manifest.json'),
             'gpu_python':sys.executable,'warp_import':wp.__file__,
             'files':{x.name:sha(x) for x in out.iterdir() if x.is_file()}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def source_hashes():
    paths=[REQ,(REPO_ROOT / 'doc/phase_05/smooth_acceleration_chain/CPU_STAGE_SCHEMA.json'),(REPO_ROOT / 'doc/phase_05/smooth_acceleration_chain/NUMERICS.md'),(REPO_ROOT / 'doc/phase_05/smooth_acceleration_chain/CONTRACT.md'),
           (GPU_ROOT / 'reference/position/mass_solve/kernel.py'),
           (REPO_ROOT / 'doc/phase_05/batch_05/COMMON.md'),
           (REPO_ROOT / 'doc/phase_05/batch_05/INTERFACES.md'),
           (REPO_ROOT / 'doc/phase_05/batch_05/NUMERICS.md'),
           (REPO_ROOT / 'doc/phase_05/batch_05/prompts/D09_SOL.md'),
           (REPO_ROOT / 'doc/phase_05/batch_05/TASKS.json'),
           (REPO_ROOT / 'doc/phase_05/batch_05/EXTERNAL_SOURCES.json')]
    paths += sorted(((GPU_ROOT / 'reference/dynamics/smooth_acceleration_chain')).glob('*.py'))
    paths += sorted(((REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain')).glob('*.py'))
    paths += sorted(((REPO_ROOT / 'tests/gpu_smooth_acceleration_chain')).glob('*.py')) if ((REPO_ROOT / 'tests/gpu_smooth_acceleration_chain')).exists() else []
    return {str(p):sha(p) for p in paths}


def stage_dir(run_dir,stage):
    p=run_dir/stage
    if p.exists():raise FileExistsError(p)
    run_dir.mkdir(parents=True,exist_ok=True)
    return p


def under_lock(cmd,env):
    return subprocess.run(['flock','-E','75','-w','3600',str(LOCK),*map(str,cmd)],
                          check=True,env=env).returncode


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--stage',choices=('cpu','gpu','negative','probes','trace','ptx','all'),default='all')
    ap.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    args=ap.parse_args()
    env=os.environ.copy();env['PYTHONPATH']=str(REPO)
    env['WARP_CACHE_PATH']=str(args.run_dir/'warp_cache')
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPYCACHEPREFIX']=str(args.run_dir/'pycache')
    env['CMAKE_BUILD_PARALLEL_LEVEL']='2'
    env['MAX_JOBS']='2'
    stages=('gpu','negative','probes','trace','ptx') if args.stage=='all' else (args.stage,)
    for stage in stages:
        free=shutil.disk_usage(args.run_dir.parent).free
        if free < 40*(1024**3):
            raise RuntimeError(('disk space below 40 GiB',free))
        target=stage_dir(args.run_dir,stage)
        before=source_hashes() if stage!='cpu' else None
        if before is not None:
            (args.run_dir/(stage+'_source_before.json')).write_text(json.dumps(before,indent=2)+'\n')
        (args.run_dir/(stage+'_disk_before.json')).write_text(json.dumps({'free_bytes':free},indent=2)+'\n')
        if stage=='cpu':
            under_lock([CPU_PYTHON,(REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/derive_cpu.py'),target],env)
        elif stage=='gpu':
            # The inner stage performs no locking; exactly one outer lock.
            under_lock([GPU_PYTHON,Path(__file__),'--inner-gpu',str(args.reference),str(target)],env)
        elif stage=='negative':
            under_lock([GPU_PYTHON,(REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/verify_negative.py'),str(args.reference),str(target)],env)
        elif stage=='probes':
            # Saved actual CUDA output is checked in a separate CPU process.
            subprocess.run([str(CPU_PYTHON),str((REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/rework_probes.py')),
                            '--mode','create','--reference',str(args.reference),
                            '--gpu-dir',str(args.run_dir/'gpu'),'--target',str(target)],
                           check=True,env=env)
        elif stage=='trace':
            under_lock([GPU_PYTHON,(REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/trace_worker.py'),str(args.reference),str(target)],env)
        elif stage=='ptx':
            target.mkdir()
            subprocess.run([str(GPU_PYTHON),str((REPO_ROOT / 'scripts/gpu_smooth_acceleration_chain/audit_ptx.py')),
                            str(args.run_dir/'warp_cache'),str(target/'audit.json')],check=True,env=env)
        after=source_hashes() if stage!='cpu' else None
        if before is not None:
            (args.run_dir/(stage+'_source_after.json')).write_text(json.dumps(after,indent=2)+'\n')
            if before!=after:raise AssertionError(('source drift',stage))
        (args.run_dir/(stage+'_disk_after.json')).write_text(json.dumps({'free_bytes':shutil.disk_usage(args.run_dir.parent).free},indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)>=2 and sys.argv[1]=='--inner-gpu':
        print(json.dumps(gpu_stage(Path(sys.argv[2]),Path(sys.argv[3])),indent=2))
    else:main()
