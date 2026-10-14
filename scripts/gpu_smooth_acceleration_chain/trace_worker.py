"""Raw Nsight 100-launch D09 production trace and overlap audit."""

import argparse
import ctypes
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np

from run_all import candidate, interval_for, load_reference, model, payload, sha, upload, workspace_class

LABEL = 'D09_smooth_acceleration_N20_100_production'


def inner(reference,output):
    import warp as wp
    SmoothAccelerationWorkspace,_=workspace_class()
    rows,topology,_,_=load_reference(reference)
    wp.init()
    ws=SmoothAccelerationWorkspace(model(topology),20)
    ids=np.arange(20,dtype=np.int32)
    p=payload(rows['real'],ids,20)
    upload(ws,p)
    ws.launch();wp.synchronize()
    nvtx=ctypes.CDLL('libnvToolsExt.so.1')
    nvtx.nvtxRangePushA.argtypes=[ctypes.c_char_p]
    nvtx.nvtxRangePop.argtypes=[]
    nvtx.nvtxRangePushA(LABEL.encode())
    for _ in range(100):ws.launch()
    wp.synchronize()
    nvtx.nvtxRangePop()
    got=candidate(ws)
    # Download occurs only after the hot range; fixed-input repeat is explicit.
    from run_all import check
    check(got,rows['real'],interval_for(rows['real'],topology),ids,p['active'],'real')
    output.write_text(json.dumps({'launches':100,'capacity':20,'fixed_input_replay':True,
                                  'production_kernels_per_launch':3,
                                  'mujoco_cpu_module_loaded':('mujoco' in sys.modules)})+'\n')


def audit(report,db):
    conn=sqlite3.connect(db)
    spans=conn.execute('select start,end from NVTX_EVENTS where text=?',(LABEL,)).fetchall()
    if len(spans)!=1:raise AssertionError(('NVTX span count',len(spans)))
    a,b=spans[0]
    tables={x[0] for x in conn.execute("select name from sqlite_master where type='table'")}
    counts={}
    for name in ('KERNEL','MEMCPY','MEMSET'):
        table='CUPTI_ACTIVITY_KIND_'+name
        counts[name.lower()]=conn.execute(f'select count(*) from {table} where start<? and end>?',(b,a)).fetchone()[0] if table in tables else 0
    names=dict(conn.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k '
                            'join StringIds s on s.id=k.shortName '
                            'where k.start<? and k.end>? group by s.value',(b,a)).fetchall())
    runtime=dict(conn.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
                              'join StringIds s on s.id=r.nameId '
                              'where r.start<? and r.end>? group by s.value',(b,a)).fetchall())
    if (counts!={'kernel':300,'memcpy':0,'memset':0} or
            len(names)!=3 or any(names.get(next((k for k in names if label in k),''))!=100
                                 for label in ('_make_force','_factor_solve','_finish')) or
            runtime.get('cuLaunchKernel')!=300 or
            any(('Memcpy' in k or 'Memset' in k or 'Alloc' in k or 'Free' in k)
                for k in runtime)):
        raise AssertionError(('production trace count/copy',counts))
    return {'duration_ns':b-a,'counts':counts,'kernel_names':names,'runtime_api':runtime,
            'overlap_rule':'start < hot_end and end > hot_start',
            'nsys_sha256':sha(report),'sqlite_sha256':sha(db)}


def outer(reference,target):
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    base=target/'d09_n20_100'
    report=base.with_suffix('.nsys-rep')
    log=target/'worker.json'
    cmd=['nsys','profile','--trace=cuda,nvtx','--force-overwrite=true','--output',str(base),
         sys.executable,str(Path(__file__)),'--inner',str(reference),str(log)]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    (target/'nsys_stdout.txt').write_text(proc.stdout)
    (target/'nsys_stderr.txt').write_text(proc.stderr)
    if proc.returncode!=0:raise AssertionError(('nsys profile',proc.returncode,proc.stderr[-2000:]))
    db=base.with_suffix('.sqlite')
    export=subprocess.run(['nsys','export','--type','sqlite','--output',str(db),str(report)],capture_output=True,text=True)
    (target/'export_stdout.txt').write_text(export.stdout)
    (target/'export_stderr.txt').write_text(export.stderr)
    if export.returncode!=0:raise AssertionError(('nsys export',export.returncode,export.stderr[-2000:]))
    summary=audit(report,db)
    worker=json.loads(log.read_text())
    if worker['mujoco_cpu_module_loaded']:
        raise AssertionError('CPU solver module loaded in production worker')
    summary.update(worker=worker,files={x.name:sha(x) for x in target.iterdir() if x.is_file()})
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('reference',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--inner',action='store_true')
    args=ap.parse_args()
    print(json.dumps(inner(args.reference,args.output) if args.inner else outer(args.reference,args.output)))
