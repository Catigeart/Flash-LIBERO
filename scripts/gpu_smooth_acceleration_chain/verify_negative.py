"""D09 device route, corruption, and saved actual-GPU-output replay tests."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_all import (OUT, REPO, GPU_PYTHON, candidate, check, exact_zero,
                     interval_for, load_reference, model, payload, residual_check,
                     out_shape, sha, upload, workspace_class)

INPUTS=('passive','bias','applied','actuator','qm','xfrc','active')
LEGACY_FIELDS={'ids','active','input_sha256'} | {'input_'+k for k in INPUTS} | {
    prefix+'_'+k for prefix in ('clean','corrupt') for k in OUT}
V2_FIELDS=LEGACY_FIELDS | {'input_schema_version','input_reference_sha256'}


def bound_digest(inputs,ids,manifest_sha):
    """Hash labeled schema, selected identities, frozen manifest and bytes."""
    h=hashlib.sha256(b'D09 saved-input v2\0')
    def update(label,array):
        value=np.asarray(array)
        meta=json.dumps({'field':label,'dtype':value.dtype.str,'shape':list(value.shape)},
                        sort_keys=True,separators=(',',':')).encode()
        for part in (meta,value.tobytes(order='C')):
            h.update(len(part).to_bytes(8,'big'));h.update(part)
    update('reference_manifest_sha256',np.bytes_(manifest_sha))
    update('ids',ids)
    for k in INPUTS:update('input_'+k,inputs[k])
    return h.hexdigest()


def _saved_input_schema(saved,rows):
    fields=set(saved)
    if fields not in (LEGACY_FIELDS,V2_FIELDS):
        raise TypeError(('saved packet fields',sorted(fields ^
                         (V2_FIELDS if 'input_schema_version' in fields else LEGACY_FIELDS))))
    v2=fields==V2_FIELDS
    if v2 and (saved['input_schema_version'].shape!=() or
               saved['input_schema_version'].dtype!=np.int32 or
               int(saved['input_schema_version'])!=2):
        raise TypeError('saved input schema version')
    for key in ('input_sha256','input_reference_sha256') if v2 else ('input_sha256',):
        if saved[key].shape!=() or saved[key].dtype.kind!='S' or len(bytes(saved[key]))!=64:
            raise TypeError(('saved input schema',key))
    ids=saved['ids'];active=saved['active']
    if ids.dtype!=np.int32 or ids.ndim!=1 or active.dtype!=np.int32 or active.ndim!=1:
        raise TypeError('saved identity schema')
    cap=len(active)
    if cap==0 or len(ids)==0 or len(ids)>cap or len(set(map(int,ids)))!=len(ids) or np.any(ids<0) or np.any(ids>=len(rows['qm'])):
        raise ValueError('saved identity selection')
    shapes={'passive':(cap,43),'bias':(cap,43),'applied':(cap,43),
            'actuator':(cap,43),'qm':(cap,153),'xfrc':(cap,38,6),'active':(cap,)}
    for k in INPUTS:
        value=saved['input_'+k]
        dtype=np.int32 if k=='active' else np.float64
        if value.shape!=shapes[k] or value.dtype!=dtype:
            raise TypeError(('saved input schema',k,value.shape,str(value.dtype)))
    if not np.array_equal(saved['input_active'],active) or not np.array_equal(active[:len(ids)],np.ones(len(ids),np.int32)) or np.any(active[len(ids):]):
        raise AssertionError('saved active routing')
    for prefix in ('clean','corrupt'):
        for k,dtype in OUT.items():
            value=saved[prefix+'_'+k]
            if value.dtype!=dtype or value.shape!=out_shape(k,cap):
                raise TypeError(('saved output schema',prefix,k,value.shape,str(value.dtype)))
    return v2


def rejected(fn, phrase):
    try:fn()
    except (AssertionError,TypeError,ValueError) as exc:
        if phrase not in str(exc):raise AssertionError(('wrong rejection',phrase,str(exc))) from exc
        return
    raise AssertionError(('negative accepted',phrase))


def replay(reference,archive):
    rows,topology,_,_=load_reference(reference)
    with np.load(archive,allow_pickle=False) as z:
        saved={k:np.array(z[k],copy=True) for k in z.files}
    v2=_saved_input_schema(saved,rows['local'])
    ids=saved['ids'];active=saved['active']
    trusted=payload(rows['local'],ids,len(active))
    for k in INPUTS:
        if saved['input_'+k].tobytes(order='C')!=trusted[k].tobytes(order='C'):
            raise AssertionError(('saved input provenance',k))
    manifest_sha=sha(reference/'manifest.json')
    trusted_digest=bound_digest(trusted,ids,manifest_sha)
    if v2:
        if bytes(saved['input_reference_sha256']).decode()!=manifest_sha:
            raise AssertionError('saved reference SHA mismatch')
        expected_sha=trusted_digest
    else:
        # Historical frozen Astra packets carry a byte-only checksum.  Their
        # schema and every input byte are bound above before this checksum is
        # considered; the returned authoritative digest is always v2.
        expected_sha=hashlib.sha256(b''.join(trusted[k].tobytes() for k in INPUTS)).hexdigest()
    if bytes(saved['input_sha256']).decode()!=expected_sha:
        raise AssertionError('saved input SHA mismatch')
    cert=interval_for(rows['local'],topology)
    clean={k:saved['clean_'+k] for k in OUT}
    corrupt={k:saved['corrupt_'+k] for k in OUT}
    check(clean,rows['local'],cert,ids,active,'local')
    clean_residual=residual_check(rows['local'],cert,clean,ids,topology)
    def verify_corrupt():
        check(corrupt,rows['local'],cert,ids,active,'local')
        residual_check(rows['local'],cert,corrupt,ids,topology)
    try:
        verify_corrupt()
    except (AssertionError,TypeError,ValueError) as exc:
        corrupt_reason=str(exc)
    else:
        raise AssertionError('saved corrupt output accepted')
    return {'saved_input_sha256':trusted_digest,'digest_version':2,
            'reference_manifest_sha256':manifest_sha,'legacy_packet':not v2,
            'clean':'passed','clean_residual':clean_residual,
            'corrupt_actual_gpu_output':'rejected','corrupt_reason':corrupt_reason,
            'archive_sha256':sha(archive)}


def run(reference,target):
    import warp as wp
    SmoothAccelerationWorkspace,validate_host_payload=workspace_class()
    if target.exists():raise FileExistsError(target)
    rows,topology,_,_=load_reference(reference)
    cert={k:interval_for(v,topology) for k,v in rows.items()}
    wp.init();target.mkdir(parents=True)
    m=model(topology)
    count=0
    def noted(fn):
        nonlocal count
        fn();count+=1
    # Original C local association probe: inspect actual device force.
    ws=SmoothAccelerationWorkspace(m,1)
    ids=np.array([6],np.int32)
    p=payload(rows['local'],ids,1);upload(ws,p);ws.launch();wp.synchronize()
    clean=candidate(ws)
    check(clean,rows['local'],cert['local'],ids,p['active'],'local');count+=1
    if clean['smooth'][0,5]!=1.0:raise AssertionError('association path not exercised')
    bad={k:v.copy() for k,v in clean.items()};bad['smooth'][0,5]=0.0
    noted(lambda:rejected(lambda:check(bad,rows['local'],cert['local'],ids,p['active'],'local'),'source-order force bit difference'))
    # The saved corruption begins with an actual GPU output array.
    corrupt={k:v.copy() for k,v in clean.items()}
    corrupt['acc'][0,5]=np.nextafter(cert['local']['factor']['solution'][1][6,0,5],np.inf)
    noted(lambda:rejected(lambda:check(corrupt,rows['local'],cert['local'],ids,p['active'],'local'),'GPU outside source interval'))
    # Residual is an independent gate even if a consumer bypassed interval membership.
    residual_bad={k:v.copy() for k,v in clean.items()};residual_bad['acc'][0,5]+=100.0
    noted(lambda:rejected(lambda:residual_check(rows['local'],cert['local'],residual_bad,ids,topology),'residual exceeds derived bound'))
    digest=bound_digest(p,ids,sha(reference/'manifest.json'))
    np.savez_compressed(target/'actual_gpu_failure.npz',ids=ids,active=p['active'],
        input_sha256=np.bytes_(digest),input_schema_version=np.int32(2),
        input_reference_sha256=np.bytes_(sha(reference/'manifest.json')),
        **{'input_'+k:v for k,v in p.items()},
        **{'clean_'+k:v for k,v in clean.items()},
        **{'corrupt_'+k:v for k,v in corrupt.items()})
    # Actual nonzero applied omission: execute with missing applied, compare to fixed input.
    ids=np.array([0],np.int32)
    p=payload(rows['local'],ids,1);p['applied'][:]=0
    upload(ws,p);ws.launch();wp.synchronize()
    got=candidate(ws)
    noted(lambda:rejected(lambda:check(got,rows['local'],cert['local'],ids,p['active'],'local'),'source-order force bit difference'))
    # Wrong sparse input: qM must drive fresh factor/solve.
    p=payload(rows['local'],np.array([2],np.int32),1)
    p['qm'][0]=rows['local']['qm'][0]
    upload(ws,p);ws.launch();wp.synchronize()
    got=candidate(ws)
    noted(lambda:rejected(lambda:check(got,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'GPU outside source interval'))
    wp.capture_begin(device=ws.device);ws.launch();graph=wp.capture_end(device=ws.device)
    wp.capture_launch(graph);wp.synchronize();got=candidate(ws)
    noted(lambda:rejected(lambda:check(got,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'GPU outside source interval'))
    correct=payload(rows['local'],np.array([2],np.int32),1)
    upload(ws,correct);wp.capture_launch(graph);wp.synchronize()
    check(candidate(ws),rows['local'],cert['local'],np.array([2],np.int32),correct['active'],'local')
    count+=1
    # Stale factor and wrong route are pure checker mutations of actual output.
    p=payload(rows['local'],np.array([2],np.int32),1);upload(ws,p);ws.launch();wp.synchronize()
    got=candidate(ws)
    check(got,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local');count+=1
    stale={k:v.copy() for k,v in got.items()};stale['ld'][0]=clean['ld'][0]
    noted(lambda:rejected(lambda:check(stale,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'GPU outside source interval'))
    index_swap={k:v.copy() for k,v in got.items()}
    a=int(topology['dof_Madr'][0]);b=int(topology['dof_Madr'][42])
    index_swap['ld'][0,a],index_swap['ld'][0,b]=index_swap['ld'][0,b],index_swap['ld'][0,a]
    noted(lambda:rejected(lambda:check(index_swap,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'GPU outside source interval'))
    route={k:v.copy() for k,v in got.items()};route['status'][0]=5
    noted(lambda:rejected(lambda:check(route,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'status first difference'))
    bad_dtype={k:v.copy() for k,v in got.items()};bad_dtype['acc']=bad_dtype['acc'].astype(np.float32)
    noted(lambda:rejected(lambda:check(bad_dtype,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'candidate schema'))
    wrong_int={k:v.copy() for k,v in got.items()};wrong_int['status']=wrong_int['status'].astype(np.int64)
    noted(lambda:rejected(lambda:check(wrong_int,rows['local'],cert['local'],np.array([2],np.int32),p['active'],'local'),'candidate schema'))
    # Device xfrc check covers every body/component, including world slot.
    p=payload(rows['real'],np.array([0],np.int32),1);p['xfrc'][0,37,5]=1.0
    upload(ws,p);ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=6 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('xfrc device reject')
    count+=1
    p=payload(rows['real'],np.array([0],np.int32),1);p['passive'][0,0]=np.nan
    upload(ws,p);ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=3 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('force nonfinite input route')
    count+=1
    p=payload(rows['real'],np.array([0],np.int32),1)
    p['passive'][0,0]=np.finfo(np.float64).max
    p['bias'][0,0]=-np.finfo(np.float64).max
    upload(ws,p);ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=4 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('force arithmetic nonfinite route')
    count+=1
    p=payload(rows['real'],np.array([0],np.int32),1);p['qm'][0,0]=np.nan
    upload(ws,p);ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=3 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('qM nonfinite input route')
    count+=1
    p=payload(rows['real'],np.array([0],np.int32),1);upload(ws,p)
    wp.copy(ws.count,wp.array(np.zeros(1,np.int32),device=ws.device))
    ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=2 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('RHS count route')
    count+=1
    # Invalid mask precedes payload reads; inactive stale/NaN is ignored.
    p=payload(rows['real'],np.array([0],np.int32),1);p['active'][0]=2
    p['passive'][0,0]=np.nan;upload(ws,p);ws.launch();wp.synchronize()
    if ws.status.numpy()[0]!=1:raise AssertionError('invalid mask first error')
    count+=1
    p['active'][0]=0;upload(ws,p);ws.launch();wp.synchronize();got=candidate(ws)
    if got['status'][0]!=0 or not exact_zero(got['smooth']) or not exact_zero(got['acc']):
        raise AssertionError('inactive control')
    count+=1
    # Dirty output must be overwritten by the three production kernels.
    p=payload(rows['real'],np.array([11],np.int32),1);upload(ws,p)
    for fill in (17.25,-93.75):
        for k in ('smooth','acc','ld','inv','sqrt_inv','pivot','status','warning_count','warning_last'):
            a=getattr(ws,k);v=np.full(a.shape,fill,dtype=np.int32 if k in ('pivot','status','warning_count','warning_last') else np.float64)
            wp.copy(a,wp.array(v,device=ws.device))
        wp.copy(ws.mass.solution,wp.array(np.full(ws.mass.solution.shape,fill,np.float64),device=ws.device))
        wp.copy(ws.mass.status,wp.array(np.full(ws.mass.status.shape,int(fill),np.int32),device=ws.device))
        ws.launch();wp.synchronize()
        check(candidate(ws),rows['real'],cert['real'],np.array([11],np.int32),p['active'],'real')
        count+=1
    wp.copy(ws.smooth,wp.array(rows['real']['smooth'][11:12],device=ws.device))
    ws.launch();wp.synchronize()
    check(candidate(ws),rows['real'],cert['real'],np.array([11],np.int32),p['active'],'real')
    count+=1
    # Host schema and capacity reject before upload or launch.
    bad_p=dict(p);bad_p['active']=p['active'].astype(np.int64)
    noted(lambda:rejected(lambda:validate_host_payload(bad_p,1),'active: expected'))
    noted(lambda:rejected(lambda:payload(rows['real'],np.array([0,1],np.int32),1),'identity selection'))
    noted(lambda:rejected(lambda:SmoothAccelerationWorkspace(m,0),'unsupported model/capacity'))
    archive=target/'actual_gpu_failure.npz'
    child=subprocess.run([str(GPU_PYTHON),str(Path(__file__)),'--replay',str(reference),str(archive)],
                         capture_output=True,text=True,env=__import__('os').environ.copy())
    (target/'replay_stdout.txt').write_text(child.stdout)
    (target/'replay_stderr.txt').write_text(child.stderr)
    if child.returncode!=0:raise AssertionError(('new process replay',child.returncode,child.stderr))
    summary={'targeted_cases':count,'new_process_replay_exit':child.returncode,
             'saved_failure_input_sha256':digest,'actual_gpu_failure_sha256':sha(archive),
             'replay':json.loads(child.stdout),'files':{x.name:sha(x) for x in target.iterdir() if x.is_file()}}
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('reference',type=Path)
    ap.add_argument('target',type=Path,nargs='?')
    ap.add_argument('--replay',action='store_true')
    args=ap.parse_args()
    if args.replay:
        print(json.dumps(replay(args.reference,args.target)))
    else:
        print(json.dumps(run(args.reference,args.target),indent=2))
