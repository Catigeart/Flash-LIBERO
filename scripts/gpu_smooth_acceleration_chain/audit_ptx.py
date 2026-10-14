"""Inspect the exact private Warp cache compiled for D09 and accepted N06."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cache,output):
    if output.exists():raise FileExistsError(output)
    files=[]
    for p in cache.rglob('*'):
        if p.is_file() and p.suffix in ('.ptx','.cu','.meta'):
            files.append(p)
    files=sorted(files)
    ptx=[p for p in files if p.suffix=='.ptx']
    chosen=[p for p in ptx if any(label in p.name for label in ('d09_smooth_acceleration_kernel','d09_accepted_n06_kernel'))]
    info={'cache':str(cache),'matching_ptx_count':len(chosen),
          'matching_ptx':{},'all_compiled_file_count':len(files)}
    for p in chosen:
        s=p.read_text(errors='replace')
        opcodes={name:len(re.findall(r'\b'+re.escape(name)+r'\b',s)) for name in
                 ('add.rn.f64','sub.rn.f64','mul.rn.f64','div.rn.f64','sqrt.rn.f64',
                  'fma.rn.f64','add.ftz.f64','mul.ftz.f64','div.approx.f64','sqrt.approx.f64')}
        info['matching_ptx'][str(p)]={'sha256':sha(p),'opcodes':opcodes,'bytes':p.stat().st_size}
    info['compiled_files']={str(p):sha(p) for p in files}
    output.write_text(json.dumps(info,indent=2)+'\n')
    if len(chosen)<2:raise AssertionError('D09/N06 compiled PTX not both located')
    if any(any(v['opcodes'][name] for name in ('fma.rn.f64','add.ftz.f64','mul.ftz.f64','div.approx.f64','sqrt.approx.f64'))
           for v in info['matching_ptx'].values()):
        raise AssertionError('PTX contains prohibited numeric opcode')
    force=[v['opcodes'] for p,v in info['matching_ptx'].items() if 'd09_smooth_acceleration_kernel' in p]
    solve=[v['opcodes'] for p,v in info['matching_ptx'].items() if 'd09_accepted_n06_kernel' in p]
    if (len(force)!=1 or force[0]['sub.rn.f64']!=1 or force[0]['add.rn.f64']!=2 or
            len(solve)!=1 or solve[0]['div.rn.f64']<1 or solve[0]['sqrt.rn.f64']<1):
        raise AssertionError('compiled RN node prerequisite')
    return info


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();print(json.dumps(run(args.cache,args.output),indent=2))
