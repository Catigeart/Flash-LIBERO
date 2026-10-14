"""Passive actual ELF and sm120 emitted-code inventory; no CUDA API imports."""
from __future__ import annotations
import hashlib
import pathlib
import re
from ptx_audit import parse_ptx

def rec(p):
    data=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def actual_code(maps,cwd):
    loaded=[]
    for raw in sorted(set(line.split()[-1] for line in maps.splitlines() if '/' in line)):
        p=pathlib.Path(raw)
        if p.is_file():
            with p.open('rb') as f:magic=f.read(4)
            if magic==b'\x7fELF':loaded.append(rec(p))
    emitted=[]
    cache=pathlib.Path(cwd)/'warp-cache'
    for p in sorted(cache.rglob('*')):
        if p.is_file() and p.suffix in ('.cu','.ptx','.meta'):
            row=rec(p)
            if p.suffix=='.ptx':
                text=p.read_text();instructions=[r['opcode'] for r in parse_ptx(text)]
                row.update({'target':re.findall(r'(?m)^\.target\s+([^\n]+)',text),
                    'ptx_version':re.findall(r'(?m)^\.version\s+([^\n]+)',text),
                    'predicated_instructions':sum(r['predicate'] is not None for r in parse_ptx(text)),
                    'f64_fma_instructions':sum(x.startswith(('fma.','mad.')) and '.f64' in x for x in instructions),
                    'f32_fma_instructions':sum(x.startswith(('fma.','mad.')) and '.f32' in x for x in instructions),
                    'ftz_instructions':sum('.ftz' in x for x in instructions),
                    'instruction_counts':{x:instructions.count(x) for x in sorted(set(instructions)) if any(y in x for y in ('.f64','.f32','cvt.','sqrt.','sin.','cos.'))}})
            emitted.append(row)
    sdk=[x for x in loaded if any(v in pathlib.Path(x['path']).name for v in ('nvrtc','warp.so','libcuda','cudart'))]
    return {'actual_loaded_elf':loaded,'actual_emitted_code':emitted,'actual_sdk_and_driver':sdk,
            'interpretation':'byte provenance and instruction census only; no hardware numerical acceptance inherited from older GPU or libdevice',
            'standalone_libdevice_bc_identity':'unknown unless a mapped compilation source explicitly supplies it; NVRTC builtins actual bytes are recorded'}
