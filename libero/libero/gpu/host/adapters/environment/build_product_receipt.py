"""Bounded cold consumption of complete compiler-produced product facts.

Self-consistency is not compilation permission. The required external authority
binds the actual successful producer/readset policy and physical product owner.
No SDK file, binary, compiler, loader or GPU operation is performed here.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

_HEX=re.compile(r'[0-9a-f]{64}\Z')
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def pin(value):
    if type(value) is not str or not _HEX.fullmatch(value):raise ValueError('exact product SHA256 required')
    return value
def record(value):
    if type(value) is not dict or set(value)!={'path','bytes','sha256'}:raise ValueError('complete file record required')
    if type(value['path']) is not str or not Path(value['path']).is_absolute():raise ValueError('actual absolute readset path required')
    if type(value['bytes']) is not int or value['bytes']<0:raise ValueError('exact readset byte length required')
    pin(value['sha256'])


@dataclass(frozen=True)
class CompleteProductFacts:
    receipt_bytes: bytes
    identity_bytes: bytes
    build_key: str
    product_sha256: str
    product_bytes: int

    @classmethod
    def inspect(cls,receipt_bytes):
        if type(receipt_bytes) is not bytes or not 0<len(receipt_bytes)<=256*1024:
            raise ValueError('bounded exact compiler receipt bytes required')
        receipt=json.loads(receipt_bytes)
        if type(receipt) is not dict or set(receipt)!={'schema','build_key','identity','identity_sha256','binary_sha256','binary_bytes'} or receipt['schema']!='complete-build-product-v1':
            raise ValueError('complete successful product receipt schema required')
        identity=receipt['identity']
        fields={'schema','source_sha256','bridges_sha256','flags','toolchain','dependencies','archive','architecture','params_launcher_abi','policy'}
        if type(identity) is not dict or set(identity)!=fields or identity['schema']!='compiler-dependency-build-v1':
            raise ValueError('complete compiler source/dependency/ABI identity required')
        pin(identity['source_sha256'])
        if type(identity['bridges_sha256']) is not list:raise ValueError('exact bridge readset required')
        for value in identity['bridges_sha256']:pin(value)
        flags=identity['flags']
        if type(flags) is not list or any(type(f) is not str or '\x00' in f for f in flags):raise ValueError('ordered exact compile flags required')
        required={'--fmad=true','--prec-div=true','--prec-sqrt=true','--ftz=false'}
        if not required.issubset(flags) or any(f in flags for f in ('--use_fast_math','--fmad=false','--prec-div=false','--prec-sqrt=false','--ftz=true')):
            raise ValueError('precise source math policy required')
        if identity['policy']!={'fma':True,'precise_div':True,'precise_sqrt':True,'ftz':False}:
            raise ValueError('exact compile math policy required')
        if type(identity['architecture']) is not int or identity['architecture']<1 or identity['params_launcher_abi']!='included-generated-source':
            raise ValueError('complete target/generated launcher ABI required')
        if type(identity['dependencies']) is not list:raise ValueError('actual compiler dependency readset required')
        for value in identity['dependencies']:record(value)
        toolchain=identity['toolchain']
        if type(toolchain) is not dict or set(toolchain)!={'compiler','version','components','environment_sha256','schema'} or toolchain['schema']!='strict-toolchain-components-env-v1':
            raise ValueError('complete compiler/toolchain environment identity required')
        if not Path(toolchain['compiler']).is_absolute() or type(toolchain['version']) is not str or not toolchain['version']:
            raise ValueError('actual compiler/version required')
        if type(toolchain['components']) is not list or not toolchain['components']:raise ValueError('complete compiler components required')
        for value in toolchain['components']:record(value)
        pin(toolchain['environment_sha256'])
        if identity['archive'] is not None:record(identity['archive'])
        encoded=canonical(identity);key=digest(encoded)
        if key!=pin(receipt['identity_sha256']) or key!=pin(receipt['build_key']):
            raise ValueError('complete receipt/identity key differs')
        product_sha=pin(receipt['binary_sha256']);product_bytes=receipt['binary_bytes']
        if type(product_bytes) is not int or product_bytes<=0:raise ValueError('successful nonempty product extent required')
        return cls(receipt_bytes,encoded,key,product_sha,product_bytes)

    def authorize(self,*,physical_owner,validate_success):
        """Only a separately qualified producer/owner authority grants reuse.

        Return the authority's opaque token unchanged. Booleans, self-authored
        dicts and cached disk flags are not accepted as a successful permit.
        """
        if physical_owner is None or not callable(validate_success):raise TypeError('actual product owner/success authority required')
        token=validate_success(self,physical_owner)
        if token is None or isinstance(token,(bool,int,float,str,bytes,dict,list,tuple)):
            raise ValueError('successful live producer/owner authority token required')
        return token
