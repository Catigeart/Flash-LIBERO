"""Unintegrated CPU-only immutable compilation-product storage.

No compiler, loader, CUDA API, ModelSpec or buffer ownership is changed here.
Callers must obtain an independently reviewed complete compile readset. Directory
roots are private trusted local directories, not adversarial concurrently mutable
namespaces. Products are copied, never hardlinked, into unique owner directories.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid

SCHEMA = "exact-compile-key-v2"
_HEX = re.compile(r"[0-9a-f]{64}\Z")

class CacheError(RuntimeError):
    pass

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _pin(value):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise CacheError("invalid SHA256")
    return value

def _text(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CacheError("missing identity string")
    return value

def _file_record(role, data):
    _text(role)
    if not isinstance(data, bytes):
        raise CacheError("readset must contain exact bytes")
    return {"logical_path": role, "sha256": digest(data), "bytes": len(data)}

def compile_key(*, source, headers, include_resolution, flags, architecture,
                compiler_files, compiler_versions, abi, environment,
                product_role, product_format, readset_receipt_sha256, source_path, working_directory, invocation):
    """Return key and JSON descriptor; no source/path normalization is allowed.

    include_resolution is the ordered complete compiler dependency resolution,
    including failed-search-sensitive include search paths. The pinned receipt
    attests completeness externally; this function cannot discover completeness.
    """
    if not isinstance(headers, dict) or not isinstance(compiler_files, dict) or not compiler_files:
        raise CacheError("headers/compiler files required")
    for seq in (include_resolution, flags, compiler_versions, invocation):
        if not isinstance(seq, (tuple, list)) or any(not isinstance(x, str) or "\x00" in x for x in seq):
            raise CacheError("ordered string sequence required")
    if not include_resolution or not compiler_versions or not invocation:
        raise CacheError("complete resolution/version required")
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in environment.items()):
        raise CacheError("explicit compile environment required")
    descriptor = {"schema": SCHEMA, "source_path": _text(source_path),
        "working_directory": _text(working_directory), "invocation": list(invocation), "source": _file_record("emitted-source", source),
        "headers": [_file_record(k, headers[k]) for k in sorted(headers)],
        "include_resolution": list(include_resolution), "flags": list(flags),
        "architecture": _text(architecture),
        "compiler_files": [_file_record(k, compiler_files[k]) for k in sorted(compiler_files)],
        "compiler_versions": list(compiler_versions), "abi": _text(abi),
        "environment": dict(environment), "product_role": _text(product_role),
        "product_format": _text(product_format),
        "readset_receipt_sha256": _pin(readset_receipt_sha256)}
    _validate_descriptor(descriptor)
    return digest(canonical(descriptor)), descriptor

def closed_artifact_storage_key(*, receipt_bytes, product_role, product_sha256):
    """Historical byte storage only; expressly NOT a compilation reuse key."""
    if not isinstance(receipt_bytes, bytes) or not receipt_bytes:
        raise CacheError("closed artifact receipt bytes required")
    descriptor={"schema":"closed-artifact-storage-v1", "compilation_reuse":False,
        "receipt_sha256":digest(receipt_bytes), "product_role":_text(product_role),
        "product_sha256":_pin(product_sha256)}
    return digest(canonical(descriptor)), descriptor

def _validate_descriptor(descriptor):
    if not isinstance(descriptor, dict): raise CacheError("descriptor must be a mapping")
    if descriptor.get("schema") == "closed-artifact-storage-v1":
        if set(descriptor) != {"schema","compilation_reuse","receipt_sha256","product_role","product_sha256"}:
            raise CacheError("invalid storage descriptor fields")
        if descriptor["compilation_reuse"] is not False: raise CacheError("storage cannot grant compile reuse")
        _pin(descriptor["receipt_sha256"]); _pin(descriptor["product_sha256"]); _text(descriptor["product_role"])
        return
    required={"schema","source","source_path","working_directory","invocation","headers",
        "include_resolution","flags","architecture","compiler_files","compiler_versions","abi",
        "environment","product_role","product_format","readset_receipt_sha256"}
    if descriptor.get("schema") != SCHEMA or set(descriptor) != required:
        raise CacheError("missing/unknown compilation identity fields")
    for field in ("source_path","working_directory","architecture","abi","product_role","product_format"):
        _text(descriptor[field])
    for field in ("source_path","working_directory"):
        if not Path(descriptor[field]).is_absolute(): raise CacheError("actual source/cwd must be absolute")
    for field in ("invocation","include_resolution","flags","compiler_versions"):
        seq=descriptor[field]
        if not isinstance(seq,list) or any(not isinstance(x,str) or "\x00" in x for x in seq):
            raise CacheError("invalid ordered compile identity")
        if field != "flags" and not seq: raise CacheError("missing ordered compile identity")
    env=descriptor["environment"]
    if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()):
        raise CacheError("invalid compile environment")
    _pin(descriptor["readset_receipt_sha256"])
    def validate_record(record):
        if not isinstance(record,dict) or set(record) != {"logical_path","sha256","bytes"}:
            raise CacheError("invalid readset record")
        _text(record["logical_path"]); _pin(record["sha256"])
        if type(record["bytes"]) is not int or record["bytes"] < 0: raise CacheError("invalid byte length")
    validate_record(descriptor["source"])
    for field in ("headers","compiler_files"):
        records=descriptor[field]
        if not isinstance(records,list) or (field == "compiler_files" and not records):
            raise CacheError("missing compile readset")
        for record in records: validate_record(record)
        names=[x["logical_path"] for x in records]
        if names != sorted(set(names)): raise CacheError("duplicate/unordered readset identities")

def _private_root(path):
    path = Path(path)
    if ".." in path.parts: raise CacheError("parent traversal in root")
    path = path.absolute()
    # Reject every existing symlink component, rather than resolving out of root.
    for part in [*reversed(path.parents), path]:
        if part.exists() or part.is_symlink():
            if part.is_symlink():
                raise CacheError("symlink root component")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    s=path.stat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077:
        raise CacheError("root must be owner-private 0700")
    return path.resolve(strict=True)

def _read(path):
    fd=os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
            raise CacheError("product must be regular unshared file")
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)

def _write(path, data):
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd,"wb",closefd=False) as f:
            f.write(data); f.flush(); os.fsync(fd)
        os.fchmod(fd,0o440)
    finally:
        os.close(fd)

def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)

class ArtifactCache:
    def __init__(self, root):
        self.root=_private_root(root)

    @contextlib.contextmanager
    def _lock(self):
        fd=os.open(self.root/".lock",os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            s=os.fstat(fd)
            if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                raise CacheError("invalid cache lock")
            fcntl.flock(fd,fcntl.LOCK_EX)
            yield
        finally: os.close(fd)

    def _entry(self,key,descriptor,product_sha256):
        _pin(key); _pin(product_sha256)
        _validate_descriptor(descriptor)
        if digest(canonical(descriptor)) != key:
            raise CacheError("descriptor/key mismatch")
        if descriptor.get("schema") == "closed-artifact-storage-v1":
            if descriptor.get("compilation_reuse") is not False or descriptor.get("product_sha256") != product_sha256:
                raise CacheError("storage-only receipt mismatch")
        entry=self.root/key
        if not entry.exists(): return None
        if entry.is_symlink() or not entry.is_dir(): raise CacheError("invalid cache entry")
        manifest=json.loads(_read(entry/"manifest.json"))
        expected={"schema":"exact-product-entry-v1","key":key,"descriptor":descriptor,
                  "product_sha256":product_sha256}
        if manifest != expected: raise CacheError("stale or tampered manifest")
        product=_read(entry/"product")
        if digest(product) != product_sha256: raise CacheError("tampered/partial product")
        return product

    def store(self,key,descriptor,product,*,product_sha256):
        """Caller pins successful source-to-product receipt, not manifest self-hash."""
        if not isinstance(product,bytes) or not product or digest(product) != _pin(product_sha256):
            raise CacheError("source-product receipt mismatch")
        with self._lock():
            if self._entry(key,descriptor,product_sha256) is not None: return False
            staging=Path(tempfile.mkdtemp(prefix=".pending-",dir=self.root))
            try:
                _write(staging/"product",product)
                _write(staging/"manifest.json",canonical({"schema":"exact-product-entry-v1",
                    "key":key,"descriptor":descriptor,"product_sha256":product_sha256}))
                _sync_dir(staging)
                os.rename(staging,self.root/key)
                _sync_dir(self.root)
            finally:
                if staging.exists(): shutil.rmtree(staging)
            return True

    def materialize(self,key,descriptor,*,product_sha256,owner_root,filename):
        """Return a new private owner path, or None on miss. Never load it."""
        if not isinstance(filename,str) or filename in ("", ".", "..") or Path(filename).name != filename or "/" in filename or "\\" in filename or "\x00" in filename:
            raise CacheError("filename must be a single path component")
        owner_root=_private_root(owner_root)
        if owner_root == self.root or self.root in owner_root.parents or owner_root in self.root.parents:
            raise CacheError("owner/cache roots must be separate")
        with self._lock():
            product=self._entry(key,descriptor,product_sha256)
            if product is None: return None
            owner=owner_root/("owner-"+uuid.uuid4().hex)
            owner.mkdir(mode=0o700)
            try:
                target=owner/filename
                _write(target,product)
                if digest(_read(target)) != product_sha256: raise CacheError("private copy verification failed")
                _sync_dir(owner); _sync_dir(owner_root)
                return target
            except BaseException:
                shutil.rmtree(owner)
                raise
