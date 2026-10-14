"""Pinned official LIBERO resources, fetched lazily outside the installed package.

Meshes, textures, initial states and BDDL files come from the original upstream
repository. Existing LIBERO_CONFIG_PATH paths take precedence over this cache.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
from urllib.parse import quote

_MANIFEST_PATH = Path(__file__).with_name("upstream_resources.json")
_GROUP_FOLDERS = {"assets": "assets", "init_states": "init_files", "bddl_files": "bddl_files"}


@lru_cache(maxsize=4)
def _read_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def resource_manifest():
    return _read_manifest(str(_MANIFEST_PATH))


def resource_root():
    """Writable resource location, optionally selected with LIBERO_RESOURCE_PATH."""
    custom = os.environ.get("LIBERO_RESOURCE_PATH")
    if custom:
        return Path(custom).expanduser().absolute()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "libero" / resource_manifest()["revision"] / "libero"


def managed_resource_path(key):
    return resource_root() / _GROUP_FOLDERS[key]


def _blob_matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha1()
    digest.update(b"blob " + str(entry["size"]).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["git_blob_sha1"]


@contextmanager
def _download_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".resource-download.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _download_file(destination, entry, manifest, folder):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe upstream resource path")
    target = destination / relative
    if _blob_matches(target, entry):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://raw.githubusercontent.com/" + manifest["repository"] + "/"
           + manifest["revision"] + "/libero/libero/" + folder + "/"
           + quote(entry["path"], safe="/"))
    last_error = None
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=".libero-download-", dir=target.parent)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Flash-LIBERO-resources/1"})
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=60) as response:
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > entry["size"]:
                        raise ValueError(f"Unexpected size for official resource {entry['path']}")
                    output.write(chunk)
            if not _blob_matches(Path(temporary), entry):
                raise ValueError(f"Official resource checksum mismatch: {entry['path']}")
            os.replace(temporary, target)
            return
        except (OSError, ValueError) as error:
            last_error = error
        finally:
            Path(temporary).unlink(missing_ok=True)
    raise RuntimeError(f"Cannot fetch official LIBERO resource {entry['path']}: {last_error}") from last_error


def ensure_resource_group(key, *, download=None):
    """Return a complete pinned cache group, resuming interrupted downloads.

    No configured external directory is written by this function. A completion
    marker is published only after every official file passed Git blob checks.
    """
    if key not in _GROUP_FOLDERS:
        raise KeyError(key)
    manifest = resource_manifest()
    destination = managed_resource_path(key)
    entries = manifest["groups"][key]
    marker = destination / ".libero-resources.json"
    stamp = {"revision": manifest["revision"], "group": key,
             "manifest_sha256": hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()}
    def ready():
        try:
            return json.loads(marker.read_text()) == stamp and all(
                (destination / entry["path"]).is_file()
                and (destination / entry["path"]).stat().st_size == entry["size"]
                for entry in entries)
        except (OSError, ValueError):
            return False
    if ready():
        return str(destination)
    if download is None:
        download = os.environ.get("LIBERO_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no"}
    if not download:
        raise FileNotFoundError(
            f"LIBERO {key} are missing from {destination}. Run "
            f"python -m libero.libero.resource_cli {key}, enable LIBERO_AUTO_DOWNLOAD, "
            "or configure an existing resource directory in LIBERO_CONFIG_PATH/config.yaml.")
    with _download_lock(resource_root()):
        if ready():
            return str(destination)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Preparing official LIBERO {key} ({len(entries)} files, "
              f"{sum(item['size'] for item in entries) / 1024**2:.1f} MiB) in {destination}")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda entry: _download_file(
                destination, entry, manifest, _GROUP_FOLDERS[key]), entries))
        # Resumable per-file atomic writes; no success marker on partial failure.
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(stamp, sort_keys=True) + "\n")
        os.replace(temporary, marker)
    return str(destination)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare official LIBERO runtime resources")
    parser.add_argument("groups", nargs="*", metavar="GROUP",
                        help="assets, init_states, bddl_files; omitted means all three")
    parser.add_argument("--check", action="store_true", help="Do not download missing groups")
    args = parser.parse_args()
    groups = args.groups or list(_GROUP_FOLDERS)
    for group in groups:
        if group not in _GROUP_FOLDERS:
            parser.error(f"unknown resource group {group!r}")
    from libero.libero import get_libero_path
    for group in groups:
        path = get_libero_path(group, download=not args.check)
        print(f"{group}: {path}")


if __name__ == "__main__":
    main()
