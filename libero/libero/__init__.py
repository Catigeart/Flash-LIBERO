"""Original LIBERO path API with noninteractive, lazy resource preparation."""
import os

# WSLg exposes the NVIDIA-backed D3D12 OpenGL route through GLFW/GLX. The
# robosuite import treats glfw as an EGL override candidate, while glx preserves
# this route. Apply before importing rendering libraries, only on WSLg with a
# display and with no user backend choice; other platforms retain their defaults.
if os.environ.get("DISPLAY") and os.path.isdir("/mnt/wslg") and "MUJOCO_GL" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "glx")
    # WSLg may otherwise select an integrated/software adapter even when the
    # NVIDIA D3D12 OpenGL driver is available. User adapter preferences win.
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

from pathlib import Path
import tempfile
import yaml

from .resources import ensure_resource_group, managed_resource_path, resource_root, resource_manifest

libero_config_path = os.path.expanduser(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"))
config_file = os.path.join(libero_config_path, "config.yaml")


def get_default_path_dict(custom_location=None):
    benchmark_root_path = (Path(__file__).resolve().parent if custom_location is None
                           else Path(custom_location).expanduser().absolute())
    def data_path(key, folder):
        bundled = benchmark_root_path / folder
        if custom_location is not None:
            return str(bundled)
        # The engineering tree intentionally contains the 130 registered task
        # definitions. Extra upstream BDDL files are not prerequisites for
        # using these bundled source definitions.
        if key == "bddl_files" and any(bundled.rglob("*.bddl")):
            return str(bundled)
        # A source checkout may contain original assets; a wheel may include only
        # XML stubs. Only a complete official group can serve as a default.
        complete = all((bundled / entry["path"]).is_file()
                       and (bundled / entry["path"]).stat().st_size == entry["size"]
                       for entry in resource_manifest()["groups"][key])
        if complete:
            return str(bundled)
        return str(managed_resource_path(key))
    return {
        "benchmark_root": str(benchmark_root_path),
        "bddl_files": data_path("bddl_files", "bddl_files"),
        "init_states": data_path("init_states", "init_files"),
        "datasets": str(benchmark_root_path.parent / "datasets"
                        if custom_location is not None or (benchmark_root_path.parent / "datasets").is_dir()
                        else resource_root().parent / "datasets"),
        "assets": data_path("assets", "assets"),
    }


def _write_config(config, *, overwrite):
    directory = Path(libero_config_path)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".config-", suffix=".yaml", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, config_file)
        else:
            # Atomic publication without replacing a config another process made.
            try:
                os.link(temporary, config_file)
            except FileExistsError:
                pass
    finally:
        Path(temporary).unlink(missing_ok=True)


def get_libero_path(query_key, *, download=None):
    with open(config_file, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"LIBERO config must be a path mapping: {config_file}")
    assert query_key in config, (
        f"Key {query_key} not found in config file {config_file}. You need to modify it. "
        f"Available keys are: {config.keys()}")
    value = config[query_key]
    if not isinstance(value, str):
        raise ValueError(f"LIBERO config path {query_key} must be a string: {config_file}")
    path = os.path.expanduser(value)
    if query_key in {"assets", "init_states", "bddl_files"}:
        if Path(path).absolute() == managed_resource_path(query_key).absolute():
            return ensure_resource_group(query_key, download=download)
        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Configured LIBERO {query_key} directory does not exist: {path}. "
                f"Update {config_file} to an existing original LIBERO resource directory, "
                f"or use {managed_resource_path(query_key)} for automatic official downloads. "
                "The existing user configuration has not been changed.")
    return path


def set_libero_default_path(custom_location=None):
    """Explicitly replace the path configuration, as in the original LIBERO API."""
    print("[Warning] You are changing the default path for Libero config. "
          "This will affect all the paths in the config file.")
    _write_config(get_default_path_dict(custom_location), overwrite=True)


if not os.path.exists(config_file):
    _write_config(get_default_path_dict(), overwrite=False)
