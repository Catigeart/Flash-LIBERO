"""Load trusted local initialization states without mandatory PyTorch.

Official numpy-only torch.save ZIP files do not contain tensor storages. Their
pickle payload can be decoded directly, preserving the original numpy value.
Like torch.load(weights_only=False), this loader must only read trusted files.
"""
from pathlib import Path
import pickle
import zipfile


def load_init_states(path):
    """Return the stored value unchanged; lazily use torch for other formats."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            payloads = [name for name in names if name.endswith('/data.pkl')]
            if len(payloads) == 1 and not any('/data/' in name for name in names):
                try:
                    return pickle.loads(archive.read(payloads[0]))
                except ModuleNotFoundError as error:
                    if error.name is None or not error.name.startswith('torch'):
                        raise
                    # A custom payload may still reference torch objects.
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f'{path} requires PyTorch to decode a tensor-storage or legacy '
            'initialization file; numpy-only official ZIPs do not require it.'
        ) from error
    return torch.load(path, map_location='cpu', weights_only=False)
