"""Require one canonical GPU namespace; never install aliases or import finders."""
import importlib
import sys
from .contracts import ContractError

def require_canonical_gpu_namespace():
    root=importlib.import_module('libero.gpu')
    forbidden='libero.libero.gpu'
    duplicate=[name for name in sys.modules if name==forbidden or name.startswith(forbidden+'.')]
    if duplicate:
        raise ContractError('GPU modules must use canonical libero.gpu namespace: '+duplicate[0])
    return root
