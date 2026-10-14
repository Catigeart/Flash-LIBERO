"""S08 complete consumer with one shared full-V Dx arena; r01 stays frozen."""
from pathlib import Path
from .reference_consumer import ReferenceHint,complete_reference_stages
from .reference_consumer import complete_reference_source as original_source


def complete_reference_source(gpu_root,hint=None):
    source=original_source(gpu_root,hint)
    if hint is None:return source
    root=Path(gpu_root)
    dense=(root/'cuda/constraints/constraints_reference.cuh').read_text().replace('CF_S_reference {','CF_S_reference_dense_fallback {')
    compact=(root/'cuda/constraints/constraints_reference_compact.cuh').read_text().replace('namespace cf_s {','namespace cf_s_grouped {')
    if source.count(dense)!=1 or source.count(compact)!=1:
        raise ValueError('Exact reference source layout changed')
    declaration='__shared__ __align__(16) char storage[cublasdx::get_shared_storage_size<ReferenceGemm>()];'
    if dense.count(declaration)!=1:raise ValueError('Original dense arena declaration changed')
    # Identical 32x1xNV/full-V Dx type, shape, layout, alignment and barriers.
    # Place the qualified full-V arena helper before the original dense body.
    shared=dense.replace(declaration,'char* storage=cf_s_grouped::reference_storage<NV>();')
    return source.replace(dense+compact,compact+shared)
