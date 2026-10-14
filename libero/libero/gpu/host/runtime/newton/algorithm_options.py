"""Stable cold policy for handwritten Newton algorithm experiments.

Options choose equivalent function implementations; they do not change solver
tolerance, iteration limits, global line search, or publication rules.
"""
from numbers import Integral
from collections.abc import Mapping


DEFAULT_NEWTON_ALGORITHM_OPTIONS={
    'cone_hessian_strategy':'auto',
    'cone_batch_threshold':None,
    'rank_rebuild_threshold':None,
    'newton_islands':'off',
}


def normalize_newton_algorithm_options(options=None):
    """Return an owned kwargs dictionary suitable for HandwrittenCudaCore."""
    if options is None:options={}
    if not isinstance(options,Mapping):
        raise TypeError('newton_algorithm_options must be a mapping')
    unknown=set(options)-set(DEFAULT_NEWTON_ALGORITHM_OPTIONS)
    if unknown:raise ValueError(('unknown Newton algorithm options',sorted(unknown)))
    result=dict(DEFAULT_NEWTON_ALGORITHM_OPTIONS,**options)
    if result['cone_hessian_strategy'] not in ('rank','batch','auto'):
        raise ValueError('cone Hessian strategy must be rank, batch or auto')
    if result['newton_islands'] not in ('off','exact'):
        raise ValueError('Newton islands must be off or exact')
    for key in ('cone_batch_threshold','rank_rebuild_threshold'):
        value=result[key]
        if value is not None:
            if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
                raise ValueError(key+' must be a nonnegative integer or None')
            result[key]=int(value)
    return result


def effective_newton_algorithm_configuration(options,nv,*,has_blas=True,has_dx=True):
    """Resolve shape defaults for manifests and safe existing-core reuse."""
    options=normalize_newton_algorithm_options(options);nv=int(nv)
    if nv<1:raise ValueError('positive NV required')
    if options['cone_hessian_strategy']=='batch' and not (has_blas and has_dx):
        raise ValueError('batch cone Hessian requires cuBLASDx and cuSOLVERDx')
    if options['newton_islands']=='exact' and not has_dx:
        raise ValueError('exact Newton islands require cuSOLVERDx')
    cone=options['cone_hessian_strategy']
    threshold=options['cone_batch_threshold']
    rebuild=options['rank_rebuild_threshold']
    return dict(cone_strategy=cone if has_blas else 'rank',requested_cone_strategy=cone,
        cone_batch_threshold=nv if threshold is None else threshold,
        rank_rebuild_threshold=(max(8,nv//2) if has_dx else 0) if rebuild is None else rebuild,
        islands=options['newton_islands'],
        threshold_status='experimental configurable crossover; not a measured winner')
