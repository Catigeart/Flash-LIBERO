"""Cold opt-in boundaries for locally measured Post tile/chain functions."""
import hashlib
import numpy as np

TILE_PROFILE='post_tile128_20261012_r01'
CHAIN_PROFILE='post_chain65_20261012_r01'
FAMILIES=((38,43,2,114,529,3174),(65,57,2,195,1049,6294))
CHAIN_PARENT_SHA256='f2a317a1b608498b3696eadc8b0a20cb89abb850eb4444121c504bd184b07387'


def select_post_function(profile,dimensions,cc,ec,*,worlds,contact_index_mode,
                         radix_min_contacts,parent,contact_topology_cache=True,
                         dataflow_profile="compatible",velocity_bias_cache=False):
    original=dict(selected=False,source_route='post_dataflow_original.cuh',
        defines={},contact_tile=32,chains=False,reason='original whole function',
        performance_scope='No optimization claim outside tested cold family')
    if profile is None:return original
    if profile not in (TILE_PROFILE,CHAIN_PROFILE):raise ValueError('unknown Post function qualification')
    family=tuple(map(int,dimensions))+(int(cc),int(ec))
    eligible=family in FAMILIES and worlds in (1,2) and contact_index_mode=='radix' and radix_min_contacts==256
    eligible=eligible and all(type(x) is int and x>0 for x in tuple(dimensions)+(cc,ec))
    eligible=eligible and type(worlds) is int and type(radix_min_contacts) is int
    eligible=eligible and contact_topology_cache is True and dataflow_profile in ('compatible','fast')
    eligible=eligible and type(velocity_bias_cache) is bool and not (dataflow_profile=='compatible' and velocity_bias_cache)
    if profile==CHAIN_PROFILE:
        a=np.asarray(parent)
        eligible=eligible and family==FAMILIES[1] and a.dtype==np.dtype('int32') and a.shape==(65,) and hashlib.sha256(a.tobytes()).hexdigest()==CHAIN_PARENT_SHA256
    if not eligible:return {**original,'profile':profile}
    return dict(selected=True,source_route='post_function_tile128.cuh' if profile==TILE_PROFILE else 'post_function_chain65.cuh',
        defines={},contact_tile=128 if profile==TILE_PROFILE else 32,chains=profile==CHAIN_PROFILE,
        profile=profile,reason='Exact tested cold family; device count guard preserves literal original fallback',
        performance_scope='Full-capacity synthetic local Post fixtures; no actual task traffic or arbitrary topology claim',
        static_shared_upper_bound_bytes=2*(32*4+9*128*8+10*4))
