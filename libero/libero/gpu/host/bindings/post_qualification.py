"""Cold, opt-in selector for the measured local Post contact-index domain.

This is a workload qualification boundary, not a topology or float identity.
The normal owner/shape/schema checks and exact per-event cache comparison still
execute. Unknown cold dimensions keep the original whole Post NativePlan.
"""
PROFILE='post_index_20261012_r01'
FAMILIES=((38,43,2,114,529,3174),(65,57,2,195,1049,6294),
          (66,57,2,198,1049,6294))


def select_post_index(profile,dimensions,contact_capacity,efc_capacity,*,
                      contact_index_mode,radix_min_contacts,block_size=64,
                      worlds=1,contact_topology_cache=True,dataflow_profile="compatible",
                      velocity_bias_cache=False):
    if profile == 'post_real_index_cache0_20261013_r05':
        values=tuple(dimensions)+(contact_capacity,efc_capacity)
        if any(type(x) is not int or x<1 for x in values):
            raise ValueError('Post real index cold integer domain')
        selected=(values==(38,43,31,270,5000,30061) and type(worlds) is int and worlds in (1,2)
            and contact_index_mode=='radix' and type(radix_min_contacts) is int and radix_min_contacts==256
            and type(block_size) is int and block_size==64 and contact_topology_cache is False
            and dataflow_profile=="fast" and velocity_bias_cache is True)
        return dict(selected=selected,reason='Qualified retained real-input fast/bias-enabled cache0 independent standalone function only',
            profile=profile,source_route='post_function_real_index_r05.cuh' if selected else 'post_dataflow_original.cuh',
            defines={},selector_revision='r05_real_consumer_fast_bias_cache0_only',count_domain=(2765,),
            cache_domain='disabled only',
            performance_domain='Single held independent input domain only; unchanged-input replay allowed. Any bias valid/status/epoch/gravity mutation ends this measured performance domain; caller must choose original/reassemble.',
            persistent_bias_qualification=False,real_family_production_selector_qualified=False,
            runtime_bias_relation='Original cache_ok re-evaluated every GPU call and original non-bias spatial branch retained; optimized integer index does not dynamically withdraw on bias=False.',
            static_shared_upper_bound_bytes=32*4+9*32*8+(10+13+13)*4)
    if profile is None:
        return dict(selected=False,reason='original profile',defines={},source_route='post_dataflow_original.cuh',selector_revision='r04_final_consumer_cold_withdrawal')
    if profile!=PROFILE:
        raise ValueError('unknown Post index qualification profile')
    values=tuple(dimensions)+(contact_capacity,efc_capacity)
    if any(type(x) is not int or x<1 for x in values):
        raise ValueError('Post index cold integer domain')
    excluded_negative=(contact_topology_cache and
                       (values[0]==65 or (values[0]==66 and worlds==2)))
    selected=(values in FAMILIES and contact_index_mode=='radix' and
              type(worlds) is int and worlds in (1,2) and not excluded_negative and
              type(radix_min_contacts) is int and radix_min_contacts==256 and
              type(block_size) is int and block_size==64 and
              dataflow_profile=="fast" and velocity_bias_cache is False)
    return dict(selected=selected,
        reason=('mixed/negative complete-consumer b65/W1-W2 or b66/W2 cache-enabled cold domain; exact original whole plan' if excluded_negative else
                'measured cold family with runtime count guard' if selected else 'unknown cold work domain; exact original whole plan'),
        selector_revision='r04_final_consumer_cold_withdrawal',
        source_route='post_dataflow.cuh' if selected else 'post_dataflow_original.cuh',
        profile=PROFILE,defines={'LIBERO_POST_QUALIFIED_CONTACT_INDEX':1} if selected else {},
        count_domain=(512,516) if values[0]==38 else (1032,1036),
        cache_domain='2-bit without once keys for cache-enabled; 2-bit plus once keys only cache-disabled',
        diagnostic_domain='legacy force/contact/sensor publication remains complete',
        static_shared_upper_bound_bytes=32*4+9*32*8+(10+13+13)*4 if selected else 32*4+9*32*8+10*4)
