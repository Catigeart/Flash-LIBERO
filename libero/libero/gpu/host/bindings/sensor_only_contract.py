"""Pure exact write-owner contract for the explicit sensor-only leaf ABI."""

WRITE_FIELDS=('g_g','r_active','r_acc','r_ext','r_intr','r_decoded','r_status',
 's_result','s_status','c_first','c_pe','c_pi','c_ps','c_valid','c_written',
 'body_index','body_offsets','body_counts','event_ping','event_pong',
 'cached_event_body','cached_contact_meta')
FAMILIES=((38,43,2,114,529,3174),(65,57,2,195,1049,6294),(66,57,2,198,1049,6294))


def validate_write_ranges(ranges,legacy_ranges):
    if not legacy_ranges:raise ValueError('explicit legacy held-owner inventory required')
    for name in WRITE_FIELDS:
        start,n=ranges[name]
        if start<=0 or n<=0:raise ValueError('invalid sensor leaf owner extent: '+name)
        for old_start,old_n in legacy_ranges:
            if max(start,old_start)<min(start+n,old_start+old_n):
                raise ValueError('sensor-only leaf writes legacy held owner: '+name)
    if ranges['c_first'][0]==ranges['g_first'][0]:
        raise ValueError('sensor-only needs distinct leaf first_error owner')
    return True
