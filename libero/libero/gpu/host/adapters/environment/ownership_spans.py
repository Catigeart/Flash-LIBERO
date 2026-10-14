"""Exact conservative span checks, without traversing or trusting GPU owners.

The fast successful path sorts intervals. A conflict falls back to the original
pair order so the public first reported overlap remains unchanged. Empty spans,
touching spans and overlaps within one world do not conflict. Strided bounding
spans are deliberately conservative, as in the original owner walk.
"""


def first_cross_world_overlap(records):
    if len(records)<2:return None
    intervals=[]
    for ordinal,record in enumerate(records):
        for field in record['fields']:
            size=field['bytes']
            if type(size) is not int or size<0 or type(field['pointer']) is not int:
                raise ValueError('integer nonnegative byte spans required')
            if size:
                intervals.append((field['pointer'],field['pointer']+size,ordinal))
    intervals.sort()
    owner=end=None
    conflict=False
    for start,stop,current_owner in intervals:
        if end is None or start>=end:
            owner,end=current_owner,stop
        elif current_owner!=owner:
            conflict=True
            break
        else:
            end=max(end,stop)
    if not conflict:return None
    # Preserve the original deterministic diagnostic pair on the error path.
    for index,a in enumerate(records):
        for b in records[index+1:]:
            for x in a['fields']:
                for y in b['fields']:
                    if (x['bytes'] and y['bytes']
                            and max(x['pointer'],y['pointer']) <
                            min(x['pointer']+x['bytes'],y['pointer']+y['bytes'])):
                        return x,y
    raise AssertionError('interval sweep and exact span predicate differ')


# No reusable owner-graph lease is exported. Complete exclusive writer coverage
# for arbitrary nested Warp owners is not available in the current authority;
# caller booleans or mutation epochs would silently miss external replacements.
