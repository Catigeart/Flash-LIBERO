"""Prepared integer accounting leaf; never invents allocation coverage.

Facts are immutable Python values captured at cold preparation, independently
of any live array metadata. A caller's authority still validates every fact on
each preflight. This is an accounting function, not a full allocator receipt.
"""
from .resident_budget import BudgetCharge, REQUIRED_CATEGORIES


def prepare_budget_leaf(charges,*,snapshot,validate_fact,serial_construction=True):
    if not callable(validate_fact):raise TypeError('allocation fact validator required')
    if type(serial_construction) is not bool:raise TypeError('explicit phase policy required')
    charges=tuple(charges)
    if any(type(c) is not BudgetCharge for c in charges):raise TypeError('BudgetCharge required')
    rows=tuple((c.key,c.category,c.steady_bytes,c.cold_bytes,c.receipt,c.cold_phase) for c in charges)
    if len({r[0] for r in rows})!=len(rows):raise ValueError('allocation aliases require one fact')
    missing=REQUIRED_CATEGORIES-{r[1] for r in rows}
    if missing:raise ValueError('incomplete resident budget categories: '+','.join(sorted(missing)))
    steady=sum(r[2] for r in rows);phases={}
    for row in rows:phases[row[5]]=phases.get(row[5],0)+row[3]
    cold=max(phases.values(),default=0) if serial_construction else sum(phases.values())
    categories=tuple(sorted(REQUIRED_CATEGORIES))
    def preflight(*,available_bytes,reserved_bytes=0):
        if any(type(x) is not int or x<0 for x in (available_bytes,reserved_bytes)):
            raise ValueError('actual nonnegative available/reserved bytes required')
        for fact in rows:
            if validate_fact(fact,snapshot) is False:
                raise ValueError('allocation authority rejected budget receipt: '+fact[0])
        peak=steady+cold+reserved_bytes
        if peak>available_bytes:raise MemoryError(f'resident peak {peak} exceeds actual available {available_bytes}')
        return {'steady_bytes':steady,'cold_peak_bytes':cold,'reserved_bytes':reserved_bytes,
                'peak_bytes':peak,'available_bytes':available_bytes,'complete_categories':list(categories),
                'receipt_count':len(rows),'serial_construction':serial_construction}
    return preflight


def archive_copy_typed_audit(fields,*,worlds):
    """Typed lower bound for exactly the independently measured A05 fixture.

    Input is the retained NPZ header descriptor, not a claim to enumerate C/S/N
    or modules/context/native allocator memory. Empty fields are retained.
    """
    if type(worlds) is not int or worlds<1:raise ValueError('positive worlds required')
    total=0;empty=[]
    for name,field in fields.items():
        if field['dtype']!='<f8':raise ValueError('exact FP64 field descriptors required')
        extent=1
        for dim in field['shape']:
            if type(dim) is not int or dim<0:raise ValueError('nonnegative actual extents required')
            extent*=dim
        total+=extent*8
        if not extent:empty.append(name)
    return {'archive_default_typed_bytes':worlds*total,'archive_root_plus_default_typed_bytes':2*worlds*total,
            'empty_manifest_fields':tuple(empty),'worlds':worlds,'complete_allocator_budget':False,
            'coverage':'only measured archive root/default arrays; no C/S/N/controller/observer/graph/module/context/native/slack/restore peak receipt',
            'missing_categories':tuple(sorted(REQUIRED_CATEGORIES))}
