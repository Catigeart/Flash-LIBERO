"""Cold conservative resident peak ledger; missing categories are never zero.

Receipts are supplied by allocation owners. Validation here checks accounting,
not completeness of a CUDA allocator's instrumentation. The caller must pin the
owner readset/model/ABI and allocator policy that produced each receipt.
"""
from dataclasses import dataclass

REQUIRED_CATEGORIES=frozenset(('arena','controller','observations','graphs',
    'modules','context','native','allocator_slack','restore','construction'))


@dataclass(frozen=True)
class BudgetCharge:
    key: str
    category: str
    steady_bytes: int
    cold_bytes: int
    receipt: str
    cold_phase: str = 'construction'

    def __post_init__(self):
        if (not isinstance(self.key,str) or not self.key or self.category not in REQUIRED_CATEGORIES or
                type(self.steady_bytes) is not int or self.steady_bytes<0 or
                type(self.cold_bytes) is not int or self.cold_bytes<0 or
                not isinstance(self.receipt,str) or not self.receipt or
                not isinstance(self.cold_phase,str) or not self.cold_phase):
            raise ValueError('nonnegative complete pinned budget charge required')


class ResidentBudget:
    def __init__(self, charges, *, snapshot, validate_receipt, serial_construction=True):
        self.charges=tuple(charges);self.snapshot=snapshot
        if not callable(validate_receipt):raise TypeError('actual receipt validator required')
        if type(serial_construction) is not bool:raise TypeError('construction policy must be explicit')
        self.validate_receipt=validate_receipt;self.serial=serial_construction
        if any(not isinstance(c,BudgetCharge) for c in self.charges):raise TypeError('BudgetCharge required')
        if len({c.key for c in self.charges})!=len(self.charges):
            raise ValueError('allocation aliases require a single shared charge')
        missing=REQUIRED_CATEGORIES-{c.category for c in self.charges}
        if missing:raise ValueError('incomplete resident budget categories: '+','.join(sorted(missing)))

    def preflight(self, *, available_bytes, reserved_bytes=0):
        if any(type(x) is not int or x<0 for x in (available_bytes,reserved_bytes)):
            raise ValueError('actual nonnegative available/reserved bytes required')
        for charge in self.charges:
            if self.validate_receipt(charge,self.snapshot) is False:
                raise ValueError('allocation authority rejected budget receipt: '+charge.key)
        steady=sum(c.steady_bytes for c in self.charges)
        phases={}
        for charge in self.charges:
            phases[charge.cold_phase]=phases.get(charge.cold_phase,0)+charge.cold_bytes
        # All simultaneously live temporaries within a phase are added. Only
        # independently certified serial phases may use a max between phases.
        cold=max(phases.values(),default=0) if self.serial else sum(phases.values())
        peak=steady+cold+reserved_bytes
        if peak>available_bytes:
            raise MemoryError(f'resident peak {peak} exceeds actual available {available_bytes}')
        return {'steady_bytes':steady,'cold_peak_bytes':cold,'reserved_bytes':reserved_bytes,
                'peak_bytes':peak,'available_bytes':available_bytes,
                'complete_categories':sorted(REQUIRED_CATEGORIES),
                'receipt_count':len(self.charges),'serial_construction':self.serial}
