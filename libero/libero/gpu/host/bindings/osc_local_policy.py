"""Cold selection for the measured OSC function scopes; no GPU imports.

The default returns the caller's original function. A positive component is
independent of the held controller, and the two task choices are alternatives.
"""
from types import MappingProxyType

CONTRACTS = MappingProxyType({
    'OSC01': 'controlled-osc-held-matrices-seeded-v1',
    'OSC02': 'osc-mass7-J6x7-X7x6-A6x6-v1',
    'OSC03': 'osc-null-raw-J6x7-X7x6-L6x6-v7-v1',
    'OSC04': 'osc-original-gate-native43-immutable-ancestor-csr-v1',
    'OSC06': 'osc-ordered-task6-3-3-v1',
    'OSC07': 'osc-certified-task6-3-3-original-pinv-fallback-v1',
    'OSC08': 'osc-full392-fp64-seed-and-two-masks-v1',
})
POSITIVE_WORLDS = MappingProxyType({key: ((1,) if key == 'OSC03' else (1, 2))
                                   for key in CONTRACTS})

def select_scope(algorithm_id, nworld, architecture, *, opt_in=False, contract=None):
    """Cold opt-in only. Unmeasured/negative/mixed scopes keep the original.

    SM120, CTA64 and W1/W2 are the measured hardware/work domains. Other
    hardware/world sizes are ordinary fallbacks, not external blockers.
    """
    if type(opt_in) is not bool:
        raise ValueError('OSC local opt_in must be a bool')
    if type(nworld) is not int or nworld < 1:
        raise ValueError('OSC local world count must be a positive int')
    if type(architecture) is not int:
        raise ValueError('OSC local architecture must be an int')
    if algorithm_id not in CONTRACTS and algorithm_id not in ('OSC05', 'OSC09'):
        raise ValueError('unknown OSC local scope')
    choice = {'algorithm_id': algorithm_id, 'nworld': nworld,
              'architecture': architecture, 'route': 'original',
              'contract': contract, 'cache_extension_policy': 'never',
              'cadence_change': False, 'combined_gain': False}
    if algorithm_id == 'OSC05':
        choice['reason'] = 'negative cold-address scope; original gather'
    elif algorithm_id == 'OSC09':
        choice['reason'] = 'negative forced-miss scope; cache extension never'
    elif not opt_in:
        choice['reason'] = 'default original function'
    elif architecture != 120 or nworld not in POSITIVE_WORLDS[algorithm_id]:
        choice['reason'] = 'outside positive hardware/world scope or mixed OSC03 W2'
    elif contract != CONTRACTS[algorithm_id]:
        choice['reason'] = 'consumer contract does not match this independent function'
    else:
        choice.update(route='candidate', reason='explicit matching local function')
    return MappingProxyType(choice)
