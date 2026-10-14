"""D09 directed intervals: force order into accepted N06 sparse loop graph.

Factor/solve loop transcription adapted from the accepted N06 independent
budget.py (baseline eae5fc88). Both source and device are enclosed from
exact input points; each arithmetic node gets outward binary64 endpoints.
"""

from fractions import Fraction
import struct

import numpy as np


def _finite(pair):
    if not np.isfinite(pair[0]).all() or not np.isfinite(pair[1]).all():
        raise ValueError('unsupported_numeric_range')
    if np.any(pair[0] > pair[1]):
        raise ValueError('unsupported_numeric_range: inverted interval')
    return pair


def point(x):
    x = np.asarray(x, np.float64)
    if not np.isfinite(x).all():
        raise ValueError('unsupported_numeric_range: nonfinite point')
    return x, x


def neg(a):
    return -a[1], -a[0]


def add(a,b):
    return _finite((np.nextafter(a[0]+b[0],-np.inf),
                    np.nextafter(a[1]+b[1],np.inf)))


def sub(a,b):
    return _finite((np.nextafter(a[0]-b[1],-np.inf),
                    np.nextafter(a[1]-b[0],np.inf)))


def mul(a,b):
    v = [a[i]*b[j] for i in (0,1) for j in (0,1)]
    return _finite((np.nextafter(np.minimum.reduce(v),-np.inf),
                    np.nextafter(np.maximum.reduce(v),np.inf)))


def div(a,b):
    if np.any((b[0] <= 0)&(b[1] >= 0)):
        raise ValueError('division interval crosses zero')
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        v = [a[i]/b[j] for i in (0,1) for j in (0,1)]
    return _finite((np.nextafter(np.minimum.reduce(v),-np.inf),
                    np.nextafter(np.maximum.reduce(v),np.inf)))


def sqrt(a):
    if np.any(a[0] < 0):
        raise ValueError('sqrt interval negative')
    return _finite((np.maximum(0,np.nextafter(np.sqrt(a[0]),-np.inf)),
                    np.nextafter(np.sqrt(a[1]),np.inf)))


def contains(a, x):
    _finite(a)
    return (np.isfinite(x).all() and np.all(a[0] <= x) and np.all(x <= a[1]))


def _rn_fraction(value):
    """Round an exact rational to binary64, nearest with ties to even.

    The integer quotient and remainder decide the rounding bit.  In
    particular, cancellation is resolved before any interval widening.
    """
    if not value:
        return 0.0
    negative = value < 0
    n, d = abs(value.numerator), value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < (d << exponent) if exponent >= 0 else (n << -exponent) < d):
        exponent -= 1
    shift = 52 - exponent if exponent >= -1022 else 1074
    numerator, denominator = ((n << shift), d) if shift >= 0 else (n, d << -shift)
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient & 1):
        quotient += 1
    if exponent >= -1022:
        if quotient == 1 << 53:
            quotient >>= 1
            exponent += 1
        if exponent > 1023:
            raise ValueError('unsupported_numeric_range: force overflow')
        bits = ((exponent + 1023) << 52) | (quotient - (1 << 52))
    else:
        bits = quotient  # 2**52 naturally encodes the smallest normal.
    if negative:
        bits |= 1 << 63
    return struct.unpack('>d',struct.pack('>Q',bits))[0]


def _rn_node(x, y, subtract=False):
    # IEEE 754 RN gives -0 only when both zero operands of the effective
    # addition are -0.  Exact cancellation of nonzero operands gives +0.
    rhs = -float(y) if subtract else float(y)
    lhs = float(x)
    exact = Fraction(*lhs.as_integer_ratio()) + Fraction(*rhs.as_integer_ratio())
    if not exact and lhs == 0.0 and rhs == 0.0 and np.signbit(lhs) and np.signbit(rhs):
        return -0.0
    return _rn_fraction(exact)


def force_rn(passive, bias, applied, actuator):
    """Independent exact-rational replay of the three source RN nodes."""
    if any(np.shape(x) != np.shape(passive) or np.asarray(x).dtype != np.float64
           for x in (bias, applied, actuator)) or np.asarray(passive).dtype != np.float64:
        raise TypeError('force input schema')
    values = [np.asarray(x) for x in (passive,bias,applied,actuator)]
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError('unsupported_numeric_range: nonfinite force input')
    result = np.empty(values[0].shape,np.float64)
    for i,(p,b,a,c) in enumerate(zip(*(x.flat for x in values))):
        t0 = _rn_node(p,b,True)
        t1 = _rn_node(t0,a)
        result.flat[i] = _rn_node(t1,c)
    return result


def force_bounds(passive, bias, applied, actuator):
    force = force_rn(passive,bias,applied,actuator)
    return point(force)


def make(rows, static, rhs_bounds):
    n = len(rows['qm'])
    madr = static['dof_Madr']
    parent = static['dof_parentid']
    nv, nm = len(madr), rows['qm'].shape[1]
    if rows['qm'].shape != (n, nm) or rows['rhs'].shape != (n, 4, nv):
        raise TypeError('reference mass/RHS schema')
    point(rows['qm']); _finite(rhs_bounds)
    if any(np.shape(side) != rows['rhs'].shape for side in rhs_bounds):
        raise TypeError('RHS interval schema')
    ld = [rows['qm'].copy(), rows['qm'].copy()]
    pivot = np.asarray(rows['pivot'],np.int32).copy()
    stable = np.zeros((n,nv),np.int32)
    for k in range(nv-1,-1,-1):
        kk = int(madr[k])
        below = ld[1][:,kk] < 1e-15
        above = ld[0][:,kk] >= 1e-15
        stable[:,k] = below.astype(np.int32) + above.astype(np.int32)
        if np.any(pivot[:,k]):
            mask = pivot[:,k].astype(bool)
            ld[0][mask,kk] = 1e-15
            ld[1][mask,kk] = 1e-15
        ki = kk+1
        i = int(parent[k])
        while i >= 0:
            tmp = div((ld[0][:,ki],ld[1][:,ki]),
                      (ld[0][:,kk],ld[1][:,kk]))
            cnt = int(madr[i+1]-madr[i]) if i < nv-1 else nm-int(madr[i])
            for a in range(cnt):
                dest = int(madr[i])+a
                src = ki+a
                value = add((ld[0][:,dest],ld[1][:,dest]),
                            mul((ld[0][:,src],ld[1][:,src]),neg(tmp)))
                ld[0][:,dest],ld[1][:,dest] = value
            ld[0][:,ki],ld[1][:,ki] = tmp
            ki += 1
            i = int(parent[i])
    inv = [np.empty((n,nv),np.float64),np.empty((n,nv),np.float64)]
    sqrt_inv = [np.empty((n,nv),np.float64),np.empty((n,nv),np.float64)]
    for i in range(nv):
        diag = (ld[0][:,int(madr[i])],ld[1][:,int(madr[i])])
        a = div(point(1.0),diag)
        b = div(point(1.0),sqrt(diag))
        inv[0][:,i],inv[1][:,i] = a
        sqrt_inv[0][:,i],sqrt_inv[1][:,i] = b
    active_rhs = np.arange(4)[None,:] < rows['count'][:,None]
    sol = [np.where(active_rhs[:,:,None],side,0.0).copy() for side in rhs_bounds]
    for i in range(nv-1,-1,-1):
        a = int(madr[i])+1
        j = int(parent[i])
        while j >= 0:
            active = np.arange(4)[None,:] < rows['count'][:,None]
            xi = (sol[0][:,:,i],sol[1][:,:,i])
            old = (sol[0][:,:,j],sol[1][:,:,j])
            candidate = sub(old,mul((ld[0][:,a,None],ld[1][:,a,None]),xi))
            exact_zero = (xi[0] == 0)&(xi[1] == 0)
            possible_zero = (xi[0] <= 0)&(xi[1] >= 0)
            lo = np.where(exact_zero,old[0],np.where(possible_zero,
                          np.minimum(old[0],candidate[0]),candidate[0]))
            hi = np.where(exact_zero,old[1],np.where(possible_zero,
                          np.maximum(old[1],candidate[1]),candidate[1]))
            sol[0][:,:,j] = np.where(active,lo,old[0])
            sol[1][:,:,j] = np.where(active,hi,old[1])
            a += 1
            j = int(parent[j])
    for i in range(nv):
        v = mul((sol[0][:,:,i],sol[1][:,:,i]),
                (inv[0][:,i,None],inv[1][:,i,None]))
        active = np.arange(4)[None,:] < rows['count'][:,None]
        sol[0][:,:,i] = np.where(active,v[0],sol[0][:,:,i])
        sol[1][:,:,i] = np.where(active,v[1],sol[1][:,:,i])
    for i in range(nv):
        a = int(madr[i])+1
        j = int(parent[i])
        while j >= 0:
            v = sub((sol[0][:,:,i],sol[1][:,:,i]),
                    mul((ld[0][:,a,None],ld[1][:,a,None]),
                        (sol[0][:,:,j],sol[1][:,:,j])))
            active = np.arange(4)[None,:] < rows['count'][:,None]
            sol[0][:,:,i] = np.where(active,v[0],sol[0][:,:,i])
            sol[1][:,:,i] = np.where(active,v[1],sol[1][:,:,i])
            a += 1
            j = int(parent[j])
    return {'ld':tuple(ld),'inv':tuple(inv),'sqrt_inv':tuple(sqrt_inv),
            'solution':tuple(sol),'pivot':pivot,'stable':stable,
            'madr':np.asarray(madr,np.int32).copy(),
            'parent':np.asarray(parent,np.int32).copy()}
