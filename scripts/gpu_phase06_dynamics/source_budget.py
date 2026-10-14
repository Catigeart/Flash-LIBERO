"""Input-only D09 -> D02 -> native sparse factor/solve interval graph.

This is a finite fixed-input graph, not a qpos-rooted producer theorem. It
never reads an expected or device answer. The three RN nodes use the original
exact rational lemma; all downstream source operations propagate outward
endpoints. Strict source pivot predicates are proved before flooring.
"""
from __future__ import annotations
import importlib.util
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "p06d_original_force_interval_lemma",
    ROOT / "scripts/gpu_smooth_acceleration_chain/budget.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

FLOAT_FIELDS = ("smooth", "qfrc_smooth", "qacc_smooth", "ld", "inv", "sqrt_inv")
INT_FIELDS = ("status", "entry_status", "d02_status", "solve_status",
              "warning_count", "warning_last", "pivot")
FIELDS = FLOAT_FIELDS + INT_FIELDS
CAPS = (1, 4, 20, 128)
OLD_NAMES = ("synthetic_repair25_original", "below_floor", "at_floor",
             "above_floor", "zero", "negative", "three_rn_original",
             "all_37_bodies_force_torque", "source_body_association", "signed_zero")
NAMES = OLD_NAMES + ("original_rn_case0", "original_rn_case1", "original_rn_case2",
                     "original_rn_case3", "original_xfrc_seed4402",
                     "original_capacity_seed", "original_capacity_changed",
                     "original_error_worlds", "original_error_relaunch",
                     "original_healthy_reinput")
INPUT_FIELDS = ("active", "passive", "bias", "applied", "actuator", "xfrc",
                "xipos", "subtree_com", "cdof", "qm")


def key(n, name, field):
    return "N%d__%s__%s" % (n, name, field)


def zero():
    return B.point(np.float64(0))


def safe_mul(a, b):
    # An exact zero factor and finite other factor have exact numerical zero.
    # Avoid decorrelating untouched sparse coefficients through subnormal halos.
    if (a[0] == 0 and a[1] == 0) or (b[0] == 0 and b[1] == 0):
        return zero()
    return B.mul(a, b)


def safe_sub(a, b):
    if a[0] == a[1] == b[0] == b[1]:
        return zero()
    if b[0] == b[1] == 0:
        return a
    return B.sub(a, b)


def safe_add(a, b):
    if b[0] == b[1] == 0:
        return a
    if a[0] == a[1] == 0:
        return b
    return B.add(a, b)


def projection(payload, static, seed):
    """Original applyFT: bodies ascending, force then torque, ancestor routing."""
    n, nv = seed.shape
    lo, hi = seed.copy(), seed.copy()
    ancestor = np.zeros((38, nv), bool)
    parent = static["body_parentid"]
    for body in range(1, 38):
        cursor = body
        while cursor and static["body_dofnum"][cursor] == 0:
            cursor = int(parent[cursor])
        if cursor:
            j = int(static["body_dofadr"][cursor] + static["body_dofnum"][cursor] - 1)
            while j >= 0:
                ancestor[body, j] = True
                j = int(static["dof_parentid"][j])
    for w in range(n):
        for body in range(1, 38):
            fx = payload["xfrc"][w, body]
            if not np.any(fx != 0):
                continue
            root = int(static["body_rootid"][body])
            offset = [safe_sub(B.point(payload["xipos"][w, body, c]),
                            B.point(payload["subtree_com"][w, root, c])) for c in range(3)]
            for j in range(nv):
                jr = [zero() for _ in range(3)]
                jp = [zero() for _ in range(3)]
                if ancestor[body, j]:
                    jr = [B.point(v) for v in payload["cdof"][w, j, :3]]
                    for c, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                        jp[c] = safe_add(B.point(payload["cdof"][w, j, c+3]),
                                     safe_sub(safe_mul(jr[a], offset[b]),
                                           safe_mul(jr[b], offset[a])))
                force, torque = zero(), zero()
                for c in range(3):
                    if fx[c] != 0:
                        force = safe_add(force, safe_mul(jp[c], B.point(fx[c])))
                value = safe_add((lo[w,j], hi[w,j]), force)
                for c in range(3):
                    if fx[c+3] != 0:
                        torque = safe_add(torque, safe_mul(jr[c], B.point(fx[c+3])))
                value = safe_add(value, torque)
                lo[w,j], hi[w,j] = value
    return lo, hi


def factor_solve(qm, rhs, static):
    """Native descending sparse elimination with independently derived pivots.

    No CPU pivot or expected factor is an input. A straddling interval is
    unsupported, preserving the original critical-neighborhood obligation.
    """
    n, nm = qm.shape
    madr, parent = static["dof_Madr"], static["dof_parentid"]
    nv = len(madr)
    ldlo, ldhi = qm.copy(), qm.copy()
    piv = np.zeros((n, nv), np.int32)
    warnings = np.zeros(n, np.int32)
    last = np.full(n, -1, np.int32)
    for w in range(n):
        for k in range(nv-1, -1, -1):
            kk = int(madr[k])
            diag = (ldlo[w,kk], ldhi[w,kk])
            if diag[1] < 1e-15:
                piv[w,k] = 1
                warnings[w] += 1
                last[w] = k
                diag = (1e-15, 1e-15)
                ldlo[w,kk] = ldhi[w,kk] = 1e-15
            elif diag[0] < 1e-15:
                raise ValueError("critical native pivot interval at world%d dof%d" % (w,k))
            ki = kk+1
            i = int(parent[k])
            while i >= 0:
                coeff = (ldlo[w,ki], ldhi[w,ki])
                tmp = zero() if coeff[0] == coeff[1] == 0 else B.div(coeff, diag)
                count = int(madr[i+1]-madr[i]) if i+1 < nv else nm-int(madr[i])
                for a in range(count):
                    dest, src = int(madr[i])+a, ki+a
                    change = safe_mul((ldlo[w,src],ldhi[w,src]), B.neg(tmp))
                    if change[0] != 0 or change[1] != 0:
                        value = B.add((ldlo[w,dest],ldhi[w,dest]), change)
                        ldlo[w,dest],ldhi[w,dest] = value
                ldlo[w,ki],ldhi[w,ki] = tmp
                ki += 1
                i = int(parent[i])
    invlo, invhi = np.empty((n,nv)), np.empty((n,nv))
    sqlo, sqhi = np.empty((n,nv)), np.empty((n,nv))
    for w in range(n):
        for i in range(nv):
            diag = (ldlo[w,int(madr[i])],ldhi[w,int(madr[i])])
            invlo[w,i],invhi[w,i] = B.div(B.point(1.),diag)
            sqlo[w,i],sqhi[w,i] = B.div(B.point(1.),B.sqrt(diag))
    sollo, solhi = rhs[0].copy(), rhs[1].copy()
    for w in range(n):
        for i in range(nv-1,-1,-1):
            a = int(madr[i])+1
            j = int(parent[i])
            while j >= 0:
                xi = (sollo[w,i],solhi[w,i])
                old = (sollo[w,j],solhi[w,j])
                coeff = (ldlo[w,a],ldhi[w,a])
                if not (xi[0] == xi[1] == 0 or coeff[0] == coeff[1] == 0):
                    candidate = B.sub(old,safe_mul(coeff,xi))
                    if xi[0] <= 0 <= xi[1]:
                        candidate = (min(old[0],candidate[0]),max(old[1],candidate[1]))
                    sollo[w,j],solhi[w,j] = candidate
                a += 1
                j = int(parent[j])
        for i in range(nv):
            sollo[w,i],solhi[w,i] = safe_mul(
                (sollo[w,i],solhi[w,i]),(invlo[w,i],invhi[w,i]))
        for i in range(nv):
            a = int(madr[i])+1
            j = int(parent[i])
            while j >= 0:
                change = safe_mul((ldlo[w,a],ldhi[w,a]),(sollo[w,j],solhi[w,j]))
                if change[0] != 0 or change[1] != 0:
                    sollo[w,i],solhi[w,i] = B.sub((sollo[w,i],solhi[w,i]),change)
                a += 1
                j = int(parent[j])
    return {"ld":(ldlo,ldhi),"inv":(invlo,invhi),"sqrt_inv":(sqlo,sqhi),
            "qacc_smooth":(sollo,solhi)}, piv, warnings, last


def derive(payload, static):
    """Derive all thirteen declared outputs from frozen source inputs only."""
    active = payload["active"]
    n = len(active)
    healthy = active == 1
    force_finite = np.ones(n, bool)
    for f in ("passive","bias","applied","actuator"):
        force_finite &= np.isfinite(payload[f]).all(axis=1)
    entry = np.where((active != 0)&(active != 1),1,
                     np.where(healthy&~force_finite,3,0)).astype(np.int32)
    # Source invalid-input zero publications do not provide native equivalence.
    p = {k:v.copy() for k,v in payload.items()}
    for f in ("passive","bias","applied","actuator"):
        p[f][~force_finite] = 0.
    smooth = B.force_rn(p["passive"],p["bias"],p["applied"],p["actuator"])
    smooth[~healthy | (entry != 0)] = 0.
    rhs = projection(p,static,smooth)
    rhs = (np.where(healthy[:,None] & (entry[:,None] == 0),rhs[0],0.),
           np.where(healthy[:,None] & (entry[:,None] == 0),rhs[1],0.))
    bounds, pivot, count, last = factor_solve(p["qm"],rhs,static)
    bounds["smooth"] = (smooth.copy(),smooth.copy())
    bounds["qfrc_smooth"] = rhs
    status = np.where(count>0,25,0).astype(np.int32)
    solve = np.where(count>0,5,0).astype(np.int32)
    d02 = np.where((active != 0)&(active != 1),1,0).astype(np.int32)
    status[entry != 0] = entry[entry != 0]
    status[active == 0] = 0
    solve[(active != 0)&(active != 1)] = 1
    for f,(lo,hi) in bounds.items():
        if f in ("smooth","qfrc_smooth","qacc_smooth"):
            mask = ~healthy | (entry != 0)
        else:
            mask = ~healthy
        lo[mask] = hi[mask] = 0.
    pivot[~healthy] = 0
    count[~healthy] = 0
    last[~healthy] = -1
    integer = {"status":status,"entry_status":entry,"d02_status":d02,
               "solve_status":solve,"warning_count":count,
               "warning_last":last,"pivot":pivot}
    return bounds, integer


def residual_prepare(payload, static, bounds, integers):
    """Original D09 fixed-input residual envelope, source-repaired diagonal.

    All current original constructed qM controls are diagonal. No non-diagonal
    repair or upstream qM perturbation theorem is inferred from this unit.
    The envelope uses only source inputs and propagated intervals, never an
    observed CPU/GPU residual. Every positive arithmetic allowance rounds up.
    """
    n=len(payload["active"]);madr=static["dof_Madr"];nv=len(madr)
    off=payload["qm"].copy();off[:,madr]=0.
    if np.any(off!=0):raise ValueError("original diagonal residual unit only; general producer proof remains required")
    diag=payload["qm"][:,madr].copy()
    diag=np.maximum(diag,1e-15)
    xl,xh=bounds["qacc_smooth"];bl,bh=bounds["qfrc_smooth"]
    xm=.5*xl+.5*xh;bm=.5*bl+.5*bh
    rx=np.nextafter(np.maximum(np.abs(xl-xm),np.abs(xh-xm)).max(axis=1),np.inf)
    rb=np.nextafter(np.maximum(np.abs(bl-bm),np.abs(bh-bm)).max(axis=1),np.inf)
    norm=np.max(np.abs(diag),axis=1)
    center=np.max(np.abs(diag*xm-bm),axis=1)
    xn=np.maximum(np.abs(xl),np.abs(xh)).max(axis=1)
    gamma=np.nextafter((43*2.**-53)/(1-43*2.**-53),np.inf)
    up=lambda a:np.nextafter(a,np.inf)
    rounding=up(gamma*up(up(norm*up(np.max(np.abs(xm),axis=1)+xn))+
                            up(np.max(np.abs(bm),axis=1)+rb)))
    allowance=up(up(center+up(norm*rx))+up(rb+rounding))
    # Underflow spacing eta is included explicitly in the source operation law.
    allowance=up(allowance+86*np.nextafter(0.,1.))
    mask=((payload["active"]==1)&(integers["entry_status"]==0)).astype(np.uint8)
    return diag,allowance,mask
