"""Original fixed-joint transmission and actuation input-only source graph.

No CPU/GPU expected value is read. Direct exact leaf fields are distinct from
the still-pending upstream qpos-root producer neighborhood certificate.
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from source_budget import B, safe_mul, safe_add

CAPS=(1,4,20,128)
MODES=("EAGER","GRAPH","REPLAY","RESET_SELECTED")
NAMES=("original_split_seed4404","ctrl_below_limits","ctrl_above_limits",
       "nan0","nan7","first_bad2_then7","plusinf_clamped","minusinf_clamped",
       "huge_clamped","unclamped_at_MAXVAL","unclamped_above_MAXVAL",
       "unclamped_below_minus_MAXVAL","unclamped_nan","disabled_nan",
       "disabled_unclamped_nan","nan_then_inactive","nan_then_active7",
       "inactive_nan","active7_nan","velocity_gate_error",
       "disabled_inf","disabled_unclamped_inf","world_distinct_badctrl")
FLOAT_FIELDS=("length","moment","velocity","force","force_pre","ctrl_used",
              "qfrc_actuator","consumer_length","consumer_moment",
              "consumer_velocity","consumer_ctrl")
INT_FIELDS=("position_status","velocity_status","status","ctrl_route","force_route",
            "warn_badctrl_count","warn_badctrl_lastinfo","warning_number",
            "warning_lastinfo","warning_reset_status","consumer_active","consumer_vel_status")
FIELDS=FLOAT_FIELDS+INT_FIELDS
SHAPES={f:(9,43) if "moment" in f else (43,) if f=="qfrc_actuator" else (9,)
        for f in FLOAT_FIELDS}
SHAPES.update({f:(9,) if f in ("ctrl_route","force_route") else () for f in INT_FIELDS})
INPUT_FIELDS=("qpos","qvel","ctrl","active","vel_status","flags","warning_reset_mask")
STATIC_FIELDS=("actuator_gear","actuator_gainprm","actuator_biasprm",
               "actuator_ctrlrange","actuator_forcerange","actuator_forcelimited",
               "actuator_biastype","jnt_qposadr","jnt_dofadr","actuator_trnid",
               "dof_parentid","dof_simplenum","dof_Madr","nM","nv")
def key(n,c,mode,f):return f"N{n}__{c}__{mode}__{f}"
def rn(a,b,op):
    x,y=Fraction.from_float(float(a)),Fraction.from_float(float(b))
    return np.float64(float(x*y if op=="mul" else x+y))
def derive(payload,static,prior_number,prior_last):
    n=len(payload["active"]);flags=int(payload["flags"])
    points={f:np.zeros((n,)+SHAPES[f],np.float64) for f in FLOAT_FIELDS}
    ints={f:np.zeros((n,)+SHAPES[f],np.int32) for f in INT_FIELDS}
    ints["warn_badctrl_lastinfo"][:]=-1
    ints["warning_number"][:]=prior_number;ints["warning_lastinfo"][:]=prior_last
    bounds={f:(v.copy(),v.copy()) for f,v in points.items()}
    masks={f:np.ones(v.shape,np.uint8) for f,v in points.items()}
    ctrl_lo,ctrl_hi=static["actuator_ctrlrange"].T
    for w in range(n):
        active=int(payload["active"][w]);vstatus=int(payload["vel_status"][w])
        if active==1:
            for i in range(9):
                a=int(static["jnt_qposadr"][static["actuator_trnid"][i,0]])
                b=int(static["jnt_dofadr"][static["actuator_trnid"][i,0]])
                # Native fixed gear=1, one scalar joint per row. Each lane's
                # other products are exact finite zeros, so copies are exact.
                points["length"][w,i]=payload["qpos"][w,a]
                points["moment"][w,i,b]=1.
                points["velocity"][w,i]=payload["qvel"][w,b]
        elif active!=0:
            ints["position_status"][w]=1;ints["velocity_status"][w]=1
        for f in ("length","moment","velocity"):
            bounds[f][0][w]=points[f][w];bounds[f][1][w]=points[f][w]
        if vstatus:
            ints["velocity_status"][w]=vstatus
        for dst,src in (("consumer_length","length"),("consumer_moment","moment"),
                        ("consumer_velocity","velocity")):
            points[dst][w]=points[src][w]
            bounds[dst][0][w]=bounds[src][0][w];bounds[dst][1][w]=bounds[src][1][w]
        points["consumer_ctrl"][w]=payload["ctrl"][w]
        fin=np.isfinite(payload["ctrl"][w])
        bounds["consumer_ctrl"][0][w]=np.where(fin,payload["ctrl"][w],0.)
        bounds["consumer_ctrl"][1][w]=np.where(fin,payload["ctrl"][w],0.)
        # Observed input NaN/inf payload copy is authorized, not a numeric output.
        masks["consumer_ctrl"][w]=fin.astype(np.uint8)
        ints["consumer_active"][w]=active
        ints["consumer_vel_status"][w]=ints["velocity_status"][w]
        if active!=1:
            ints["status"][w]=0 if active==0 else 1;continue
        if ints["velocity_status"][w]!=0:
            ints["status"][w]=2;continue
        if flags & 1024:continue
        used=payload["ctrl"][w].copy()
        if not(flags & 128):
            for i in range(9):
                if used[i]<ctrl_lo[i]:
                    used[i]=ctrl_lo[i];ints["ctrl_route"][w,i]=-1
                elif used[i]>ctrl_hi[i]:
                    used[i]=ctrl_hi[i];ints["ctrl_route"][w,i]=1
        bad=next((i for i,x in enumerate(used) if np.isnan(x) or abs(x)>1e10),None)
        if bad is not None:
            used[:]=0.
            ints["warn_badctrl_count"][w]=1;ints["warn_badctrl_lastinfo"][w]=bad
            ints["warning_number"][w]+=1;ints["warning_lastinfo"][w]=bad
        points["ctrl_used"][w]=used
        bounds["ctrl_used"][0][w]=used;bounds["ctrl_used"][1][w]=used
        for i,u in enumerate(used):
            g=static["actuator_gainprm"][i,0]
            fi=safe_mul(B.point(g),B.point(u)) if g!=1 else B.point(u)
            fp=rn(g,u,"mul")
            if static["actuator_biastype"][i]==1:
                b0,b1,b2=static["actuator_biasprm"][i,:3]
                l,v=points["length"][w,i],points["velocity"][w,i]
                bi=safe_add(safe_add(B.point(b0),safe_mul(B.point(b1),B.point(l))),
                            safe_mul(B.point(b2),B.point(v)))
                fi=safe_add(fi,bi)
                bp=rn(rn(b0,rn(b1,l,"mul"),"add"),rn(b2,v,"mul"),"add")
                fp=rn(fp,bp,"add")
            points["force_pre"][w,i]=fp
            bounds["force_pre"][0][w,i],bounds["force_pre"][1][w,i]=fi
            if static["actuator_forcelimited"][i]:
                lo,hi=static["actuator_forcerange"][i]
                # Exact-rational RN source-point establishes original strict
                # predicate even at exact boundaries; interval remains outward.
                if fp<lo:ints["force_route"][w,i]=-1
                elif fp>hi:ints["force_route"][w,i]=1
                fp=min(max(fp,lo),hi)
                fi=(min(max(fi[0],lo),hi),min(max(fi[1],lo),hi))
            points["force"][w,i]=fp
            bounds["force"][0][w,i],bounds["force"][1][w,i]=fi
            b=int(static["jnt_dofadr"][static["actuator_trnid"][i,0]])
            points["qfrc_actuator"][w,b]=fp
            bounds["qfrc_actuator"][0][w,b],bounds["qfrc_actuator"][1][w,b]=fi
    points.update(ints)
    return points,bounds,masks
