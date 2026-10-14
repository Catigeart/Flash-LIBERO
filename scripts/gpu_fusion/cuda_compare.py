"""Bounded FP64 result comparison; discrete and held outputs remain exact."""
import numpy as np


def compare(a,b,*,rtol=1e-10,atol=1e-11):
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(passed=False,layout=True)
    if a.tobytes()==b.tobytes():return dict(passed=True,byte_equal=True,max_abs=0.)
    if a.dtype.kind!='f':return dict(passed=False,discrete=True)
    finite=np.isfinite(a)&np.isfinite(b)
    matching_special=(np.isnan(a)&np.isnan(b))|((a==b)&~finite)
    error=np.abs(a[finite]-b[finite]);limit=atol+rtol*np.abs(a[finite])
    return dict(passed=bool(np.all(finite|matching_special) and np.all(error<=limit)),
                byte_equal=False,max_abs=float(error.max(initial=0.)),
                max_scaled=float((error/limit).max(initial=0.)),rtol=rtol,atol=atol)
