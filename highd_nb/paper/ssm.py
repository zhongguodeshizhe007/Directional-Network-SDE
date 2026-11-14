"""Auditable SSMs. Constant motion/acceleration assumptions are explicit.
Sources: Wang et al. (AAP 2024), section 3.2; Jiao et al. (AMAR 2025), Table 4.
No valid non-closing pair is deleted: its TTC/DRAC-derived risk is zero.
"""
from __future__ import annotations
import numpy as np
from scipy.special import log_ndtr


def following_metrics(gap, vf, vl, af=None, al=None, braking=3.924):
    """Vectorised same-lane metrics, SI units. Invalid observations remain NaN.

    MTTC is the smallest positive solution of gap-dv*t-da*t*t/2=0.
    A solution beyond either vehicle's predicted stopping time is not extrapolated
    into reverse motion: that root is marked NaN (unsupported), not safe zero.
    No positive crossing root under the constant-acceleration formula is infinity.
    PSD uses the available braking rate supplied by the study (not an estimated
    friction coefficient). At vf=0 its inverse-risk representation is exactly zero.
    """
    if not np.isfinite(braking) or braking <= 0:
        raise ValueError('braking must be a positive finite m/s^2 value')
    g,vf,vl=np.broadcast_arrays(np.asarray(gap,float),np.asarray(vf,float),np.asarray(vl,float))
    good=np.isfinite(g)&np.isfinite(vf)&np.isfinite(vl)&(g>0)&(vf>=0)&(vl>=0)
    dv=vf-vl
    with np.errstate(divide='ignore',invalid='ignore'):
        ttc=np.where(good,np.where(dv>0,g/dv,np.inf),np.nan)
        inv=np.where(good,np.maximum(dv,0)/g,np.nan)
        drac=np.where(good,np.maximum(dv,0)**2/(2*g),np.nan)
        ipsd=np.where(good,vf**2/(2*braking*g),np.nan)
        psd=np.where(good,np.where(vf>0,2*braking*g/vf**2,np.inf),np.nan)
    out={'ttc':ttc,'inv_ttc':inv,'drac':drac,'psd':psd,'inv_psd':ipsd}
    if af is None or al is None:
        out.update(mttc=np.full_like(g,np.nan),inv_mttc=np.full_like(g,np.nan));return out
    af,al=np.broadcast_arrays(np.asarray(af,float),np.asarray(al,float),g)[:2]
    ok=good&np.isfinite(af)&np.isfinite(al); da=af-al
    disc=dv**2+2*da*g; linear=np.abs(da)<1e-8
    root=np.sqrt(np.maximum(disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=(-dv+root)/da; r2=(-dv-root)/da
        # Stable form of r1 when subtraction would cancel.
        stable=2*g/(dv+root)
        r1=np.where((np.abs(dv+root)>1e-10)&(disc>=0),stable,r1)
        r1=np.where((r1>0)&(disc>=0),r1,np.inf)
        r2=np.where((r2>0)&(disc>=0),r2,np.inf)
        mttc=np.where(linear,ttc,np.minimum(r1,r2))
        stop_f=np.where(af<0,-vf/af,np.inf); stop_l=np.where(al<0,-vl/al,np.inf)
        mttc=np.where(np.isfinite(mttc)&(mttc>np.minimum(stop_f,stop_l)+1e-7),np.nan,mttc)
        mttc=np.where(ok,mttc,np.nan)
        im=np.where(ok,1/mttc,np.nan)
    out.update(mttc=mttc,inv_mttc=im)
    return out


def aabb_ttc(relative_position, relative_velocity, half_extent_sum):
    """Constant-velocity TTC between axis-aligned vehicle boxes in 2D.
    relative_position=target_center-ego_center; relative_velocity=v_target-v_ego.
    This is an axis-aligned highD approximation, NOT the authors' rotated-box code.
    Existing overlap -> 0; no future overlap -> inf; invalid coordinates -> NaN.
    """
    r,v,b=np.broadcast_arrays(np.asarray(relative_position,float),np.asarray(relative_velocity,float),
                              np.asarray(half_extent_sum,float))
    if r.shape[-1]!=2: raise ValueError('last coordinate dimension must be 2')
    valid=np.isfinite(r).all(-1)&np.isfinite(v).all(-1)&np.isfinite(b).all(-1)&(b>0).all(-1)
    static=np.abs(v)<1e-12
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(-b-r)/v; z=(b-r)/v
    entry=np.where(static,np.where(np.abs(r)<=b,-np.inf,np.inf),np.minimum(a,z))
    exit=np.where(static,np.where(np.abs(r)<=b,np.inf,-np.inf),np.maximum(a,z))
    lo=np.maximum(entry.max(-1),0); hi=exit.min(-1)
    return np.where(valid,np.where(hi>=lo,lo,np.inf),np.nan)


def conflict_probability(s,mu,sigma,n=1.):
    """Jiao Eq.(14): [1-F_lognormal(s|context)]**n, not future crash probability."""
    s,mu,sigma,n=np.broadcast_arrays(np.asarray(s,float),np.asarray(mu,float),np.asarray(sigma,float),np.asarray(n,float))
    if np.any((s<0)|(sigma<=0)|(n<1)): raise ValueError('s>=0, sigma>0 and n>=1 required')
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.exp(n*log_ndtr(-(np.log(s)-mu)/sigma))


def conflict_intensity(s,mu,sigma,p=.5):
    """Continuous intensity at probability p, Jiao Eq.(15b), including its .5 limit.
    n<1 is left as n<1 (ordinary interactions); no silent promotion to a conflict.
    """
    if not 0<p<1: raise ValueError('p must be in (0,1)')
    s,mu,sigma=np.broadcast_arrays(np.asarray(s,float),np.asarray(mu,float),np.asarray(sigma,float))
    if np.any((s<0)|(sigma<=0)): raise ValueError('s>=0 and sigma>0 required')
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        return np.log(p)/log_ndtr(-(np.log(s)-mu)/sigma)
