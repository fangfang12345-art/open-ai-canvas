"""Clean-observation continuous tensor field and screened graph harmonic solve.

Build through watermark/native_build.py. The NumPy oracle is available through
R02_DISABLE_NATIVE=1; no prior-frame state is retained by either implementation.
"""
import ctypes
import os
from pathlib import Path
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_tensor')
try:
    NATIVE = ctypes.CDLL(str(LIBRARY))
    U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    NATIVE.tensor_solve.argtypes = [U8, U8, U8, U8, F32, F32, F32, F32,
                                  ctypes.c_int, ctypes.c_int, U8, F64]
    NATIVE.tensor_solve.restype = ctypes.c_int
    NATIVE.tensor_allocator_last_stats.argtypes = [F64]
    NATIVE.tensor_allocator_last_stats.restype = None
except (OSError, AttributeError):
    NATIVE = None


def tensor_field(image, exclusion, write):
    clean = cv2.erode((~exclusion).astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * .5
    gy[1:-1] = (gray[2:] - gray[:-2]) * .5
    distance = cv2.distanceTransform(write.astype(np.uint8), cv2.DIST_L2, 5)
    sigma = float(np.clip(1.5*np.percentile(distance[write], 85), 3, 12))
    radius = int(np.ceil(3*sigma))
    kernel = cv2.getGaussianKernel(2*radius+1, sigma, cv2.CV_32F)
    def smooth(a):
        return cv2.sepFilter2D(a, -1, kernel, kernel, borderType=cv2.BORDER_CONSTANT)
    support = smooth(clean)
    jxx, jxy, jyy = [smooth(clean*v)/np.maximum(support, 1e-6)
                     for v in (gx*gx, gx*gy, gy*gy)]
    a = jxx+jyy
    coherence = np.clip(np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(a+1e-6), 0, 1)
    confidence = np.clip((support-.10)/.25, 0, 1)
    flatness = np.exp(-a/9)
    c = confidence*(1-flatness)*coherence
    q = confidence*(flatness+(1-flatness)*coherence**2)
    theta = .5*np.arctan2(2*jxy, jxx-jyy)
    return tuple(np.ascontiguousarray(v, np.float32) for v in (c, q, -np.sin(theta), np.cos(theta)))


def numpy_solve(base, unknown, halo, c, q, tx, ty):
    """Vectorized independent graph/PCG oracle, intentionally slower than C++."""
    h, w = unknown.shape; size=h*w
    pixels=np.flatnonzero(unknown); n=len(pixels)
    index=np.full(size, -1, np.int32);index[pixels]=np.arange(n)
    sources=np.flatnonzero(halo); y,x=np.divmod(sources,w)
    keys=[]; values=[]
    def add(xx,yy,weight):
        good=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(weight>0)
        p=sources[good];z=(yy[good]*w+xx[good]).astype(np.int64);v=weight[good]
        good=(p!=z)&((index[p]>=0)|(index[z]>=0))
        keys.append(np.minimum(p[good],z[good])*size+np.maximum(p[good],z[good]));values.append(v[good]*.5)
    cv=c.ravel()[sources]; axis=(1-cv+.10)/4
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):add(x+dx,y+dy,axis)
    for sign in (-1,1):
        px=x+sign*3*tx.ravel()[sources].astype(float);py=y+sign*3*ty.ravel()[sources].astype(float)
        x0=np.floor(px).astype(int);y0=np.floor(py).astype(int);dx=px-x0;dy=py-y0
        for ox,oy,v in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):add(x0+ox,y0+oy,cv*v/2)
    keys=np.concatenate(keys);values=np.concatenate(values)
    unique,inverse=np.unique(keys,return_inverse=True);weights=np.bincount(inverse,weights=values)
    p,z=np.divmod(unique,size);a=index[p];b=index[z]
    degree=np.bincount(np.concatenate((a[a>=0],b[b>=0])),weights=np.concatenate((weights[a>=0],weights[b>=0])),minlength=n)
    fixed=np.zeros((n,3)); colors=base.reshape(-1,3).astype(float)
    for src,dst,weight in ((a[b<0],z[b<0],weights[b<0]),(b[a<0],p[a<0],weights[a<0])):
        np.add.at(fixed,src,weight[:,None]*colors[dst])
    both=(a>=0)&(b>=0);aa=a[both];bb=b[both];ww=weights[both]
    labels=np.arange(n)
    while True:
        old=labels.copy();np.minimum.at(labels,aa,old[bb]);np.minimum.at(labels,bb,old[aa]);labels=labels[labels]
        if np.array_equal(old,labels):break
    _,groups=np.unique(labels,return_inverse=True);ng=groups.max()+1
    confidence=q.ravel()[pixels];lam=degree*(.05+((1-confidence)/np.maximum(confidence,.001))**2);diag=degree+lam
    u=colors[pixels].copy();rhs=lam[:,None]*u+fixed
    def mat(v):
        out=diag[:,None]*v;np.add.at(out,aa,-ww[:,None]*v[bb]);np.add.at(out,bb,-ww[:,None]*v[aa]);return out
    def reduce(v):
        out=np.zeros((ng,3));np.add.at(out,groups,v);return out
    initial=float(np.sum(.5*u*mat(u)-rhs*u))
    single=np.bincount(groups)[groups]==1;u[single]=rhs[single]/diag[single,None]
    r=rhs-mat(u);z=r/diag[:,None];d=z.copy();rz=reduce(r*z);norm=np.maximum(np.sqrt(reduce(rhs*rhs)),1)
    limit=4+min(20,2000000//max(len(aa),1));iters=np.zeros((ng,3),int)
    for _ in range(limit):
        live=np.sqrt(reduce(r*r))/norm>1e-4
        if not live.any():break
        d*=live[groups];ad=mat(d);pap=reduce(d*ad)
        if np.any(pap[live]<=0):raise ArithmeticError('Nonpositive PCG denominator')
        alpha=np.divide(rz,pap,out=np.zeros_like(rz),where=live);iters+=live
        u+=alpha[groups]*d;r-=alpha[groups]*ad;z=r/diag[:,None];new=reduce(r*z)
        beta=np.divide(new,rz,out=np.zeros_like(rz),where=live);d=z+beta[groups]*d;rz=new
    final=float(np.sum(.5*u*mat(u)-rhs*u));res=np.sqrt(reduce((rhs-mat(u))**2))/norm
    if not np.isfinite(u).all() or final>initial+1e-9*max(abs(initial),1):raise ArithmeticError('Invalid PCG energy')
    output=base.copy();output.reshape(-1,3)[pixels]=np.clip(np.rint(u),0,255).astype(np.uint8)
    return output,np.array([n,len(unique),len(aa),ng,limit,0,0,iters.max(),res.max(),(res>1e-4).sum(),initial,final],float)


def repair(image, exclusion, write, base, flat):
    started=time.perf_counter();image=np.ascontiguousarray(image);base=np.ascontiguousarray(base)
    exclusion=np.asarray(exclusion,bool);write=np.asarray(write,bool)
    stats=dict(tensor_seconds=0.,graph_seconds=0.,solve_seconds=0.,solve_vertices=0,
               graph_edges=0,unknown_edges=0,components=0,iteration_limit=0,iterations=0,
               terminal_residual=0.,unconverged_channel_components=0,q_mean=0.,
               filled_pixels=0,failure_fallbacks=0,native=NATIVE is not None)
    if not write.any():return base.copy(),stats
    c,q,tx,ty=tensor_field(image,exclusion,write)
    interior=cv2.erode(write.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    unknown=exclusion & ~(write & ~interior) & (flat==0) & (q>0)
    stats['tensor_seconds']=time.perf_counter()-started
    if not unknown.any():return base.copy(),stats
    halo=cv2.dilate(unknown.astype(np.uint8),np.ones((9,9),np.uint8))
    stats['q_mean']=float(q[unknown].mean()); output=base.copy();raw=np.zeros(12,float)
    try:
        if NATIVE is not None and os.environ.get('R02_DISABLE_NATIVE')!='1':
            status=NATIVE.tensor_solve(image,base,np.ascontiguousarray(unknown,np.uint8),halo,c,q,tx,ty,*unknown.shape,output,raw)
            allocator=np.zeros(5,float);NATIVE.tensor_allocator_last_stats(allocator)
            stats['graph_allocator']=dict(zip(('requests','system_allocations','peak_bytes','fallbacks','enabled'),map(int,allocator)))
            if status:raise ArithmeticError('Native status '+str(status))
        else:
            stats['native']=False;begin=time.perf_counter();output,raw=numpy_solve(base,unknown,halo,c,q,tx,ty);raw[6]=time.perf_counter()-begin
    except (ArithmeticError,MemoryError) as exc:
        output=base.copy();stats.update(failure_fallbacks=1,fallback_reason=str(exc))
    for name,value in zip(('solve_vertices','graph_edges','unknown_edges','components','iteration_limit','graph_seconds','solve_seconds','iterations','terminal_residual','unconverged_channel_components','initial_energy','final_energy'),raw):
        stats[name]=float(value) if name.endswith(('seconds','residual','energy')) else int(value)
    stats['filled_pixels']=int(np.any(output!=base,axis=2)[write].sum())
    return output,stats


if __name__=='__main__':
    raise SystemExit('Use watermark/native_build.py to build portable kernels outside the source tree')
