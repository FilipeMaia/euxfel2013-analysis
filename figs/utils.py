import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from geom import pixel_maps_from_geometry_file, apply_geom_ij_yx

def plot_bg_full(b, gx, gy):
    fig = plt.figure(figsize=[20,20])
    ax = fig.gca()
    lt = 5
    mm = apply_geom_ij_yx( (gy,gx), ~b.bl.mask )
    bb = apply_geom_ij_yx( (gy,gx), b.bl)
    bb = np.ma.masked_array(bb, ~mm)
    ax.matshow(bb, vmin=-0.15*64, norm=SymLogNorm(0.15*64,lt, base=10))
    ax.set_xticks([])
    ax.set_yticks([])

def rdf(ri, vi, si, dk=3, l=[200e-6,200e-6], lmd=1.34):
    r = ri * [[l[0], l[1]]]
    
    si2 = si*si
        
    qi = np.sqrt(2. - 2. / np.sqrt(np.sum(r * r, 1) + 1.)) / lmd
    dq = dk*np.sqrt(2. - 2. / np.sqrt(l[0]*l[0] + l[1]*l[1] + 1.)) / lmd
    
    #qi = np.sqrt(np.sum(ri * ri, 1) + 1.)
    #dq = 5
    
    qmn, qmx = qi.min(), qi.max()
    nbin = int(np.floor((qmx - qmn) / dq))

    qk = np.round((qi - qi.min()) / dq).astype(int)

    n = np.bincount(qk)
    m = np.argmax(n) + 1
    n = n[:m]
    idx, = np.where(n > 10)
    n = n[idx]
    
   
    q = np.bincount(qk, qi)[idx]
    
    S = np.bincount(qk, 1./si2)[idx]
    
    v = np.bincount(qk, vi/si2)[idx]
    v1 = np.bincount(qk, vi)[idx]
    v2 = np.bincount(qk, vi*vi/si2)[idx]
    
    sv = np.bincount(qk, si2)[idx]
        
    q /= n
    v /= S
    v2 /= S
        
    return q, v, np.sqrt((v2-v*v))

def make_indexed_rep(gx,gy,b,b2,s,saturated=False):
    mi = ~b.mask
    mi[8,:,:]=False
    mi[9,64*6+6:64*6+9,64:]=False
    if saturated:
        mi[3,448:,64:128-16]=False
        mi[4,64*7+44:64*7+47,6:9]=False

    
    ri = np.vstack([gx[mi.reshape(16*512,128)],gy[mi.reshape(16*512,128)]]).T
    vi = b[mi]

    di = b2[mi]
    si = s[mi]
    return ri, vi, di, si

def read_reduced_pw(fn):
    a = type('BC', (object,), {})
    with h5py.File(fn, 'r') as f:
        m = f['bgcum/bg_msk'][:]
        a.bg = np.ma.masked_array(f['bgcum/bg'][:], mask=~m)
        a.bg2 = np.ma.masked_array(f['bgcum/bg2'][:], mask=~m)
        a.sbg = np.ma.masked_array(f['bgcum/bg_sig'][:], mask=~m)
        m = f['bgcum/pw_msk'][:]
        a.pw = np.ma.masked_array(f['bgcum/pw'][:], mask=~m)
        a.pw2 = np.ma.masked_array(f['bgcum/pw2'][:], mask=~m)
        a.spw = np.ma.masked_array(f['bgcum/pw_sig'][:], mask=~m)
        x0 = f['geom/x0'][()]
        y0 = f['geom/y0'][()]
    return a, (x0, y0)

def read_reduced_bg(fn):
    a = type('BC', (object,), {})
    with h5py.File(fn, 'r') as f:
        m = f['bgcum/gas/bg_msk'][:]
        a.bg = np.ma.masked_array(f['bgcum/gas/bg'][:], mask=~m)
        a.bg2 = np.ma.masked_array(f['bgcum/gas/bg2'][:], mask=~m)
        a.sg = np.ma.masked_array(f['bgcum/gas/bg_sig'][:], mask=~m)
        m = f['bgcum/line/bg_msk'][:]
        a.bl = np.ma.masked_array(f['bgcum/line/bg'][:], mask=~m)
        a.bl2 = np.ma.masked_array(f['bgcum/line/bg2'][:], mask=~m)
        a.sl = np.ma.masked_array(f['bgcum/line/bg_sig'][:], mask=~m)
        x0 = f['geom/x0'][()]
        y0 = f['geom/y0'][()]
    return a, (x0, y0)
