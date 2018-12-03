from __future__ import print_function

import math
import numpy as np

from scipy.special import erf
from scipy.interpolate import interp1d,  splrep,splev
from scipy.stats import binned_statistic_2d, iqr, poisson, binom, erlang, planck

import h5py

import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, LogFormatterExponent
from mpl_toolkits.axes_grid1 import make_axes_locatable

        

class SprFitSet:
    def __init__(self, l, L, m=6):
        
        self.l = l
        self.L = L
        self.ntrain = 1000
        
        self.m = m
        self.hidx = np.zeros([m, m], dtype=int)
        self.hidx[np.triu_indices(m)] = range(m*(m+1)//2)
        self.hidx[np.tril_indices(m, -1)] = (self.hidx.T)[np.tril_indices(m, -1)]
        
        self.runs = []
        self.grps = []
        self.nrun = 0
        self.nhit = []
        self.nfit = []
        self.ncell = []
        self.asf2 = []
        
        self.P = np.empty([0, m], dtype=float)
        self.H = np.empty([0, m, m], dtype=float)
        self.C = np.empty([0, m, m], dtype=float)
        
        #self.asf = np.empty(0, dtype=float)
        self.grp = np.empty(0, dtype=int)
        self.hit = np.empty(0, dtype=int)
        self.frm = np.empty(0, dtype=int)
        self.cell = np.empty(0, dtype=int)
        self.run = np.empty(0, dtype=int)
        self.np = np.empty(0, dtype=int)
        self.chi2 = np.empty(0, dtype=float)
        self.en = np.empty(0, dtype=float)
        
        self.hit_grp = np.empty(0, dtype=int)
        self.hit_frm = np.empty(0, dtype=int)
        self.hit_cell = np.empty(0, dtype=int)
        self.hit_train = np.empty(0, dtype=int)
        self.hit_run = np.empty(0, dtype=int)
        
        self.lbl = ("$b_0$, photons", "$I^0$, photons/\AA$^2$", "$R$, \AA", "x, pixels", "y, pixels")

    
    def read(self, runs, asf, fnptrn, x2mx=1.1, fn_intens="data/intens.h5"):
        
        F, I, R, N, X, P, H, C, J, T, E, G = [], [], [], [], [], [], [], [], [], [], [], []
        HF, HR, HI, HT, HG = [], [], [], [], []
        print(" run file                             Nfit   Nhit Rf/h,% Ncell      Imx")
        for i in range(len(runs.i)):
            run = runs.i[i]
            grp = runs.g[i]
            a = asf[runs.s[i]]
            fn = fnptrn.format(run)
            print("{:04d} {:30s}".format(run, fn[-30:]), end='')
            asf2 = a * a
            with h5py.File(fn, 'r') as f:

                npnt = f['sphr/np'][:]
                chi2 = f['sphr/chi2'][:]
                param = f['sphr/param'][:,:]
                param[:,1] /= asf2

    
                goodi, = np.where(np.logical_and(chi2 <= x2mx, param[:,3] >= 650.))
                ngood = goodi.size
            
                hessian = f['sphr/hessian'][goodi, :]
                hessian = hessian[:, self.hidx]
                corr = f['sphr/corr'][goodi, :]
                corr = corr[:, self.hidx]
                
                hiti = f['sphr/hits'][:]
                hiti = hiti[goodi]
                frmi = f['hits/frames'][:]
                #nhit = frmi.size
                cells = f['hits/cells'][:]
                ncell = cells.size
                
                frm = frmi[goodi]
                cell = frm % ncell
                train = frm // ncell
                intens = f['sphr/en'][:]
                nhit = hiti.max()
                
            #with h5py.File('hits/hits-r{:04d}.h5'.format(run), 'r') as f:
            #    cellid = f['hits/cellId'][:]
            #    trainid = f['hits/trainId'][:]
            #    trmin = np.min(trainid)
                
            #    cellid = cellid[hiti]
            #    cell = (cellid - 2) // 2
            #    train = trainid[hiti] - trmin
            #    frm = train * ncell + cell
                
            #with h5py.File(fn_intens, 'r') as f:
            #    r = f['intensities/run'][:]
            #    k, = np.where(r == run)
            #    k = k[hiti]
                
            #    intens = f['intensities/energy'][:]
            #    intens = intens[k]
            #    cnv = f['intensities/conversion'][:]
            #    cnv = cnv[k]
            #    j = intens > 0
                
            #    #print(intens[intens<0])
            #    
            #    #intens[j] = intens[j]
            #    intens[np.logical_not(j)] = -1.
            

            Imx = np.max(param[goodi,1])
            print(" {:6d} {:6d} {:6.1f} {:5d}   {:6.2f}".format(ngood, nhit, 100.*ngood / nhit, ncell, Imx))
            G.append(np.ones(ngood, dtype=int) * grp)
            J.append(hiti)
            F.append(frm)
            I.append(cell)
            T.append(train)
            R.append(np.ones(ngood, dtype=int) * i)
            N.append(npnt[goodi])
            X.append(chi2[goodi])
            P.append(param[goodi,:])
            H.append(hessian)
            C.append(corr)
            E.append(intens)
            
            #HF.append(frmi)
            HR.append(np.ones(nhit, dtype=int) * i)
            #HI.append(frmi % ncell)
            #HT.append(frmi // ncell)
            HG.append(np.ones(nhit, dtype=int) * grp)
            
            self.nfit.append(ngood)
            self.nhit.append(nhit)
            self.ncell.append(ncell)
            self.grps.append(grp)
                
            self.nrun += 1
            self.runs.append(run)
            self.asf2.append(asf2)
             
            
        self.P = np.concatenate(P, 0)
        self.H = np.concatenate(H, 0)
        self.C = np.concatenate(C, 0)
        
        self.chi2 = np.concatenate(X)
        
        self.grp = np.concatenate(G)
        self.hit = np.concatenate(J)
        self.frm = np.concatenate(F)
        self.cell = np.concatenate(I)
        self.train = np.concatenate(T)
        self.np = np.concatenate(N)
        self.run = np.concatenate(R)
        self.en = np.concatenate(E)
        
        self.hit_grp = np.concatenate(HG)
        #self.hit_frm = np.concatenate(HF)
        self.hit_run = np.concatenate(HR)
        #self.hit_cell = np.concatenate(HI)
        #self.hit_train = np.concatenate(HT)


        
    def get_nfit_by_pulse(self, k = None):
        if k is None:
            return np.bincount(self.cell)
        else:
            return np.bincount(self.cell[k])
        
    def get_nhit_by_pulse(self, k = None):
        if k is None:
            return np.bincount(self.hit_cell)
        else:
            return np.bincount(self.hit_cell[k])
        
    def get_stat_by_pulse(self, v, k = None):
        if k is None:
            k = np.arange(self.P.shape[0], dtype=int)
        i = k[v[k] != -1]
        n = np.bincount(self.cell[i])
        Mv = np.bincount(self.cell[i], weights=v[i])
        Dv = np.bincount(self.cell[i], weights=v[i]*v[i])
        j, = np.where(n != 0)
        Mv[j] /= n[j]
        Dv[j] = Dv[j] / n[j] - Mv[j] * Mv[j]
        return Mv, np.sqrt(Dv), n

    def get_stat_by_run(self, v, k = None):
        if k is None:
            k = np.arange(self.P.shape[0], dtype=int)
        i = k[v[k] != -1]
        n = np.bincount(self.run[i])
        Mv = np.bincount(self.run[i], weights=v[i])
        Dv = np.bincount(self.run[i], weights=v[i]*v[i])
        j, = np.where(n != 0)
        Mv[j] /= n[j]
        Dv[j] = Dv[j] / n[j] - Mv[j] * Mv[j]
        return Mv, np.sqrt(Dv), n
    
    def get_consiquent_fits(self):
        return np.where(np.logical_and(np.diff(self.frm) == 1, self.cell[:-1] < 29))[0]
    
    def get_ntrains_by_nfit(self):
        trainid = self.run * self.ntrain + self.train
        
        nmx = np.max(self.ncell)
        #nn[0] = self.ntrain * self.nrun - np.sum(nn[1:])

        mu = np.empty(self.nrun, dtype=float)
        n = np.empty([self.nrun, nmx], dtype=int)
        p = np.empty([self.nrun, nmx], dtype=float)
        x = np.arange(nmx, dtype=int)
        for j in range(self.nrun):
            mu[j] = self.nfit[j] / self.ntrain

            nn = np.unique(self.train[self.run == j], return_counts=True)[1]
            
            n[j,:] = np.bincount(nn, minlength=nmx)
            #n[j,n[j,:] < 5] = 0
            p[j,:] = poisson.pmf(x, mu[j]) * self.ntrain
            #kmx = np.where(p[j,:] < .5)[0][0]
            #n[j,kmx] = np.sum(n[j,kmx:])
            #n[j,kmx:] = 0
            n[j,0] = self.ntrain - np.sum(n[j,1:])


            #p[j,:] = binom.pmf(x, nmx, mu[j]/nmx) * self.ntrain
            #p[j,:] = erlang.pdf(x, mu[j]) * self.ntrain
            
        return mu, n, p
       
    def get_param_by_run(self, pid, by='runs', ranges=None, logscale=False, k = None):
        if k is None:
            p = self.P[:,pid]
        else:
            p = self.P[k,pid]
        n = p.size
        nbin = int(2*n**(1./3.))

        if ranges is None:
            p0, pn = p.min(), p.max()
        else:
            p0, pn = ranges
        if logscale:
            bins = np.geomspace(p0, pn, nbin)
        else:
            bins = np.linspace(p0, pn, nbin)

        if by == 'runs':
            m = self.nrun
            select = lambda k, i: self.run[k] == i
        elif by == 'cells':
            m = max(self.ncell)
            select = lambda k, i: self.cell[k] == i
        
        H = np.empty([m, nbin-1], dtype=float)
        for i in range(m):
            #l, = np.where(select(i))
            l = k[select(k, i)]
            pk = self.P[l,pid]
            spk = self.C[l,pid,pid]
            
            q = np.subtract(*np.meshgrid(bins, pk, indexing='ij'))
            h = 0.5*(1 + erf(q/np.sqrt(2.*spk)))

            h = np.mean(h,1)
            H[i,:] = h[1:]-h[:-1]
            
            #d = np.histogram(Ik, bins=bins, density=True)[0]
            #w = np.histogram(Ik, bins=bins, weights=1./sI)[0]
            #x = np.histogram(Ik, bins=bins, weights=Ik/sI)[0]            
            
            
        return H, bins
    
    def get_IR_distribution(self, k = None, Imx = None, Rmx = None):
        if k is None:
            I = self.P[:,2]
            R = self.P[:,3]
        else:
            I = self.P[k,2]
            R = self.P[k,3]            
        
        n = I.size
        nbin = int(2*n**(1./3))
        
        if Imx is None:
            Imx = I.max()
        if Rmx is None:
            Rmx = R.max()
            
        Imn = I.min()
        Rmn = R.min()
        
        Rb = np.geomspace(Rmn, Rmx, nbin)
        Ib = np.geomspace(Imn, Imx, nbin)
        #Rb = np.linspace(Rmn, Rmx, nbin)
        #Ib = np.linspace(Imn, Imx, nbin)
        
        H = np.histogram2d(I, R, bins=(Ib, Rb), normed=False)[0]
        
        return H, Rb, Ib
   
    def corrcoef(self, h1, h2=None):
        n, m = h1.shape
        
        M1 = np.mean(h1, 1, keepdims=True)
        D1 = np.std(h1, 1)
        if h2 is None:
            h2, M2, D2 = h1, M1, D1
        else:
            M2 = np.mean(h2, 1, keepdims=True)
            D2 = np.std(h2, 1)

        return (np.matmul((h1 - M1), (h2 - M2).T)) / np.outer(D1, D2) / m

    
    def mean_center(self, i=None):
        if i is None:
            i = np.arange(self.P.shape[0], dtype=int)
        n = i.size
        ri = self.P[i,4:]
        Hi = self.H[i,4:,4:] * (self.np[i] - self.m).reshape(n,1,1)

        r0 = np.tensordot(Hi, ri, ((0,1),(0,1)))
        M1 = np.linalg.inv(np.sum(Hi, 0))
        r0 = np.dot(M1,r0)
    
        ri2 = np.einsum('ij,ik->ijk', ri-r0,ri-r0)
        K = np.sum(Hi * ri2, 0) / np.sum(Hi,0)
        
        C0 = np.triu(K)
        std_r0 = np.sqrt(np.diag(K) / (n - 1))
        
        return r0, C0, n
    
    def get_center_distribution(self, pid, statistics, i=None, k=1e6, d=1, ranges=None):
        if i is None:
            i = np.arange(self.P.shape[0], dtype=int)
                     
        r0, C0, n = self.mean_center(i)
        
#        if weights is None:
#            weights = np.ones(n, dtype=float)
        
        v = self.P[i,pid]
        ri = self.P[i,4:] - r0
        #ai = np.arctan2(ri * self.l, self.L) * k
        
        if ranges is None:
            ranges = (
                (np.min(ri[:,0]), np.max(ri[:,0])),
                (np.min(ri[:,1]), np.max(ri[:,1])),
            )
        #amx = np.max(np.abs(ai))
        #amx = 750.0
        
        nbin = int(4*n**(1./3.)) / d
        
        
        #bins = np.geomspace(np.sqrt(amx)/nbin, amx, nbin//2) 
        #bins = np.concatenate([-bins[::-1], bins])
        
        #bins_x = np.linspace(-amx, amx, nbin)
        bins_x = np.linspace(ranges[0][0], ranges[0][1], nbin)
        bins_y = np.linspace(ranges[1][0], ranges[1][1], nbin)
        H, x_edge, y_edge, idx = binned_statistic_2d(ri[:,1], ri[:,0], v, statistics, bins=(bins_y,bins_x))#[0::3]
        
        #print(H.shape)
        
        m = (len(x_edge) + 1) * (len(y_edge) + 1)
        D = np.bincount(idx, minlength=m)
        D = D.reshape(len(y_edge)+1, len(x_edge)+1)
        D = D[1:-1,1:-1]
        
        return H, D, bins_x, bins_y
    
    def get_bandwidth_center_dist(self, k, pi, bins=10):
        x0, y0 = np.median(self.P[k,4]), np.median(self.P[k,5])

        rmin, rmax = self.P[k,pi].min(), self.P[k,pi].max()
        rst = np.geomspace(rmin,rmax,bins)
        #drst = rst[1:]-rst[:-1]

        pos = []
        hx = []
        hy = []
        drst = []

        for i in range(bins-1):
            l = k[np.where(np.logical_and(self.P[k,pi]>rst[i],self.P[k,pi]<=rst[i+1]))[0]]
            if not l.size:
                continue
            hx.append(self.P[l,4]-x0)
            hy.append(self.P[l,5]-y0)
            pos.append(np.mean(self.P[l,pi]))
            drst.append(rst[i+1]-rst[i])
            
        return np.array(hx), np.array(hy), np.array(pos), np.array(drst), (x0, y0)

    def plot_bandwidth_dist(self, ax, k, rlim, ilim, plim, labelleft=True):
        # particle size

        datx, daty, pos, drst, r0 = self.get_bandwidth_center_dist(k, 3)
        x0, y0 = r0
 

        #ax = fig.subplots(2,2)

        axi = ax[0]
        axi.boxplot(datx, notch=False, sym='', vert=False, positions=pos, widths=drst*.35, whis=[5, 95])
        axi.semilogy()
        axi.set_ylim(*rlim)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        if labelleft:
            axi.set_ylabel('$R$, \AA')
        else:
            axi.tick_params(labelleft=False)
        #axi.set_xlabel('$x$, pixel')
        axi.set_xlim(*plim)
        axi.tick_params(labelbottom=False)

        axi = ax[1]
        axi.boxplot(daty, notch=False, sym='', vert=False, positions=pos, widths=drst*.35, whis=[5, 95])
        axi.semilogy()
        axi.set_ylim(*rlim)
        axi.tick_params(labelleft=False, left=False, which='both',labelbottom=False)
        axi.spines['left'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        #axi.set_xlabel('$y$, pixel')
        axi.set_xlim(*plim)

        # incident photon intesity
        datx, daty, pos, drst, r0 = self.get_bandwidth_center_dist(k, 2)

        axi = ax[2]
        axi.boxplot(datx, notch=False, sym='', vert=False, positions=pos, widths=drst*.35, whis=[5, 95])
        axi.semilogy()
        axi.set_ylim(*ilim)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        if labelleft:
            axi.set_ylabel('$I^0$, photons/\AA$^2$')
        else:
            axi.tick_params(labelleft=False)
        axi.set_xlabel('$x$, pixel')
        axi.set_xlim(*plim)

        axi = ax[3]
        axi.boxplot(daty, notch=False, sym='', vert=False, positions=pos, widths=drst*.35, whis=[5, 95])
        axi.semilogy()
        axi.set_ylim(*ilim)
        axi.tick_params(labelleft=False, left=False, which='both')
        axi.spines['left'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.set_xlabel('$y$, pixel')
        axi.set_xlim(*plim)
       

    def plot_center_distribution(self, bins, H, ranges=None, ax=None):
        x, y = np.meshgrid(bins[0],bins[1])

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.gca()


        cmap = plt.get_cmap()
        #cmap = plt.cm.jet
        cmap.set_bad(cmap(0))
        
        if ranges is None:
            goodv = np.logical_not(np.isnan(H))
            vmin = np.min(H[goodv])
            vmax = np.max(H[goodv])
        else:
            vmin, vmax = ranges

        im = ax.pcolor(x, y, H, norm=LogNorm(),cmap=cmap,vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        #ax.set_xlabel("$\gamma_\mathrm{h}, \mu\mathrm{rad}$")
        #ax.set_ylabel("$\gamma_\mathrm{v}, \mu\mathrm{rad}$")
        ax.set_xlabel("$x, \mathrm{pixel}$")
        ax.set_ylabel("$y, \mathrm{pixel}$")
        
        return im
        
        
    
    def plot_run_correlation(self, C, ticks=None, lbl="", ax=None):
        
        ny, nx = C.shape
        i = np.linspace(0.5, ny + 0.5, ny + 1)
        j = np.linspace(0.5, nx + 0.5, nx + 1)
        if ticks is None:
            tx = range(1, nx + 1)
            ty = range(1, ny + 1)
        elif isinstance(ticks, tuple):
            tx, ty = ticks[0], ticks[1]
        else:
            tx, ty = ticks, ticks
        if isinstance(lbl, tuple):
            lx, ly = lbl[0], lbl[1]
        else:
            lx, ly = lbl, lbl

        
        #runs = np.linspace(self.runs[0]-0.5, self.runs[-1]+0.5, self.nrun+1)
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.gca()
        
        x, y = np.meshgrid(j, i, indexing='ij')
        im = ax.pcolor(x, y, C, vmin=0, vmax=1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(im, ax=ax,cax=cax)
        
        ax.set_ylim(i[-1], i[0])
        ax.set_xlim(j[0], j[-1])
        
        ax.set_yticks(i[:-1]+0.5)
        ax.set_yticklabels(ty, fontsize=14)
        ax.set_xticks(j[:-1]+0.5)
        ax.set_xticklabels(tx, fontsize=14)
        ax.tick_params("x", labelrotation=90, bottom=False, top=True, labelbottom=False, labeltop=True)
        
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_aspect("equal")



    def plot_pulse_hist(self, ax, pi, k, lim):
        for i in range(len(k)):
            ki = k[i]

            hI, bI = self.get_param_by_run(pi, 'cells', lim, logscale=True, k=ki)
            xx,yy = np.meshgrid(bI, np.arange(31)+0.5)
            axi = ax[i]
            axi.pcolor(yy,xx,hI)
            axi.set_yscale('log')
            axi.tick_params(labelleft=False)
            axi.set_xticks([1,10, 20, 30])
            axi.set_xlabel("pulse")
            #xi.add_line(plt.Line2D([0.5, 30.5], [90.]*2, c="C%d"%i, linewidth=8))
            #axi.set_ylim(Imn, 100)

        axi=ax[0]
        axi.tick_params(labelleft=True)
        axi.set_ylabel(self.lbl[pi-1])
       
        
    def plot_IR_distribution(self, bR, bI, H, ax=None):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()

        cmap = plt.get_cmap()
        cmap.set_bad(cmap(0))
            
        x, y = np.meshgrid(bR, bI)
        im = ax.pcolor(x, y, H, norm=LogNorm(), cmap=cmap)
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        ax.set_ylabel(self.lbl[1])
        ax.set_xlabel(self.lbl[2])

        return im
        
        
    
    def plot_param_by_run(self, bins, h, lbl, logscale=False, ax=None):
        
        #runs = np.linspace(self.runs[0]-0.5, self.runs[-1]+0.5, self.nrun+1)
        runs = np.linspace(0.5, self.nrun+0.5, self.nrun+1)
        hmn = np.min(h[h>0])
        hmn = 1e-4
        
        y, x = np.meshgrid(runs, bins, indexing='ij')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(16,4))
            ax = fig.gca()
            
            
        cmap = plt.get_cmap()
        cmap.set_bad(cmap(0))
        
        ax.pcolor(x, y, h, norm=LogNorm(), vmin=hmn, cmap=cmap)
        if logscale:
            ax.set_xscale('log')
        ax.set_ylim(self.nrun+0.5, 0.5)
        
        ax.set_xlabel(lbl)
        ax.set_ylabel("run")
        
        #ax.get_xaxis().set_major_formatter(LogFormatterExponent())
        #ax.get_xaxis().set_minor_formatter(ScalarFormatter())
        
        ax.set_yticks(runs[:-1]+0.5)
        ax.set_yticklabels(self.runs, fontsize=14)
        
        

    
    def plot_ntrains_by_nfit(self, mu, n, p, ax=None):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()
        
        nn = np.sum(n, 0)
        pp = np.sum(p, 0)
               
        i, = np.where(nn)
        
        imx = np.where(pp > 0.75)[0][-1] + 1
        #imx = i.max() + 1
        j = range(0,imx)
        
        ax.stem(i, nn[i], bottom=1, linefmt="C0", markerfmt="C0o")
        ax.plot(j, pp[j], "C1P")
        ax.set_yscale('log')
        
        ax.set_xticks(j)
        #ax.set_ylabel("trains")
        #ax.set_xlabel("fits per train")
         
    def plot_nfit_by_pulse(self, ax=None):
        n = self.get_nfit_by_pulse()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()
        
        cells = np.arange(1,max(self.ncell)+1, dtype=int)
        ax.stem(cells, n)
        ax.set_xticks(cells)
        ax.set_xlabel("impulse")
        ax.set_ylabel("fits")
        ax.tick_params("x", labelrotation=90)
        ax.set_ylim(0, np.max(n)*1.1)
        
        
    def plot_n_by_run(self, y, lbl=None, ax=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)
        
        mrk = ('o', 's', 'v', '^', '<', '>')
        nmrk = len(mrk)

        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()
            
            
        grps = np.array(self.grps)
        i = np.arange(len(self.runs)) + np.array(grps)
        
        vmx = float("-inf")
        for j in range(len(y)):
            if lbl is not None:
                l = lbl[j]
            else:
                l = None
            ax.stem(i, y[j], linefmt="C%d"%j, markerfmt="C%d"%j+mrk[j%nmrk], basefmt="None", label=l)
            m = np.max(y[j])
            vmx = max(m, vmx)
        
        for j in range(np.max(grps)+1):
            k, = np.where(grps == j)
            k += grps[k]
            k0, kn = k[0], k[-1]            
            ax.add_line(plt.Line2D([k0, kn], [0.]*2, c="C%d"%j, linewidth=5))
            
        
        ax.set_xticks(i)
        ax.set_xticklabels(self.runs)
        ax.set_xlabel("run")
        ax.tick_params("x", labelrotation=90)
        #ax.set_ylim(-0.1 * vmx, vmx*1.1)

    def plot_fit_ratio_by_run(self, ax=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=16)

        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()
        
        r = 100.*np.array(self.nfit) / np.array(self.nhit)
        i = np.arange(len(self.runs)) + np.array(self.grps)
        ax.stem(i, r)
        ax.set_xticks(i)
        ax.set_xticklabels(self.runs)
        ax.set_xlabel("run")
        ax.set_ylabel("fit to hit ratio, \%")
        ax.tick_params("x", labelrotation=90)
        ax.set_ylim(0, np.max(r)*1.1)


        