import math
import numpy as np

from scipy import stats

from matplotlib.pylab import plt
from .utils import *


class SprModelRadial:
    def __init__(self, ri, vi, bi, si, ei, lmd, l):
        
        self.k2 = l[0] * l[1]
        self.lmd = lmd
        self.ri = ri
        self.vi = vi
        self.bi = bi
        self.si = si
        self.ei = ei
        
        r = ri * [[l[0], l[1]]]
        
        self.qi = np.sqrt(2. - 2. / np.sqrt(np.sum(r * r, 1) + 1.)) / lmd
        self.dq = np.sqrt(2. - 2. / np.sqrt(l[0]*l[0] + l[1]*l[1] + 1.)) / lmd
             
        self.q, self.v, self.sv, self.b, self.sb, self.n = SprModelRadial._rdf(self.qi, self.dq, vi, bi, si, ei)

        self.zi, = np.where(self.q == .0)
        self.ni, = np.where(self.q != .0)

        self.nq = self.q.size
        self.q = self.q.reshape(self.nq, 1)
        self.v = self.v.reshape(self.nq, 1)
        self.sv = self.sv.reshape(self.nq, 1)
        self.b = self.b.reshape(self.nq, 1)
        self.sb = self.sb.reshape(self.nq, 1)
        self.n = self.n.reshape(self.nq, 1)
        
        self.T = self._factors(self.q)

        
    def model(self, I0, R, model_kind):
        
        #qlmd = self.q * self.lmd
        #C = 1. - 0.5 * qlmd * qlmd
        #C2 = C*C
        
        F = model_kind(self, R, self.q)
 
        # volume of ball
        V = math.pi * (R*R*R) / 6.

        F = re * V * F
        F2 = F*F

        # vignetting
        #T = C2 * C * self.k2
        # polarization
        #P = 0.5 + 0.5 * C2
        
        return I0 * F2 * self.T #* P * T
    
    def model_spr(self, I0, R):
        return self.model(I0, R, SprModelRadial._model_spr)

    def model_gauss(self, I0, R):
        return self.model(I0, R, SprModelRadial._model_gauss)

    def model_slope(self, I0, R):
        return self.model(I0, R, SprModelRadial._model_slope)
    
    def model_q(self, I0, R, q, model_kind):
        F = model_kind(self, R, q)
 
        # volume of ball
        V = math.pi * (R*R*R) / 6.

        F = re * V * F
        F2 = F*F
        
        return I0 * F2 * self._factors(q)
        

    
    def chi2(self, model_kind, Rmn=None):
        
        qn = self.q[-1]
        
        #mnR = self.lmd
        if Rmn is None:
            Rmn = j1[0,0] / (math.pi * self.q[-1])

        Rmx = (j1[1,0] - j1[0,0]) / (3.*math.pi*self.dq)
        nR = int(np.floor((Rmx - Rmn) / 5.))
        
        
        R = np.linspace(Rmn, Rmx, nR+1).reshape(1, nR+1)       
        I = self.model(1., R, model_kind)
        
        #y = self.v - self.b
        y = self.v
        b = self.b
        s2 = self.sv * self.sv
        e2 = self.sb * self.sb
        
        M1 = np.sum(1./ s2).repeat(nR+1)
        My = np.sum(y / s2).repeat(nR+1)
        MI = np.sum(I / s2, 0)
        Mb = np.sum(b / s2).repeat(nR+1)
        
        #DI = np.sum(I * I / s2, 0)
        #Dy = np.sum(I * (y / s2), 0)
       
        #D = DI * M1 - MI * MI
        #I0 = (Dy * M1 - MI * My) / D
        #b0 = (DI * My - Dy * MI) / D
        
        #rs = I0 * I + b0 - y
        
        DI2 = np.sum(I * I / s2, 0)
        DIb = np.sum(I * (b / s2), 0)
        Db2 = np.sum(b * b / s2).repeat(nR+1)
        DyI = np.sum(I * (y / s2), 0)
        Dyb = np.sum(b * y / s2).repeat(nR+1)
        
        # I0 == 0
        D = Db2 * M1 - Mb * Mb
        a0 = (Dyb * M1 - Mb * My) / D
        b0 = (Db2 * My - Dyb * Mb) / D
        I0 = np.zeros(nR+1, dtype=float)
        
        F = a0 * b
        rs = F + b0 - y
        chi2 = np.mean(rs*rs / (s2 + e2 + F / self.n), 0)
        
        # complete model
        A = np.vstack([DI2, DIb, MI, DIb, Db2, Mb, MI, Mb, M1]).T.reshape(nR+1,3,3)
        C = np.vstack([DyI, Dyb, My]).T

        x = np.linalg.solve(A, C)
        j = np.where(np.logical_and(x[:,0] >= 0., x[:,1] >= 0))[0]
        
        I1, a1, b1 = x[j,0], x[j,1], x[j,2]
        
        F = I1 * I[:,j] + a1 * b
        rs = F + b1 - y
        chi2a = np.mean(rs * rs / (s2 + e2 + F / self.n), 0)
        
        l = chi2a < chi2[j]
        j = j[l]
        chi2[j], I0[j], a0[j], b0[j] = chi2a[l], I1[l], a1[l], b1[l]
        
        # a0 == 0
        j = np.where(x[:,1] < 0)[0]
        
        D = DI2[j] * M1[j] - MI[j] * MI[j]
        I1 = (DyI[j] * M1[j] - MI[j] * My[j]) / D
        b1 = (DI2[j] * My[j] - DyI[j] * MI[j]) / D
        a1 = np.zeros(j.size, dtype=float)
        
        F = I1 * I[:,j]
        rs = F + b1 - y
        chi2a = np.mean(rs * rs / (s2 + F / self.n), 0)
        
        l = chi2a < chi2[j]
        j = j[l]
        chi2[j], I0[j], a0[j], b0[j] = chi2a[l], I1[l], a1[l], b1[l]        
        
        return R.flatten(), I0, b0, a0, chi2
    
    
    def chi2_slope(self):

        I = self.q[0,0] / self.q
        I *= I
        I *= I
        
        #y = self.v - self.b
        b = self.b
        y = self.v
        s2 = self.sv * self.sv + self.sb * self.sb
        
        M1 = np.sum(1./ s2)
        My = np.sum(y / s2)
        MI = np.sum(I / s2)
        Mb = np.sum(b / s2)
        
        DI2 = np.sum(I * I / s2)
        DIb = np.sum(I * (b / s2))
        Db2 = np.sum(b * b / s2)
        DyI = np.sum(I * (y / s2))
        Dyb = np.sum(b * y / s2)
        
        A = np.array([
            [DI2, MI, DIb],
            [DIb, Mb, Db2],
            [MI, M1, Mb]
        ], dtype = float)
        
        C = np.array([DyI, Dyb, My], dtype=float)
        
        
        x = np.linalg.solve(A, C)
        
        I0, b0, a0 = x[:,0], x[:,1], x[:,2]
        
       
        #D = DI * M1 - MI * MI
        #I0 = (Dy * M1 - MI * My) / D
        #b0 = (DI * My - Dy * MI) / D
        
        #rs = I0 * I + b0 - y
        rs = I0 * I + a0 * b + b0 - y
        chi2 = np.mean(rs * rs / s2)
        
        a0 = My / M1
        chi2_ln = np.mean((a0 - y) * (a0 - y) / s2)
                
        return chi2, chi2_ln


    def chi2_spr(self, Rmn=None):
        return self.chi2(SprModelRadial._model_spr, Rmn)
    
    def chi2_gauss(self, Rmn=None):
        return self.chi2(SprModelRadial._model_gauss, Rmn)

    
    def f_test(self, sx2, ex2, alpha=.5):
        
        i, j, idx = peaks(sx2)
        
        fa, fb = stats.f.interval(alpha, self.q.size, self.q.size)
        
        r = sx2[i] / ex2[i]
        k, = np.where(np.logical_or(r < fa, r > fb))
        
        if k.size == 0:
            return 1, None
        elif k.size == 1:
            return 0, i[k[0]]
        else:
            T = 0.5*(fb - fa)
            Z = np.sum(np.exp(-r[k]/T))
            P = np.exp(-r[k]/T) / Z
            
            l, = np.where(P > .5)
            
            if l.size == 1:
                return 0, i[k[l[0]]]
            else:
                return 2, None
            
#    def solve(self, Rmn=None, alpha=.5):
#        R, I0, b0, sx2 = self.chi2(SprModelRadial._model_spr, Rmn)
#        ex2 = self.chi2(SprModelRadial._model_gauss, Rmn)[3]
        
#        k0, = np.where(I0[::-1] > 0.)
#        if k0.size:
#            k0 = k0[0]
#            R = R[-k0:]
#            I0 = I0[-k0:]
#            b0 = b0[-k0:]
#            sx2 = sx2[-k0:]
#            ex2 = ex2[-k0:]
#        
#        flg, k = self.f_test(sx2, ex2, alpha)
#        if flg == 0:
#            X = sx2[k], b0[k], I0[k], R[k] 
#        else:
#            X = None
#            
#        return flg, X

#    def solve(self, alpha=0.75):
#        nrm2, ln2 = self.chi2_slope()
#        R, I0, b0, chi2 = self.chi2(SprModelRadial._model_gauss)
        
#        i, j, idx = peaks(chi2)
        
#        print(nrm2/ln2)
#        print(nrm2/ln2 > 0.95)
#        if (nrm2/ln2 > 0.95) or (i.size == 1 and j.size == 1):
#            return 1, None
#        elif i.size == 1 and j.size  == 2 and nrm2 / ln2 > 0.786661:
#            return 1, None
       
#        i1 = i[-1]
#        j1 = j[j < i1][-1]
#        k, = np.where(chi2[j1:i1] > nrm2)
#        if k.size > 0:
#            Rmn = R[j1 + k[-1]]
#        else:
#            Rmn = R[j1]
                    
#        R, I0, b0, chi2 = self.chi2(SprModelRadial._model_spr, Rmn)
#        i, j, idx = peaks(chi2)
#        chi2 /= nrm2
        
#        fa, fb = stats.f.interval(alpha, self.q.size, self.q.size)
        
#        ss = np.concatenate([i, j])
#        T = np.std(chi2[ss])
#        if T == 0.:
#            return 1, None
        
#        i1 = i[idx[0]]
#        p = np.exp(-(chi2[i1]/T)) / np.sum(np.exp(-(chi2[i]/T)))
        
#        if chi2[i1] > fa:
#            return 1, None
#        elif p < 0.5:
#            return 2, None
#        else:
#            return 0, (chi2[i1], b0[i1], I0[i1], R[i1])
        
    def solve(self, Rmn = 700, t1 = 1.75, t2 = 0.7):
        R, I0, b0, a0, chi2 = self.chi2(SprModelRadial._model_gauss, Rmn)
        
        chi2_max = chi2.min()
        
        R, I0, b0, a0, chi2 = self.chi2(SprModelRadial._model_spr, Rmn)
        
        if np.all(chi2_max < 0.9 * chi2):
            return 1, None
        
        i, j, idx = peaks(chi2)
        
        # no minima
        if i.size == 0:
            return 1, None
        
        # the first min
        i0 = i[0]
        # the dippest min
        i1 = i[idx[0]]
        # the second min by dip
        i2 = i[idx[1]] if i.size > 1 else chi2.max()
        
        good = (R[i1] > Rmn) and (chi2[i1] < 0.9 * chi2_max) and (chi2[i1] < 10.)
            
        
#        ij = np.concatenate([i, j])
#        
#        mean_chi2, std_chi2 = chi2[ij].mean(), chi2[ij].std()
#        
#        nmin = np.sum(chi2[i] < mean_chi2 - t1 * std_chi2) # t1 = 1.5
#
#        # the first min
#        i0 = i[0]
#        # the dippest min
#        i1 = i[idx[0]]
#        # indexes of minimums after first maxima
#        k = np.where(i[idx] > j[0])[0][0]
#        ik = i[idx[k]]
#             
#        good = (nmin == 1) and (R[ik] > Rmn) and ((chi2[ik]/mean_chi2) < t2) # t2 = 0.68, Rmn == 700

        if good:
            ik = i1
            return 0, (chi2[ik], b0[ik], a0[ik], I0[ik], R[ik])
        else:
            return 1, None
        

        
        
    def plot_rdf(self, ax, R=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)
        
        if ax is None:
            ax = plt.gca()
 
        idx = self.v > 0.
        if R is None:
            ax.loglog(self.q[idx], self.v[idx])
            ax.loglog(self.q[idx], self.v[idx]+self.sv[idx]+self.sb[idx])
            ax.set_xlabel("$q, \mathrm{\AA}^{-1}$")
        else:
            ax.loglog(self.q[idx] * R, self.v[idx])
            ax.loglog(self.q[idx], self.v[idx]+self.sv[idx]+self.sb[idx])
            ax.set_xlabel("$q$")

        ax.set_ylabel("$I, \mathrm{photon}/\mathrm{\AA}^{-2}$")
        
    def plot_model(self, ax, I0, R, b0, a0, model_kind=None, dimless=True, rdf=True, bg=False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)
        
        if ax is None:
            ax = plt.gca()
            
        if model_kind is None:
            model_kind = SprModelRadial._model_spr
            
        q0, qn = self.q[0,0], self.q[-1,0]
        q = np.linspace(q0, qn, 300).reshape(300,1)
        q1 = np.linspace(0, q0, 50).reshape(50,1)
        
        I = self.model_q(I0, R, q, model_kind) + a0*np.interp(q, self.q.flat, self.b.flat) + b0
        I1 = self.model_q(I0, R, q1, model_kind) + a0*self.b[0,0] + b0
        #q = q.reshape(200,1)
        #print("I", I.shape)
        #print("q", q.shape)
        
        dq = (7.72525183693770716-4.49340945790906418) / (math.pi * R) / 9
        if dq < self.dq:
            dq = self.dq

        
        qr, v, sv, b, sb, nn = SprModelRadial._rdf(self.qi, dq, self.vi, self.bi, self.si, self.ei)

        if dimless:
            q = q * R
            q1 = q1 * R
            qr = qr * R
            ax.set_xlabel("$q$")
        else:
            #q = self.q
            ax.set_xlabel("$q, \mathrm{\AA}^{-1}$")
        
        #idx = I > 0.
        ax.semilogy(q, I-b0, 'C0', linewidth=2)
        ax.semilogy(q1, I1-b0, 'C0:', linewidth=2)
        if rdf:
            idx = v > 0.
            ax.semilogy(qr, v-b0, 'C1--', linewidth=2)
        if bg:
            idx = b > 0.
            ax.semilogy(qr, a0*b, 'C2', linewidth=2)            

        ax.set_xlim(0, q[-1])
        ax.set_ylabel("$I, \mathrm{photons}$")

    def _model_spr(self, R, q):
        Z = np.outer(math.pi*q, R)
        snZ, csZ, Z2 = np.sin(Z), np.cos(Z), Z*Z
        
        F = np.zeros_like(Z)

        zi, = np.where(q.flat == .0)
        ni, = np.where(q.flat != .0)

        F[ni,:] = 3. * (snZ[ni] - Z[ni]*csZ[ni]) / (Z[ni]*Z2[ni])
        F[zi,:] = 1.

        return F
    
    def _model_slope(self, R, q):
        Z = np.outer(math.pi*q, R)
        return 1./(Z*Z)
    def _model_gauss(self, R, q):
        Z = np.outer(math.pi*q, R)
        return np.exp(-Z*Z/5.)

    
    @staticmethod
    def _rdf(qi, dq, vi, bi, si, ei):
        
        si2 = si * si
        ei2 = ei * ei
        
        qmn, qmx = qi.min(), qi.max()
        nbin = int(np.floor((qmx - qmn) / dq))
    
        qk = np.round((qi - qi.min()) / dq).astype(int)
    
        n = np.bincount(qk)
        m = np.argmax(n) + 1
        n = n[:m]
        idx, = np.where(n > 30)
        n = n[idx]
    
        q = np.bincount(qk, qi)[idx]
    
        S = np.bincount(qk, 1./si2)[idx]
        Se = np.bincount(qk, 1./ei2)[idx]
    
        v = np.bincount(qk, vi/si2)[idx]
        v2 = np.bincount(qk, vi*vi/si2)[idx]
    
        b = np.bincount(qk, bi/ei2)[idx]
        b2 = np.bincount(qk, bi*bi/ei2)[idx]
        
        sv = np.bincount(qk, si2)[idx]
        sb = np.bincount(qk, ei2)[idx]

        
        q /= n
        v /= S
        b /= Se
        
        sv = np.sqrt(sv / n / (n-1))
        sb = np.sqrt(sb / n / (n-1))
        
        #sv = np.sqrt((v2 / S - v*v) / (n - 1))
        #sb = np.sqrt((b2 / S - b*b) / (n - 1))
        
        
        #e = np.sqrt(e / n / (n-1))
        #print(e.min())
        
        #sv[sv < e] = e[sv < e]
        #sv = e
    
        m = np.argmax(n) + 1
        
        return q, v, sv, b, sb, n

    def _factors(self, q):
        
        qlmd = q * self.lmd
        C = 1. - 0.5 * qlmd * qlmd
        C2 = C*C
        
        # vignetting
        T = C2 * C * self.k2
        # polarization
        P = 0.5 + 0.5 * C2
        
        return T*P


