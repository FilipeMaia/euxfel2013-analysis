import math
import numpy as np
from scipy.optimize import minimize

from .utils import *

class SprModel:
    def __init__(self, ri, vi, bi, si, ei, lmd, l, n=[.0, .0]):
        self.n = vi.size
        
        a = re * math.pi
        self.a = (5./6.) * a * a * l[0] * l[1]
        self.b = math.pi / lmd
        
        self.pn = np.array([[n[1], -n[0]]])
        self.p1 = n[0]*n[0] + n[1]*n[1]
        nrm = 1. - np.sqrt(self.p1)
        if nrm < .0:
            raise(ValueError("|n| > 1"))
        self.p2 = 0.5 * nrm * nrm
        self.p1 += self.p2

        self.l = np.array([[l[0], l[1]]], dtype=float)
        self.l2 = np.array([[l[0]*l[0], .0, l[1]*l[1]]], dtype=float)
        self.lp = -2.*np.array([[l[0]*l[0]*n[1]*n[1], -l[0]*l[1]*n[1]*n[0], l[1]*l[1]*n[0]*n[0]]])
        
        self.ri = ri
        self.vi = vi.reshape(self.n, 1)
        self.bi = bi.reshape(self.n, 1) - bi.min()
        self.si2 = (si * si).reshape(self.n, 1)
        self.ei2 = (ei * ei).reshape(self.n, 1)

        self.z = np.zeros([self.n, 1], dtype=float)
        self.p = np.array([float("nan")] * 6)
        self.L = None
        self.dL = None
        self.d2L = None
        self.nev = 0

    def model(self, I0, R, x0, y0, exponential=False):

        ri = (self.ri - [[x0, y0]]) * self.l
    
        # r(xk, yk)
        r = np.linalg.norm(ri, axis=1, keepdims=True)
        dr = -self.l * ri / r
        d2r = (self.l2 - dsymaa(dr)) / r
      
        # S(r) = 1 / (r(x,y)^2 + 1)
        C2 = 1. / (r * r + 1.)
        C = np.sqrt(C2)
        C3 = C * C2
        dCdr = -r * C3
        d2Cdr2 = dCdr * (1./r + 3. * dCdr / C)
    
        d2C = d2Cdr2 * dsymaa(dr) + dCdr * d2r
        dC = dCdr * dr
    
    
        # Q(S) = sqrt(2 - 2S(r)) / lmd
        Q = np.sqrt(2. - 2.*C)
        d2Q = -(dsymaa(dC) / (Q*Q) + d2C) / Q
        dQ = -dC / Q
    
        # Z(Q, R) = math.pi * R * Q(S)
        Z = self.b*Q*R
        d2Z = self.b*np.hstack([self.z, dQ, R * d2Q])   
        dZ = self.b*np.hstack([Q, R * dQ])

        # F(Z(Q,R))
        snZ, csZ, Z2 = np.sin(Z), np.cos(Z), Z*Z
        F = 3. * (snZ - Z*csZ) / (Z*Z2)
        dFdZ = 3. *(snZ / Z2 - F / Z)
        d2FdZ2 = -F - 4. * dFdZ / Z

        d2F = d2FdZ2 * dsymaa(dZ)  + dFdZ * d2Z
        dF = dFdZ * dZ
               
        # polarisation
        B = np.sum(self.pn * ri, 1, keepdims=True)
        dB2 = B * (2.*self.pn * self.l)
        B2 = self.p2 - B*B
        
        dC2 = 2.*C*dC
        d2C2 = 2.*(dsymaa(dC)+C*d2C)
             
        P = self.p1 + B2 * C2
        dP = B2 * dC2 + C2 * dB2
        d2P = B2 * d2C2 + C2 * self.lp + dsymab(dB2,dC2)


        
        # vitnetting
        # T(x,y) = P(x,y)*S(x,y)^3
        T = P * C3
        dT = C2 * (3. * P * dC + C * dP)
        d2T = C2 * (3.*P * d2C + C * d2P + 3.*dsymab(dP, dC)) + 6.* P * C * dsymaa(dC)
        
        # sphere volume
        # A(R,x,y)=V(R)*T(x,y)
        d2V = R * R
        d2V = self.a * d2V * d2V
        dV = d2V * R / 5.
        V = dV * R / 6.
        
        A = T * V
        dA = np.hstack([T * dV, V * dT])
        d2A = np.hstack([T * d2V, dV * dT, V * d2T])
        
        # E(A,F)=A(R,x,y) * F(R,x,y)^2
        F2 = F*F
        E = A * F2
        dE = F2 * dA + 2.* A * F * dF
        d2E = F2 * d2A + 2.* (F * dsymab(dA, dF) + A * dsymaa(dF) + A * F * d2F)
        
        # I(I0,R,x,y) = I0 * E(R,x,y)
        I = I0 * E
        d2I = np.hstack([self.z, dE, I0 * d2E])
        dI = np.hstack([E, I0 * dE])

        return I, dI, d2I


    def likelihood_gauss_stab(self, b0, ec, I, dI, d2I):
        m = dI.shape[1] + 2
                     
        lmd = np.abs(I) + np.abs(ec) * self.bi
        
        s2 = lmd + self.si2 + self.ei2 #ec*ec*self.ei2
        
        A = lmd + b0 - self.vi
        S = s2
        
#        B = 0.5 / S
#        C = 0.5 * (1. - A*A / S)
#        D = (A + C) / S
        
#        AS = A/S
        
         
        #E = 2.* ec * self.ei2
#        E = 0.
        
#        L = 0.5 * np.mean(A*A/S + np.log(2. * math.pi * S), 0)
        
#        dSda = self.bi + E        
#        dLda = D * self.bi + (C / S) * E 
        
#        dL = np.mean(np.hstack([AS, dLda, D * dI]), 0)

#        d2Ld12 = (self.bi - AS * dSda) / S
#        d2Ld13 = (1. - AS) / S * dI
        
        #d2Ld22 = (2. * C * self.ei2 + self.bi*self.bi + (B * dSda - 2.*dLda) * dSda) / S
#        d2Ld22 = (self.bi*self.bi + (B * dSda - 2.*dLda) * dSda) / S
#        d2Ld23 = (self.bi - dLda + (B - D) * dSda) / S * dI

#        d2Ld33 = D * d2I + (1. + B - 2.*D) / S * dsymaa(dI)
        
#        d2L = np.mean(np.hstack([1./S, d2Ld12, d2Ld13, d2Ld22, d2Ld23, d2Ld33]), 0)
        
        
        B = A / s2
        C = 1. / s2
        B1 = 1. - B
        D = (C - B*B) * self.ei2
       
        L = 0.5 * np.mean(A * B + np.log(2. * math.pi * s2), 0)
        
        dLdI = B * (1. - 0.5 * B) + 0.5 * C        
        d2LdI2 = (B1 * B1 - 0.5 * C) * C * dsymaa(dI) + dLdI * d2I
        
        dLde = D * ec
        d2Lda2 = (B1 * B1 - 0.5 * C) * C * self.bi * self.bi
        
        d2L = np.mean(np.hstack([C, C * B1 * self.bi + D, C * B1 * dI,
                                 d2Lda2, self.bi * C * B1 * dI,
                                 d2LdI2]), 0)
        dL = np.mean(np.hstack([B, dLdI*self.bi + D*ec, dLdI * dI]), 0)

        idx = np.zeros([m, m], dtype=int)
        idx[np.triu_indices(m)] = range(d2L.size)
        idx[np.tril_indices(m, -1)] = (idx.T)[np.tril_indices(m, -1)]
        
        return L, dL, d2L[idx]

    def residuals(self, b0, ec, I):
        return (I + b0 + ec * self.bi - self.vi) / np.sqrt(I + self.si2)
    
    def chi2(self, b0, ec, I):
        s2 = I + self.si2 + ec*self.ei2
        d = (I + b0 + ec * self.bi - self.vi)
        Chi2 = d * d / s2
        
        return np.mean(Chi2)

    def _recompute(self, pk, process="gauss-stab"):
        pn = np.array(pk)
        if self.p.size != np.size or np.any(pn != self.p):
            b0, ec, I0, R, x0, y0 = pk
            I, dI, d2I = self.model(np.abs(I0), R, x0, y0)
            if process=="gauss-stab":
                self.L, self.dL, self.d2L = self.likelihood_gauss_stab(b0, ec, I, dI, d2I)
            else:
                raise(ValueError())
            self.p = pn
            self.nev += 1
        
    def obj(self, pk, process="gauss-stab"):
        self._recompute(pk, process)
        return self.L
            
    def jac(self, pk, process="gauss-stab"):
        self._recompute(pk, process)
        return self.dL

    def hess(self, pk, process="gauss-stab"):
        self._recompute(pk, process)
        return self.d2L
    
    def solve(self, xinit, tol=1e-7):
        
        fobj = lambda x, m: m.obj(x)
        fjac = lambda x, m: m.jac(x)
        fhess = lambda x, m: m.hess(x)

        res = minimize(fobj, xinit, self, 'trust-exact', jac=fjac, hess=fhess,
                       options={'gtol': tol})

        v, w = np.linalg.eigh(res.hess)
        
        res.positive = np.all(v > 0.)
        res.corr = np.matmul((w / v), w.T) / (self.n - res.x.size)
        
        b0, ec, I0, R, x, y = res.x

        I = self.model(I0, R, x, y)[0]
        res.gof = self.chi2(b0, ec, I)
        
        return res



