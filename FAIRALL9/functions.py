import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import chisquare

def Voigt(x, x0, sigma, gamma):
    '''
    x should be an array.
    sigma is the Doppler parameter.
    gamma is the Gaussian sigma.
    x0 is the center position.
    '''
    z = ((x-x0) + 1j * abs(gamma)) / abs(sigma) / 2**0.5

    return (special.wofz(z)).real/abs(sigma)/(2*np.pi)**0.5

def vline(x, p):
    '''
    p = [sigma, gamma, tau, v0, covering_factor]
    saturated_factor is a newly defined parameter to describe the thermal radiation varying in 0 to 1.
    The simplified model is c_f * exp(-tau) + (1-c_f).
    '''
    x = np.array(x)
    pro = abs(p[2]) * Voigt(x, p[3], p[0], abs(p[1]))

    return abs(p[4]) * np.exp(-pro) + (1 - abs(p[4]))

def convolve(model, lsf):
    Nlsf = len(lsf) / 2
    ext_model = np.append(np.ones(Nlsf)*model[0], model)
    ext_model = np.append(ext_model, np.ones(Nlsf)*model[-1])
    cmodel = np.convolve(ext_model, lsf, 'valid')
    return cmodel

def N2tau(N, wlr, f):
    tau = N * 2.654e-15 * f * wlr
    return tau

def model(p0):
    m1 = np.ones(v1.shape)
    m2 = np.ones(v2.shape)
    Ncomp = len(p0)/3
    for comp_i in range(Ncomp):
        N, b, vc = p0[comp_i*3: comp_i*3+3]
        m1 = m1*vline(v1, [b/1.414, gamma1, N2tau(10**N, wl1, f1), vc, 1])
        m2 = m2*vline(v2, [b/1.414, gamma2, N2tau(10**N, wl2, f2), vc, 1])

    m1 = convolve(m1, lsf)
    m2 = convolve(m2, lsf)

    return m1, m2

def fitting(p0):
    m1, m2 = model(p0)
    chi1 = (fn1 - m1) / fne1
    chi2 = (fn2 - m2) / fne2

    chi1 = chi1[(v1 < 200) & (v1 > -200)]
    chi2 = chi2[(v2 < 200) & (v2 > -200)]

    return np.append(chi1, chi2)

def plot_model(p0):
    m1, m2 = model(p0)
    plt.step(v1, fn1+1, 'k-', lw=2, where='mid')
    plt.plot(v1, m1+1, 'r--', lw=2)
    plt.hlines(1, -400, 400, linestyles='dashed', colors='c')
    plt.step(v2, fn2, 'k-', lw=2, where='mid')
    plt.plot(v2, m2, 'r--', lw=2)
    plt.hlines(2, -400, 400, linestyles='dashed', colors='c')

    Ncomp = len(p0)/3
    for comp_i in range(Ncomp):
        N, b, vc = p0[comp_i*3: comp_i*3+3]
        m1 = vline(v1, [b/1.414, gamma1, N2tau(10**N, wl1, f1), vc, 1])
        m2 = vline(v2, [b/1.414, gamma2, N2tau(10**N, wl2, f2), vc, 1])

        m1 = convolve(m1, lsf)
        m2 = convolve(m2, lsf)
        plt.plot(v1, m1+1, 'y:', lw=2)
        plt.plot(v2, m2, 'y:', lw=2)

    plt.xlim(-400, 400)
    plt.show()

def V2Wave(v, wlr):
    return ((v * (1e-13 *wlr) / 3e5) + (wlr * 1e-13))*1e13

def Nv(l1, l2):
    wlr1, f1, g1, elem, state = (1548.187, 0.19, 0.00324, 'C', 'IV')
    wlr2, f2, g2, elem, state = (1550.772, 0.0952, 0.00325, 'C', 'IV')

    dv1, fl1, fle1, ctn1 = l1.transpose()
    dv2, fl2, fle2, ctn2 = l2.transpose()

    ctn1 = correct_ctn(fl1, fle1, ctn1)
    ctn2 = correct_ctn(fl2, fle2, ctn2)

    Nv1 = -np.log(fl1/ctn1) / 2.654e-15 / wlr1 / f1
    Nv2 = -np.log(fl2/ctn2) / 2.654e-15 / wlr2 / f2
    Nve1 = fle1/fl1 / 2.654e-15 / wlr1 / f1
    Nve2 = fle2/fl2 / 2.654e-15 / wlr2 / f2

    flag1 = (dv1 < 200) & (dv1 > -200)
    flag2 = (dv2 < 200) & (dv2 > -200)
    N1 = np.nansum(Nv1[flag1] * 6.4313003967213112)
    N2 = np.nansum(Nv2[flag2] * 6.3899308606557375)
    Ne1 = np.nansum(Nve1[flag1]**2)**0.5 * 6.4313003967213112
    Ne2 = np.nansum(Nve2[flag2]**2)**0.5 * 6.3899308606557375
    logN1 = int(np.log10(N1) * 1000)/1000.
    logN2 = int(np.log10(N2) * 1000)/1000.
    logNe1 = int(Ne1 / N1 * 0.43429448190325182 * 1000)/1000.
    logNe2 = int(Ne2 / N2 * 0.43429448190325182 * 1000)/1000.

    plt.plot(dv1, Nv1, 'ko')
    plt.plot(dv2, Nv2, 'ro')
    plt.plot(dv1, Nve1, 'k:')
    plt.plot(dv2, Nve2, 'r:')

    Nv_max = max(np.max(Nv1), np.max(Nv2))
    plt.text(-300, 4e12, 'Strong\nlogN='+str(logN1)+'\n$\\sigma$='+str(logNe1))
    plt.text(200, 4e12, 'Weak\nlogN='+str(logN2)+'\n$\\sigma$='+str(logNe2))

    name = glob.glob('*_CIV.pdf')[0]
    target = name.split('_CIV')[0]
    plt.text(-200, 5e12, target)

    plt.hlines(0, -600, 600, color='c', linestyle='dashed')
    plt.vlines(-200, -1e11, 1e12, color='g', linestyle='dashed')
    plt.vlines(200, -1e11, 1e12, color='g', linestyle='dashed')
    plt.xlabel('$V$ (km/s)', fontsize=14)
    plt.ylabel('$N(v)$ (cm$^2$ km$^{-1}$ s)', fontsize=14)
    plt.xlim(-400, 400)
    plt.ylim(-1e12, 6e12)
    plt.savefig('CIV.Nv.pdf')
    plt.close()
