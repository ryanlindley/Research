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

lsf = np.loadtxt('CIV.lsf')

v1, f1, fe1, ctn1 = np.loadtxt('CIV1548.bin3.linespec').transpose()
fn1 = f1 / ctn1
fne1 = fe1 / ctn1
v2, f2, fe2, ctn2 = np.loadtxt('CIV1550.bin3.linespec').transpose()
fn2 = f2 / ctn2
fne2 = fe2 / ctn2



wl1, f1, gamma1, elem, state = (1548.187, 0.19, 0.00324, 'C', 'IV')
wl2, f2, gamma2, elem, state = (1550.772, 0.0952, 0.00325, 'C', 'IV')

def model(p0):
    m1 = np.ones(v1.shape)
    m2 = np.ones(v2.shape)
    Ncomp = int(len(p0)/3)
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

def print_p0(p0):
    print( ' No    logN       b      vc ')
    Ncomp = len(p0)/3
    for comp_i in range(Ncomp):
        N, b, vc = p0[comp_i*3: comp_i*3+3]
        print "%3d  %6.3f %7.2f %7.2f" % (comp_i, N, b, vc)


x0 = [13.5, 20.3, 20] + [13.5, 20.3, 40] + [14, 20.3, 150] + [13, 20.3, 170]
p0, cov, a, b, c = leastsq(fitting, x0, full_output=1)
print_p0(p0)
plot_model(p0)

def V2W(v, wlr):
    wl = []
    for v in v:
        wl.append(functions.V2Wave(v, wlr))
    return wl

def V2Wave(v, wlr):
    return ((v * (1e-13 *wlr) / 3e5) + (wlr * 1e-13))*1e13

'''
x0 = [13.5, 20.3, 20] + [13.5, 20.3, 40] + [14, 20.3, 150]
x1 = [13.5, 20.3, 20] + [13.5, 20.3, 40] + [14, 20.3, 150] + [13, 20.3, 170]

p0, cov, a, b, c = leastsq(fitting, x0, full_output=1)
p1, cov1, a1, b1, c1 = leastsq(fitting, x1, full_output=1)
#print_p0(p0)
#print_p0(p1)
#plot_model(p0)
#plot_model(p1)

m1 = [model(p0)[0], 9]
m2 = [model(p1)[0], 12]

#plt.step(v1, fn1, 'k-', lw=2, where='mid')
#plt.show()
'''

def Chi_sq_diff(data, m1, m2, er1):
    '''
    m = [model, dof]
    '''

    return Chi_sq(m1[0], data, er1) - Chi_sq(m2[0], data, er1)

def Chi_sq(model, data, error):
    return np.sum((model-data)**2 / error**2)

#print(Chi_sq(m1[0], fn1, fne1))
#print 'Chi Square Difference: ', Chi_sq_diff(fn1, m1, m2, fne1)
