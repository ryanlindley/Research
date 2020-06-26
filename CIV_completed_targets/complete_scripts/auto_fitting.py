import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy import special
from scipy.interpolate import UnivariateSpline
import glob

def continuum(wl, fl, fe, threshold):
    fl_out, wl_out, fe_out = trim(wl, fl, 2, np.mean(fl), fe)
    ctn_func = UnivariateSpline(wl_out, fl_out)
    ctn = ctn_func(wl_out)
    i, fl_prev = 0, fl
    while(len(fl_prev) != len(fl_out) or i > 20):
        fl_prev = fl_out
        fl_out, wl_out, fe_out = trim(wl_out, fl_out, threshold, ctn, fe_out)
        ctn_func = UnivariateSpline(wl_out, fl_out)
        ctn = ctn_func(wl_out)
        i += 1
    return wl_out, fl_out, fe_out, ctn_func
    
def plot_continuum(wl, wl_out, fl, fl_out, ctn_given, ctn_func, name, t):
    plt.step(wl, fl, 'g-', lw=2, where='mid', label='Data')
    plt.plot(wl_out, fl_out, 'k-', label='Trimmed Data')
    plt.plot(wl, ctn_func(wl), '-r', label='Fitted Continuum')
    plt.plot([],[],' ', label=('Threshold: ' + str(t)))
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    plt.title(name)
    plt.legend()
    plt.show()
    
def plot_chi_histogram(chi, chi_mean):
    mean = 'Mean: ' + str(chi_mean)
    plt.hist(chi)
    plt.plot([],[],'', label=mean)
    plt.title('Chi Values')
    plt.xlabel('Chi')
    plt.legend()
    plt.show()

def find_chi(wl, fl, fe, ctn):
    lower = np.min(np.where(wl > 1547))
    upper = np.max(np.where(wl < 1552))
    flag = np.invert((wl > 1547) & (wl < 1552))
    chi = Chi(ctn[flag], fl[flag], fe[flag])
    return chi, np.mean(chi)

def trim(wl, fl, threshold, reference, fe):
    flag = fl > reference - threshold*fe
    fl_new = fl[flag]
    wl_new = wl[flag]
    fe_new = fe[flag]
    return fl_new, wl_new, fe_new
    
def Chi_sq(model, data, error):
    return np.sum((model-data)**2 / error**2)

def Chi(model, data, error):
    return (data - model) / error

def find_continuum(name, wl, fl, fe, ctn, flag):
    threshold = np.arange(1.2, 2, .1)
    t = threshold[0]
    wl_out, fl_out, fe_out, ctn_func = continuum(wl[flag], fl[flag], fe[flag], t)
    chi, chi_mean = find_chi(wl, fl, fe, ctn_func(wl))
    for th in threshold:
        wl_1, fl_1, fe_1, ctn_function = continuum(wl[flag], fl[flag], fe[flag], th)
        chi_1, chi_mean_1 = find_chi(wl, fl, fe, ctn_function(wl))
        if (np.abs(chi_mean_1) < np.abs(chi_mean)):
            wl_out, fl_out, fe_out, ctn_func = wl_1, fl_1, fe_1, ctn_function
            t = th
            chi, chi_mean = chi_1, chi_mean_1
    plot_continuum(wl, wl_out, fl, fl_out, ctn, ctn_func, name, round(t,1))
#    plot_chi_histogram(chi, chi_mean)
    return ctn_func(wl)

def V2Wave(v, wlr):
    return ((v * (1e-13 *wlr) / 3e5) + (wlr * 1e-13))*1e13

def Wave2V(wl, wlr):
    return ((wl / 1e13) - (wlr * 1e-13)) * 3e5 / (1e-13 *wlr)

def model(p0, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf):
    m = np.ones(wl.shape)
    
    Ncomp = len(p0)//3
    for comp_i in range(Ncomp):
        N, b, vc = p0[comp_i*3: comp_i*3+3]
        if feat[comp_i] == 0:
            m1 = vline(Wave2V(wl, wl1), [b/1.414, gamma1, N2tau(10**N, wl1, f1), vc, 1])
            m2 = vline(Wave2V(wl, wl2), [b/1.414, gamma2, N2tau(10**N, wl2, f2), vc, 1])
            m*= m1 * m2
        elif feat[comp_i] == 1:
            m1 = vline(Wave2V(wl, wl1), [b/1.414,  gamma1, N2tau(10**N, wl1, f1), vc, 1])
            m2 = np.ones(wl.shape)
            m*= m1 * m2
        elif feat[comp_i] == 2:
            m1 = np.ones(wl.shape)
            m2 = vline(Wave2V(wl, wl2), [b/1.414,  gamma2, N2tau(10**N, wl2, f2), vc, 1])
            m*= m1 * m2
    m = convolve(m, lsf)
    return m
    
def fitting(p0, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf, fn, fne, limits):
    m = model(p0, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf)
    chi = (fn - m) / fne
#    chi1 = chi[(wl > 1547) & (wl < 1550)]
#    chi2 = chi[(wl > 1550) & (wl < 1553)]
    chi1 = chi[(wl > limits[0]) & (wl < 1550)]
    chi2 = chi[(wl > 1550) & (wl < limits[1])]
    return np.append(chi1, chi2)

def convolve(model, lsf):
    Nlsf = len(lsf) // 2
    ext_model = np.append(np.ones(Nlsf)*model[0], model)
    ext_model = np.append(ext_model, np.ones(Nlsf)*model[-1])
    cmodel = np.convolve(ext_model, lsf, 'valid')
    return cmodel

def N2tau(N, wlr, f):
    tau = N * 2.654e-15 * f * wlr
    return tau

def vline(x, p):
    '''
    p = [sigma, gamma, tau, v0, covering_factor]
    saturated_factor is a newly defined parameter to describe the thermal radiation varying in 0 to 1.
    The simplified model is c_f * exp(-tau) + (1-c_f).
    '''
    x = np.array(x)
    pro = abs(p[2]) * Voigt(x, p[3], p[0], abs(p[1]))

    return abs(p[4]) * np.exp(-pro) + (1 - abs(p[4]))

def Voigt(x, x0, sigma, gamma):
    '''
    x should be an array.
    sigma is the Doppler parameter.
    gamma is the Gaussian sigma.
    x0 is the center position.
    '''
    z = ((x-x0) + 1j * abs(gamma)) / abs(sigma) / 2**0.5

    return (special.wofz(z)).real/abs(sigma)/(2*np.pi)**0.5

def plot_model(p0, feat, wl, fn, wl1, wl2, f1, f2, gamma1, gamma2, lsf, limits):
    m = model(p0, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf)
    plt.step(wl, fn, 'k-', lw=2, where='mid')
    plt.plot(wl, m, 'r--', lw=2)
    
    Ncomp = len(p0)//3
    
    for comp_i in range(Ncomp):
        N, b, vc = p0[comp_i*3: comp_i*3+3]
        if feat[comp_i] == 0:
            m1 = vline(Wave2V(wl, wl1), [b/1.414, gamma1, N2tau(10**N, wl1, f1), vc, 1])
            m2 = vline(Wave2V(wl, wl2), [b/1.414, gamma2, N2tau(10**N, wl2, f2), vc, 1])
            m = m1 * m2
            m = convolve(m, lsf)
            plt.plot(wl, m, 'y:', lw=2)
        elif feat[comp_i] == 1:
            m1 = vline(Wave2V(wl, wl1), [b/1.414,  gamma1, N2tau(10**N, wl1, f1), vc, 1])
            m2 = np.ones(wl.shape)
            m = m1 * m2
            m = convolve(m, lsf)
            plt.plot(wl, m, 'y:', lw=2)
        elif feat[comp_i] == 2:
            m1 = np.ones(wl.shape)
            m2 = vline(Wave2V(wl, wl2), [b/1.414, gamma2, N2tau(10**N, wl2, f2), vc, 1])
            m = m1 * m2
            m = convolve(m, lsf)
            plt.plot(wl, m, 'y:', lw=2)
    
    plt.xlim(limits[0], limits[1])
    plt.show()

def make_features(p0, feat, wl, wl1, wl2, lsf, gamma1, gamma2):
    length = len(p0) // 3
    Ns_tot = 0
    Nw_tot = 0
    for i in range(length):
        if feat[i] == 0:
            Ns_tot += 10**(p0[3 * i])*Voigt(Wave2V(wl, wl1), p0[2 + 3*i], p0[1 + 3*i]/np.sqrt(2), gamma1)
            Nw_tot += 10**(p0[3 * i])*Voigt(Wave2V(wl, wl2), p0[2 + 3*i], p0[1 + 3*i]/np.sqrt(2), gamma2)
        elif feat[i] == 1:
            Ns_tot += 10**(p0[3 * i])*Voigt(Wave2V(wl, wl1), p0[2 + 3*i], p0[1 + 3*i]/np.sqrt(2), gamma1)
        elif feat[i] == 2:
            Nw_tot += 10**(p0[3 * i])*Voigt(Wave2V(wl, wl2), p0[2 + 3*i], p0[1 + 3*i]/np.sqrt(2), gamma2)
    Ns = convolve(Ns_tot, lsf)
    Nw = convolve(Nw_tot, lsf)
    return Ns, Nw

def add_residual(p0, N1, N2, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf, fn):
    m = model(p0, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf)
    r = fn / m
    N1 += N_v(r, wl1, f1)
    N2 += N_v(r, wl2, f2)
    return N1, N2

def N_v(fn, wlr, f):
    return -np.log(fn)/wlr/f/2.654e-15

def nfle2Nev(nfl, nfle, f, wlr):
    Nev = nfle/nfl / 2.654e-15/f/wlr
    return Nev
    
def final_data(v_bins, Nv_bins, Nve_bins):
    v_new = []
    Nv_new = []
    Nve_new = []
    for i in np.arange(len(v_bins)):
        if len(v_bins[i]) == 0:
            v_new.append(0)
            Nv_new.append(0)
            Nve_new.append(0)
        else:
            v_new.append(np.mean(v_bins[i]))
            Nv_new.append(np.sum((Nv_bins[i] * (1 / Nve_bins[i]**2)) / np.sum((1 / Nve_bins[i]**2))))
            Nve_new.append(1 / np.sqrt(np.sum((1 / Nve_bins[i]**2))))
    
    return v_new, Nv_new, Nve_new

def plot_features(wl, wl1, wl2, N1, N2, Ne1, Ne2, limits):
    plt.plot(Wave2V(wl, wl1), N1, '-r', label='Strong Feature')
    plt.plot(Wave2V(wl, wl2), N2, '-k', label='Weak Feature')
    plt.plot(Wave2V(wl, wl1), Ne1, '-b', label='Strong Error')
    plt.plot(Wave2V(wl, wl2), Ne2, '-m', label='Weak Error')
    plt.legend()
    plt.xlim(limits[0], limits[1])
    #plt.ylim(-0.5e13, 1e13)
    plt.show()

def remove_regions(strong_flag, weak_flag, wl, wl1, wl2, N1r, N2r, Ne1, Ne2):
    v1 =  Wave2V(wl, wl1)[strong_flag]
    Nv1 = N1r[strong_flag]
    Nve1 = Ne1[strong_flag]

    v2 =  Wave2V(wl, wl2)[weak_flag]
    Nv2 = N2r[weak_flag]
    Nve2 = Ne2[weak_flag]
    return v1, v2, Nv1, Nv2, Nve1, Nve2
    
def make_bins(v1, v2, Nv1, Nv2, Nve1, Nve2):
    
    Nv = np.append(Nv1, Nv2)
    Nve = np.append(Nve1, Nve2)
    v = np.append(v1, v2)

    idx = np.argsort(v)
    v_sorted = v[idx]
    Nv_sorted = Nv[idx]
    Nve_sorted = Nve[idx]

    New_v = np.linspace(-400, 400, 101) # dv ~ 8
    indices = np.searchsorted(v_sorted, New_v)
    v_bins = np.split(v_sorted, indices)
    Nv_bins = np.split(Nv_sorted, indices)
    Nve_bins = np.split(Nve_sorted, indices)
    return v_bins, Nv_bins, Nve_bins

#v_bins, Nv_bins, Nve_bins = make_bins(v1, v2, Nv1, Nv2, Nve1, Nve2)



#v_final, Nv_final, Nve_final =  final_data(v_bins, Nv_bins, Nve_bins)

def plot_final_data(wl, wl1, wl2, N1r, N2r, v_final, Nv_final, Nve_final, name, limits):
    plt.plot(v_final, Nv_final,'k-', lw=3, label='Combined Nv')
    plt.plot(v_final, Nve_final, 'b:', lw=3, label='Combined Error')
    plt.plot(Wave2V(wl, wl1), N1r, '--m', label='Strong Feature')
    plt.plot(Wave2V(wl, wl2), N2r, '--g', label='Weak Feature')
    plt.xlim(limits[0], limits[1])
    plt.title(name)
    plt.axhline(0)
    plt.legend()
    plt.show()

#plot_final_data(wl, wl1, wl2, N1r, N2r, v_final, Nv_final, Nve_final)

def mask_wl(wl, ranges):
    length = len(ranges)
    flag = [True] * len(wl)
    if length == 0:
        return flag
    width = length // 2
    for i in range(width):
        flag &= np.invert((wl > ranges[2 * i]) & (wl < ranges[2 * i + 1]))
    return flag
    
def mask_v(wl, w, ranges):
    length = len(ranges)
    flag = [True] * len(wl)
    if length == 0:
        return flag
    width = length // 2
    for i in range(width):
        flag &= np.invert((Wave2V(wl, w) > ranges[2 * i]) & (Wave2V(wl, w) < ranges[2 * i + 1]))
    return flag
        
def save_final_data_plot(wl, wl1, wl2, N1r, N2r, v_final, Nv_final, Nve_final, name, limits):
    plt.plot(v_final, Nv_final,'k-', lw=3, label='Combined Nv')
    plt.plot(v_final, Nve_final, 'b:', lw=3, label='Combined Error')
    plt.plot(Wave2V(wl, wl1), N1r, '--m', label='Strong Feature')
    plt.plot(Wave2V(wl, wl2), N2r, '--g', label='Weak Feature')
    plt.xlim(limits[0], limits[1])
    plt.title(name)
    plt.axhline(0)
    plt.legend()
    plt.savefig('CIV.N_v.pdf')
    plt.close()

def save_final_continuum_data(wl, fl, ctn, name):
    plt.step(wl, fl, 'g-', lw=2, where='mid')
    plt.plot(wl, ctn, '-k', lw=2)
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    plt.title(name)
    plt.savefig('CIV.flux_wl.pdf')
    plt.close()

def find_N(min, max, v_final, Nv_final):
    v = np.array(v_final)
    Nv = np.array(Nv_final)
    flag = (v > min) & (v < max)
    dv = np.mean(np.diff(v))
    return np.log10(np.sum(Nv[flag] * dv))
