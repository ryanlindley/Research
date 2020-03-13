import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import UnivariateSpline


def gnrt_splctn(dv, fl, fle, sigma=1.5):
    """
    Generate continuum using Spline Fitting.
    """
    flag = fl - np.median(fl) > -1.5 * fle
    spl = UnivariateSpline(dv[flag], fl[flag], 1.0 / fle[flag])
    spl.set_smoothing_factor(len(dv[flag]))
    ctn = spl(dv)
    flag = fl - ctn > -sigma * fle
    len_ctn = len(flag)
    i = 0
    while np.sum(flag) != len_ctn and i < 50:
        len_ctn = np.sum(flag)
        spl = UnivariateSpline(dv[flag], fl[flag], 1.0 / fle[flag])
        spl.set_smoothing_factor(len(dv[flag]))
        ctn = spl(dv)
        flag = fl - ctn > -sigma * fle
        i += 1

    return spl(dv)

def rebin(input_f, bins=3):
    line = np.loadtxt(input_f)
    dv, fl, fle, wl, ctn = line.transpose()
    N_new = len(fl) / bins
    dv_n = dv[1:bins*N_new:bins]
    fl_n = np.mean(fl[:bins*N_new].reshape(-1, bins), axis=1)
    fle_n = np.sum(fle[:bins*N_new].reshape(-1, bins)**2, axis=1)**0.5 / bins
    ctn_n = gnrt_splctn(dv_n, fl_n, fle_n)
    prefix = input_f.split('.')[0]
    output_f = prefix+'.bin'+str(bins)+'.linespec'
    np.savetxt(output_f, zip(dv_n, fl_n, fle_n, ctn_n), fmt='%.9e')

    name = glob.glob('*_CIV.pdf')[0]
    target = name.split('_CIV')[0]
    plt.text(-200, np.mean(ctn_n)/2., target)

    plt.step(dv, fl, 'k', where='mid', lw=3)
    plt.step(dv_n, fl_n, 'r', where='mid', lw=3)
    plt.plot(dv_n, ctn_n, 'y--', lw=2)
    plt.plot(dv_n, fle_n, 'c-', lw=2)
    plt.xlabel('$V$ (km/s)', fontsize=14)
    plt.ylabel(r'Flux (erg/s/cm$^2$/$\rm~\AA$)', fontsize=14)
    plt.xlim(-400, 400)
    plt.savefig(prefix+'_binned.pdf')
    plt.close()

def correct_ctn_norm(fl, fle, ctn):
    flag = fl - ctn > -1.5 * fle
    mean_shift = np.median(fl[flag] - ctn[flag])
    return ctn + mean_shift

def correct_ctn_none(fl, fle, ctn):
    return ctn

correct_ctn = correct_ctn_norm

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
    #plt.savefig('CIV.Nv.pdf')
    plt.show()
    #plt.close()

if __name__ == '__main__':
    rebin('CIV1548.linespec')
    rebin('CIV1550.linespec')
    l1 = np.loadtxt('CIV1548.bin3.linespec')
    l2 = np.loadtxt('CIV1550.bin3.linespec')
    Nv(l1, l2)
