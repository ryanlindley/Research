import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import chisquare

v1, f1, fe1, ctn1 = np.loadtxt('CIV1548.bin3.linespec').transpose()
fn1 = f1 / ctn1
fne1 = fe1 / ctn1

v2, f2, fe2, ctn2 = np.loadtxt('CIV1550.bin3.linespec').transpose()
fn2 = f2 / ctn2
fne2 = fe2 / ctn2

import functions

wlr1, f_val_1, gamma1, elem, state = (1548.187, 0.19, 0.00324, 'C', 'IV')
wlr2, f_val_2, gamma2, elem, state = (1550.772, 0.0952, 0.00325, 'C', 'IV')

wl1 = []
for v in v1:
    wl1.append(functions.V2Wave(v, wlr1))

wl2 = []
for v in v2:
    wl2.append(functions.V2Wave(v, wlr2))

def V2W(v, wlr):
    wl = []
    for v in v:
        wl.append(functions.V2Wave(v, wlr))
    return wl

plt.plot(wl1, fn1)
plt.plot(wl2, fn2, 'r')
plt.show()

#plt.step(wl1, fn1, 'k-', lw=2, where='mid')
#plt.step(wl2, fn2, 'r-', lw=2, where='mid')
#plt.show()
