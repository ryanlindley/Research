#!/usr/bin/env python
# coding: utf-8

# In[1]:


import auto_fitting as auto
import glob
import os
auto.plt.rcParams['figure.figsize'] = [11, 8]


# In[2]:


filenames = glob.glob('/Users/ryanlindley/Research/CIV/*')
CIV_targets = []
for f in filenames:
    CIV_targets.append(f.replace('/Users/ryanlindley/Research/CIV/', ''))
    
#print(CIV_targets)


# In[3]:


name = CIV_targets[73]
print(name)
print(len(CIV_targets))


# In[4]:


directory = '/Users/ryanlindley/Research/CIV/' + name
os.chdir(directory)
os.getcwd()


# In[5]:


wl1, f1, gamma1, elem, state = (1548.187, 0.19, 0.00324, 'C', 'IV')
wl2, f2, gamma2, elem, state = (1550.772, 0.0952, 0.00325, 'C', 'IV')

path = '/Users/ryanlindley/Research/CIV/' + name + '/CIV.bin3.linespec'
wl, fl, fe, ctn = auto.np.loadtxt(path).transpose()
lsf = auto.np.loadtxt('/Users/ryanlindley/Research/CIV.old_setup/CIV.lsf')


# In[6]:


#new_ctn = auto.find_continuum(name, wl, fl, fe, ctn, [True]*len(wl))

ctn_flag = auto.mask_wl(wl, [1546.2, 1547, 1547.5, 1548.5, 1550, 1551])  # add custom ctn if needed 
manual_ctn = auto.find_continuum_polyfit(name, wl, fl, fe, ctn, ctn_flag)


# In[7]:


def check_larger_continuum(name, wl, ctn):
    spec = auto.np.loadtxt('/Users/ryanlindley/Research/CIV/' + name + '/full.spec')
    auto.plt.plot(spec[:,0], spec[:,1])
    auto.plt.plot(wl, ctn)
    auto.plt.xlim(1542, 1556)
    auto.plt.ylim(0, 0.7e-14)
    auto.plt.show
    
check_larger_continuum(name, wl, manual_ctn)


# In[8]:


new_ctn = manual_ctn # only include when manual continuum us found

fn = fl / new_ctn
fne = fe / new_ctn


# In[9]:


#chi, chi_mean = auto.find_chi(wl, fl, fe, new_ctn)
#auto.plot_chi_histogram(chi, chi_mean)

#flag_chi = auto.mask_wl(wl, [1546, 1546.7, 1548, 1551.2]) # mask out regions for better calculation of chi histogram

#new_chi, new_chi_mean = auto.find_chi(wl[flag_chi], fl[flag_chi], fe[flag_chi], new_ctn[flag_chi])
#auto.plot_chi_histogram(new_chi, new_chi_mean)


# In[10]:


x0 = [13.5, 20.3, auto.Wave2V(1547.7, wl1)] + [13.5, 20.3, auto.Wave2V(1548, wl1)] + [13.3, 20.3, auto.Wave2V(1548.5, wl1)] + [13.3, 20.3, auto.Wave2V(1550.7, wl2)] #+ [13.5, 20.3, auto.Wave2V(1549.8, wl1)]  + [13.2, 20.3, auto.Wave2V(1552.6, wl2)] #+ [13.4, 20.3, auto.Wave2V(1550.6, wl2)] #+ [13.2, 20.3, auto.Wave2V(1552.5, wl2)] + [13.3, 20.3, auto.Wave2V(1551.5, wl2)] + [13.2, 20.3, auto.Wave2V(1550.6, wl2)] #+ [13.5, 20.3, auto.Wave2V(1549.6, wl2)]
feat = [0, 0, 1, 2] #which features used to model 0 - both, 1 - strong, 2 - weak
p0, cov, a, b, c = auto.leastsq(auto.fitting, x0, full_output=1, args=(feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf, fn, fne, [1547, 1553]))

auto.plot_model(p0, feat, wl, fn, wl1, wl2, f1, f2, gamma1, gamma2, lsf, [1547, 1553]) #normally [1547, 1553]
print(p0)


# In[11]:


p0 = auto.make_ten(p0)
N1, N2 = auto.make_features(p0, feat, wl, wl1, wl2, lsf, gamma1, gamma2)
Ne1 = auto.nfle2Nev(fn, fne, f1, wl1)
Ne2 = auto.nfle2Nev(fn, fne, f2, wl2)
auto.plot_features(wl, wl1, wl2, N1, N2, Ne1, Ne2, [-500, 500])


print(p0)


# In[12]:


N1r, N2r = auto.add_residual(p0, N1, N2, feat, wl, wl1, wl2, f1, f2, gamma1, gamma2, lsf, fn)
auto.plot_features(wl, wl1, wl2, N1r, N2r, Ne1, Ne2, [-500, 500])


# In[13]:


strong_flag = auto.mask_v(wl, wl1,[]) # add masking for any regions not to be used in combined data
weak_flag = auto.mask_v(wl, wl2, [-60, 0])
v1, v2, Nv1, Nv2, Nve1, Nve2 = auto.remove_regions(strong_flag, weak_flag, wl, wl1, wl2, N1r, N2r, Ne1, Ne2)

v_bins, Nv_bins, Nve_bins = auto.make_bins(v1, v2, Nv1, Nv2, Nve1, Nve2)

v_final, Nv_final, Nve_final =  auto.final_data(v_bins, Nv_bins, Nve_bins)

auto.plot_final_data(wl, wl1, wl2, N1r, N2r, v_final, Nv_final, Nve_final, name, [-300, 300])


# In[14]:


data = auto.np.c_[v_final, Nv_final, Nve_final]
CIV_regions = [[-150, 75]]
cont_regions = [[-60, 0]]

auto.np.savetxt('CIV.data', data)
auto.np.savetxt('CIV.regions', CIV_regions, fmt='%1.3i')
auto.np.savetxt('contamintion.regions', cont_regions, fmt='%1.3i')
auto.save_final_data_plot(wl, wl1, wl2, N1r, N2r, v_final, Nv_final, Nve_final, name, [-300, 300])
auto.save_final_continuum_data(wl, fl, new_ctn, name)


# In[15]:


#remember to save file 


# In[ ]:





# In[ ]:




