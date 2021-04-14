# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:24:15 2021

@author: kurma
"""

#%% PhaseSpace
import numpy as np
from matplotlib import pyplot as plt
from topas2numpy import read_ntuple

#%%
x = read_ntuple('/Applications/topas/examples/Thesis/ASCIIOutput.header')

#%%
X=np.zeros(x.shape)
Y=np.zeros(x.shape)
Z=np.zeros(x.shape)
dircosX=np.zeros(x.shape)
dircosY=np.zeros(x.shape)
E=np.zeros(x.shape)
Weight=np.zeros(x.shape)
ID=np.zeros(x.shape)

#%%
for i in range(len(x)):
    X[i],Y[i],Z[i],dircosX[i],dircosY[i],E[i],Weight[i],ID[i],*rest=x[I]

#%% 
proton_energy=E[ID==2212]
proton_x=X[ID==2212]
proton_y=Y[ID==2212]
proton_dircosX=dircosX[ID==2212]
proton_dircosY=dircosY[ID==2212]
#%%
plt.hist(proton_energy)
plt.xlabel('Energy, MeV')
plt.ylabel('Histories')
plt.title('Proton energy')
plt.show()

#%%

plt.figure()
plt.scatter(proton_x,proton_y)
plt.xlabel('X, cm')
plt.ylabel('Y, cm')
plt.title('Proton distibution')
plt.grid()
plt.show()

#%%

plt.figure()
plt.scatter(proton_x,np.arccos(proton_dircosY))
plt.xlabel('X,cm')
plt.ylabel('X`, rad')
plt.title('Beam Phase-Space distibution')
plt.grid()
plt.show()
#%%

plt.figure()
plt.scatter(proton_y,np.arccos(proton_dircosY))
plt.xlabel('Y,cm')
plt.ylabel('Y`, rad')
plt.title('Beam Phase-Space distibution')
plt.grid()
plt.show()