@author: almakurmanova
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from topas2numpy import read_ntuple
import pandas as pd
import pylab as plb
from lmfit import Parameters, minimize, report_fit, Model
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import seaborn as sns
from scipy.optimize import curve_fit

#%%
data= read_ntuple('/Applications/topas/examples/Thesis/PhSp1.phsp')

#%%Array creation

X=np.zeros(data.shape)
Y=np.zeros(data.shape)
Z=np.zeros(data.shape)
dircosX=np.zeros(data.shape)
dircosY=np.zeros(data.shape)
E=np.zeros(data.shape)
Weight=np.zeros(data.shape)
Particle=np.zeros(data.shape)
Flag_to_tell_if_third_direction_cosine_is_negative=np.zeros(data.shape)
Flag_to_tell_if_this_is_the_first_scored_particle_from_this_history=np.zeros(data.shape)

#%% Separation of the data 

for i in range(len(data)):
    X[i],Y[i],Z[i],dircosX[i],dircosY[i],E[i],Weight[i],Particle[i],Flag_to_tell_if_third_direction_cosine_is_negative[i],Flag_to_tell_if_this_is_the_first_scored_particle_from_this_history[i],*rest=data[i]

#%% Selection of proton information

proton_energy=E[Particle==2212]
proton_x=X[Particle==2212]
proton_y=Y[Particle==2212]
proton_dircosX=dircosX[Particle==2212]
proton_dircosY=dircosY[Particle==2212]

#%%Energy distribution

plt.hist(proton_energy,bins=100)
plt.grid()
plt.xlabel('Energy [MeV]')
plt.ylabel('Histories')
plt.title('Proton energy')
plt.show()

#%%2D Histogram of particle distribution

H,xedges,yedges=np.histogram2d(proton_x,proton_y,bins=10)
X_val=(xedges[:-1]+xedges[1:])/2
Y_val=(yedges[:-1]+yedges[1:])/2

x,y=np.meshgrid(X_val,Y_val)

fig = plt.figure(figsize =(10,10))
ax = plt.axes(projection ='3d')
surf=ax.plot_surface(x,y,H,cmap=cm.coolwarm,linewidth=0,antialiased=False) 
plt.xlabel('X [cm]',fontweight='bold')
plt.ylabel('Y [cm]',fontweight='bold')
fig.colorbar(surf,ax=ax,shrink=0.7,aspect=7)
plt.show()

#%% LeastSQ Fitting 
delX=X_val[1]-X_val[0]
delY=Y_val[1]-Y_val[0]

def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y):
    return np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0) 

def residuals(p, x, y, z):
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    return (z - (height*delX*delY/(2*3.14*sigma_x*sigma_y))*gaussian2D(x,y, cen_x, cen_y, sigma_x, sigma_y))

initial = Parameters()
initial.add("height",value=1e6)
initial.add("centroid_x",value=0.001)
initial.add("centroid_y",value=0.001)
initial.add("sigma_x",value=.15)
initial.add("sigma_y",value=.15)

fit = minimize(residuals, initial, args=(x, y, H))
print(report_fit(fit))

#%% Test of the model
test=fit.params['height'].value*delX*delY/(2*3.14*fit.params['sigma_x'].value*fit.params['sigma_y'].value)*gaussian2D(x,y,fit.params['centroid_x'].value,fit.params['centroid_y'].value,fit.params['sigma_x'].value,fit.params['sigma_y'].value)
                         
fig = plt.figure(figsize =(10,10))
ax = plt.axes(projection ='3d')
surf=ax.plot_surface(x,y,test,cmap=cm.coolwarm,linewidth=0,antialiased=False) 
plt.xlabel('X [cm]',fontweight='bold')
plt.ylabel('Y [cm]',fontweight='bold')
fig.colorbar(surf,ax=ax,shrink=0.7,aspect=7)
plt.show()

#%% Visualize the results (model and data)
fig = plt.figure(figsize = (50,50))
ax = Axes3D(fig)
ax.plot_wireframe(x,y,test, label = 'Model')
ax.scatter(x, y, H,color ='r', label = 'Data')
plt.xlabel('X [cm]',fontweight='bold')
plt.ylabel('Y [cm]',fontweight='bold')

plt.legend()

#%% Particle distribution

plt.figure()
plt.scatter(proton_x,proton_y,label="data")
plt.xlabel('X, cm')
plt.ylabel('Y, cm')
plt.title('Proton distibution')
plt.grid()
plt.show()

#%% Joint graph for partcile distribution

df=pd.DataFrame({"x":proton_x,"y":proton_y})
sns.jointplot(data=df,x='x',y='y',kind='resid')
plt.grid()

#%% PHASE SPACE IN X

plt.figure()
plt.scatter(proton_x,np.arccos(proton_dircosY))
plt.xlabel('X,cm')
plt.ylabel('X`, rad')
plt.title('Beam Phase-Space distibution')
plt.grid()
plt.show()
#%% PHASE SPACE IN Y

plt.figure()
plt.scatter(proton_y,np.arccos(proton_dircosY))
plt.xlabel('Y,cm')
plt.ylabel('Y`, rad')
plt.title('Beam Phase-Space distibution')
plt.grid()
plt.show()
