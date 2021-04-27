@author: almakurmanova
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
#%%
infile=open('/Applications/topas/examples/Thesis/Energy1.csv')
#%%
data=pd.read_csv(infile, skiprows=8, names=('X','Y','Z', 'E'))
data['E']=pd.to_numeric(data['E'])                                           
Energy=data['E']                                       
z=data['Z']
#%%
plt.figure()
plt.grid()
plt.plot(data['Z'],data['E'])
plt.xlabel('z,bin of 0.0001 cm cm')
plt.ylabel('E, MeV')
plt.title('Energy deposited along z axis')
plt.show()

#%%
sum_column = data.sum(axis=0)
print (sum_column)


