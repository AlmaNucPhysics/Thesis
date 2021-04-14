#Import the libraries

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

infile=open('C:/Users/kurma/Desktop/Master thesis/DepositedEnergy.csv')

E=pd.read_csv(infile, names=('Z', 'Energy'))
E['Energy']=pd.to_numeric(E['Energy'], errors='coerce')                                           
Energies=E['Energy']                                       
z=E['Z']


plt.figure()
plt.grid()
plt.plot(E['Z'],E['Energy'])
plt.xlabel('z,bin of 4 cm')
plt.ylabel('Energy, MeV')
plt.title('Deposited energy along z axis')
plt.show()



