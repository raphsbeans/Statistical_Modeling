#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:20:31 2019

@author: macbookpro
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
df_ora=pd.read_csv('ORA.PA.csv')
df_ora['ORA.PA'] = (df_ora['Close']-df_ora['Open'])*100/df_ora['Open'] 
data=df_ora[['ORA.PA']]
data.index=df_ora['Date']
s=['AC.PA.csv','SAN.PA.csv','RI.PA.csv','ENGI.PA.csv','LR.PA.csv','EN.PA.csv','SU.PA.csv','VIE.PA.csv','UG.PA.csv','ML.PA.csv','ATO.PA.csv','CAP.PA.csv','DG.PA.csv','OR.PA.csv','CA.PA.csv','AIR.PA.csv','BN.PA.csv','AI.PA.csv','MC.PA.csv','BNP.PA.csv','SGO.PA.csv','FP.PA.csv','VIV.PA.csv','ACA.PA.csv','GLE.PA.csv','KER.PA.csv','FR.PA.csv','SW.PA.csv','FTI.PA.csv']
for x in s:
    df1=pd.read_csv(x)
    df1[x[:-4]] = (df1['Close']-df1['Open'])*100/df1['Open'] 
    data[x[:-4]]=df1[x[:-4]].values
data[['AC.PA','ORA.PA','SAN.PA']].plot(kind='line')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x = StandardScaler().fit_transform(data)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.index=df_ora['Date']
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
for i in range(10):
    ax.scatter(principalDf['principal component 1'][i],principalDf['principal component 2'][i],s=5)
    ax.text(principalDf['principal component 1'][i]+0.03,principalDf['principal component 2'][i]+0.03,principalDf.index[i])
    ax.grid()
ax.plot()
(fig, ax) = plt.subplots(figsize=(12, 12))

for i in range(0,10):
    ax.arrow(0, 0,  # Start the arrow at the origin
             pca.components_[0, i], pca.components_[1, i],  # 0 and 1 correspond to dimension 1 and 2
             head_width=0.03,head_length=0.03)
    plt.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map ')
plt.show()

'''for i in range(10,20):
    ax.arrow(0, 0,  # Start the arrow at the origin
             pca.components_[0, i], pca.components_[1, i],  # 0 and 1 correspond to dimension 1 and 2
             head_width=0.03,head_length=0.03)
    plt.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map ')
plt.show()'''

'''for i in range(20,30):
    ax.arrow(0, 0,  # Start the arrow at the origin
             pca.components_[0, i], pca.components_[1, i],  # 0 and 1 correspond to dimension 1 and 2
             head_width=0.03,head_length=0.03)
    plt.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, data.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map ')
plt.show()'''