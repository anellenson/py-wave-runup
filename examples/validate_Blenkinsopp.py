import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import py_wave_runup

#generate datasets
Hs_salt = np.linspace(0.65, 2.54,100)
steepness = np.linspace(0.004, 0.033, 100)
Lp_salt = Hs_salt/steepness
Tp_salt = np.sqrt((Lp_salt*2*np.pi)/9.81)
beta_berm = np.ones(100) * 0.18
beta_sand = np.ones(100) * 0.015
dtoe = np.ones(100) *0.45

df = pd.DataFrame({'Hs':Hs_salt, 'Tp':Tp_salt, 'bberm':beta_berm, 'bsand':beta_sand, 'dtoe':dtoe})
blen22 = py_wave_runup.models.Blenkinsopp2022(Hs = df.Hs,beta=df.bberm,Tp=df.Tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoe)
r2_salt = blen22.R2_eq21
dtoetanberm = blen22.dtoe * blen22.bberm

fig, ax = pl.subplots(3,1)
ax[0].plot(blen22.dtoe,blen22.Htoe,'b.',markersize=2)
ax[0].set_xlabel('dtoe')
ax[0].set_ylabel('Htoe')
ax[0].set_xlim((0,2))
ax[0].set_title('Figure 6')
ax[0].set_ylim((0,2.5))

ax[1].plot(dtoetanberm, r2_salt,'b.', markersize=2)
ax[1].set_xlabel('dtoe*tan(berm)')
ax[1].set_ylabel('r2%')
ax[1].set_title('Figure 10')
ax[1].set_xlim((0,0.6))
ax[1].set_ylim((0,3.5))

ax[2].plot(df.Hs, blen22.setup_toe,'b.', markersize=2)
ax[2].set_xlabel('H0')
ax[2].set_xlabel('eta')
ax[2].set_title('Figure 12')
ax[2].set_xlim((0,6))
ax[2].set_ylim((0,1))
