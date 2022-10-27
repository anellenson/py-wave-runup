import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import py_wave_runup
import seaborn as sns

#Validate with two points on dtoe*tanberm
Tp_returnvals = pd.read_csv('data\Tp_return_vals.csv')
Tp=np.reshape(Tp_returnvals.drop(columns='return period').values, (15,1))
Hs_returnvals = pd.read_csv('data\Hs_return_vals.csv')
Hs=np.reshape(Hs_returnvals.drop(columns='return period').values, (15,1))
bberm = [0.24]*Hs.shape[0]
bsand = [0.008]*Hs.shape[0]
dtoeSWL = [0.1]*Hs.shape[0]
df = pd.DataFrame({'hs':Hs, 'tp':Tp, 'bsand':bsand,'bberm':bberm, 'dtoeSWL':dtoeSWL})
blen22 = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand,spectral_wave_period=True, dtoeSWL =df.dtoeSWL,h=10)
eurotop = py_wave_runup.models.EurOtop2018(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoeSWL, spectral_wave_period=True,h=10)
r2 = blen22.R2_eq21
r2_eurotop = eurotop.R2()

df['r2'] = r2
