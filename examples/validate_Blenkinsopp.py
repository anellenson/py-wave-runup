import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import py_wave_runup
import seaborn as sns

#Validate with two points on dtoe*tanberm
bberm = [0.24,0.3]
bsand = [0.06,0.067]
Hs = [0.8,0.8]
Tp = [6,6]
dtoeSWL = [0.2,0.2]
df = pd.DataFrame({'hs':Hs, 'tp':Tp, 'bsand':bsand,'bberm':bberm, 'dtoeSWL':dtoeSWL})
blen22 = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoeSWL)
r2 = blen22.R2_eq21
dtoetanberm = blen22.dtoe * np.tan(blen22.bberm)

df['r2'] = r2
df['dtoe_tanberm'] = dtoetanberm
df['Htoe']=blen22.Htoe
df['setup_toe']=blen22.setup_toe
df['dtoe'] = blen22.dtoe

print('r2% = {0:.2f}, {1:.2f} for DR and 2DR, similar to values found in Figure 10 in Belkinsopp 2022'.format(r2[0], r2[1]))
