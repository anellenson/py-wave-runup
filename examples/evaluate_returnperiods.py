import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import py_wave_runup
import seaborn as sns

#Validate with two points on dtoe*tanberm
Tp_returnvals = pd.read_csv('data\Tp_return_vals.csv')
Tp=np.reshape(Tp_returnvals.drop(columns='return period').values, (15,))
Hs_returnvals = pd.read_csv('data\Hs_return_vals.csv')
Hs=np.reshape(Hs_returnvals.drop(columns='return period').values, (15,))
bberm = [0.14]*Hs.shape[0]
bsand = [0.012]*Hs.shape[0]
dtoeSWL = [0.8]*Hs.shape[0]
returnperiod = [5]*3 + [10] * 3 + [20]*3 + [50]*3 + [100]*3
df = pd.DataFrame({'hs':Hs, 'tp':Tp, 'bsand':bsand,'bberm':bberm, 'dtoeSWL':dtoeSWL})
blen22 = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand,spectral_wave_period=True, dtoeSWL =df.dtoeSWL,h=10)
eurotop = py_wave_runup.models.EurOtop2018(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoeSWL, spectral_wave_period=True,h=10)
stock = py_wave_runup.models.Stockdon2006(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoeSWL, spectral_wave_period=True,h=10)
rug01 = py_wave_runup.models.Ruggiero2001(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.bsand, dtoeSWL =df.dtoeSWL, spectral_wave_period=True,h=10)

df_eurotop = df.copy(deep=True)
df_stock = df.copy(deep=True)
df_rug = df.copy(deep=True)

df['r2'] =  blen22.R2_eq21*3.3
df['method'] = 'Blenkinsopp'
df['return period'] = returnperiod

df_eurotop['r2'] =  eurotop.R2(gamma_f=0.75)*3.3
df_eurotop['method'] = 'TAW'
df_eurotop['return period'] = returnperiod

df_stock['r2'] = stock.R2*3.3
df_stock['method'] = 'Stockdon'
df_stock['return period'] = returnperiod

df_rug['r2'] = rug01.R2*3.3
df_rug['method'] = 'Ruggiero2001'
df_rug['return period'] = returnperiod


df = pd.concat((df, df_eurotop,df_stock))

with open('df_setup_blenkinsopp.pickle', 'rb') as f:
    df_blenset = pickle.load(f)

df_blenset['setup_method'] = 'Blenkinsopp fit'
df['setup_method'] = 'FEMA estimation'

df = pd.concat((df,df_blenset))
df.drop(df[df.method == 'Stockdon'])
fig, ax = pl.subplots(1,1)
sns.barplot(x='method',y='r2',hue='setup_method',data=df)
ax.set_xlabel('return period (yrs)')
ax.set_ylabel('r2% (ft)')
pl.legend(bbox_to_anchor=(.6, 1.1),ncol=2)

with open('blenkinsopp2022_setup.pickle', 'rb') as f:
    blensetup = pickle.load(f)

blensetup = blensetup['eta']
pl.figure()
pl.scatter(Hs, blensetup, marker = '*')
pl.scatter(Hs, blen22.setup_toe, marker = '^')
pl.xlabel('Hs [m]')
pl.ylabel(r'$\overline{\eta}$')
pl.legend(('Blenkinsopp 2022 Fit', 'FEMA guidelines'))
