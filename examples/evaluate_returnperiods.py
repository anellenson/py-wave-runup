import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import py_wave_runup
import seaborn as sns

#Wave dataset
Tp_returnvals = pd.read_csv('data\Tp_return_vals.csv')
Tp=np.reshape(Tp_returnvals.drop(columns='return period').values, (15,))
Hs_returnvals = pd.read_csv('data\Hs_return_vals.csv')
Hs=np.reshape(Hs_returnvals.drop(columns='return period').values, (15,))

#Transect data
existing = 'data\Adamson_M_Z_existing.csv'
existing_data = pd.read_csv(existing)

returnperiod = [5]*3 + [10] * 3 + [20]*3 + [50]*3 + [100]*3
returnperiod = np.tile(returnperiod, existing_data['beta'].unique().shape[0])
Hs_repeated = np.tile(Hs, existing_data['beta'].unique().shape[0])
Tp_repeated = np.tile(Tp, existing_data['beta'].unique().shape[0])

beta = []
transect = []
for tt,bb in zip(existing_data.Transect, existing_data.beta):
    transect += [tt]*len(Hs)
    beta += [bb]*len(Hs)

df = pd.DataFrame({'hs':Hs_repeated, 'tp':Tp_repeated, 'beta':beta, 'transect':transect, 'returnperiod':returnperiod})
df['beta'] = df.beta.values.round(3)
stock = py_wave_runup.models.Stockdon2006(Hs=df.hs,beta=df.beta,Tp=df.tp,spectral_wave_period=True,h=10)

df['r2'] = stock.R2*3.3
df['method'] = 'Stockdon'

fig, ax = pl.subplots(3,1, tight_layout = {'rect':[0,0,1,0.98]})
fig.set_size_inches(3,9)
pl.tight_layout()
for i in range(3):
    sns.barplot(x='returnperiod',y='r2',hue='beta',data=df[df.transect == i+1], ax =ax[i])
    ax[i].set_xlabel('return period (yrs)')
    ax[i].set_ylabel('r2% (ft)')
    ax[i].set_ylim((0,12))
    ax[i].set_title('Transect {0}'.format(i+1))
    legend_labels, _= ax[i].get_legend_handles_labels()
    ax[i].legend(title=r'$\beta_{sand}$')

    if i < 2:
        ax[i].set_xticks([])
        ax[i].set_xlabel('')
pl.suptitle("Stockdon Existing Transect")
pl.savefig('stockdon_existing.png')
########
''' Design conditions '''
design = 'data\Adamson_M_Z_design.csv'
import itertools
design_data = pd.read_csv(design)
transectnums = []
paramsdict = {'bberm':[], 'bsand':[],'dtoeSWL':[]}
hs = []
tp = []
returnperiod = []
for i in design_data.transect.unique():
    num_unique_conds = design_data[design_data.transect==i].bsand.unique().shape[0]*design_data[design_data.transect==i].bberm.unique().shape[0]*design_data[design_data.transect==i].dtoeSWL.unique().shape[0]
    rp1 = [5]*3 + [10] * 3 + [20]*3 + [50]*3 + [100]*3
    rp = np.tile(rp1, num_unique_conds)
    Hs_repeated = np.tile(Hs, num_unique_conds)
    Tp_repeated = np.tile(Tp, num_unique_conds)
    returnperiod += list(rp)
    hs += list(Hs_repeated)
    tp += list(Tp_repeated)
    bbermlist = design_data[(design_data.transect==i)]['bberm'].unique()
    bsandlist = design_data[design_data.transect==i]['bsand'].unique()
    dtoelist = design_data[design_data.transect==i]['dtoeSWL'].unique()
    res = list(itertools.product(*[bbermlist, bsandlist, dtoelist, Hs]))
    for pi,param in enumerate(['bberm', 'bsand', 'dtoeSWL']):
        paramsdict[param] += [res[i][pi] for i in range(len(res))]
    transectnums += [i]*len(Hs_repeated)

df = pd.DataFrame({'hs':hs, 'tp':tp, 'beta':paramsdict['bberm'],'bberm':paramsdict['bberm'],'bsand':paramsdict['bsand'],'dtoeSWL':paramsdict['dtoeSWL'],'transect':transectnums,'returnperiod':returnperiod})

df['beta'] = df.bberm.values.round(3)
df['bsand'] = df.bsand.values.round(3)
df['bberm'] = df.bberm.values.round(3)
df['dtoeSWL'] = np.round(df.dtoeSWL.values/3.3,3)

blen = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
euro = py_wave_runup.models.EurOtop2018(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
poate = py_wave_runup.models.Poate2016(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
df['r2'] = blen.R2_eq21*3.3
df['r2_euro'] = euro.R2(gamma_f=0.75)*3.3
df['r2_poate'] = poate.R2(D50=.102)*3.3

fig, ax = pl.subplots(3,2,tight_layout = {'rect':[0,0,1,0.98]})
fig.set_size_inches((7,7))
pl.tight_layout()
a = sns.barplot(x='returnperiod',y='r2',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[0,0],hue='dtoeSWL')
sns.barplot(x='returnperiod',y='r2_euro',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[1,0],hue='dtoeSWL')
sns.barplot(x='returnperiod',y='r2_poate',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[2,0],hue='dtoeSWL')

sns.barplot(x='returnperiod',y='r2',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[0,1],hue='dtoeSWL')
sns.barplot(x='returnperiod',y='r2_euro',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[1,1],hue='dtoeSWL')
sns.barplot(x='returnperiod',y='r2_poate',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[2,1],hue='dtoeSWL')
for i in range(6):
    ax.ravel('F')[i].set_ylim((0,15))
    if i >0:
        ax.ravel('F')[i].get_legend().remove()
    if i == 0:
        legend_labels, _= ax.ravel('F')[i].get_legend_handles_labels()
        ax.ravel('F')[i].legend(legend_labels,['2ft','3ft','4ft'],title=r'$d_{toe,SWL}$',ncol=3)
ax[0,0].set_xlabel('')
ax[0,1].set_xlabel('')
ax[1,0].set_ylabel('')
ax[1,1].set_ylabel('')
ax[1,1].set_xlabel('')
ax[1,0].set_xlabel('')
ax[2,1].set_ylabel('')
ax[0,0].set_ylabel('Blenkinsopp R2% [ft]')
ax[1,0].set_ylabel('TAW R2% [ft]')
ax[2,0].set_ylabel('Poate R2% [ft]')
ax[2,0].set_xlabel('Return Period (yrs)')
ax[2,1].set_xlabel('Return Period (yrs)')
ax[0,0].set_title(r'$\beta_{berm} = 1:4$')
ax[0,1].set_title(r'$\beta_{berm} = 1:7$')
pl.suptitle('Transect 1')
pl.savefig('Transect1_design_alternatives.png')

pl.tight_layout()
for bi,bermval in enumerate(df.bberm.unique()):
    for ri,rp in enumerate(df.returnperiod.unique()):
        r2vals = df[(df.transect==i+1)&(df.bberm==bermval)&(df.returnperiod==rp)].pivot_table(columns="bsand",index="dtoeSWL",values="r2")
        b=sns.heatmap(data=r2vals,ax=ax[ri,bi],vmin=0,vmax=3)
