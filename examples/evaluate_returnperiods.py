import sys
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import py_wave_runup
import seaborn as sns

def load_design_data(designcsvname, Hs, Tp, returnperiod):
    '''
    Loads the design data from the design csv.
    Args:
    -----
    designcsvname:  filename of the csv with design slopes
    Hs:             l x 1, Tp, where the values are for each return year: 95-ci, 95, 95+ci
    Tp:             l x 1, Hs, where the values are for each return year: 95-ci, 95, 95+ci
    ReturnPeriod:   l x 1, return period, signifying return year period for each value for Hs and Tp, e.g., 5 5 5 10 10 10
    '''

    import itertools
    design_data = pd.read_csv(designcsvname)
    transectnums = []
    paramsdict = {'bberm':[], 'bsand':[],'dtoeSWL':[]}
    hs = []
    tp = []
    rplist = []
    sclist = []
    for i in design_data.transect.unique():
        num_unique_conds = design_data[design_data.transect==i].bsand.unique().shape[0]*design_data[design_data.transect==i].bberm.unique().shape[0]*design_data[design_data.transect==i].dtoeSWL.unique().shape[0]
        rp = np.tile(returnperiod, num_unique_conds)
        Hs_repeated = np.tile(Hs, num_unique_conds)
        Tp_repeated = np.tile(Tp, num_unique_conds)
        rplist += list(rp)
        hs += list(Hs_repeated)
        tp += list(Tp_repeated)
        sclist = design_data[design_data.transect==i]['scenario'].unique()
        bbermlist = design_data[(design_data.transect==i)]['bberm'].unique()
        bsandlist = design_data[design_data.transect==i]['bsand'].unique()
        dtoelist = design_data[design_data.transect==i]['dtoeSWL'].unique()
        res = list(itertools.product(*[bbermlist, bsandlist, dtoelist, Hs]))

        for pi,param in enumerate(['bberm', 'bsand', 'dtoeSWL']):
            paramsdict[param] += [res[i][pi] for i in range(len(res))]
        transectnums += [i]*len(Hs_repeated)

    df = pd.DataFrame({'hs':hs, 'tp':tp, 'beta':paramsdict['bberm'],'bberm':paramsdict['bberm'],'bsand':paramsdict['bsand'],'dtoeSWL':paramsdict['dtoeSWL'],'transect':transectnums,'returnperiod':rplist, 'scenario':scenario})
    df['beta'] = df.bberm.values.round(3)
    df['bsand'] = df.bsand.values.round(3)
    df['bberm'] = df.bberm.values.round(3)
    df['dtoeSWL'] = np.round(df.dtoeSWL.values/3.3,3)

    return df

def load_existing_data(existingcsvname, Hs, Tp, returnperiod):

    existing_data = pd.read_csv(existingcsvname)
    returnperiod = np.tile(returnperiod, existing_data['beta'].unique().shape[0])
    Hs_repeated = np.tile(Hs, existing_data['beta'].unique().shape[0])
    Tp_repeated = np.tile(Tp, existing_data['beta'].unique().shape[0])

    paramlist = data.columns
    beta = []
    transect = []
    sclist = []
    for tt,bb,ss in zip(existing_data.Transect, existing_data.beta, existing_data.scenario):
        transect += [tt]*len(Hs)
        beta += [bb]*len(Hs)
        sclist += [ss]*len(Hs)

    df = pd.DataFrame({'hs':Hs_repeated, 'tp':Tp_repeated, 'beta':beta, 'transect':transect, 'returnperiod':returnperiod, 'scenario':sclist})
    df['beta'] = df.beta.values.round(3)

    return df

def load_timeseries_data(Hscsv, Tpcsv, slope=1/7, dtoe=3):
    Hs_csv = pd.read_csv(Hscsv)
    Tp_csv = pd.read_csv(Tpcsv)
    Hs = Hs_csv['2781']
    Tp = Tp_csv['2781']
    beta = [slope]*len(Hs)
    bsand = [slope]*len(Hs)
    bberm = [slope]*len(Hs)
    dtoeSWL = [dtoe/3.3]*len(Hs)
    df = pd.DataFrame({'hs':Hs, 'tp':Tp, 'beta':beta, 'bsand':bsand, 'bberm':bberm, 'dtoeSWL':dtoeSWL})
    return df

Tp_returnvals = pd.read_csv('data\Tp_return_vals.csv')
Tp=np.reshape(Tp_returnvals.drop(columns='return period').values, (15,))
Hs_returnvals = pd.read_csv('data\Hs_return_vals.csv')
Hs=np.reshape(Hs_returnvals.drop(columns='return period').values, (15,))
returnperiod = [5]*3 + [10] * 3 + [20]*3 + [50]*3 + [100]*3
MHW = 4.3#m
existing = 'data\Adamson_M_Z_existing.csv'
design = 'data\Adamson_M_Z_design.csv'
df_stock = load_existing_data(existing, Hs, Tp, returnperiod)
df = load_design_data(design, Hs, Tp, returnperiod)

df = pd.concat([df_stock,df])

blen = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
euro = py_wave_runup.models.EurOtop2018(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
poate = py_wave_runup.models.Poate2016(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)
stock = py_wave_runup.models.Stockdon2006(Hs=df.hs,beta=df.beta,bsand=df.bsand,bberm=df.bberm,dtoeSWL=df.dtoeSWL,Tp=df.tp,spectral_wave_period=True,h=10)

df_taw = df.copy(deep=True)
df_poate = df.copy(deep=True)
df_stock = df.copy(deep=True)

df_stock['r2'] = stock.R2*3.3
df['r2'] = blen.R2_eq21*3.3
df_taw['r2'] = euro.R2(gamma_f=0.70)*3.3
df_poate['r2'] = poate.R2()*3.3

df_poate['method'] = ['Poate 2016'] * len(df.hs)
df_taw['method'] = ['TAW'] * len(df.hs)
df['method'] = ['Blenkinsopp 2022'] * len(df.hs)
df_stock['method'] = ['Stockdon'] * len(df.hs)

df_full = pd.concat([df, df_taw, df_poate, df_stock])
sns.barplot(x='method',y='r2',data=df_full)

fig, ax = pl.subplots(1,1)
ax.scatter(df_stock.r2, df_taw.r2, color = 'r')
ax.scatter(df_stock.r2, df_poate.r2, color = 'b')
ax.scatter(df_stock.r2, df.r2, color = 'g')
#
#
#

#
#
#
# ################More plotting stuff
# fig, ax = pl.subplots(3,1, tight_layout = {'rect':[0,0,1,0.98]})
# fig.set_size_inches(3,9)
# pl.tight_layout()
# for i in range(3):
#     sns.barplot(x='returnperiod',y='r2',hue='beta',data=df[df.transect == i+1], ax =ax[i])
#     ax[i].set_xlabel('return period (yrs)')
#     ax[i].set_ylabel('r2% (ft)')
#     ax[i].set_ylim((0,12))
#     ax[i].set_title('Transect {0}'.format(i+1))
#     legend_labels, _= ax[i].get_legend_handles_labels()
#     ax[i].legend(title=r'$\beta_{sand}$')
#
#     if i < 2:
#         ax[i].set_xticks([])
#         ax[i].set_xlabel('')
# pl.suptitle("Stockdon Existing Transect")
# pl.savefig('stockdon_existing.png')
# ########
#
# fig, ax = pl.subplots(3,2,tight_layout = {'rect':[0,0,1,0.98]})
# fig.set_size_inches((7,7))
# pl.tight_layout()
# a = sns.barplot(x='returnperiod',y='r2',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[0,0],hue='dtoeSWL')
# sns.barplot(x='returnperiod',y='r2_euro',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[1,0],hue='dtoeSWL')
# sns.barplot(x='returnperiod',y='r2_poate',data=df[(df.bsand==0.085)&(df.bberm==0.25)],ax =ax[2,0],hue='dtoeSWL')
#
# sns.barplot(x='returnperiod',y='r2',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[0,1],hue='dtoeSWL')
# sns.barplot(x='returnperiod',y='r2_euro',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[1,1],hue='dtoeSWL')
# sns.barplot(x='returnperiod',y='r2_poate',data=df[(df.bsand==0.085)&(df.bberm==0.14)],ax =ax[2,1],hue='dtoeSWL')
# for i in range(6):
#     ax.ravel('F')[i].set_ylim((0,15))
#     if i >0:
#         ax.ravel('F')[i].get_legend().remove()
#     if i == 0:
#         legend_labels, _= ax.ravel('F')[i].get_legend_handles_labels()
#         ax.ravel('F')[i].legend(legend_labels,['2ft','3ft','4ft'],title=r'$d_{toe,SWL}$',ncol=3)
# ax[0,0].set_xlabel('')
# ax[0,1].set_xlabel('')
# ax[1,0].set_ylabel('')
# ax[1,1].set_ylabel('')
# ax[1,1].set_xlabel('')
# ax[1,0].set_xlabel('')
# ax[2,1].set_ylabel('')
# ax[0,0].set_ylabel('Blenkinsopp R2% [ft]')
# ax[1,0].set_ylabel('TAW R2% [ft]')
# ax[2,0].set_ylabel('Poate R2% [ft]')
# ax[2,0].set_xlabel('Return Period (yrs)')
# ax[2,1].set_xlabel('Return Period (yrs)')
# ax[0,0].set_title(r'$\beta_{berm} = 1:4$')
# ax[0,1].set_title(r'$\beta_{berm} = 1:7$')
# pl.suptitle('Transect 1')
# pl.savefig('Transect1_design_alternatives.png')
#
# pl.tight_layout()
# for bi,bermval in enumerate(df.bberm.unique()):
#     for ri,rp in enumerate(df.returnperiod.unique()):
#         r2vals = df[(df.transect==i+1)&(df.bberm==bermval)&(df.returnperiod==rp)].pivot_table(columns="bsand",index="dtoeSWL",values="r2")
#         b=sns.heatmap(data=r2vals,ax=ax[ri,bi],vmin=0,vmax=3)
