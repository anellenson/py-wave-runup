import sys
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import py_wave_runup
import seaborn as sns

def load_scenario_data(scenariocsv, Hs, Tp, returnperiod):
    '''
    Loads the design data from the scenario csv.
    Args:
    -----
    scenariocsv:    filename of the csv with scenario information
    Hs:             l x 1, Tp, where the values are for each return year: 95-ci, 95, 95+ci
    Tp:             l x 1, Hs, where the values are for each return year: 95-ci, 95, 95+ci
    ReturnPeriod:   l x 1, return period, signifying return year period for each value for Hs and Tp, e.g., 5 5 5 10 10 10
    '''


    data = pd.read_csv(scenariocsv)
    returnperiod = np.tile(returnperiod, data.shape[0])
    Hs_repeated = np.tile(Hs, data.shape[0])
    Tp_repeated = np.tile(Tp, data.shape[0])
    #Repeat the row of data for each Hs value
    data_repeated = np.repeat(data.values, len(Hs), axis = 0)
    #Concatenate with Hs and Tp
    full_data = np.concatenate((data_repeated, np.expand_dims(Hs_repeated,axis=1), np.expand_dims(Tp_repeated,axis=1), np.expand_dims(returnperiod,axis=1)), axis = 1)
    #Turn into dictionary and dataframe
    df_cols = dict.fromkeys(list(data.columns.values) + ['hs', 'tp', 'returnperiod'])

    df = pd.DataFrame(columns = df_cols,data=full_data)
    df['beta'] = df.beta.values.round(3)
    df['dtoeSWL'] = df.dtoeSWL/3.3 #ft to m

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
scenario = 'data\Adamson_M_Z_scenarios.csv'
df = load_scenario_data(scenario, Hs, Tp, returnperiod)
#
df_stock = df.copy(deep=True)
df_stock = df_stock.loc[(df_stock.scenario == 1) | (df_stock.scenario==2)]
df = df.loc[(df.scenario==3) | (df.scenario==4)]

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
# sns.barplot(x='method',y='r2',data=df_full)
#
# fig, ax = pl.subplots(1,1)
# ax.scatter(df_stock.r2, df_taw.r2, color = 'r')
# ax.scatter(df_stock.r2, df_poate.r2, color = 'b')
# ax.scatter(df_stock.r2, df.r2, color = 'g')
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
