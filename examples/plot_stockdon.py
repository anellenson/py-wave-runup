# -*- coding: utf-8 -*-
"""
Assessing Stockdon et al (2006) runup model
===========================================


In this example, we will evaluate the accuracy of the Stockdon et al (2006) runup
model. To do this, we will use the compiled wave runup observations provided by Power
et al (2018).

The Stockdon et al (2006) wave runup model comprises of two relationships, one for
dissipative beaches (i.e. :math:`\\zeta < 0.3`) Eqn (18):

.. math:: R_{2} = 0.043(H_{s}L_{p})^{0.5}

and a seperate relationship for intermediate and reflective beaches (i.e.
:math:`\\zeta > 0.3`):

.. math::

    R_{2} = 1.1 \\left( 0.35 \\beta (H_{s}L_{p})^{0.5} + \\frac{H_{s}L_{p}(
    0.563 \\beta^{2} +0.004)^{0.5}}{2} \\right)

First, let's import our required packages:
"""
#############################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import py_wave_runup

#############################################
# Let's import the Power et al (2018) runup data, which is included in this
# package.

df = py_wave_runup.datasets.load_power18()
print(df.head())
# add depth, berm and sand slopes for Blenkinsopp
dtoeSWL = np.ones(df.shape[0]) * 0.1
bberm = np.ones(df.shape[0]) * 0.25

df['dtoeSWL'] = dtoeSWL
df['bberm'] = bberm
df['bsand'] = bsand

#############################################
# We can see that this dataset gives us :math:`H_{s}` (significant wave height),
# :math:`T_{p}` (peak wave period), :math:`\tan \beta` (beach slope). Let's import
# the Stockdon runup model and calculate the estimated :math:`R_{2}` runup value for
# each row in this dataset.

# Initalize the Stockdon 2006 model with values from the dataset
sto06 = py_wave_runup.models.Stockdon2006(Hs=df.hs, Tp=df.tp, beta=df.beta)
blen22 = py_wave_runup.models.Blenkinsopp2022(Hs=df.hs, Tp=df.tp, beta=df.beta, dtoeSWL=df.dtoeSWL, bberm=df.bberm, bsand=df.beta, spectral_wave_period=True)
eurotop = py_wave_runup.models.EurOtop2018(Hs=df.hs,beta=df.bberm,Tp=df.tp,bberm=df.bberm,bsand=df.beta, dtoeSWL =df.dtoeSWL, spectral_wave_period=True)

# Append a new column at the end of our dataset with Stockdon 2006 R2 estimations
df["sto06_r2"] = sto06.R2
df["blen22"] = blen22.R2_eq21
df["TAW"] = eurotop.R2(gamma_f=1)
df["TAW_gamma"] = eurotop.R2(gamma_f=0.75)

#############################################
# Now let's create a plot of observed R2 values vs. predicted R2 values:

# Plot data
fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
ax1.plot(df.r2, df.sto06_r2, "b.", markersize=2, linewidth=0.5)
ax1.plot(df.r2, df.blen22, "r.", markersize=2, marker='*', linewidth=0.5)
ax1.plot(df.r2, df.TAW, "g.", markersize=2, marker='^', linewidth=0.5)
ax1.plot(df.r2, df.TAW_gamma, "y.", markersize=2, marker='^', linewidth=0.5)
# Add 1:1 line to indicate perfect fit
ax1.plot([0, 12], [0, 12], "k-")

# Add axis labels
ax1.set_xlabel("Observed R2 (m)")
ax1.set_ylabel("Modelled R2 (m)")
ax1.set_title("Runup Model")
ax1.legend(['Stockdon', 'Blenkinsopp', 'TAW', 'TAW with Gamma'])
plt.savefig('stockdonvsblenkinsopp.png')
plt.tight_layout()

#############################################
# We can see there is a fair amount of scatter, especially as we get larger wave
# runup heights. This indicates that the model might not be working as well as we
# might have hoped.
#
# Let's also check RMSE and coefficient of determination values:

print(f"R2 Score: {r2_score(df.r2, df.sto06_r2):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(df.r2, df.sto06_r2)):.2f} m")
