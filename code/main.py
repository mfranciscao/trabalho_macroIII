# -*- coding: UTF-8 -*-

"""
Macro III Project
"""
from math import log
import numpy as np
from numpy import std, var, cov, corrcoef, mean
from gensys import gensys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

np.set_printoptions(formatter={'all':lambda x: str(x)}, suppress=True)

#######################################
# Parameters
#######################################

ε    = 6         #elasticity of substitution between varieties (within any country)
α    = 0.4       #inverse of home bias preference or openess natural index
η    = 1         #substitutability between domestic and foreign goods
γ    = 1         #substitutability between goods produced in different countries
σ    = 1         #coefficient of relative risk aversion
φ    = 3         #inverse Frisch elasticity of labor supply
β    = 0.99      #intertemporal discount factor
θ    = 0.75      #price stickiness / share of firms which need to keep prices unchanged
φ_π  = 1.5       #Taylor rule response to inflation
ρ_a  = 0.66      #AR(1) coeffiecient of productivity
ρ_y  = 0.86      #AR(1) coefficient of Rest of World GDP
σ_a  = 0.0071    #Variance of productivity shocks
σ_y  = 0.0078    #Variance of shocks Rest of World GDP
corr = 0.3       #Correlation of productivity shocks and Rest of World GDP shocks


ρ = 1/β - 1
M = ε/(ε-1)
τ = 1 - 1/(M*(1-α))
μ    = log(M)
ν    = -log(1-τ)
λ    = (1-θ)*(1-β*θ)/θ
ω    = σ*γ + (1-α)*(σ*η-1)
σ_α  = σ/((1-α)+α*ω)
Θ    = ω - 1
Ω    = (ν-μ)/(σ_α+φ)
Γ    = (1+φ)/(σ_α+φ)
Ψ    = (-Θ*σ_α)/(σ_α+φ)
ζ    = -α*η+α/σ

#######################################
# Steady State
#######################################

def steady_state_equations(x):
    [Q, S, Y, C, N] = x
    return [
        (((1-α)*S**(η-1)+α)**(1/(η-1)) if η != 1 else S**(1-α)) - Q,
        N - Y,
        (C**σ)*(Y**φ)*(((1-α)+α*S**(η-1))**(1/(η-1)) if η != 1 else S**α)*(1-α) -1,
        Q**(1/σ) - C,
        C*(((1-α)+α*S**(η-1))**(η/(η-1)) if η != 1 else S**α)*((1-α) + α*(S**(γ-η))*(Q**(η - 1/σ))) - Y
    ]

steady_state = fsolve(steady_state_equations, x0=(1, 1, 1, 1, 1))

[Q, S, Y, C, N] = steady_state

print("Steady State Values")
print("Y = %.5f" % Y)
print("C = %.5f" % C)
print("C/Y = %.5f" % (C/Y))
print("S = %.5f" % S)
print("NX/Y = %.5f" % ((Y - C*(((1-α)+α*S**(η-1))**(1/(η-1)) if η != 1 else S**α))/Y))
print("(R⁴ -1) = %.5f" % (β**(-4)-1))

#######################################
# Defining Sims Matrices
#######################################

#Gamma_0
Gamma_0 = np.array([[0,  0,    0,   0,  0, 0,   0,   0,   0,   0,  1, 1/σ, 0, 0,  0],
                    [0,  0,    β,   0,  0, 0,   0,   0,   0,   0,  0,  0,  0, 0,  0],
                    [0,  0,    0,   0,  0, 0,   0,   0,   0,   0,  0,  0,  0, 0,  0],
                    [0,  0,    0,   1,  0, 0,   0,   0,   0,   0,  0,  0,  0, 0,  0],
                    [0,  1,    0,   0,  0, 0,   0,   0,   0,   0,  0,  0,  0, 0,  0],
                    [0, -α*Ψ,  0,  -Γ,  0, 1,   0,   0,   0,   0,  0,  0,  0, 0,  0],
                    [1,  0,    0,   0,  0, 1,  -1,   0,   0,   0,  0,  0,  0, 0,  0],
                    [0,  1,    0,   0,  0, 0,   0,  -1,   0,   0,  0,  0,  0, 0,  0],
                    [0,  0,    0,   0,  0, 0,   0,   1,   0, 1/σ, -1,  0,  0, 0,  0],
                    [0,  0,    0,   0,  0, 0,   0,   0,  α-1,  1,  0,  0,  0, 0,  0],
                    [0,  0,    0,   0,  0, 0,   1,   0, -α*γ,  ζ, -1,  0,  0, 0,  0],
                    [0,  0,    1,   0,  0, 0,   0,   0,   α,   0,  0, -1,  0, 0,  0],
                    [0,  0,    1,   0,  0, 0,   0,   0,   1,   0,  0,  0, -1, 0,  0],
                    [0, σ-σ_α, 0, -1-φ, 0, 0, σ_α+φ, 0,   0,   0,  0,  0,  0, -1, 0],
                    [0,  0,    0,   1,  0, 0,   -1,  0,   0,   0,  0,  0,  0, 0,  1]])

#Gamma_1
Gamma_1  = np.array([[0,  0,  0,  0, 1/σ, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0,  0,  1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0,-λ, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0, ρ_a, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, ρ_y, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, α, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#C
C = np.array([[-ρ/σ],
              [0],
              [0],
              [0],
              [0],
              [Ω],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0]])

#PI
PI = np.array([[-1, -1/σ, 0],
               [0,    0, -β],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0],
               [0,    0,  0]])

#PSI
PSI = np.array([[0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]])

#######################################
# Defining IRFs
#######################################

def irfs(G1, impact, c, nperiods):

    # Calculate the shock impact in the next periods
    resp = np.zeros((c.shape[0], nperiods))
    resp[:, 0] = impact[:, 0]
    for j in range(1, nperiods):
        resp[:, [j]] = G1 @ (resp[:, [j-1]] + c)

    #Define nominal variables
    pt = np.cumsum(resp[11, :])
    pHt = np.cumsum(resp[2, :])
    et = np.cumsum(resp[12, :])

    #Return irfs series
    return [resp[2, :], resp[0, :], resp[11, :], resp[8, :], et, resp[4, :], pHt, pt]


#######################################
# Specifying monetary rules
#######################################

#Optimal policy
Gamma_0[2] = [0, -α*σ_α*(Θ+Ψ),0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Gamma_1[2] = [0, -α*σ_α*(Θ+Ψ), φ_π, -Γ*(1-ρ_a)*σ_α, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C[2] = [ρ]
PI[2] = [0, -1, 0]
[G1_opt, impact_opt, RC_opt, c_opt] = gensys(Gamma_0, Gamma_1, PSI, PI, C)

#DITR
Gamma_0[2] = [0, 0, φ_π, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C[2] = [ρ]
PI[2] = [0, 0, 0]
[G1_ditr, impact_ditr, RC_ditr, c_ditr] = gensys(Gamma_0, Gamma_1, PSI, PI, C)

#CITR
Gamma_0[2] = [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, φ_π, 0, 0, 0]
Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C[2] = [ρ]
PI[2] = [0, 0, 0]
[G1_citr, impact_citr, RC_citr, c_citr] = gensys(Gamma_0, Gamma_1, PSI, PI, C)

#PEG
Gamma_0[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C[2] = [0]
PI[2] = [0, 0, 0]
[G1_peg, impact_peg, RC_peg, c_peg] = gensys(Gamma_0, Gamma_1, PSI, PI, C)


#######################################
# Creating charts and defining style
#######################################

figure1 = plt.figure(figsize=(6,6))
linewidth  = 2
markersize = 5
nperiods = 20
x_axis = range(1, nperiods+1)
lines = []
charts = [figure1.add_subplot(4, 2, j+1) for j in range(8)]
plot_titles = ["Domestic inflation", "Output gap", "CPI Inflation",
               "Terms of trade", "Nominal exchange rate", "Nominal interest rate",
               "Domestic price level", "CPI level"]
limits = [(-.4,.4), (-1,.5), (-.4,.4), (0,1), (-2,1), (-.3,.1), (-1.5,.5), (-1.5,.5)]
ticks = [(-.4,.6,.2), (-1,1,.5), (-.4,.6,.2), (0,1.5,.5), (-2,2,1), (-.3,.2,.1), (-1.5,1,.5), (-1.5,1,.5)]

for j in range(8):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    charts[j].set_ylim(limits[j])
    charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
    charts[j].set_xticks(range(0,nperiods+1,5))
    charts[j].set_xlim(0,nperiods+1)
    charts[j].grid(color="#000000", linestyle=':',  dashes=(1,4,1,4))

# Calculate irfs
series_opt = irfs(G1_opt, impact_opt, c_opt, nperiods)
series_ditr = irfs(G1_ditr, impact_ditr, c_ditr, nperiods)
series_citr = irfs(G1_citr, impact_citr, c_citr, nperiods)
series_peg = irfs(G1_peg, impact_peg, c_peg, nperiods)

# Plot irfs
lines.append([charts[j].plot(x_axis, series_opt[j], linewidth=linewidth,
                             color="#0000FF")[0] for j in range(8)][0])
lines.append([charts[j].plot(x_axis, series_ditr[j], linewidth=linewidth, color="#007E00",
                linestyle="--")[0] for j in range(8)][0])
lines.append([charts[j].plot(x_axis, series_citr[j], linewidth=linewidth, color="#FF0000",
                marker="X", markersize=markersize)[0] for j in range(8)][0])
lines.append([charts[j].plot(x_axis, series_peg[j], linewidth=linewidth, color="#00BFBF",
                marker="o", markersize=markersize)[0] for j in range(8)][0])

# Create legend and title
figure1.legend(lines, ["Optimal", "DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.95), loc=9, frameon=False)
figure1.suptitle("Impulse Responses\n", fontweight="bold")

# Draw charts
plt.tight_layout()
plt.draw()
plt.show()

# --------------------------------------------------------------------------
# Stochastic Simulations
# --------------------------------------------------------------------------

lenghtsimul = 201
numsimul = 500

std_series = np.zeros((6,numsimul))

G1 = [G1_opt, G1_ditr, G1_citr, G1_peg]
impact = [impact_opt, impact_ditr, impact_citr, impact_peg]
c = [c_opt, c_ditr, c_citr, c_peg]
results = []

for k in range(4):
    for i in range(numsimul):
        shocks = np.random.multivariate_normal([0,0],
                      [[σ_a**2, corr*σ_a*σ_y],
                      [corr*σ_a*σ_y, σ_y**2]], lenghtsimul).transpose()

        endog = np.zeros((c_opt.shape[0], lenghtsimul))

        for t in range(0, lenghtsimul):
            endog[:, t] = G1[k].dot(endog[:, t-1]) + impact[k].dot(shocks[:, t]) + c[k].reshape(c_opt.shape[0], )

        [x_t, dcpi, cpi, nrate, s_t, de_t] = [endog[j, :] for j in [6,2,11,4,8,12]]

        std_series[:, i] = [std(j[1:])*100 for j in [x_t, dcpi, cpi, nrate, s_t, de_t]]

    results.append([round(mean(std_series[j, :]), 2) for j in range(std_series.shape[0])])

table1 = pd.DataFrame(results, index=["Optimal", "DI Taylor", "CPI Taylor", "Peg"],
                    columns=["Output", "Domestic Inflation", "CPI Inflation",
                    "Nominal int. rate", "Terms of trade", "Nominal depr. rate"]).transpose()

print(table1)

var_domestic_inflation = list(map(lambda x: (1-α)/2*(x/100)**2*ε/λ, [results[j][1] for j in range(1, 4)]))
var_output = list(map(lambda x: (1-α)/2*(x/100)**2*(1+φ), [results[j][0] for j in range(1, 4)]))

