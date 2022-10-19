# -*- coding: UTF-8 -*-

"""
Macro III Project
"""
from math import log
import numpy as np
from numpy import std, var, cov, corrcoef, mean, exp
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
ρ_a  = 0.90      #AR(1) coeffiecient of productivity
ρ_y  = 0.86      #AR(1) coefficient of Rest of World GDP
σ_a  = 0.0071    #Variance of productivity shocks
σ_y  = 0.0078    #Variance of shocks Rest of World GDP
ρ_ay = 0.3       #Correlation of productivity shocks and Rest of World GDP shocks
rho_pi = 0.5
sigma_pi = 0.3
cor = 0.4
coef = 0.3


ρ    = 1/β - 1
λ    = (1-θ)*(1-β*θ)/θ
ω    = σ*γ + (1-α)*(σ*η-1)
σ_α  = σ/((1-α)+α*ω)
Θ    = ω - 1
M    = ε/(ε-1)
τ    = 1 - 1/(M*(1-α))
μ    = log(M)
ν    = μ + log(1-α)
Ω    = (ν-μ)/(σ_α+φ)
Γ    = (1+φ)/(σ_α+φ)
Ψ    = (-Θ*σ_α)/(σ_α+φ)


#######################################
# Defining Sims Matrices
#######################################

def create_matrices(monetary_rule):
    """
    Creates the matrices in the Sims format given the monetary rule
    :param monetary_rule: "optimal", "ditr", "citr" or "peg"
    :return: [Gamma_0, Gamma_1, PSI, PI, CONST]
    """

    Gamma_0 = np.array([[0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      1, 1/σ, 0, 0,  0, 0, 0],
                        [0,  0,    β,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0,  0,    0,   1,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0,  1,    0,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0, -α*Ψ,  0,  -Γ,  0, 1,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [1,  0,    0,   0,  0, 1,  -1,   0,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0,  1,    0,   0,  0, 0,   0,  -1,   0,     0,      0,  0,  0, 0,  0, 0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   1,   0,    1/σ,    -1,  0,  0, 0,  0, 0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,  α-1,    1,      0,  0,  0, 0,  0, 0, 0],
                        [0,  0,    0,   0,  0, 0,   1,   0, -α*γ, -α*η+α/σ, -1,  0,  0, 0,  0, 0, 0],
                        [0,  0,    1,   0,  0, 0,   0,   0,   α,     0,      0, -1,  0, 0,  0, 0, 0],
                        [0,  0,    1,   0,  0, 0,   0,   0,   1,     0,      0,  0, -1, 0,  0,-1, 0],
                        [0, σ-σ_α, 0, -1-φ, 0, 0, σ_α+φ, 0,   0,     0,      0,  0,  0, -1, 0, 0, 0],
                        [0,  0,    0,   1,  0, 0,   -1,  0,   0,     0,      0,  0,  0, 0,  1, 0, 0],
                        [0,  0,    0,   0,  0, 0,    0,  0,   0,     0,      0,  0,  0, 0,  0, 1, 0],
                        [0,  0,    0,   0,  0, 0,    0,  0,   0,     0,      0,  0,  0, 0,  0, 0, 1]])

    Gamma_1  = np.array([[0,  0,  0,  0, 1/σ, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0,  0,  1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0,-λ, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0, ρ_a, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, ρ_y, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, α, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rho_pi, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rho_pi]])

    CONST = np.array([[-ρ/σ],
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
                  [0],
                  [0],
                  [0]])

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
                   [0,    0,  0],
                   [0,    0,  0],
                   [0,    0,  0]])

    PSI = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    if monetary_rule == "optimal":
        Gamma_0[2] = [0, -α * σ_α * (Θ + Ψ), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, -α*σ_α*(Θ+Ψ), φ_π, -Γ*(1-ρ_a)*σ_α, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, -1, 0]
    elif monetary_rule == "ditr":
        Gamma_0[2] = [0, 0, φ_π, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0]
    elif monetary_rule == "citr":
        Gamma_0[2] = [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, φ_π, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0]
    elif monetary_rule == "peg":
        Gamma_0[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [0]
        PI[2] = [0, 0, 0]
    else:
        raise Exception("Unrecognized monetary rule")

    return [Gamma_0, Gamma_1, PSI, PI, CONST]


#######################################
# Running gensys
#######################################

G1_opt,  impact_opt,  RC_opt,  C_opt = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg = gensys(*create_matrices("peg"))

#######################################
# Defining IRFs
#######################################

def irfs(G1, impact, C, nperiods, shock):
    # Calculate the shock impact in the next periods
    resp = np.zeros((C.shape[0], nperiods))
    resp[:, 0] = impact[:, 2]
    for j in range(1, nperiods):
        resp[:, [j]] = G1 @ (resp[:, [j-1]] + C)

    #Define nominal variables
    pt = np.cumsum(resp[11, :])
    pHt = np.cumsum(resp[2, :])
    et = np.cumsum(resp[12, :])

    #Return irfs series
    return [resp[15 if shock=="a" else 1, :], resp[2, :], resp[0, :],
            resp[11, :], resp[8, :], et, resp[4, :], pHt, pt]



#######################################
# Creating charts
#######################################
linewidth  = 2
markersize = 5
nperiods = 20
figsize= (9,5)
x_axis = range(1, nperiods+1)

# Figure 1 : World Inflation Shock
figure1 = plt.figure(figsize=figsize)
lines = []
charts = [figure1.add_subplot(3, 3, j+1) for j in range(9)]
limits = [(0,1.1), (-.4,.4), (-1,.5), (-.4,.4), (0,1), (-2,1), (-.3,.1), (-1.5,.5), (-1.5,.5)]
ticks = [(0,1.5,.5), (-.4,.6,.2), (-1,1,.5), (-.4,.6,.2), (0,1.5,.5), (-2,2,1),
         (-.3,.2,.1), (-1.5,1,.5), (-1.5,1,.5)]
plot_titles = ["Global Inflation", "Domestic inflation", "Output gap", "CPI Inflation",
               "Terms of trade", "Nominal exchange rate", "Nominal interest rate",
               "Domestic price level", "CPI level"]

for j in range(9):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    #charts[j].set_ylim(limits[j])
    #charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
    charts[j].set_xticks(range(0,nperiods+1,5))
    charts[j].set_xlim(0,nperiods+1)
    charts[j].grid(color="#000000", linestyle=':',  dashes=(1,4,1,4))

# Calculate irfs
series_opt = irfs(G1_opt, impact_opt, C_opt, nperiods, "a")
series_ditr = irfs(G1_ditr, impact_ditr, C_ditr, nperiods, "a")
series_citr = irfs(G1_citr, impact_citr, C_citr, nperiods, "a")
series_peg = irfs(G1_peg, impact_peg, C_peg, nperiods, "a")

# Plot irfs
lines.append([charts[j].plot(x_axis, series_opt[j], linewidth=linewidth,
                             color="#0000FF")[0] for j in range(9)][0])
lines.append([charts[j].plot(x_axis, series_ditr[j], linewidth=linewidth, color="#007E00",
                linestyle="--")[0] for j in range(9)][0])
lines.append([charts[j].plot(x_axis, series_citr[j], linewidth=linewidth, color="#FF0000",
                marker="X", markersize=markersize)[0] for j in range(9)][0])
lines.append([charts[j].plot(x_axis, series_peg[j], linewidth=linewidth, color="#00BFBF",
                marker="o", markersize=markersize)[0] for j in range(9)][0])

# Create legend and title
figure1.legend(lines, ["Optimal", "DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.95), loc=9, frameon=False)
figure1.suptitle("Impulse Responses - World Inflation Shock\n", fontweight="bold")

# Draw chart 1
plt.tight_layout()
plt.draw()
plt.show()

