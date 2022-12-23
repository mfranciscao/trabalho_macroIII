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
np.random.seed(1)

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
ρ_ay = 0.3       #Correlation of productivity shocks and Rest of World GDP shocks

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
# Steady State
#######################################

def steady_state_equations(x):
    Q, S, Y, C, N, Wr = x
    return [
        (C**σ)*(Y**φ) - Wr,
        (((1-α)*S**(η-1)+α)**(1/(η-1)) if η != 1 else S**(1-α)) - Q,
        N - Y,
        (C**σ)*(Y**φ)*(((1-α)+α*S**(η-1))**(1/(η-1)) if η != 1 else S**α)*(1-α) -1,
        Q**(1/σ) - C,
        C*(((1-α)+α*S**(η-1))**(η/(η-1)) if η != 1 else S**α)*((1-α) + α*(S**(γ-η))*(Q**(η - 1/σ))) - Y
    ]

steady_state = fsolve(steady_state_equations, x0=(1, 1, 1, 1, 1, 1))

[Q_steady, S_steady, Y_steady, C_steady, N_steady, Wr_steady] = steady_state

print("Steady State Values")
print("Y = %.5f" % Y_steady)
print("C = %.5f" % C_steady)
print("Wr = %.5f" % Wr_steady)
print("C/Y = %.5f" % (C_steady/Y_steady))
print("S = %.5f" % S_steady)
print("NX/Y = %.5f" % ((Y_steady - C_steady*(((1-α)+α*S_steady**(η-1))**(1/(η-1)) if 
                                             η != 1 else S_steady**α))/Y_steady))
print("(R⁴ -1) = %.5f" % (β**(-4)-1))

#######################################
# Defining Sims Matrices
#######################################

def create_matrices(monetary_rule):
    """
    Creates the matrices in the Sims format given the monetary rule
    :param monetary_rule: "optimal", "ditr", "citr" or "peg"
    :return: [Gamma_0, Gamma_1, PSI, PI, CONST]
    """

    Gamma_0 = np.array([[0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      1, 1/σ, 0, 0,  0, 0],
                        [0,  0,    β,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0,  0,    0,   1,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0,  1,    0,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0, -α*Ψ,  0,  -Γ,  0, 1,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [1,  0,    0,   0,  0, 1,  -1,   0,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0,  1,    0,   0,  0, 0,   0,  -1,   0,     0,      0,  0,  0, 0,  0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   1,   0,    1/σ,    -1,  0,  0, 0,  0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,  α-1,    1,      0,  0,  0, 0,  0, 0],
                        [0,  0,    0,   0,  0, 0,   1,   0, -α*γ, -α*η+α/σ, -1,  0,  0, 0,  0, 0],
                        [0,  0,    1,   0,  0, 0,   0,   0,   α,     0,      0, -1,  0, 0,  0, 0],
                        [0,  0,    1,   0,  0, 0,   0,   0,   1,     0,      0,  0, -1, 0,  0, 0],
                        [0,  0,    0,  -1,  0, 0,   0,   0,   α,     0,      0,  0,  0, -1, 0, 1],
                        [0,  0,    0,   1,  0, 0,  -1,   0,   0,     0,      0,  0,  0, 0,  1, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,   0,     0,     -σ,  0,  0, 0, -φ, 1]])

    Gamma_1  = np.array([[0,  0,  0,  0, 1/σ, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0,  0,  1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0,-λ, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0, ρ_a, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, ρ_y, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, α, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

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
                  [-ν+μ],
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
                   [0,    0,  0]])

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
                    [0, 0],
                    [0, 0]])

    if monetary_rule == "optimal":
        Gamma_0[2] = [0, -α * σ_α * (Θ + Ψ), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, -α * σ_α * (Θ + Ψ), φ_π, -Γ * (1 - ρ_a) * σ_α, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, -1, 0]
    elif monetary_rule == "ditr":
        Gamma_0[2] = [0, 0, φ_π, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0]
    elif monetary_rule == "citr":
        Gamma_0[2] = [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, φ_π, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0]
    elif monetary_rule == "peg":
        Gamma_0[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [0]
        PI[2] = [0, 0, 0]
    else:
        raise Exception("Unrecognized monetary rule")

    return [Gamma_0, Gamma_1, PSI, PI, CONST]


#######################################
# Running gensys
#######################################

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

# --------------------------------------------------------------------------
# Stochastic Simulations
# --------------------------------------------------------------------------

lenghtsimul = 201
numsimul = 1000

std_series = np.zeros((6,numsimul))

G1 = [G1_opt, G1_ditr, G1_citr, G1_peg]
impact = [impact_opt, impact_ditr, impact_citr, impact_peg]
C = [C_opt, C_ditr, C_citr, C_peg]
results = []

for k in range(4): #iterate over models
    for i in range(numsimul):
        shocks = np.random.multivariate_normal([0,0],
                      [[σ_a**2, ρ_ay*σ_a*σ_y],
                      [ρ_ay*σ_a*σ_y, σ_y**2]], lenghtsimul).transpose()

        endog = np.zeros((16, lenghtsimul))

        for t in range(0, lenghtsimul):
            endog[:, t] = G1[k].dot(endog[:, t-1]) + impact[k].dot(shocks[:, t]) + C[k].reshape(16, )

        std_series[:, i] = [std(j[1:])*100 for j in [endog[j, :] for j in [6,2,11,4,8,12]]]

    results.append([round(mean(std_series[j, :]), 2) for j in range(std_series.shape[0])])

table1 = pd.DataFrame(results, index=["Optimal", "DI Taylor", "CPI Taylor", "Peg"],
                    columns=["Output", "Domestic Inflation", "CPI Inflation",
                    "Nominal int. rate", "Terms of trade", "Nominal depr. rate"]).transpose()

print("\nDynamic Properties")
print(table1)


# --------------------------------------------------------------------------
# Welfare Losses
# --------------------------------------------------------------------------

def welfare_simulations():
    var_output_ditr = []
    var_output_citr = []
    var_output_peg = []
    var_dcpi_ditr = []
    var_dcpi_citr = []
    var_dcpi_peg = []

    for i in range(numsimul):
        shocks = np.random.multivariate_normal([0, 0],
                   [[σ_a ** 2, ρ_ay * σ_a * σ_y],
                   [ρ_ay * σ_a * σ_y, σ_y ** 2]], lenghtsimul).transpose()

        endog_ditr = np.zeros((16, lenghtsimul))
        endog_citr = np.zeros((16, lenghtsimul))
        endog_peg = np.zeros((16, lenghtsimul))

        for t in range(0, lenghtsimul):
            endog_ditr[:, t] = G1_ditr.dot(endog_ditr[:, t - 1]) + \
                               impact_ditr.dot(shocks[:, t]) + C_ditr.reshape(16, )
            endog_citr[:, t] = G1_citr.dot(endog_citr[:, t - 1]) + \
                               impact_citr.dot(shocks[:, t]) + C_citr.reshape(16, )
            endog_peg[:, t] = G1_peg.dot(endog_peg[:, t - 1]) + \
                               impact_peg.dot(shocks[:, t]) + C_peg.reshape(16, )

        var_dcpi_ditr.append(100*(1-α)/2*(ε/λ*var(endog_ditr[2, :])))
        var_dcpi_citr.append(100*(1-α)/2*(ε/λ*var(endog_citr[2, :])))
        var_dcpi_peg.append(100*(1-α)/2*(ε/λ*var(endog_peg[2, :])))

        var_output_ditr.append(100*(1-α)/2*((1+φ)*var(endog_ditr[0, :])))
        var_output_citr.append(100*(1-α)/2*((1+φ)*var(endog_citr[0, :])))
        var_output_peg.append(100*(1-α)/2*((1+φ)*var(endog_peg[0, :])))

    return [[mean(var_dcpi_ditr), mean(var_dcpi_citr), mean(var_dcpi_peg)],
            [mean(var_output_ditr), mean(var_output_citr), mean(var_output_peg)],
            [mean(var_dcpi_ditr) + mean(var_output_ditr), mean(var_dcpi_citr) +
             mean(var_output_citr), mean(var_dcpi_peg) + mean(var_output_peg)]]

# Case 1 - Benchmark:  μ = log(1.2), φ = 3
table2 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)", "Total"],
                    columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 1 - Benchmark:  μ = log(1.2), φ = 3")
print(table2)

# Case 2 - Low steady state markup:  μ = ln(1.1), φ = 3
μ = log(1.1)
ε = exp(μ)/(exp(μ) - 1)
ν = μ
Ω = (ν-μ)/(σ_α+φ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table3 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)", "Total"],
                    columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 2 - Low steady state markup:  μ = ln(1.1), φ = 3")
print(table3)

# Case 3 - Low elasticity of labor supply:  μ = log(1.2), φ = 10
μ = log(1.2)
φ = 10
ε = exp(μ)/(exp(μ) - 1)
ν = μ
Ω = (ν-μ)/(σ_α+φ)
Γ = (1+φ)/(σ_α+φ)
Ψ = (-Θ*σ_α)/(σ_α+φ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table4 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)", "Total"],
                    columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 3 - Low elasticity of labor supply:  μ = log(1.2), φ = 10")
print(table4)


# Case 4 - Low mark-up and elasticity of labour supply:  μ = log(1.1), φ = 10
μ = log(1.1)
φ = 10
ε = exp(μ)/(exp(μ) - 1)
ν = μ
Ω = (ν-μ)/(σ_α+φ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table5 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)", "Total"],
                    columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 4 - Low mark-up and elasticity of labour supply:  μ = log(1.1), φ = 10")
print(table5)


#######################################
# Running gensys with ρ_a  = 0.90
#######################################

# Change ρ_a and restore μ = log(1.2) and φ = 3
ρ_a  = 0.90
μ = 1.2
φ = 3
ε = exp(μ)/(exp(μ) - 1)
ν = μ
Ω = (ν-μ)/(σ_α+φ)
Γ = (1+φ)/(σ_α+φ)
Ψ = (-Θ*σ_α)/(σ_α+φ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

#######################################
# Defining IRFs
#######################################

def irfs(G1, impact, C, nperiods, shock):
    # Calculate the shock impact in the next periods
    resp = np.zeros((C.shape[0], nperiods))
    resp[:, 0] = impact[:, 0 if shock=="a" else 1]
    for j in range(1, nperiods):
        resp[:, [j]] = G1 @ (resp[:, [j-1]] + C)

    #Return irfs series
    return [resp[3 if shock=="a" else 1, :], resp[0, :],  resp[6, :], resp[10, :],
            resp[11, :], resp[2, :], resp[14, :], resp[15, :], 
            resp[4, :], resp[12, :], resp[8, :], resp[13, :]]



#######################################
# Creating charts
#######################################
linewidth  = 2
markersize = 5
nperiods = 20
figsize= (16,9)
x_axis = range(1, nperiods+1)

# Figure 1 : Productivity shock
figure1 = plt.figure(figsize=figsize)
lines = []
charts = [figure1.add_subplot(3, 4, j+1) for j in range(12)]
limits = [(0,1.1), (-.8,0.1), (0,1.2), (0,0.8), 
          (-.4,.6), (-0.6,0.2), (-0.8,0.2), (-2,1), 
          (-0.3,0.1), (-.4,1.2), (0,1.2), (-2,0.2)]
ticks = [(0,1.2,.2), (-.8,.2,.2), (0,1.2,.2),(0,0.8,.2), 
         (-.4,.6,.2), (-.4,.4,.2), (-0.8,0.2,0.2), (-2,2,1), 
         (-.3,.2,.1), (-.5,1.5,.5), (0,1.2,.2), (-2.5,0.5,.5)]
plot_titles = ["Productivity", "Output Gap", "Output", "Consumption", 
               "CPI Inflation", "Domestic Inflation","Hours worked", "Real Wage", 
               "Nominal Interest Rate", "Exchange rate deprec (Δe)", "Terms of Trade","Marginal Cost Gap"]

for j in range(12):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    charts[j].set_ylim(limits[j])
    charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
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
                             color="#0000FF")[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_ditr[j], linewidth=linewidth, color="#007E00",
                linestyle="--")[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_citr[j], linewidth=linewidth, color="#FF0000",
                marker="X", markersize=markersize)[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_peg[j], linewidth=linewidth, color="#00BFBF",
                marker="o", markersize=markersize)[0] for j in range(12)][0])

# Create legend and title
figure1.legend(lines, ["Optimal", "DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.97), loc=9, frameon=False)
figure1.suptitle("Impulse Responses - Productivity Shock\n", fontweight="bold")

# Draw chart 1
plt.tight_layout()
plt.draw()
plt.show()


# Figure 2 : World Output shock
figure2 = plt.figure(figsize=figsize)
lines = []
charts = [figure2.add_subplot(3, 4, j+1) for j in range(12)]
limits = [(0,1.1), (-0.2,0.8), (-0.2,0.8), (0,0.8),
          (-.6,.4), (-0.2,0.4), (-0.2,0.8), (-0.5,3), 
          (-0.4,0.1), (-1.1,0.2), (-1.1,0.1), (-0.5,2.6)]
ticks = [(0,1.2,.2), (-0.2,.8,.2), (-0.2,.8,.2),(0,0.8,.2),
         (-.6,.4,.2), (-.2,.4,.2), (-0.2,0.8,0.2), (-0.5,3,0.5), 
         (-.4,.1,.1), (-1,0.2,.2), (-1,0.2,0.2), (0,3,.5)]
plot_titles[0] = "World Output"

for j in range(12):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    charts[j].set_ylim(limits[j])
    charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
    charts[j].set_xticks(range(0,nperiods+1,5))
    charts[j].set_xlim(0,nperiods+1)
    charts[j].grid(color="#000000", linestyle=':',  dashes=(1,4,1,4))

# Calculate irfs
series_opt = irfs(G1_opt, impact_opt, C_opt, nperiods, "y")
series_ditr = irfs(G1_ditr, impact_ditr, C_ditr, nperiods, "y")
series_citr = irfs(G1_citr, impact_citr, C_citr, nperiods, "y")
series_peg = irfs(G1_peg, impact_peg, C_peg, nperiods, "y")

# Plot irfs
lines.append([charts[j].plot(x_axis, series_opt[j], linewidth=linewidth,
                             color="#0000FF")[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_ditr[j], linewidth=linewidth, color="#007E00",
                linestyle="--")[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_citr[j], linewidth=linewidth, color="#FF0000",
                marker="X", markersize=markersize)[0] for j in range(12)][0])
lines.append([charts[j].plot(x_axis, series_peg[j], linewidth=linewidth, color="#00BFBF",
                marker="o", markersize=markersize)[0] for j in range(12)][0])

# Create legend and title
figure2.legend(lines, ["Optimal", "DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.97), loc=9, frameon=False)
figure2.suptitle("Impulse Responses - World Output Shock\n", fontweight="bold")

# Draw chart 2
plt.tight_layout()
plt.draw()
plt.show()


figure1.savefig("..\\latex\\img\\figure1.svg")
figure2.savefig("..\\latex\\img\\figure2.svg")