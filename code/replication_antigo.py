# -*- coding: UTF-8 -*-

"""
Macro III Project
"""
from math import log
import numpy as np
from numpy import std, var, cov, corrcoef, mean, exp
from gensys import gensys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

np.set_printoptions(formatter={'all':lambda x: str(x)}, suppress=True)
np.random.seed(1)
pd.set_option('display.max_columns', 7)


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
ζ    = 4         #elasticity of substitution between labor types
ς    = 0.75      #wage stickiness / share of unions which must keep wages unchanged

ρ    = 1/β - 1
λ    = (1-θ)*(1-β*θ)/θ
ω    = σ*γ + (1-α)*(σ*η-1)
σ_α  = σ/((1-α)+α*ω)
Θ    = ω - 1
M    = ε/(ε-1)
τ    = 1 - 1/(M*(1-α))
μ    = log(M)
ν    = μ #+ log(1-α)
Ω    = (ν-μ)/(σ_α+φ)
Γ    = (1+φ)/(σ_α+φ)
Ψ    = (-Θ*σ_α)/(σ_α+φ)
Ξ    = ζ/(ζ-1)
ξ    = log(Ξ)
Λ    = (1-ς)*(1-β*ς)/(ς*(1+ζ*φ))

#######################################
# Steady State
#######################################

def steady_state_equations(x):
    Q, S, Y, C, N, Wr = x
    return [
        Ξ*(C**σ)*(Y**φ) - Wr,
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
                        [0,  0,    1,   0,  0, 0,   0,   0,   1,     0,      0,  0, -1, 0,  0, 0, 0],
                        [0,  0,    0,  -1,  0, 0,   0,   0,   α,     0,      0,  0,  0, -1, 0, 1, 0],
                        [0,  0,    0,   1,  0, 0,  -1,   0,   0,     0,      0,  0,  0, 0,  1, 0, 0],
                        [0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      0,  0,  0, 0,  0, 0, β],
                        [0,  0,    0,   0,  0, 0,   0,   0,   0,     0,      0,  1,  0, 0,  0,-1, 1]])

    Gamma_1  = np.array([[0,  0,  0,  0, 1/σ, 0, 0, 0, 0, 0,   1, 0, 0, 0,  0,  0, 0],
                         [0,  0,  1,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0,-λ,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0, ρ_a, 0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0, ρ_y, 0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, α, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 1, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0,  0, 0],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,-Λ*σ, 0, 0, 0,-Λ*φ, Λ, 1],
                         [0,  0,  0,  0,  0,  0, 0, 0, 0, 0,   0, 0, 0, 0,  0, -1, 0]])

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
                  [-Λ*ξ],
                  [0]])

    PI = np.array([[-1, -1/σ, 0,  0],
                   [0,    0, -β,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0,  0],
                   [0,    0,  0, -β],
                   [0,    0,  0,  0]])

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
                    [0, 0],
                    [0, 0]])

    if monetary_rule == "optimal":
        Gamma_0[2] = [0, -α*σ_α*(Θ+Ψ), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, -α*σ_α*(Θ+Ψ), φ_π, -Γ*(1-ρ_a)*σ_α, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, -1, 0, 0]
    elif monetary_rule == "ditr":
        Gamma_0[2] = [0, 0, φ_π, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0, 0]
    elif monetary_rule == "citr":
        Gamma_0[2] = [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, φ_π, 0, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [ρ]
        PI[2] = [0, 0, 0, 0]
    elif monetary_rule == "peg":
        Gamma_0[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        Gamma_1[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        CONST[2] = [0]
        PI[2] = [0, 0, 0, 0]
    else:
        raise Exception("Unrecognized monetary rule")

    return [Gamma_0, Gamma_1, PSI, PI, CONST]


# --------------------------------------------------------------------------
# Stochastic Simulations
# --------------------------------------------------------------------------

lenghtsimul = 201
numsimul = 1000

std_series = np.zeros((6,numsimul))

Ξ = 1
Λ = 1000000000000
ξ = log(Ξ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

G1 = [[G1_opt, G1_ditr, G1_citr, G1_peg]]
impact = [[impact_opt, impact_ditr, impact_citr, impact_peg]]
C = [[C_opt, C_ditr, C_citr, C_peg]]

Ξ = ζ/(ζ-1)
Λ = (1-ς)*(1-β*ς)/(ς*(1+ζ*φ))
ξ = log(Ξ)

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

G1.append([G1_opt, G1_ditr, G1_citr, G1_peg])
impact.append([impact_opt, impact_ditr, impact_citr, impact_peg])
C.append([C_opt, C_ditr, C_citr, C_peg])

standard_dev = [[], []]

for h in range(2):
    for k in range(4): #iterate over models
        for i in range(numsimul):
            shocks = np.random.multivariate_normal([0,0],
                          [[σ_a**2, ρ_ay*σ_a*σ_y],
                          [ρ_ay*σ_a*σ_y, σ_y**2]], lenghtsimul).transpose()

            endog = np.zeros((17, lenghtsimul))

            for t in range(0, lenghtsimul):
                endog[:, t] = G1[h][k].dot(endog[:, t-1]) + impact[h][k].dot(shocks[:, t]) + C[h][k].reshape(17, )

            std_series[:, i] = [std(j[1:])*100 for j in [endog[j, :] for j in [6,2,11,4,8,12]]]

        standard_dev[h].append([round(mean(std_series[j, :]), 2) for j in range(std_series.shape[0])])

table1a = pd.DataFrame(standard_dev[0], index=["Optimal_Model", "DITR_Model", "CITR_Model", "Peg_Model"],
                    columns=["Output", "Domestic Inflation", "CPI Inflation",
                    "Nominal int. rate", "Terms of trade", "Nominal depr. rate"]).transpose()

table1b = pd.DataFrame(standard_dev[1], index=["Optimal_Modif", "DITR_Modif", "CITR_Modif", "Peg_Modif"],
                    columns=["Output", "Domestic Inflation", "CPI Inflation",
                    "Nominal int. rate", "Terms of trade", "Nominal depr. rate"]).transpose()

print("\nDynamic Properties")
table1 = pd.concat([table1a, table1b], axis=1)
print(table1[table1.columns[[1,5,2,6,3,7]]])


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
    var_wage_ditr = []
    var_wage_citr = []
    var_wage_peg = []

    for i in range(numsimul):
        shocks = np.random.multivariate_normal([0, 0],
                   [[σ_a ** 2, ρ_ay * σ_a * σ_y],
                   [ρ_ay * σ_a * σ_y, σ_y ** 2]], lenghtsimul).transpose()

        endog_ditr = np.zeros((17, lenghtsimul))
        endog_citr = np.zeros((17, lenghtsimul))
        endog_peg = np.zeros((17, lenghtsimul))

        for t in range(0, lenghtsimul):
            endog_ditr[:, t] = G1_ditr.dot(endog_ditr[:, t - 1]) + \
                               impact_ditr.dot(shocks[:, t]) + C_ditr.reshape(17, )
            endog_citr[:, t] = G1_citr.dot(endog_citr[:, t - 1]) + \
                               impact_citr.dot(shocks[:, t]) + C_citr.reshape(17, )
            endog_peg[:, t] = G1_peg.dot(endog_peg[:, t - 1]) + \
                               impact_peg.dot(shocks[:, t]) + C_peg.reshape(17, )

        var_dcpi_ditr.append(100*(1-α)/2*(ε/λ*var(endog_ditr[2, :])))
        var_dcpi_citr.append(100*(1-α)/2*(ε/λ*var(endog_citr[2, :])))
        var_dcpi_peg.append(100*(1-α)/2*(ε/λ*var(endog_peg[2, :])))

        var_output_ditr.append(100*(1-α)/2*((1+φ)*var(endog_ditr[0, :])))
        var_output_citr.append(100*(1-α)/2*((1+φ)*var(endog_citr[0, :])))
        var_output_peg.append( 100*(1-α)/2*((1+φ)*var(endog_peg[0, :])))

        var_wage_ditr.append(100*(1-α)/2*(ζ/Λ*var(endog_ditr[16, :])))
        var_wage_citr.append(100*(1-α)/2*(ζ/Λ*var(endog_citr[16, :])))
        var_wage_peg.append( 100*(1-α)/2*(ζ/Λ*var(endog_peg[16, :])))

    return [[mean(var_dcpi_ditr), mean(var_dcpi_citr), mean(var_dcpi_peg)],
            [mean(var_output_ditr), mean(var_output_citr), mean(var_output_peg)],
            [mean(var_wage_ditr), mean(var_wage_citr), mean(var_wage_peg)],
            [mean(var_dcpi_ditr) + mean(var_output_ditr) + mean(var_wage_ditr),
             mean(var_dcpi_citr) + mean(var_output_citr) + mean(var_wage_citr),
             mean(var_dcpi_peg) + mean(var_output_peg) + mean(var_wage_peg)]]

# Case 1 - No wage rigidity: ς = 0
ς = 0
Λ = 10000000

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table2 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)",
                    "Var(Wage Inf.)", "Total"], columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nNo wage rigiditiy: ς = 0")
print(table2)

# Case 2 - Low wage rigidity: ς = 0.25
ς = 0.25
Λ = (1-ς)*(1-β*ς)/(ς*(1+ζ*φ))

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table3 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)",
                    "Var(Wage Inf.)", "Total"], columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 2 - Low wage rigidity: ς = 0.25")
print(table3)

# Case 3 - Moderate wage rigidity: ς = 0.50
ς = 0.50
Λ = (1-ς)*(1-β*ς)/(ς*(1+ζ*φ))

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table4 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)",
                    "Var(Wage Inf.)", "Total"], columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 3 - Moderate wage rigidity: ς = 0.50")
print(table4)

# Case 4 - Benchmark wage rigidity: ς = 0.75
ς = 0.75
Λ = (1-ς)*(1-β*ς)/(ς*(1+ζ*φ))

G1_opt,  impact_opt,  RC_opt,  C_opt  = gensys(*create_matrices("optimal"))
G1_ditr, impact_ditr, RC_ditr, C_ditr = gensys(*create_matrices("ditr"))
G1_citr, impact_citr, RC_citr, C_citr = gensys(*create_matrices("citr"))
G1_peg,  impact_peg,  RC_peg,  C_peg  = gensys(*create_matrices("peg"))

table5 = pd.DataFrame(welfare_simulations(), index=["Var(Domest. Inf.)", "Var(Output)",
                    "Var(Wage Inf.)", "Total"], columns=["DI Taylor", "CPI Taylor", "Peg"])
print("\nCase 4 - Benchmark wage rigidity: ς = 0.75")
print(table5)

#######################################
# Defining IRFs
#######################################

nperiods = 20

def irfs(G1, impact, RC, C, nperiods, shock):
    # Calculate the shock impact in the next periods
    resp = np.zeros((C.shape[0], nperiods))
    resp[:, 0] = impact[:, 0 if shock=="a" else 1]
    for j in range(1, nperiods):
        resp[:, [j]] = G1 @ (resp[:, [j-1]] + C)

    #Return irfs series
    return [resp[3 if shock=="a" else 1, :], resp[0, :], resp[6, :],
            resp[11, :], resp[2, :], resp[15, :], resp[4, :], resp[12, :], resp[8, :]]


#######################################
# Calculating IRFs
#######################################

# Change ρ_a
ρ_a  = 0.90

# Calculate irfs

series_ditr_modification_prod = irfs(*gensys(*create_matrices("ditr")), nperiods, "a")
series_citr_modification_prod = irfs(*gensys(*create_matrices("citr")), nperiods, "a")
series_peg_modification_prod  = irfs(*gensys(*create_matrices("peg")),  nperiods, "a")

series_ditr_modification_world = irfs(*gensys(*create_matrices("ditr")), nperiods, "y")
series_citr_modification_world = irfs(*gensys(*create_matrices("citr")), nperiods, "y")
series_peg_modification_world  = irfs(*gensys(*create_matrices("peg")),  nperiods, "y")

#Calculating the original model
Ξ = 1
Λ = 1000000
ξ = log(Ξ)

# Calculate irfs

series_ditr_original_prod = irfs(*gensys(*create_matrices("ditr")), nperiods, "a")
series_citr_original_prod = irfs(*gensys(*create_matrices("citr")), nperiods, "a")
series_peg_original_prod  = irfs(*gensys(*create_matrices("peg")),  nperiods, "a")

series_ditr_original_world = irfs(*gensys(*create_matrices("ditr")), nperiods, "y")
series_citr_original_world = irfs(*gensys(*create_matrices("citr")), nperiods, "y")
series_peg_original_world  = irfs(*gensys(*create_matrices("peg")),  nperiods, "y")

#######################################
# Creating charts
#######################################
figsize= (9,5)
x_axis = range(1, nperiods+1)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#007E00", "#FF0000", "#00BFBF"],
                                             linestyle=["--", "solid", "solid"],
                                             marker=["None", "X", "o"])

# Figure 1 : Productivity shock
figure3 = plt.figure(figsize=figsize)
lines = []
charts = [figure3.add_subplot(3, 3, j+1) for j in range(9)]
limits = [(0,1.1), (-1,0.5), (0,1.5), (-.25,.25), (-.4,0.2), (-.5,.5), (-.25,.25), (-.5,.5), (0,1.5)]
ticks = [(0,1.5,.5), (-1,1,.5), (0,2,.5), (-.4,.6,.2), (-.5,.5,.25), (-.5,1.,0.5),
         (-.25,.5,.25), (-.5,1,.5), (0,2,.5)]
plot_titles = ["Productivity", "Output Gap", "Output", "CPI Inflation", "Domestic Inflation",
               "Real Wage", "Nominal Interest Rate", "Exchange rate deprec (Δe)", "Terms of Trade"]

for j in range(9):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    charts[j].set_ylim(limits[j])
    charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
    charts[j].set_xticks(range(0,nperiods+1,5))
    charts[j].set_xlim(0,nperiods+1)
    charts[j].grid(color="#000000", linestyle=':',  dashes=(1,4,1,4))

# Plot irfs
lines.append([charts[j].plot(x_axis, series_ditr_modification_prod[j])[0] for j in range(9)][0])
lines.append([charts[j].plot(x_axis, series_citr_modification_prod[j])[0] for j in range(9)][0])
lines.append([charts[j].plot(x_axis, series_peg_modification_prod[j])[0] for j in range(9)][0])

# Create legend and title
figure3.legend(lines, ["DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.95), loc=9, frameon=False)
figure3.suptitle("Impulse Responses - Productivity Shock\n", fontweight="bold")

# Draw chart 1
plt.tight_layout()
plt.draw()
plt.show()

# Figure 4 : World Output shock
figure4 = plt.figure(figsize=figsize)
charts = [figure4.add_subplot(3, 3, j+1) for j in range(9)]
limits = [(0,1.1), (-.5,1), (-.5,1), (-.3,.1), (-.2,.1), (-.2,0.4), (-0.4,.2), (-.5,.25), (-.5,0.25)]
ticks = [(0,1.5,.5), (-0.5,1.5,.5), (-.5,1.5,.5), (-.3,.2,.1), (-.2,.2,.1), (-.2,0.6,.2),
         (-.4,.4,.2), (-.5,.5,.25), (-.5,.5,.25)]
plot_titles[0] = "World Output"

for j in range(9):
    charts[j].set_title(plot_titles[j], fontsize = 10)
    charts[j].set_ylim(limits[j])
    charts[j].set_yticks(np.arange(ticks[j][0], ticks[j][1], ticks[j][2]))
    charts[j].set_xticks(range(0,nperiods+1,5))
    charts[j].set_xlim(0,nperiods+1)
    charts[j].grid(color="#000000", linestyle=':',  dashes=(1,4,1,4))

# Plot irfs
[charts[j].plot(x_axis, series_ditr_modification_world[j]) for j in range(9)]
[charts[j].plot(x_axis, series_citr_modification_world[j]) for j in range(9)]
[charts[j].plot(x_axis, series_peg_modification_world[j]) for j in range(9)]

# Create legend and title
figure4.legend(lines, ["DITR", "CITR", "PEG"], ncol=4,
               bbox_to_anchor=(0.5,0.95), loc=9, frameon=False)
figure4.suptitle("Impulse Responses - World Output Shock\n", fontweight="bold")

# Draw chart 2
plt.tight_layout()
plt.draw()
plt.show()

############################################################ Chart

# Figure 5 : Comparison
figure5 = plt.figure(figsize=figsize)
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0000FF", "#FF0000"], linestyle=["solid", "--"])

figure5.suptitle("Productivity Shock - Impulse Responses Comparison \n", fontweight="bold")
charts = [figure5.add_subplot(3, 3, j+1) for j in range(9)]
lines = []

for j in range(9):
    charts[j].grid(color="#000000", linestyle=':', dashes=(1, 4, 1, 4))
    charts[j].set_xticks(range(0, nperiods + 1, 5))
    charts[j].set_xlim(0, nperiods + 1)

charts[0].plot(x_axis, series_ditr_original_prod[1], label="Gali & Monacelli")
charts[0].plot(x_axis, series_ditr_modification_prod[1], label="Our Modification")
charts[0].set_ylim((-0.25, 0.25))
charts[0].set_yticks(np.arange(-0.25,0.5,0.25))
charts[0].set_title("DITR - Output Gap")

charts[1].plot(x_axis, series_ditr_original_prod[2])
charts[1].plot(x_axis, series_ditr_modification_prod[2])
charts[1].set_ylim((0, 1.1))
charts[1].set_yticks(np.arange(0,1.5,.5))
charts[1].set_title("DITR - Output")

charts[2].plot(x_axis, series_ditr_original_prod[4])
charts[2].plot(x_axis, series_ditr_modification_prod[4])
charts[2].set_ylim((-0.2, 0.1))
charts[2].set_yticks(np.arange(-.2,.2,.1))
charts[2].set_title("DITR - Domestic Inflation")

charts[3].plot(x_axis, series_citr_original_prod[1])
charts[3].plot(x_axis, series_citr_modification_prod[1])
charts[3].set_ylim((-0.5, 0.25))
charts[3].set_yticks(np.arange(-0.5,0.5,0.25))
charts[3].set_title("CITR - Output Gap")

charts[4].plot(x_axis, series_citr_original_prod[2])
charts[4].plot(x_axis, series_citr_modification_prod[2])
charts[4].set_ylim((0, 1))
charts[4].set_yticks(np.arange(0,1.5,.5))
charts[4].set_title("CITR - Output")

charts[5].plot(x_axis, series_citr_original_prod[4])
charts[5].plot(x_axis, series_citr_modification_prod[4])
charts[5].set_ylim((-.3, 0.05))
charts[5].set_yticks(np.arange(-.3,.1,.1))
charts[5].set_title("CITR - Domestic Inflation")

charts[6].plot(x_axis, series_peg_original_prod[1])
charts[6].plot(x_axis, series_peg_modification_prod[1])
charts[6].set_ylim((-1, 0.55))
charts[6].set_yticks(np.arange(-1,0.75,0.5))
charts[6].set_title("PEG - Output Gap")

charts[7].plot(x_axis, series_peg_original_prod[2])
charts[7].plot(x_axis, series_peg_modification_prod[2])
charts[7].set_ylim((0, 1.05))
charts[7].set_yticks(np.arange(0,1.5,.5))
charts[7].set_title("PEG - Output")

charts[8].plot(x_axis, series_peg_original_prod[4])
charts[8].plot(x_axis, series_peg_modification_prod[4])
charts[8].set_ylim((-.5, 0.25))
charts[8].set_yticks(np.arange(-.5,.5,.25))
charts[8].set_title("PEG - Domestic Inflation")


handles, labels = charts[0].get_legend_handles_labels()
figure5.legend(handles, labels, ncol=2, bbox_to_anchor=(0.5,0.95), loc=9, frameon=False)

# Draw chart 1
plt.tight_layout()
plt.draw()
plt.show()

figure3.savefig("..\\latex\\img\\figure3.svg")
figure4.savefig("..\\latex\\img\\figure4.svg")
figure5.savefig("..\\latex\\img\\figure5.svg")