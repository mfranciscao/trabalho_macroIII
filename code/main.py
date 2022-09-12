# -*- coding: UTF-8 -*-

"""
Macro III Project
"""
from math import log

#Parameters
epsilon  = 6         #elasticity of substitution between varieties (within any country)
alpha    = 0.4       #inverse of home bias preference or openess natural index
eta      = 1         #substitutability between domestic and foreign goods
gamma    = 1         #substitutability between goods produced in different countries
sigma    = 1         #coefficient of relative risk aversion
phi      = 3         #inverse Frisch elasticity of labor supply
beta     = 0.99      #intertemporal discount factor
v        = 1         #initial conditions relative to net asset positions (1 = ssymmetrical)
theta    = 0.75      #price stickiness / share of firms which need to keep prices unchanged
phi_pi   = 1.5       #Taylor rule response to inflation
rho_a    = 0.66      #AR(1) coeffiecient of productivity
rho_row  = 0.86      #AR(1) coefficient of Rest of World GDP
sigma_a  = 0.0071    #Variance of productivity shocks
sigma_y  = 0.0078    #Variance of shocks Rest of World GDP
correl   = 0.3       #Correlation of productivity shocks and Rest of World GDP shocks

rho      = 1/beta + 1
mu       = log(epsilon/(epsilon-1))
nu       = mu + log((1-alpha))
lambda_  = (1-theta)*(1-beta*theta)/theta
omega    = sigma*gamma + (1-alpha)*(sigma*eta-1)
sigma_al = sigma/((1-alpha)+alpha*omega)
Theta    = omega - 1
Omega    = (nu-mu)/(sigma_al+phi)
Gamma    = (1+phi)/(sigma_al+phi)
Psi      = (-Theta*sigma_al)/(sigma_al+phi)
kappa_al = lambda_*(sigma_al+phi)