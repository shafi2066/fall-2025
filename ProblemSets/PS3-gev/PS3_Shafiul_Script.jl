using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS3_Shafiul_Source.jl")

allwrap()

##########################################################
#Question :2 Interpretation of the estimated coefficient γ
##########################################################
#estimated gamma is -0.09419444797845519
#Gamma represents the change in latent utility
#With 1 unit change in the relative E(log wage)
#in occupation j (relative to other)
#positive gamma is intuitive as people prefer higher wages
#negative gamma is surprising as it suggests people prefer lower wages
#probably encountered mis-specification