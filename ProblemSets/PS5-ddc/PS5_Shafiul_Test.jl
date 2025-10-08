using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

# Load the student's source (this will also include create_grids.jl via the source)
include("PS5_Shafiul_Source.jl")

# ---------- Basic API and include checks ----------
@testset "API and include" begin
    @test isdefined(Main, :load_static_data)
    @test isdefined(Main, :load_dynamic_data)
    @test isdefined(Main, :estimate_static_model)
    @test isdefined(Main, :compute_future_value!)
    @test isdefined(Main, :log_likelihood_dynamic)
    @test isdefined(Main, :create_grids)
end

# ---------- Static data loading tests ----------
@testset "Static data loading" begin
    df_long = load_static_data()
    @test isa(df_long, DataFrame)
    @test :bus_id in names(df_long)
    @test :RouteUsage in names(df_long)
    @test :Branded in names(df_long)
    @test :Y in names(df_long)
    @test :Odometer in names(df_long)
    @test nrow(df_long) >= 1000   # sanity: should be many rows (typical dataset 20k)
    # first rows have expected column types
    @test isa(df_long.bus_id[1], Integer)
end

# ---------- Dynamic data tests ----------
@testset "Dynamic data loading" begin
    d = load_dynamic_data()
    @test isa(d, NamedTuple)
    fldnames = fieldnames(typeof(d))
    required = (:Y, :X, :B, :Xstate, :Zstate, :N, :T, :xval, :xbin, :zbin, :xtran, :β)
    @test all(s -> s in fldnames, required)
    @test d.N > 0
    @test d.T > 0
    @test size(d.Y,1) == d.N
    @test size(d.Y,2) == d.T
end

# ---------- create_grids() checks ----------
@testset "create_grids" begin
    zval, zbin, xval, xbin, xtran = create_grids()
    @test isa(zval, AbstractVector)
    @test isa(xval, AbstractVector)
    @test isa(zbin, Integer)
    @test isa(xbin, Integer)
    @test isa(xtran, AbstractMatrix)
    # transition matrix shape sanity: rows should equal zbin*xbin
    @test size(xtran,1) == zbin * xbin
    # each row of xtran should sum to (approximately) 1.0
    rowsums = sum(xtran, dims=2)
    @test all(isfinite, rowsums)
    @test maximum(abs.(rowsums .- 1.0)) < 1e-8
end

# ---------- compute_future_value! on small toy problem ----------
@testset "compute_future_value! (toy problem)" begin
    # Build a tiny deterministic toy d for fast checks
    d_small = (; 
        xval = [0.0, 1.0, 2.0], 
        xbin = 3, 
        zbin = 2, 
        T = 4, 
        β = 0.9,
        # xtran: (zbin*xbin) × xbin transition rows (rows sum to 1)
        xtran = vcat(fill(1/3, (2*3,1)) .* ones(1,3)) .+ 0.0
    )
    # ensure xtran has correct shape: (zbin*xbin, xbin)
    d_small = merge(d_small, (; xtran = reshape(fill(1/3, d_small.xbin*d_small.zbin*d_small.xbin), d_small.xbin*d_small.zbin, d_small.xbin) ))
    FV = zeros(d_small.xbin * d_small.zbin, 2, d_small.T + 1)
    θ = [0.5, -0.1, 0.2]
    # Should run quickly and produce finite numbers
    compute_future_value!(FV, θ, d_small)
    @test size(FV) == (d_small.xbin * d_small.zbin, 2, d_small.T + 1)
    @test all(isfinite, FV)
    # terminal condition: last time-slice should be finite (initially zeros but algorithm may fill earlier slices)
    @test all(isfinite, FV[:, :, end])
end

# ---------- log_likelihood_dynamic on toy dataset ----------
@testset "log_likelihood_dynamic (toy dataset)" begin
    # Construct a small dataset compatible with log_likelihood_dynamic
    N = 2; T = 3; xbin = 3; zbin = 2
    # simple deterministic transitions: uniform next-state distribution
    xtran = fill(1.0/xbin, zbin * xbin, xbin)
    # Observed matrices/vectors (indices for states should be 1..xbin)
    Y = zeros(Int, N, T)
    Y[1,1] = 1; Y[2,3] = 1
    X = ones(Int, N, T)                 # simple odometer (not used heavily)
    Xstate = ones(Int, N, T)            # pick first mileage bin for simplicity
    Zstate = ones(Int, N)               # route usage index 1
    B = zeros(Int, N)                   # brand 0 for all
    d2 = (Y=Y, X=X, B=B, Xstate=Xstate, Zstate=Zstate, N=N, T=T, xval=[0.,1.,2.], xbin=xbin, zbin=zbin, xtran=xtran, β=0.9)
    θ = [0.1, -0.05, 0.2]
    val = log_likelihood_dynamic(θ, d2)
    @test isa(val, Float64)
    @test isfinite(val)
end

# ---------- Static estimation smoke test ----------
@testset "Static estimation smoke" begin
    df_long = load_static_data()
    # The student's function currently prints and returns nothing; we at least ensure it runs
    @test estimate_static_model(df_long) === nothing
    # Also verify glm directly on the data produces finite coefficients (sanity)
    mdl = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    coefs = coef(mdl)
    @test all(isfinite, coefs)
end

# ---------- API presence but do not run heavy optimization ----------
@testset "Optimization presence (no-run)" begin
    @test typeof(estimate_dynamic_model) <: Function
    # Do NOT call estimate_dynamic_model() here; it runs Optim and is slow.
    # Instead, check that its method exists and can be called with a wrong argument type to raise a predictable error.
    @test_throws MethodError estimate_dynamic_model("not_a_named_tuple")
end

println("\nPS5 tests defined. Run with `include(\"PS5_Shafiul_Test.jl\")` or via Julia --project -e 'using Test; include(\"PS5_Shafiul_Test.jl\")' to execute.")
