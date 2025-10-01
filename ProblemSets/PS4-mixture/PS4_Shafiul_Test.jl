using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions

cd(@__DIR__)

Random.seed!(1234)

include("PS4_Shafiul_Source.jl")

# Test data loading
@testset "Data Loading" begin
    df, X, Z, y = load_data()
    @test size(X, 2) == 3
    @test size(Z, 2) == 8
    @test length(y) == size(X, 1)
end

# Test mlogit_with_Z
@testset "Multinomial Logit" begin
    df, X, Z, y = load_data()
    K = size(X, 2)
    J = length(unique(y))
    theta = [zeros(K*(J-1)); 0.0]
    val = mlogit_with_Z(theta, X, Z, y)
    @test isfinite(val)
    @test val >= 0
    # Type check
    @test isa(val, Float64)
    # Small data test
    Xs = [1.0 0.0 0.0; 0.0 1.0 0.0]
    Zs = [1.0 0.0; 0.0 1.0]
    ys = [1,2]
    thetas = [0.0, 0.0, 0.0, 0.0]
    @test isfinite(mlogit_with_Z(thetas, Xs, Zs, ys))
    # Error: wrong theta length
    @test_throws DimensionMismatch mlogit_with_Z([0.0,0.0], X, Z, y)
    # More granular tests for multinomial logit
    # Test with all ones theta
    theta_ones = ones(K*(J-1)+1)
    val_ones = mlogit_with_Z(theta_ones, X, Z, y)
    @test isfinite(val_ones)
    # Test with random theta
    theta_rand = randn(K*(J-1)+1)
    val_rand = mlogit_with_Z(theta_rand, X, Z, y)
    @test isfinite(val_rand)
    # Test with negative theta
    theta_neg = -ones(K*(J-1)+1)
    val_neg = mlogit_with_Z(theta_neg, X, Z, y)
    @test isfinite(val_neg)
    # Test with a range of theta values (moderate)
    for v in -5.0:2.5:5.0
        theta_mod = fill(v, K*(J-1)+1)
        val_mod = mlogit_with_Z(theta_mod, X, Z, y)
        @test isfinite(val_mod)
    end
    # Add more granular tests for multinomial logit
end

# Test quadrature function
@testset "Quadrature" begin
    nodes, weights = lgwt(7, -4, 4)
    @test length(nodes) == 7
    @test length(weights) == 7
    # Check that weights sum to approx 8 (interval length)
    @test abs(sum(weights) - 8) < 1e-6
    # Check node range
    @test minimum(nodes) >= -4
    @test maximum(nodes) <= 4
    # Check weights are positive
    @test all(weights .> 0)
    # Add more granular tests for different numbers of points
    for n in 2:10
        nodesn, weightsn = lgwt(n, -4, 4)
        @test length(nodesn) == n
        @test length(weightsn) == n
        @test abs(sum(weightsn) - 8) < 1e-6
    end
    # Type check
    @test isa(nodes, Vector{Float64})
    @test isa(weights, Vector{Float64})
end

# Test practice_quadrature and variance_quadrature (just check they run)
@testset "Quadrature Practice" begin
    @test isnothing(practice_quadrature())
    @test isnothing(variance_quadrature())
    # Add a test for output type
    nodes, weights = lgwt(7, -4, 4)
    d = Normal(0, 1)
    integral_density = sum(weights .* pdf.(d, nodes))
    @test abs(integral_density - 1) < 0.01
    # Add more tests for expectation
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    @test abs(expectation) < 0.01
end

# Test practice_monte_carlo (just check it runs)
@testset "Monte Carlo Practice" begin
    @test isnothing(practice_monte_carlo())
    # Add a test for MC integration output
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    function mc_integrate(f, a, b, D)
        draws = rand(D) * (b - a) .+ a
        return (b - a) * mean(f.(draws))
    end
    mc_var = mc_integrate(x -> x^2 * pdf(d, x), a, b, 10000)
    @test abs(mc_var - σ^2) < 0.5
    # Add more granular MC tests
    mc_mean = mc_integrate(x -> x * pdf(d, x), a, b, 10000)
    @test abs(mc_mean) < 0.2
    mc_density = mc_integrate(x -> pdf(d, x), a, b, 10000)
    @test abs(mc_density - 1) < 0.05
end

# Test mixed logit setup functions (just check they run)
@testset "Mixed Logit Setup" begin
    df, X, Z, y = load_data()
    @test length(optimize_mixed_logit_quad(X, Z, y)) > 0
    @test length(optimize_mixed_logit_mc(X, Z, y, 10)) > 0
    # Error: wrong input types
    @test_throws Union{BoundsError, DomainError} mixed_logit_quad([0.0], X, Z, y)
    @test_throws Union{BoundsError, DomainError} mixed_logit_mc([0.0], X, Z, y, 10)
    # Test with random starting values
    K = size(X, 2)
    J = length(unique(y))
    theta_rand = [randn(K*(J-1)+1); abs(randn()) + 0.1]  # last element positive for sigma_gamma
    val_quad = mixed_logit_quad(theta_rand, X, Z, y)
    @test isfinite(val_quad)
    val_mc = mixed_logit_mc(theta_rand, X, Z, y, 10)
    @test isfinite(val_mc)
end
