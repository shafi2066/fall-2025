using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
cd(@__DIR__)
include("PS2_Shafiul_Source.jl")  

@testset "Problem Set 2 Unit Tests" begin

    # Test Question 1: Optimization of quartic function
    @testset "Q1: Optimization" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        minusf(x) = -f(x)
        result = optimize(minusf, [0.0], BFGS())
        @test isapprox(Optim.minimizer(result)[1], -7.37824, atol=1e-5)
        @test isapprox(Optim.minimum(result), -964.31338, atol=1e-5)
    end

    # Test Question 2: OLS estimation
    @testset "Q2: OLS Function" begin
        X_test = rand(500, 5)
        β = [1., 2., 3., 4., 5.]
        y_test = X_test * β + randn(500)
        beta_hat_ols = optimize(b -> ols(b, X_test, y_test), rand(size(X_test,2)), LBFGS()).minimizer
        bols = inv(X_test' * X_test) * X_test' * y_test

        # Check optimizer approximate solution vs closed form and true coefficients
        @test isapprox(beta_hat_ols, bols, atol=1e-10)
        @test isapprox(beta_hat_ols, β, atol=1e0)
        @test isapprox(bols, β, atol=1e0)
    end

    # Test Question 3: Logit function
    @testset "Q3: Logit Function" begin
        function logit_test(alpha, X, y)
            P = exp.(X * alpha) ./ (1 .+ exp.(X * alpha))
            loglike = -sum((y .== 1) .* log.(P) .+ (y .== 0) .* log.(1 .- P))
            return loglike
        end
        X_test = randn(1000, 5)
        β = [1., 0.5, -0.5, 2., -2.]
        P_test = 1 ./ (1 .+ exp.(-X_test * β))
        y_test = rand.(Bernoulli.(P_test))
        alpha_hat = optimize(a -> logit_test(a, X_test, y_test), rand(size(X_test,2)), LBFGS()).minimizer

        @test isapprox(alpha_hat, β, atol=1e0)
        @test sign.(alpha_hat) == sign.(β)
    end

    # Test Question 5: Multinomial Logit
    @testset "Q5: Multinomial Logit Function" begin
        function mlogit_test(alpha, X, y)
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j in 1:J
                bigY[:,j] .= y .== j
            end
            bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]
            num = zeros(N, J)
            dem = zeros(N)
            for j in 1:J
                num[:,j] .= exp.(X * bigAlpha[:,j])
                dem .+= num[:,j]
            end
            P = num ./ dem
            loglike = -sum(bigY .* log.(P))
            return loglike
        end

        function generate_multinomial_logit_data(N, K)
            X = randn(N, K)
            X = hcat(ones(N), X)
            β = [1.0 0.1 0.0;
                 2.0 1.1 0.0;
                 -1.5 1.0 0.0;
                 0.5 -0.5 0.0;
                 1.0 -1.0 0.0]

            @assert size(β, 1) == K + 1
            @assert size(β, 2) == 3

            U = X * β
            P = exp.(U) ./ sum(exp.(U), dims=2)
            y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]

            return X, y, β
        end

        N, K = 1000, 4
        X_test, y_test, β = generate_multinomial_logit_data(N, K)
        alpha_start = rand((K + 1) * (3 - 1))
        alpha_hat = optimize(a -> mlogit_test(a, X_test, y_test), alpha_start, LBFGS()).minimizer

        @test size(alpha_hat) == size(β[:,1:2][:])
        @test isapprox(alpha_hat, β[:,1:2][:], atol=1e0)
        @test sign.(alpha_hat) == sign.(β[:,1:2][:])
    end

end

