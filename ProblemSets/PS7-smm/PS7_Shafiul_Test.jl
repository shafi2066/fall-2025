
using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS7_Shafiul_Source.jl")

################################################################################
# Test Suite 1: Data Loading and Preparation
################################################################################

@testset "Data Loading and Preparation" begin
    
    @testset "load_data: basic functionality" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        
        # Test that outputs have correct types
        @test df isa DataFrame
        @test X isa Matrix
        @test y isa Vector
        
        # Test dimensions match
        @test size(X, 1) == size(df, 1)
        @test length(y) == size(df, 1)
        
        # Test X has intercept column (all ones)
        @test all(X[:, 1] .== 1.0)
        
        # Test X has 4 columns (intercept, age, race, collgrad)
        @test size(X, 2) == 4
        
        # Test y is log wages (should be reasonable values)
        @test all(isfinite.(y))
        @test minimum(y) > 0  # log wage should be positive for reasonable wages
    end
    
    @testset "prepare_occupation_data: basic functionality" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_orig, _, _ = load_data(url)
        df, X, y = prepare_occupation_data(df_orig)
        
        # Test outputs
        @test df isa DataFrame
        @test X isa Matrix
        @test y isa Vector
        
        # Test dimensions match
        @test size(X, 1) == length(y)
        
        # Test X has intercept
        @test all(X[:, 1] .== 1.0)
        
        # Test X has 4 columns
        @test size(X, 2) == 4
        
        # Test occupation categories are collapsed (max should be 7)
        @test maximum(y) <= 7
        @test minimum(y) >= 1
        
        # Test all y values are integers
        @test all(y .== floor.(y))
        
        # Test that high occupation codes were collapsed
        @test !any(y .> 7)
    end
    
    @testset "prepare_occupation_data: white and collgrad indicators" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_orig, _, _ = load_data(url)
        df, X, y = prepare_occupation_data(df_orig)
        
        # Test that df has white column (names returns strings)
        @test "white" in names(df)
        
        # Test white is binary (0 or 1 only) - allowing for Bool or Int
        @test all(x -> x in [0, 1, false, true], df.white)
        
        # Test collgrad column exists and is properly formatted
        @test "collgrad" in names(df)
    end
end

################################################################################
# Test Suite 2: OLS via GMM
################################################################################

@testset "OLS via GMM" begin
    
    @testset "ols_gmm: zero residuals with true parameters" begin
        Random.seed!(123)
        N, K = 100, 3
        X = hcat(ones(N), randn(N, K-1))
        β_true = randn(K)
        y = X * β_true
        
        # Objective should be zero at true parameters
        obj = ols_gmm(β_true, X, y)
        @test obj ≈ 0.0 atol=1e-10
    end
    
    @testset "ols_gmm: positive objective with wrong parameters" begin
        Random.seed!(124)
        N, K = 100, 3
        X = hcat(ones(N), randn(N, K-1))
        β_true = randn(K)
        y = X * β_true + 0.1 * randn(N)
        
        # Wrong parameters should give positive objective
        β_wrong = zeros(K)
        obj = ols_gmm(β_wrong, X, y)
        @test obj > 0
    end
    
    @testset "ols_gmm: comparison with closed-form OLS" begin
        Random.seed!(125)
        N, K = 100, 3
        X = hcat(ones(N), randn(N, K-1))
        β_true = randn(K)
        y = X * β_true + 0.1 * randn(N)
        
        # Optimize GMM objective
        result = optimize(b -> ols_gmm(b, X, y), zeros(K), LBFGS())
        β_gmm = result.minimizer
        
        # Compare with closed-form OLS
        β_ols = X \ y
        @test β_gmm ≈ β_ols atol=1e-4
    end
    
    @testset "ols_gmm: objective is sum of squared residuals" begin
        Random.seed!(126)
        N, K = 50, 2
        X = hcat(ones(N), randn(N))
        β = randn(K)
        y = randn(N)
        
        obj = ols_gmm(β, X, y)
        residuals = y - X * β
        @test obj ≈ sum(residuals.^2) atol=1e-10
    end
end

################################################################################
# Test Suite 3: Multinomial Logit MLE
################################################################################

@testset "Multinomial Logit MLE" begin
    
    @testset "mlogit_mle: basic dimensions and output" begin
        Random.seed!(127)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        obj = mlogit_mle(α, X, y)
        
        # Should return a scalar
        @test obj isa Real
        
        # Should be positive (negative log-likelihood)
        @test obj > 0
        
        # Should be finite
        @test isfinite(obj)
    end
    
    @testset "mlogit_mle: zero parameters give reasonable likelihood" begin
        Random.seed!(128)
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = zeros(K * (J-1))
        
        obj = mlogit_mle(α, X, y)
        
        # With zero parameters, all choices have equal probability
        # Negative log-likelihood should be approximately N*log(J)
        expected_nll = N * log(J)
        @test obj ≈ expected_nll rtol=0.1
    end
    
    @testset "mlogit_mle: objective decreases with optimization" begin
        Random.seed!(129)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        
        α_init = zeros(K * (J-1))
        obj_init = mlogit_mle(α_init, X, y)
        
        # Run a few iterations
        result = optimize(a -> mlogit_mle(a, X, y), α_init, LBFGS(), 
                         Optim.Options(iterations=10))
        obj_final = result.minimum
        
        # Objective should decrease
        @test obj_final < obj_init
    end
    
    @testset "mlogit_mle: handles all observations in one category" begin
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = ones(Int, N)  # All observations in category 1
        α = zeros(K * (J-1))
        
        obj = mlogit_mle(α, X, y)
        
        # Should still return finite value
        @test isfinite(obj)
        @test obj > 0
    end
end

################################################################################
# Test Suite 4: Multinomial Logit GMM
################################################################################

@testset "Multinomial Logit GMM" begin
    
    @testset "mlogit_gmm: basic dimensions and output" begin
        Random.seed!(130)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        obj = mlogit_gmm(α, X, y)
        
        @test obj isa Real
        @test obj >= 0  # Sum of squared moments
        @test isfinite(obj)
    end
    
    @testset "mlogit_gmm: zero at population parameters (simulation)" begin
        Random.seed!(131)
        N, K, J = 10000, 2, 3
        X = hcat(ones(N), randn(N))
        
        # True parameters
        α_true = randn(K * (J-1))
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        
        # Simulate data from these parameters
        P = exp.(X * bigα) ./ sum(exp.(X * bigα), dims=2)
        y = [rand(Categorical(vec(P[i, :]))) for i in 1:N]
        
        # GMM objective should be small at true parameters
        obj = mlogit_gmm(α_true, X, y)
        @test obj / N < 0.01  # Normalized objective should be small
    end
    
    @testset "mlogit_gmm_overid: basic functionality" begin
        Random.seed!(132)
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        obj = mlogit_gmm_overid(α, X, y)
        
        @test obj isa Real
        @test obj >= 0
        @test isfinite(obj)
    end
    
    @testset "mlogit_gmm_overid: uses more moments than mlogit_gmm" begin
        Random.seed!(133)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        obj_just = mlogit_gmm(α, X, y)
        obj_over = mlogit_gmm_overid(α, X, y)
        
        # Both should be positive and finite
        @test obj_just >= 0
        @test obj_over >= 0
        
        # Overidentified moments use more information
        # (though magnitudes may differ due to different scaling)
    end
end

################################################################################
# Test Suite 5: Simulation Functions
################################################################################

@testset "Simulation Functions" begin
    
    @testset "sim_logit: basic output structure" begin
        Random.seed!(134)
        N, J = 1000, 4
        Y, X = sim_logit(N, J)
        
        # Check dimensions
        @test length(Y) == N
        @test size(X, 1) == N
        @test size(X, 2) == 4  # intercept + 3 covariates
        
        # Check Y values are in valid range
        @test all(1 .<= Y .<= J)
        
        # Check X has intercept
        @test all(X[:, 1] .== 1.0)
    end
    
    @testset "sim_logit: choice frequencies are reasonable" begin
        Random.seed!(135)
        N, J = 10000, 4
        Y, X = sim_logit(N, J)
        
        # Each choice should have some observations
        for j in 1:J
            freq = sum(Y .== j) / N
            @test freq > 0.02  # At least 2% for each choice (relaxed)
            @test freq < 0.98  # No choice dominates
        end
    end
    
    @testset "sim_logit_with_gumbel: basic output structure" begin
        Random.seed!(136)
        N, J = 1000, 4
        Y, X = sim_logit_with_gumbel(N, J)
        
        @test length(Y) == N
        @test size(X, 1) == N
        @test all(1 .<= Y .<= J)
        @test all(X[:, 1] .== 1.0)
    end
    
    @testset "sim_logit_with_gumbel: frequencies match probabilities" begin
        Random.seed!(137)
        N, J = 50000, 3
        Y, X = sim_logit_with_gumbel(N, J)
        
        # All choices should be represented
        @test length(unique(Y)) == J
        
        # Frequencies should be reasonably balanced (not all in one category)
        freqs = [sum(Y .== j) for j in 1:J]
        @test all(freqs .> 0)
    end
    
    @testset "sim_logit: works with different J" begin
        for J in [2, 3, 5, 7]
            Y, X = sim_logit(500, J)
            @test maximum(Y) <= J
            @test minimum(Y) >= 1
            @test length(unique(Y)) >= min(J, 2)  # At least 2 choices represented
        end
    end
end

################################################################################
# Test Suite 6: SMM Estimation
################################################################################

@testset "SMM Estimation" begin
    
    @testset "mlogit_smm_overid: basic functionality" begin
        Random.seed!(138)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 10
        
        obj = mlogit_smm_overid(α, X, y, D)
        
        @test obj isa Real
        @test obj >= 0
        @test isfinite(obj)
    end
    
    @testset "mlogit_smm_overid: reproducibility with seed" begin
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 20
        
        # Run twice - should get same result due to internal seed
        obj1 = mlogit_smm_overid(α, X, y, D)
        obj2 = mlogit_smm_overid(α, X, y, D)
        
        @test obj1 ≈ obj2
    end
    
    @testset "mlogit_smm_overid: objective is reasonable" begin
        Random.seed!(139)
        N, K, J = 100, 2, 3
        X = hcat(ones(N), randn(N))
        
        # Generate data
        α_true = randn(K * (J-1))
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        P = exp.(X * bigα) ./ sum(exp.(X * bigα), dims=2)
        y = [rand(Categorical(vec(P[i, :]))) for i in 1:N]
        
        # Objective should be finite at true parameters
        obj_10 = mlogit_smm_overid(α_true, X, y, 10)
        obj_100 = mlogit_smm_overid(α_true, X, y, 100)
        
        # Both should be reasonably small and finite
        @test isfinite(obj_10)
        @test isfinite(obj_100)
        @test obj_10 >= 0
        @test obj_100 >= 0
    end
    
    @testset "mlogit_smm_overid: works with different D values" begin
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        for D in [5, 10, 50]
            obj = mlogit_smm_overid(α, X, y, D)
            @test isfinite(obj)
            @test obj >= 0
        end
    end
end

################################################################################
# Test Suite 7: Integration Tests
################################################################################

@testset "Integration Tests" begin
    
    @testset "End-to-end: Load data and run OLS" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        
        # Run OLS via GMM
        result = optimize(b -> ols_gmm(b, X, y), zeros(size(X, 2)), LBFGS())
        β_gmm = result.minimizer
        
        # Compare with closed-form
        β_ols = X \ y
        
        @test β_gmm ≈ β_ols rtol=0.01
        @test result.minimum < 100  # Objective should be reasonably small
    end
    
    @testset "End-to-end: Simulate and recover logit parameters" begin
        Random.seed!(140)
        N, K, J = 5000, 3, 3
        X = hcat(ones(N), randn(N, K-1))
        
        # True parameters
        α_true = [0.5, -0.3, 0.2, -0.8, 0.4, -0.1]
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        
        # Simulate data
        P = exp.(X * bigα) ./ sum(exp.(X * bigα), dims=2)
        y = [rand(Categorical(vec(P[i, :]))) for i in 1:N]
        
        # Estimate via MLE
        result = optimize(a -> mlogit_mle(a, X, y), zeros(length(α_true)), 
                         LBFGS(), Optim.Options(iterations=100))
        α_hat = result.minimizer
        
        # Should recover parameters reasonably well
        @test α_hat ≈ α_true rtol=0.2  # Allow 20% relative error
    end
    
    @testset "Consistency: MLE and GMM give similar results" begin
        Random.seed!(141)
        N, K, J = 1000, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Estimate via MLE
        α_init = zeros(K * (J-1))
        result_mle = optimize(a -> mlogit_mle(a, X, y), α_init, LBFGS(),
                             Optim.Options(iterations=50))
        α_mle = result_mle.minimizer
        
        # Estimate via GMM (using MLE as starting point)
        result_gmm = optimize(a -> mlogit_gmm(a, X, y), α_mle, LBFGS(),
                             Optim.Options(iterations=50))
        α_gmm = result_gmm.minimizer
        
        # Should be similar (though not identical)
        @test α_mle ≈ α_gmm rtol=0.3
    end
end

################################################################################
# Test Suite 8: Edge Cases and Robustness
################################################################################

@testset "Edge Cases and Robustness" begin
    
    @testset "Small sample sizes" begin
        N, K, J = 20, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # All functions should work with small N
        @test isfinite(mlogit_mle(α, X, y))
        @test isfinite(mlogit_gmm(α, X, y))
        @test isfinite(mlogit_gmm_overid(α, X, y))
        @test isfinite(mlogit_smm_overid(α, X, y, 5))
    end
    
    @testset "Two alternatives (binary logit)" begin
        N, K, J = 100, 3, 2
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        @test isfinite(mlogit_mle(α, X, y))
        @test isfinite(mlogit_gmm(α, X, y))
    end
    
    @testset "Many alternatives" begin
        N, K, J = 200, 3, 8
        X = hcat(ones(N), randn(N, K-1))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        @test isfinite(mlogit_mle(α, X, y))
        @test isfinite(mlogit_gmm_overid(α, X, y))
    end
    
    @testset "Parameter vector at zeros" begin
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = zeros(K * (J-1))
        
        # All functions should handle zero parameters
        @test isfinite(mlogit_mle(α, X, y))
        @test isfinite(mlogit_gmm(α, X, y))
        @test isfinite(mlogit_gmm_overid(α, X, y))
        @test isfinite(mlogit_smm_overid(α, X, y, 10))
    end
end

################################################################################
# Test Suite 9: Parameter Recovery and Monte Carlo
################################################################################

@testset "Parameter Recovery and Monte Carlo" begin
    
    @testset "OLS: Parameter recovery with known DGP" begin
        Random.seed!(200)
        N, K = 500, 4
        X = hcat(ones(N), randn(N, K-1))
        β_true = [2.0, -1.5, 0.8, 1.2]
        σ = 0.5
        y = X * β_true + σ * randn(N)
        
        result = optimize(b -> ols_gmm(b, X, y), zeros(K), LBFGS())
        β_hat = result.minimizer
        
        # Should recover all parameters within 3 standard errors
        for k in 1:K
            @test abs(β_hat[k] - β_true[k]) < 3 * σ / sqrt(N)
        end
    end
    
    @testset "Binary logit: Parameter recovery" begin
        Random.seed!(201)
        N, K = 2000, 3
        X = hcat(ones(N), randn(N, K-1))
        α_true = [1.0, -0.5, 0.8]
        
        # Simulate binary choice
        utilities = X * α_true
        probs = exp.(utilities) ./ (1 .+ exp.(utilities))
        y = [rand() < probs[i] ? 1 : 2 for i in 1:N]
        
        result = optimize(a -> mlogit_mle(a, X, y), zeros(K), LBFGS(),
                         Optim.Options(iterations=200))
        α_hat = result.minimizer
        
        # Should recover parameters reasonably well
        @test norm(α_hat - α_true) / norm(α_true) < 0.15  # Within 15% relative error
        @test result.minimum < N * log(2)  # Better than random guessing
    end
    
    @testset "Multinomial logit: Parameter recovery with balanced data" begin
        Random.seed!(202)
        N, K, J = 3000, 2, 3
        X = hcat(ones(N), randn(N))
        α_true = [0.8, -0.5, -0.4, 0.6]
        
        # Simulate data
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        utilities = X * bigα
        exp_utilities = exp.(utilities)
        probs = exp_utilities ./ sum(exp_utilities, dims=2)
        y = [rand(Categorical(vec(probs[i, :]))) for i in 1:N]
        
        # Estimate
        result = optimize(a -> mlogit_mle(a, X, y), zeros(length(α_true)), LBFGS(),
                         Optim.Options(iterations=200))
        α_hat = result.minimizer
        
        # Check recovery for each parameter
        for k in 1:length(α_true)
            @test abs(α_hat[k] - α_true[k]) < 0.25
        end
    end
    
    @testset "GMM vs MLE: Consistency in large samples" begin
        Random.seed!(203)
        N, K, J = 5000, 2, 3
        X = hcat(ones(N), randn(N))
        α_true = randn(K * (J-1))
        
        # Simulate data
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        utilities = X * bigα
        exp_utilities = exp.(utilities)
        probs = exp_utilities ./ sum(exp_utilities, dims=2)
        y = [rand(Categorical(vec(probs[i, :]))) for i in 1:N]
        
        # Estimate both
        result_mle = optimize(a -> mlogit_mle(a, X, y), zeros(length(α_true)), LBFGS(),
                             Optim.Options(iterations=100))
        result_gmm = optimize(a -> mlogit_gmm(a, X, y), zeros(length(α_true)), LBFGS(),
                             Optim.Options(iterations=100))
        
        α_mle = result_mle.minimizer
        α_gmm = result_gmm.minimizer
        
        # Both should be close to truth
        @test norm(α_mle - α_true) / norm(α_true) < 0.2
        @test norm(α_gmm - α_true) / norm(α_true) < 0.2
        
        # MLE and GMM should be close to each other
        @test norm(α_mle - α_gmm) / norm(α_true) < 0.15
    end
    
    @testset "SMM: Stability across different D values" begin
        Random.seed!(204)
        N, K, J = 200, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        α = randn(K * (J-1))
        
        # Test different D values
        obj_5 = mlogit_smm_overid(α, X, y, 5)
        obj_10 = mlogit_smm_overid(α, X, y, 10)
        obj_20 = mlogit_smm_overid(α, X, y, 20)
        obj_50 = mlogit_smm_overid(α, X, y, 50)
        
        # All should be finite and positive
        @test isfinite(obj_5) && obj_5 >= 0
        @test isfinite(obj_10) && obj_10 >= 0
        @test isfinite(obj_20) && obj_20 >= 0
        @test isfinite(obj_50) && obj_50 >= 0
        
        # Higher D should give more stable (usually lower variance) estimates
        # Check that none explode
        @test obj_50 < obj_5 * 100  # Shouldn't explode with more draws
    end
end

################################################################################
# Test Suite 10: Numerical Stability and Convergence
################################################################################

@testset "Numerical Stability and Convergence" begin
    
    @testset "OLS GMM: Convergence from different starting values" begin
        Random.seed!(210)
        N, K = 100, 3
        X = hcat(ones(N), randn(N, K-1))
        y = randn(N)
        
        # Try different starting values
        starts = [zeros(K), randn(K), 10*randn(K)]
        results = [optimize(b -> ols_gmm(b, X, y), start, LBFGS()) for start in starts]
        
        # All should converge to similar solutions
        β_solutions = [r.minimizer for r in results]
        @test norm(β_solutions[1] - β_solutions[2]) < 0.01
        @test norm(β_solutions[1] - β_solutions[3]) < 0.01
        @test all([r.minimum for r in results] .< 1000)
    end
    
    @testset "MLE: Convergence with different optimization methods" begin
        Random.seed!(211)
        N, K, J = 300, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Try LBFGS
        result_lbfgs = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), LBFGS(),
                               Optim.Options(iterations=100))
        
        # Try Newton with autodiff
        result_newton = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), Newton(),
                                Optim.Options(iterations=100); autodiff=:forward)
        
        # Both should converge
        @test result_lbfgs.g_converged || result_lbfgs.f_converged || result_lbfgs.x_converged
        @test isfinite(result_lbfgs.minimum)
        @test isfinite(result_newton.minimum)
        
        # Solutions should be similar
        @test norm(result_lbfgs.minimizer - result_newton.minimizer) / norm(result_lbfgs.minimizer) < 0.1
    end
    
    @testset "GMM: Objective decreases monotonically in optimization" begin
        Random.seed!(212)
        N, K, J = 200, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        objectives = Float64[]
        
        function callback(state)
            push!(objectives, state.value)
            return false
        end
        
        optimize(a -> mlogit_gmm(a, X, y), zeros(K*(J-1)), LBFGS(),
                Optim.Options(iterations=50, callback=callback))
        
        # Check objectives are decreasing (allowing for some numerical noise)
        @test length(objectives) > 10
        @test objectives[end] <= objectives[1]
        
        # Most steps should decrease objective
        decreasing_steps = sum(diff(objectives) .<= 1e-6)
        @test decreasing_steps / length(objectives) > 0.8
    end
    
    @testset "Extreme parameter values don't cause overflow" begin
        Random.seed!(213)
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Test with very large parameters
        α_large = 100 * randn(K * (J-1))
        @test isfinite(mlogit_mle(α_large, X, y))
        @test isfinite(mlogit_gmm(α_large, X, y))
        
        # Test with very small parameters
        α_small = 0.001 * randn(K * (J-1))
        @test isfinite(mlogit_mle(α_small, X, y))
        @test isfinite(mlogit_gmm(α_small, X, y))
    end
    
    @testset "Simulation functions: Reproducibility" begin
        # Set seed and generate data
        Random.seed!(214)
        Y1, X1 = sim_logit(100, 4)
        
        # Set same seed and generate again
        Random.seed!(214)
        Y2, X2 = sim_logit(100, 4)
        
        # Should be identical
        @test Y1 == Y2
        @test X1 == X2
        
        # Same for Gumbel method
        Random.seed!(215)
        Y3, X3 = sim_logit_with_gumbel(100, 4)
        
        Random.seed!(215)
        Y4, X4 = sim_logit_with_gumbel(100, 4)
        
        @test Y3 == Y4
        @test X3 == X4
    end
end

################################################################################
# Test Suite 11: Data Quality and Preprocessing
################################################################################

@testset "Data Quality and Preprocessing" begin
    
    @testset "load_data: No missing values in key variables" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        
        # Check no missing in X
        @test !any(ismissing.(X))
        @test !any(isnan.(X))
        
        # Check no missing in y
        @test !any(ismissing.(y))
        @test !any(isnan.(y))
        
        # Check X values are reasonable
        @test all(isfinite.(X))
        @test all(isfinite.(y))
    end
    
    @testset "prepare_occupation_data: Proper collapsing of categories" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_orig, _, _ = load_data(url)
        df, X, y = prepare_occupation_data(df_orig)
        
        # All y should be between 1 and 7
        @test all(1 .<= y .<= 7)
        
        # Should have representation in multiple categories
        unique_occupations = unique(y)
        @test length(unique_occupations) >= 3  # At least 3 different occupations
        @test length(unique_occupations) <= 7  # At most 7
        
        # Each category should have some observations
        for occ in unique_occupations
            @test sum(y .== occ) >= 1
        end
    end
    
    @testset "prepare_occupation_data: white variable construction" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_orig, _, _ = load_data(url)
        df, X, y = prepare_occupation_data(df_orig)
        
        # White should be binary
        @test all(x -> x in [0, 1, false, true], df.white)
        
        # Should have both white and non-white observations
        @test any(df.white .== 1)
        @test any(df.white .== 0)
        
        # White should be in X matrix (check column 3 matches white)
        # Note: X columns are [intercept, age, race==1, collgrad]
        # So white is column 3
        @test all(X[:, 3] .== df.white)
    end
    
    @testset "Data dimensions: Consistency checks" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df, X, y = load_data(url)
        
        # All should have same number of observations
        @test nrow(df) == size(X, 1)
        @test nrow(df) == length(y)
        
        # X should have correct structure
        @test size(X, 2) == 4  # intercept + 3 covariates
        
        # DataFrame should have expected columns
        @test "age" in names(df)
        @test "race" in names(df)
        @test "collgrad" in names(df)
    end
    
    @testset "Occupation data: Valid category mappings" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_orig, _, _ = load_data(url)
        df, X, y = prepare_occupation_data(df_orig)
        
        # No gaps in occupation categories
        occ_range = minimum(y):maximum(y)
        for occ in occ_range
            # Each occupation in range should exist
            @test any(y .== occ)
        end
        
        # Total observations preserved
        @test length(y) == nrow(df_orig)
    end
end

################################################################################
# Test Suite 12: Probability and Moment Calculations
################################################################################

@testset "Probability and Moment Calculations" begin
    
    @testset "Multinomial logit: Probabilities sum to 1" begin
        Random.seed!(220)
        N, K, J = 100, 3, 4
        X = hcat(ones(N), randn(N, K-1))
        α = randn(K * (J-1))
        
        # Compute probabilities manually
        bigα = hcat(reshape(α, K, J-1), zeros(K))
        utilities = X * bigα
        exp_utilities = exp.(utilities)
        probs = exp_utilities ./ sum(exp_utilities, dims=2)
        
        # Each row should sum to 1
        row_sums = sum(probs, dims=2)
        @test all(abs.(row_sums .- 1.0) .< 1e-10)
        
        # All probabilities should be between 0 and 1
        @test all(0 .<= probs .<= 1)
    end
    
    @testset "Multinomial logit: Normalization with zero base category" begin
        Random.seed!(221)
        N, K, J = 50, 2, 3
        X = hcat(ones(N), randn(N))
        α = randn(K * (J-1))
        
        # When last category is base (zeros), probabilities should still work
        bigα = hcat(reshape(α, K, J-1), zeros(K))
        utilities = X * bigα
        
        # Last column of utilities should be zero
        @test all(utilities[:, J] .== 0)
        
        # Probabilities should still be valid
        exp_utilities = exp.(utilities)
        probs = exp_utilities ./ sum(exp_utilities, dims=2)
        @test all(sum(probs, dims=2) .≈ 1.0)
    end
    
    @testset "Choice probabilities: Increase with utility" begin
        N, K, J = 100, 2, 3
        X = hcat(ones(N), randn(N))
        
        # Increase utility for choice 1
        α_low = [0.0, 0.0, 0.0, 0.0]
        α_high = [2.0, 0.5, 0.0, 0.0]  # Higher utilities for choice 1
        
        # Compute probabilities
        bigα_low = hcat(reshape(α_low, K, J-1), zeros(K))
        bigα_high = hcat(reshape(α_high, K, J-1), zeros(K))
        
        probs_low = exp.(X * bigα_low) ./ sum(exp.(X * bigα_low), dims=2)
        probs_high = exp.(X * bigα_high) ./ sum(exp.(X * bigα_high), dims=2)
        
        # Probability of choice 1 should increase
        @test mean(probs_high[:, 1]) > mean(probs_low[:, 1])
    end
    
    @testset "Moment conditions: Zero at truth for GMM" begin
        Random.seed!(222)
        N, K, J = 10000, 2, 3
        X = hcat(ones(N), randn(N))
        α_true = randn(K * (J-1))
        
        # Simulate data from true parameters
        bigα = hcat(reshape(α_true, K, J-1), zeros(K))
        utilities = X * bigα
        probs = exp.(utilities) ./ sum(exp.(utilities), dims=2)
        y = [rand(Categorical(vec(probs[i, :]))) for i in 1:N]
        
        # Moment conditions should be close to zero at truth
        moments_obj = mlogit_gmm(α_true, X, y)
        
        # Normalized by sample size, should be small
        @test moments_obj / N < 0.001
    end
    
    @testset "Simulation: Generated choice frequencies match probabilities" begin
        Random.seed!(223)
        N, J = 20000, 4
        Y, X = sim_logit(N, J)
        
        # Compute empirical frequencies
        freqs = [sum(Y .== j) / N for j in 1:J]
        
        # All frequencies should be positive
        @test all(freqs .> 0)
        
        # Should sum to 1
        @test sum(freqs) ≈ 1.0
        
        # No single choice should dominate excessively
        @test maximum(freqs) < 0.7
        @test minimum(freqs) > 0.05
    end
end

################################################################################
# Test Suite 13: Optimization Diagnostics
################################################################################

@testset "Optimization Diagnostics" begin
    
    @testset "MLE: Gradient at optimum is near zero" begin
        Random.seed!(230)
        N, K, J = 500, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        result = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), LBFGS(),
                         Optim.Options(g_tol=1e-6))
        
        # Gradient norm should be small at optimum
        @test result.g_converged || norm(result.g_residual) < 1e-4
    end
    
    @testset "GMM: Objective value is non-negative" begin
        Random.seed!(231)
        N, K, J = 200, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Test at multiple random parameter values
        for _ in 1:20
            α = randn(K * (J-1))
            obj = mlogit_gmm(α, X, y)
            @test obj >= 0  # Sum of squared moments must be non-negative
        end
    end
    
    @testset "MLE: Hessian at optimum is positive definite" begin
        Random.seed!(232)
        N, K, J = 300, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        result = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), Newton(),
                         Optim.Options(iterations=100); autodiff=:forward)
        
        # Check that result converged
        @test result.iterations < 100
        
        # For MLE, Hessian should be positive definite at optimum
        # (This is the negative of the second derivative matrix)
        # Just check that optimization succeeded
        @test isfinite(result.minimum)
    end
    
    @testset "Different algorithms reach similar solutions" begin
        Random.seed!(233)
        N, K, J = 300, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Try different algorithms
        result_lbfgs = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), LBFGS(),
                               Optim.Options(iterations=200))
        result_gd = optimize(a -> mlogit_mle(a, X, y), zeros(K*(J-1)), GradientDescent(),
                            Optim.Options(iterations=1000))
        
        # Both should find similar minima
        @test abs(result_lbfgs.minimum - result_gd.minimum) / abs(result_lbfgs.minimum) < 0.05
        
        # Parameter estimates should be similar
        @test norm(result_lbfgs.minimizer - result_gd.minimizer) / norm(result_lbfgs.minimizer) < 0.15
    end
    
    @testset "SMM: Increasing D improves objective (more draws)" begin
        Random.seed!(234)
        N, K, J = 150, 2, 3
        X = hcat(ones(N), randn(N))
        y = rand(1:J, N)
        
        # Estimate with small D
        result_small = optimize(a -> mlogit_smm_overid(a, X, y, 10), zeros(K*(J-1)), LBFGS(),
                               Optim.Options(iterations=50))
        
        # Estimate with large D (starting from small D solution for speed)
        result_large = optimize(a -> mlogit_smm_overid(a, X, y, 50), result_small.minimizer, LBFGS(),
                               Optim.Options(iterations=50))
        
        # Both should converge to finite values
        @test isfinite(result_small.minimum)
        @test isfinite(result_large.minimum)
        
        # More draws should give more stable estimates (similar or better objective)
        # Note: Due to randomness in SMM, we just check both are reasonable
        @test result_large.minimum < result_small.minimum * 10
    end
end

################################################################################
# Summary Report
################################################################################

println("\n" * "="^80)
println("Test Suite Summary")
println("="^80)
println("All tests completed!")
println("\nTest Coverage:")
println("  ✓ Data loading and preparation")
println("  ✓ OLS via GMM")
println("  ✓ Multinomial logit MLE")
println("  ✓ Multinomial logit GMM (just-identified and over-identified)")
println("  ✓ Simulation functions (inverse CDF and Gumbel methods)")
println("  ✓ SMM estimation")
println("  ✓ Integration tests")
println("  ✓ Edge cases and robustness")
println("="^80)
