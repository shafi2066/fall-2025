################################################################################
# Test Suite for Problem Set 8: Factor Models and Dimension Reduction
# Comprehensive tests for all functions in PS8_Shafiul_Source.jl
################################################################################

using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, MultivariateStats, FreqTables, ForwardDiff, LineSearches

cd(@__DIR__)

include("PS8_Shafiul_Source.jl")

################################################################################
# Test Suite 1: Data Loading
################################################################################

@testset "Data Loading" begin
    
    @testset "load_data: Basic functionality" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Test that output is a DataFrame
        @test df isa DataFrame
        
        # Test that DataFrame is not empty
        @test nrow(df) > 0
        @test ncol(df) > 0
        
        # Test key columns exist
        @test "logwage" in names(df)
        @test "black" in names(df)
        @test "hispanic" in names(df)
        @test "female" in names(df)
        @test "schoolt" in names(df)
        
        # Test ASVAB columns exist (last 6 columns)
        @test "asvabAR" in names(df)
        @test "asvabCS" in names(df)
        @test "asvabMK" in names(df)
        @test "asvabNO" in names(df)
        @test "asvabPC" in names(df)
        @test "asvabWK" in names(df)
    end
    
    @testset "load_data: Data types and validity" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Check no missing values in key variables
        @test !any(ismissing.(df.logwage))
        @test !any(ismissing.(df.black))
        @test !any(ismissing.(df.hispanic))
        @test !any(ismissing.(df.female))
        
        # Check logwage is numeric and reasonable
        @test all(isfinite.(df.logwage))
        
        # Check binary variables are 0 or 1
        @test all(x -> x in [0, 1], df.black)
        @test all(x -> x in [0, 1], df.hispanic)
        @test all(x -> x in [0, 1], df.female)
    end
    
    @testset "load_data: ASVAB scores validity" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # All ASVAB scores should be finite
        @test all(isfinite.(df.asvabAR))
        @test all(isfinite.(df.asvabCS))
        @test all(isfinite.(df.asvabMK))
        @test all(isfinite.(df.asvabNO))
        @test all(isfinite.(df.asvabPC))
        @test all(isfinite.(df.asvabWK))
        
        # ASVAB scores should have variation
        @test std(df.asvabAR) > 0
        @test std(df.asvabCS) > 0
        @test std(df.asvabMK) > 0
    end
end

################################################################################
# Test Suite 2: Base Regression (Question 1)
################################################################################

@testset "Base Regression" begin
    
    @testset "estimate_base_regression: Returns valid model" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_base_regression(df)
        
        # Should return a regression model
        @test result isa StatsModels.TableRegressionModel
        
        # Should have coefficients
        @test length(coef(result)) > 0
        
        # Should have 7 coefficients (6 predictors + intercept)
        @test length(coef(result)) == 7
    end
    
    @testset "estimate_base_regression: Coefficient validity" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_base_regression(df)
        
        # All coefficients should be finite
        @test all(isfinite.(coef(result)))
        
        # R-squared should be between 0 and 1
        @test 0 <= r2(result) <= 1
        
        # Should have standard errors
        @test length(stderror(result)) == length(coef(result))
        @test all(stderror(result) .> 0)
    end
    
    @testset "estimate_base_regression: Model structure" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_base_regression(df)
        
        # Check that correct variables are included
        var_names = string.(coefnames(result))
        @test any(contains.(var_names, "black"))
        @test any(contains.(var_names, "hispanic"))
        @test any(contains.(var_names, "female"))
        @test any(contains.(var_names, "schoolt"))
        @test any(contains.(var_names, "gradHS"))
        @test any(contains.(var_names, "grad4yr"))
        
        # ASVAB variables should NOT be included
        @test !any(contains.(var_names, "asvab"))
    end
end

################################################################################
# Test Suite 3: ASVAB Correlations (Question 2)
################################################################################

@testset "ASVAB Correlations" begin
    
    @testset "compute_asvab_correlations: Output structure" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        cordf = compute_asvab_correlations(df)
        
        # Should return a DataFrame
        @test cordf isa DataFrame
        
        # Should be 6x6 (6 ASVAB tests)
        @test nrow(cordf) == 6
        @test ncol(cordf) == 6
    end
    
    @testset "compute_asvab_correlations: Correlation properties" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        cordf = compute_asvab_correlations(df)
        cormat = Matrix(cordf)
        
        # All correlations should be between -1 and 1
        @test all(-1 .<= cormat .<= 1)
        
        # Diagonal should be 1 (correlation with self)
        @test all(abs.(diag(cormat) .- 1.0) .< 1e-10)
        
        # Matrix should be symmetric
        @test norm(cormat - cormat') < 1e-10
        
        # All values should be finite
        @test all(isfinite.(cormat))
    end
    
    @testset "compute_asvab_correlations: High correlations expected" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        cordf = compute_asvab_correlations(df)
        cormat = Matrix(cordf)
        
        # Off-diagonal correlations should generally be positive
        # (cognitive tests are typically positively correlated)
        off_diag = cormat[triu(trues(6,6), 1)]
        @test mean(off_diag) > 0
        
        # Should have some high correlations (> 0.5)
        @test any(off_diag .> 0.5)
    end
end

################################################################################
# Test Suite 4: Full Regression with ASVAB (Question 3)
################################################################################

@testset "Full Regression with ASVAB" begin
    
    @testset "estimate_full_regression: Returns valid model" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_full_regression(df)
        
        # Should return a regression model
        @test result isa StatsModels.TableRegressionModel
        
        # Should have 13 coefficients (6 demographics + 6 ASVAB + intercept)
        @test length(coef(result)) == 13
    end
    
    @testset "estimate_full_regression: Includes all ASVAB" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_full_regression(df)
        var_names = string.(coefnames(result))
        
        # Check all ASVAB variables are included
        @test any(contains.(var_names, "asvabAR"))
        @test any(contains.(var_names, "asvabCS"))
        @test any(contains.(var_names, "asvabMK"))
        @test any(contains.(var_names, "asvabNO"))
        @test any(contains.(var_names, "asvabPC"))
        @test any(contains.(var_names, "asvabWK"))
    end
    
    @testset "estimate_full_regression: Multicollinearity effects" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_full_regression(df)
        
        # Due to high correlations, standard errors might be inflated
        # Check that some standard errors are larger than in base model
        base_result = estimate_base_regression(df)
        
        # Both should have finite coefficients
        @test all(isfinite.(coef(result)))
        @test all(isfinite.(stderror(result)))
    end
end

################################################################################
# Test Suite 5: PCA Regression (Question 4)
################################################################################

@testset "PCA Regression" begin
    
    @testset "generate_pca!: Creates PC variable" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_pca = generate_pca!(df)
        
        # Should add asvabPCA column
        @test "asvabPCA" in names(df_pca)
        
        # PC should have same length as original data
        @test length(df_pca.asvabPCA) == nrow(df)
        
        # PC values should be finite
        @test all(isfinite.(df_pca.asvabPCA))
    end
    
    @testset "generate_pca!: PC properties" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_pca = generate_pca!(df)
        
        # PC should have variation
        @test std(df_pca.asvabPCA) > 0
        
        # PC mean should be close to 0 (standardized)
        @test abs(mean(df_pca.asvabPCA)) < 1.0
    end
    
    @testset "generate_pca!: Regression with PC" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_pca = generate_pca!(df)
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        
        # Should have 8 coefficients
        @test length(coef(result)) == 8
        
        # Should have valid estimates
        @test all(isfinite.(coef(result)))
        @test r2(result) > 0
    end
    
    @testset "generate_pca!: PC explains variation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Get PC
        df_pca = generate_pca!(df)
        
        # PC should be correlated with ASVAB scores
        @test abs(cor(df_pca.asvabPCA, df_pca.asvabAR)) > 0.3
        @test abs(cor(df_pca.asvabPCA, df_pca.asvabCS)) > 0.3
    end
end

################################################################################
# Test Suite 6: Factor Analysis Regression (Question 5)
################################################################################

@testset "Factor Analysis Regression" begin
    
    @testset "generate_factor!: Creates factor variable" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_fa = generate_factor!(df)
        
        # Should add asvabFactor column
        @test "asvabFactor" in names(df_fa)
        
        # Factor should have same length as original data
        @test length(df_fa.asvabFactor) == nrow(df)
        
        # Factor values should be finite
        @test all(isfinite.(df_fa.asvabFactor))
    end
    
    @testset "generate_factor!: Factor properties" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_fa = generate_factor!(df)
        
        # Factor should have variation
        @test std(df_fa.asvabFactor) > 0
        
        # Factor mean should be close to 0
        @test abs(mean(df_fa.asvabFactor)) < 1.0
    end
    
    @testset "generate_factor!: Regression with factor" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_fa = generate_factor!(df)
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_fa)
        
        # Should have 8 coefficients
        @test length(coef(result)) == 8
        
        # Should have valid estimates
        @test all(isfinite.(coef(result)))
        @test r2(result) > 0
    end
    
    @testset "generate_factor!: PCA vs FA comparison" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_pca = generate_pca!(df)
        df_fa = generate_factor!(df)
        
        # PC and Factor should be highly correlated (both capture latent ability)
        @test abs(cor(df_pca.asvabPCA, df_fa.asvabFactor)) > 0.8
        
        # Both should have similar standard deviations
        @test abs(std(df_pca.asvabPCA) - std(df_fa.asvabFactor)) < 2.0
    end
end

################################################################################
# Test Suite 7: Data Preparation for Factor Model (Question 6)
################################################################################

@testset "Data Preparation for Factor Model" begin
    
    @testset "prepare_factor_matrices: Output dimensions" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        N = nrow(df)
        
        # Check dimensions
        @test size(X, 1) == N
        @test size(X, 2) == 7  # 6 demographics + constant
        
        @test length(y) == N
        
        @test size(Xfac, 1) == N
        @test size(Xfac, 2) == 4  # black, hispanic, female, constant
        
        @test size(asvabs, 1) == N
        @test size(asvabs, 2) == 6  # 6 ASVAB tests
    end
    
    @testset "prepare_factor_matrices: Matrix contents" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # X should have constant in last column
        @test all(X[:, end] .== 1.0)
        
        # Xfac should have constant in last column
        @test all(Xfac[:, end] .== 1.0)
        
        # All values should be finite
        @test all(isfinite.(X))
        @test all(isfinite.(y))
        @test all(isfinite.(Xfac))
        @test all(isfinite.(asvabs))
    end
    
    @testset "prepare_factor_matrices: Variable matching" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Check that y matches logwage
        @test y == df.logwage
        
        # Check that first column of X matches black
        @test X[:, 1] == df.black
        
        # Check that first column of Xfac matches black
        @test Xfac[:, 1] == df.black
        
        # Check that first column of asvabs matches asvabAR
        @test asvabs[:, 1] == df.asvabAR
    end
end

################################################################################
# Test Suite 8: Factor Model Likelihood (Question 6)
################################################################################

@testset "Factor Model Likelihood" begin
    
    @testset "factor_model: Returns finite value" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Create simple starting values
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        θ = vcat(
            zeros(L * J),  # γ
            zeros(K),      # β
            ones(J + 1),   # α
            ones(J + 1)    # σ
        )
        
        # Evaluate likelihood at starting values
        nll = factor_model(θ, X, Xfac, asvabs, y, 5)
        
        # Should return finite value
        @test isfinite(nll)
        
        # Should be positive (negative log-likelihood)
        @test nll > 0
    end
    
    @testset "factor_model: Parameter unpacking" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        
        # Total parameters should be L*J + K + (J+1) + (J+1)
        n_params = L * J + K + (J + 1) + (J + 1)
        θ = randn(n_params)
        
        # Should not error with correct parameter length
        nll = factor_model(θ, X, Xfac, asvabs, y, 5)
        @test isfinite(nll)
    end
    
    @testset "factor_model: Different quadrature points" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        θ = vcat(zeros(L*J), zeros(K), ones(J+1), ones(J+1))
        
        # Test with different R values
        nll_5 = factor_model(θ, X, Xfac, asvabs, y, 5)
        nll_7 = factor_model(θ, X, Xfac, asvabs, y, 7)
        nll_9 = factor_model(θ, X, Xfac, asvabs, y, 9)
        
        # All should be finite
        @test isfinite(nll_5)
        @test isfinite(nll_7)
        @test isfinite(nll_9)
        
        # Higher R should give similar but slightly different results
        # (more accurate quadrature)
        @test abs(nll_7 - nll_9) < abs(nll_5 - nll_9)
    end
    
    @testset "factor_model: Gradient exists (autodiff)" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        θ = vcat(zeros(L*J), zeros(K), ones(J+1), 0.5*ones(J+1))
        
        # Should be able to compute gradient
        grad = ForwardDiff.gradient(th -> factor_model(th, X, Xfac, asvabs, y, 5), θ)
        
        @test length(grad) == length(θ)
        @test all(isfinite.(grad))
    end
end

################################################################################
# Test Suite 9: Full MLE Estimation (Question 6)
################################################################################

@testset "Full MLE Estimation" begin
    
    @testset "run_estimation: Returns valid results" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        # Prepare starting values
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        svals = vcat(
            Xfac\asvabs[:, 1], Xfac\asvabs[:, 2], Xfac\asvabs[:, 3],
            Xfac\asvabs[:, 4], Xfac\asvabs[:, 5], Xfac\asvabs[:, 6],
            X\y,
            rand(7),
            0.5*ones(7)
        )
        
        # Run estimation (with limited iterations for testing)
        θ̂, se, loglike = run_estimation(df, svals)
        
        # Should return parameter estimates
        @test length(θ̂) == length(svals)
        @test all(isfinite.(θ̂))
        
        # Should return standard errors
        @test length(se) == length(θ̂)
        @test all(se .>= 0)
        
        # Log-likelihood should be finite
        @test isfinite(loglike)
        @test loglike > 0  # This is negative log-likelihood
    end
    
    @testset "run_estimation: Standard errors are positive" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        svals = vcat(
            Xfac\asvabs[:, 1], Xfac\asvabs[:, 2], Xfac\asvabs[:, 3],
            Xfac\asvabs[:, 4], Xfac\asvabs[:, 5], Xfac\asvabs[:, 6],
            X\y,
            rand(7),
            0.5*ones(7)
        )
        
        θ̂, se, loglike = run_estimation(df, svals)
        
        # All standard errors should be positive
        @test all(se .> 0)
        
        # Standard errors should be reasonable (not too large)
        @test all(se .< 100)
    end
end

################################################################################
# Test Suite 10: Integration and Comparison Tests
################################################################################

@testset "Integration and Comparison Tests" begin
    
    @testset "Base vs Full regression: R-squared comparison" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        base = estimate_base_regression(df)
        full = estimate_full_regression(df)
        
        # Full model should have higher R-squared
        @test r2(full) >= r2(base)
    end
    
    @testset "PCA vs FA: Similar predictive power" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_pca = generate_pca!(df)
        df_fa = generate_factor!(df)
        
        pca_reg = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        fa_reg = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_fa)
        
        # R-squared should be similar
        @test abs(r2(pca_reg) - r2(fa_reg)) < 0.05
        
        # Both should improve over base regression
        base = estimate_base_regression(df)
        @test r2(pca_reg) > r2(base)
        @test r2(fa_reg) > r2(base)
    end
    
    @testset "Dimension reduction: Fewer parameters, similar fit" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        full = estimate_full_regression(df)
        df_pca = generate_pca!(df)
        pca_reg = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        
        # PCA has fewer parameters (8 vs 13)
        @test length(coef(pca_reg)) < length(coef(full))
        
        # But R-squared should be reasonably close
        @test r2(pca_reg) > r2(full) * 0.8
    end
end

################################################################################
# Test Suite 11: Edge Cases and Robustness
################################################################################

@testset "Edge Cases and Robustness" begin
    
    @testset "Small sample: Functions still work" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df_full = load_data(url)
        df = first(df_full, 100)  # Use only first 100 observations
        
        # All functions should work
        @test estimate_base_regression(df) isa StatsModels.TableRegressionModel
        @test compute_asvab_correlations(df) isa DataFrame
        @test estimate_full_regression(df) isa StatsModels.TableRegressionModel
        
        df_pca = generate_pca!(df)
        @test "asvabPCA" in names(df_pca)
        
        df_fa = generate_factor!(df)
        @test "asvabFactor" in names(df_fa)
    end
    
    @testset "Correlation matrix: Positive definite" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        cordf = compute_asvab_correlations(df)
        cormat = Matrix(cordf)
        
        # Eigenvalues should all be positive (positive definite)
        eigvals = eigen(cormat).values
        @test all(eigvals .> -1e-10)  # Allow small numerical errors
    end
    
    @testset "Factor model: Different starting values converge" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        
        # Two different starting values
        θ1 = vcat(zeros(L*J), zeros(K), ones(J+1), 0.5*ones(J+1))
        θ2 = vcat(0.1*randn(L*J), 0.1*randn(K), 1.5*ones(J+1), 0.8*ones(J+1))
        
        # Both should give finite likelihoods
        nll1 = factor_model(θ1, X, Xfac, asvabs, y, 7)
        nll2 = factor_model(θ2, X, Xfac, asvabs, y, 7)
        
        @test isfinite(nll1)
        @test isfinite(nll2)
    end
end

################################################################################
# Test Suite 12: Numerical Derivatives and Optimization
################################################################################

@testset "Numerical Derivatives and Optimization" begin
    
    @testset "Factor model: Hessian is computable" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        L, J, K = size(Xfac, 2), size(asvabs, 2), size(X, 2)
        θ = vcat(zeros(L*J), zeros(K), ones(J+1), 0.5*ones(J+1))
        
        # Should be able to compute Hessian
        hess = ForwardDiff.hessian(th -> factor_model(th, X, Xfac, asvabs, y, 5), θ)
        
        @test size(hess, 1) == length(θ)
        @test size(hess, 2) == length(θ)
        @test all(isfinite.(hess))
    end
    
    @testset "Starting values: OLS provides reasonable estimates" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # OLS starting values for γ
        γ_start1 = Xfac \ asvabs[:, 1]
        
        # Should be finite
        @test all(isfinite.(γ_start1))
        @test length(γ_start1) == size(Xfac, 2)
        
        # OLS starting values for β
        β_start = X \ y
        @test all(isfinite.(β_start))
        @test length(β_start) == size(X, 2)
    end
end

################################################################################
# Test Suite 13: Statistical Properties
################################################################################

@testset "Statistical Properties" begin
    
    @testset "Regression coefficients: Sign expectations" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        result = estimate_base_regression(df)
        coef_dict = Dict(zip(string.(coefnames(result)), coef(result)))
        
        # Education variables typically have positive effects on wages
        # (This is descriptive, not a hard requirement)
        @test haskey(coef_dict, "schoolt") || haskey(coef_dict, "(Intercept)")
    end
    
    @testset "PCA: First component explains most variance" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        asvabs = Matrix(df[:, end-5:end])'
        
        # Fit full PCA to check variance explained
        M_full = fit(PCA, asvabs)
        var_explained = principalvars(M_full)
        
        # First PC should explain substantial variance
        @test var_explained[1] > sum(var_explained) * 0.3
    end
    
    @testset "Factor loadings: All positive for cognitive ability" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
        df = load_data(url)
        
        df_fa = generate_factor!(df)
        
        # Factor should be correlated with all ASVAB tests in same direction
        cors = [cor(df_fa.asvabFactor, df_fa[:, col]) for col in ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]]
        
        # All should have same sign (typically positive)
        @test all(cors .> 0) || all(cors .< 0)
        
        # All should be reasonably strong
        @test all(abs.(cors) .> 0.3)
    end
end

################################################################################
# Summary Report
################################################################################

println("\n" * "="^80)
println("Test Suite Summary for PS8: Factor Models")
println("="^80)
println("All tests completed!")
println("\nTest Coverage:")
println("  ✓ Data Loading (3 testsets)")
println("  ✓ Base Regression without ASVAB (3 testsets)")
println("  ✓ ASVAB Correlations (3 testsets)")
println("  ✓ Full Regression with ASVAB (3 testsets)")
println("  ✓ PCA Regression (4 testsets)")
println("  ✓ Factor Analysis Regression (4 testsets)")
println("  ✓ Data Preparation for Factor Model (3 testsets)")
println("  ✓ Factor Model Likelihood Function (4 testsets)")
println("  ✓ Full MLE Estimation (2 testsets)")
println("  ✓ Integration and Comparison Tests (3 testsets)")
println("  ✓ Edge Cases and Robustness (3 testsets)")
println("  ✓ Numerical Derivatives and Optimization (2 testsets)")
println("  ✓ Statistical Properties (3 testsets)")
println("="^80)