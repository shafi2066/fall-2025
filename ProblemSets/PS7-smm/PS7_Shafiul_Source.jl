
################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    x = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad.== 1]
    y = log.(df.wage)
    return df, x, y
end

"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS via GMM
################################################################################
"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression.
Uses identity weighting matrix.

Mathematical form:
    J(β) = g'Wg where W = I and g = (y - Xβ)

Arguments:
- β: coefficient vector
- X: N×K design matrix
- y: N×1 outcome vector

Returns: scalar objective function value
"""
function ols_gmm(β, X, y)
    ŷ = X * β
    g = y - ŷ
    J = dot(g, g)
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(α, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- α: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # TODO: Create N×J matrix of choice indicators (bigY)
    # bigY[i,j] = 1 if y[i]==j, 0 otherwise
    bigY = zeros(N, J)
     for j = 1:J
        bigY[:,j] = y .== j
    end

    # TODO: Reshape α into K×(J-1) matrix and append zeros for normalization
     bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Compute choice probabilities P
     P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
    
    # Compute negative log-likelihood: -Σ_i Σ_j d_ij * log(P_ij)
    loglike = -sum(bigY .* log.(P))

    return loglike
end

"""
    mlogit_gmm(α, X, y)

GMM objective function for multinomial logit using just-identified moments.

Moment conditions:
    E[X_i(d_ij - P_ij(α))] = 0 for j = 1,...,J-1

This gives K*(J-1) moments for K*(J-1) parameters.
"""
function mlogit_gmm(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
     for j = 1:J
        bigY[:,j] = y .== j
    end

    # Reshape coefficients
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # Compute choice probabilities
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    # Compute moment vector g of dimension K*(J-1)
    # Each element is: mean((d_ij - P_ij) * X_ik)
    g = zeros((J-1)*K)
     for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K + k] = mean((bigY[:,j] - P[:,j]) .* X[:,k])
        end
    end

    # Compute objective function J = N * g'g
    J = N * dot(g, g)

    return J
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    # Reshape and compute probabilities
    bigα = [reshape(α, K, J-1) zeros(K)]
    P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

    # Stack moments as vector: g = vec(d - P)
    # This creates an N*J dimensional vector
    g = bigY[:] .- P[:]

    # Compute objective J = g'Wg with W = I
    J = dot(g, g)

    return J
end

################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################
"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF method.

Steps:
1. Generate X ~ N(μ, Σ)
2. Set coefficient matrix β (K×J)
3. Compute choice probabilities P_ij
4. Draw uniform random variable ε_i
5. Assign y_i based on which cumulative probability bracket ε_i falls in

Arguments:
- N: number of observations (default: 100,000)
- J: number of choice alternatives (default: 4)

Returns: (Y, X) where Y is N×1 choice vector, X is N×K covariate matrix
"""
function sim_logit(N=100_000, J=4)
    #  Generate X matrix
    # Include intercept, and K-1 other covariates with different distributions
    # Example: X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    X = hcat(ones(N), randn(N), rand(N) .> 0.5, 10 .* rand(N))
    #  Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    if J == 4
        β = hcat([1, -1, 0.5, 0.25], [0, 0.5, 0.3, -0.4], [0, -0.5, 2, 1], zeros(4))
    else
        # Generate random coefficients
        β = -2 .+ 4 .* randn(size(X, 2), J)
        #β[:,end] .= 0.0
        β = β .-β[:,end]  # Normalize last column to zero
    end

    # Compute choice probabilities P (dimension N×J)
    # P_ij = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    P = exp.(X * β) ./ sum.(eachrow(exp.(X * β)))
    @assert size(P) == (N, J)

    #  Draw uniform random numbers
     draw = rand(N)

    # Generate choices based on cumulative probabilities
    Y = zeros(N)
     for j = 1:J
         Y += (sum(P[:,1:j], dims=2) .>= draw)
     end

    return Y, X
end
"""
    sim_logit_with_gumbel(N, J)

Simulate multinomial logit data using Gumbel shocks (alternative method).

This method directly uses the fact that if ε ~ Gumbel(0,1), then
    y_i = argmax_j (X_i'β_j + ε_ij)
generates choices from a multinomial logit model.

This is often simpler and more numerically stable than the inverse CDF method.

"""
function sim_logit_with_gumbel(N=100_000, J=4)
    # Generate X matrix
    X = hcat(ones(N), rand(N), rand(N) .> 0.5, 10 .* rand(N))

    if J == 4
         β = hcat([1, -1, 0.5, 0.25], [0, 0.5, 0.3, -0.4], [0, -0.5, 2, 1], zeros(4))
    else
        β = rand(-2. +4*rand(size(X,2), J))
       # β[:,end] .= 0
        β=β .-β[:,end]
    end

    #Generate chocies based on Gumbel errors
    #Draw Gumbel Errors 
    ε = rand(Gumbel(0,1), N, J) 
    #Choosing alternatives that maximizes utility
    Y = argmax.(eachrow(X * β .+ ε) ) 

    return Y, X
end

##############################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D)

Simulated Method of Moments for multinomial logit.

Algorithm:
1. For given parameters α, compute model-implied probabilities
2. Simulate D datasets using the Gumbel method
3. Compute average simulated choice frequencies
4. Match simulated frequencies to actual frequencies

Arguments:
- α: parameter vector
- X: covariate matrix
- y: actual choices
- D: number of simulation draws

Returns: SMM objective function value
"""
function mlogit_smm_overid(α, X, y, D)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create actual choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end

    # Initialize simulated choice frequencies
    bigỸ = zeros(N, J)

    # Reshape parameters
    bigα =[reshape(α, K, J-1) zeros(K)]


    # Simulate D datasets and accumulate frequencies
    Random.seed!(1234)  # For reproducibility
    for d = 1:D
        # Draw Gumbel shocks
        ε = rand(Gumbel(0,1), N, J) 

        # Generate simulated choices
        ỹ = argmax.(eachrow(X*bigα .+ ε))
    
    # Update frequency counts
     for j = 1:J
       bigỸ[:,j] .+= (ỹ .== j) * (1/D)
         end
    end 
    #  Compute moment vector (actual - simulated frequencies)
    g = bigY[:] .- bigỸ[:]
    
    # Compute objective function
    J = dot(g,g) 

    return J
end
################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################

"""
    main()

Main function that runs all estimation procedures.
Keeps everything out of global scope for better performance.
"""
function main()
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, X_wage, y_wage = load_data(url)
    df, X, y = prepare_occupation_data(df)
    
    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)

    # Estimate β using GMM
    # Hint: Use optimize() with LBFGS algorithm
     β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage), 
                         rand(size(X_wage,2)),
                         LBFGS(),
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

    # Compare with closed-form OLS: (X'X)^(-1)X'y
     β_ols = X_wage \ y_wage
    
     println("GMM estimates: ", β_hat_gmm.minimizer)
     println("OLS estimates: ", β_ols)

    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)
    
    # Get starting values from series of binary logits
    # Create dummy variables for each occupation
    # Run binary logits for each occupation
     svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
        # Use glm directly instead of lm with @formula macro
        model = glm(X, df[:, tempname], Binomial(), LogitLink())
        svals[:,j] = coef(model)
    end
    
    # Take differences relative to base category
     svals = svals[:,1:6] .- svals[:,7]
     svals = svals[:]
    
    # Estimate via MLE
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y),
                          svals,
                          LBFGS(),
                          Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
        println("MLE estimates: ", α_hat_mle.minimizer)
        return

    starter=[0.191006, 0.0129826, -0.0492051, 0.378569, 0.000490569, 0.258659, -0.00499296, 0.233641, -0.000737866, 0.176803, -0.00262248, 0.111769, -0.00176349]

    # Estimate via GMM using MLE estimates as starting values
     α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                    #α_hat_mle.minimizer,
                                    starter,
                                    LBFGS(),
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
    # Estimate via GMM using random starting values
     α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                    rand(length(svals)),
                                    LBFGS(),
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))

    # Compare estimates and objective function values
    # Discuss: Is the objective function globally concave?
#--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80) 
    # Simulate data
     ySim, XSim = sim_logit(100_000, 4)
    println([mean(ySim .== j; dims=1) for j = 1:4])
    println(mean(XSim, dims=1))

    #  Try alternative simulation method with Gumbel
    ySim, XSim = sim_logit_with_gumbel(100_000, 4)
     println([mean(ySim .== j; dims=1) for j = 1:4])
    println(mean(XSim, dims=1))
  

    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)
    
    # Estimate via SMM
    
    starter=[0.19101716593910428, -0.03352609117521728, 0.5963962451896369, 0.4165044529729242, -0.16983274603146667, -0.03597846574054922, 1.306839908174689, -0.43099894843244585, 0.689479978380685, -0.01045802127010514, 0.5231641223390338, -1.4924775865308129, -2.2674877189456253, -0.005299894385579393, 1.3914027540548604, -0.9849632645162181, -1.3985348715766366, -0.014295247902017449, -0.017651511604991305, -1.4951325302066962, 0.24549582859880864, -0.006726902845960213, -0.5382873380942454, -3.7897907309937615]
    α_hat_smm = optimize(th -> mlogit_smm_overid(th, X, y, 100),  # Small D for testing
                          starter, # Use MLE as starting values
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=1000, show_trace=true))

    #  Compare SMM estimates to MLE and GMM estimates
    println("MLE estimates: ", α_hat_mle.minimizer)
    println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
    println("SMM estimates: ", α_hat_smm.minimizer)
    
    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)
    
    # Return estimates for testing
    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm

end