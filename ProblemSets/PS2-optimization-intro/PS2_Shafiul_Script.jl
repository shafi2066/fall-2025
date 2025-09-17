using Optim, DataFrames, CSV, HTTP, GLM, FreqTables
cd(@__DIR__)

include("PS2_Shafiul_Source.jl")

function run_all_questions()
    # Question 1
    f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
    minusf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
    startval = rand(1)
    result = optimize(minusf, startval, BFGS())
    println("argmin (minimizer) is ", Optim.minimizer(result)[1])
    println("min is ", Optim.minimum(result))

    # Question 2
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(),
        Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    println("OLS closed-form: ", bols)
    df.white = df.race .== 1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println("OLS via GLM: ", coef(bols_lm))

    # Question 3
    alpha_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(),
        Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println("Logit via Optim: ", alpha_hat_logit.minimizer)

    # Question 4
    df.white = df.race .== 1
    logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println("Logit coefficients via GLM: ", coef(logit_glm))

    # Question 5
    freqtable(df, :occupation)
    df = dropmissing(df, :occupation)
    for val in 8:13
        df[df.occupation .== val, :occupation] .= 7
    end
    freqtable(df, :occupation)

    X2 = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]
    y2 = df.occupation

    K = size(X2, 2)
    J = 7

    init_params = zeros(K * (J - 1))
    result = optimize(a -> mlogit(a, X2, y2), init_params, LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println("Estimated coefficients:")
    println(reshape(result.minimizer, K, J - 1))
end

run_all_questions()
