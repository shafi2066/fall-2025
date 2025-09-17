function ols(beta, X, y)
    ssr = (y .- X*beta)' * (y .- X*beta)
    return ssr
end

function logit(alpha, X, d)
    η = X * alpha
    p = 1 ./(1 .+ exp.(-η))
    loglike = -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
    return loglike
end

function mlogit(alpha, X, d)
    K = size(X, 2)
    J = length(unique(d))
    beta = reshape(alpha, K, J - 1)
    n = size(X, 1)
    utilities = X * beta
    utilities_full = hcat(utilities, zeros(n))
    denom = sum(exp.(utilities_full), dims=2)
    p_mat = exp.(utilities_full) ./ denom
    ind = zeros(n, J)
    for i in 1:n
        ind[i, d[i]] = 1.0
    end
    loglike = -sum(ind .* log.(p_mat))
    return loglike
end
