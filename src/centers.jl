
"""
    randomcenters(X, k) -> Vector{Float}

    Return a vector of `k` cluster centers chosen from `X` data matrix.
"""
function randomcenters(X::AbstractMatrix{<:Real},
                       k::Integer;)
    d, n = size(X)
    i = sample(1:size(X, 2), k)
    return X[:, i]
end
