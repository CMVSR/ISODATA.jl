
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

function kmppcenters(
        X::AbstractMatrix{<:Real},
        k::Integer;
        metric::PreMetric = SqEuclidean())
    n = size(X, 2)

    D = eltype(X)
    iseeds = Vector{Int}(undef, k)
    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = colwise(metric, X, view(X, :, p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p

            # update mincosts
            c = view(X, :, p)
            colwise!(tmpcosts, metric, X, view(X, :, p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end

    return X[:, iseeds]
end

function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end