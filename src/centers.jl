
"""
    randomcenters(X, k) -> Vector{Float}

    Return a vector of `k` cluster centers chosen from `X` data matrix.
"""
function randomcenters(X::AbstractMatrix{<:Real}, k::Integer;)
    d, n = size(X)
    i = sample(1:size(X, 2), k)
    return X[:, i]
end

function kmppcenters(
    X::AbstractMatrix{<:Real},
    k::Integer;
    metric::PreMetric = SqEuclidean(),
)
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

"""
    isodata_s(X, k) -> Vector{Float}

    Return a vector of `k` cluster centers calculated from `X` data matrix. 
    Starting with one cluster center, this function assigns the data to the clusters,
    then splits the clusters until there are `k` clusters.
"""
function isodata_s(X, k)
    d, n = size(X)
    assignments = Vector{Int}(undef, n)
    centers = randomcenters(X, 1)

    dist = pairwise(Euclidean(), centers, X, dims = 2)
    D = eltype(dist)

    costs = Vector{D}(undef, n)
    counts = Vector{Int}(undef, k)

    while true
        # compute number of clusters
        NROWS = size(centers, 2)

        # assign cluster members
        counts = Vector{Int}(undef, NROWS)
        dist = pairwise(Euclidean(), centers, X, dims = 2)

        update_assignments!(dist, assignments, costs, counts)
        update_centers!(X, assignments, centers, counts)

        # check number of centroids
        if NROWS == k
            return centers
        elseif NROWS > k
            while NROWS > k
                # discard small clusters (fewest assignments), ties broken arbitrarily
                to_delete = findfirst(x -> x ∈ sort(counts)[1:NROWS-k], counts)
                centers = discard([to_delete], centers)
                NROWS = size(centers, 2)
                
                counts = Vector{Int}(undef, NROWS)
                dist = pairwise(Euclidean(), centers, X, dims = 2)
                update_assignments!(dist, assignments, costs, counts)
                update_centers!(X, assignments, centers, counts)
            end
            return centers
        end

        # split clusters
        sd = Array{D}(undef, d, NROWS)
        update_sd!(X, assignments, centers, counts, sd)

        centers = split!(collect(1:NROWS), centers, sd)
    end

    return centers
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
