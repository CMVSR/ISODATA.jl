
struct ClusteringResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real}
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
end

const _default_θn   = Float64(0.1) 
const _default_θe   = Float64(1)
const _default_θc   = Float64(0.5)
const _default_L    = Integer(2)
const _default_iter = Int(10)

"""
    base(X, k, [...]) -> ClusteringResult

    ISODATA clustering of the ``d×n`` data matrix `X` (each column of `X`
    is a ``d``-dimensional data point) into maximum of `k` clusters.

    # Arguments
    - `θn`:     minimum number of cluster members (as percentage)
                    *determines discarding clusters
    - `θe`:     maximum standard deviation allowed for a cluster,
                    *determines splitting clusters
    - `θc`:     minimum distance between clusters, 
                    *determines lumping clusters
    - `L`:      number of clusters to merge during lumping
    - `iter`:   number of iterations to run
"""
function base(X::AbstractMatrix{<:Real},
              k::Integer;
              θn::Float64=_default_θn,
              θe::Float64=_default_θe,
              θc::Float64=_default_θc,
              L::Integer,
              iter::Integer=_default_iter)
    d, n = size(X)

    # assign initial cluster centers
    centers = randomcenters(X, k)
    assignments = Vector{Int}(undef, n)
    counts = Vector{Int}(undef, k)
    dist = pairwise(Euclidean(), centers, X, dims=2)
    D = eltype(dist)
    costs = Vector{D}(undef, n)
    AVEDIST = Vector{Float64}(undef, k)
    AD = Float64(undef)
    
    for i = 1:iter
        # compute number of clusters
        NROWS = size(centers, 2)

        # assign cluster members
        update_assignments!(dist, assignments, costs, counts)
        update_centers!(X, assignments, centers, counts)
        dist = pairwise(Euclidean(), centers, X, dims=2)
        update_avedist!(AVEDIST, AD, dist, centers, assignments)

        # compute std dev of each component in each cluster
        sd = Array{D}(undef, d, NROWS)
        update_sd(X, assignments, centers, counts, sd)
    
        # discard small clusters
        discard_clusters(X, centers, counts, θn)

        # split or lump
        if mod(i, 2) != 0 || i == iter
            lump_clusters(centers, counts, assignments, L, θc)
        elseif mod(i, 2) == 0
            split_clusters(centers, counts, assignments, sd, k, AD, θn, θe)
    end

    return ClusteringResult(centers, assignments, costs, counts)
end

function update_assignments!(
        dist::Matrix{<:Real},
        assignments::Vector{Int},
        costs::Vector{<:Real},
        counts::Vector{Int})
    k, n = size(dist)

    fill!(counts, 0)
    fill!(costs, 0)

    for j in 1:n
        c, a = findmin(view(dist, :, j))
        assignments[j] = a
        costs[j] += c
        counts[a] += 1
    end
end

function update_centers!(
        X::AbstractMatrix{<:Real},
        assignments::Vector{Int},
        centers::AbstractMatrix{<:AbstractFloat},
        counts::Vector{Int})
    d, n = size(X)
    k = size(centers, 2)

    fill!(centers, 0)

    # for each data point
    for j in 1:n
        cj = assignments[j]
        # for each dimension
        for i in 1:d
            centers[i, cj] += X[i, j]
        end
    end

    # for each cluster
    for j in 1:k
        # for each dimension
        for i in 1:d
            centers[i, j] /= counts[j]
        end
    end
end

function update_avedist!(
        AVEDIST::Vector{Float64},
        AD::Float64,
        costs::Vector{<:Real},
        counts::Vector{Int})
    k = size(counts, 2)

    fill!(AVEDIST, 0)
    AD = 0

    for j in 1:k
        AVEDIST[j] = costs[j]/counts[j]
        AD += costs[j]*counts[j]
    end

    AD /= k
end

function update_sd(
        X::AbstractMatrix{<:Real},
        assignments::Vector{Int},
        centers::AbstractMatrix{<:AbstractFloat},
        counts::Vector{Int},
        sd::AbstractMatrix{<:AbstractFloat})
    d, n = size(X)
    k = size(centers, 2)

    fill!(sd, 0)

    # for each data point
    for j in 1:n
        cj = assignments[j]
        # for each dimension
        for i in 1:d
            sd[i, cj] += (X[i, j]-centers[cj])^2
        end
    end

    # for each cluster
    for j in 1:k
        # for each dimension
        for i in 1:d
            sd[i, j] = sqrt(sd[i, j]/counts[j])
        end
    end
end

