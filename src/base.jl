
struct ClusteringResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real}
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    sd::C                      # standard deviation of each cluster (k) in each dimension (d)
end

const _default_θn = Float64(0.15)
const _default_θe = Float64(1)
const _default_θc = Float64(0.5)
const _default_L = Integer(2)
const _default_iter = Int(10)

"""
    base(X, k, [...]) -> ClusteringResult

    ISODATA clustering of the ``d×n`` data matrix `X` (each column of `X`
    is a ``d``-dimensional data point) into maximum of `k` clusters.

    # Arguments
    - `θn`:     minimum number of cluster members (as percentage)
                    *determines discarding clusters
    - `θe`:     maximum standard deviation allowed for a cluster
                    *determines splitting clusters
    - `θc`:     minimum distance between clusters, 
                    *determines lumping clusters
    - `L`:      number of clusters to merge during lumping
    - `iter`:   number of iterations to run
"""
function base(
    X::AbstractMatrix{<:Real},
    k::Integer;
    θn::Float64 = _default_θn,
    θe::Float64 = _default_θe,
    θc::Float64 = _default_θc,
    L::Integer = 2,
    iter::Integer = _default_iter,
)
    d, n = size(X)

    # assign initial cluster centers
    # centers = kmppcenters(X, k)
    centers = randomcenters(X, k)
    assignments = Vector{Int}(undef, n)

    dist = pairwise(Euclidean(), centers, X, dims = 2)
    D = eltype(dist)

    counts = Vector{Int}(undef, k)
    costs = Vector{D}(undef, n)
    sd = Array{D}(undef, d, k)

    for i = 1:iter
        # compute number of clusters
        NROWS = size(centers, 2)
        #println(NROWS)

        # assign cluster members
        counts = Vector{Int}(undef, NROWS)
        dist = pairwise(Euclidean(), centers, X, dims = 2)
        update_assignments!(dist, assignments, costs, counts)
        update_centers!(X, assignments, centers, counts)

        dist = pairwise(Euclidean(), centers, X, dims = 2)
        AVEDIST = Vector{Float64}(undef, NROWS)
        AD = Float64(0)
        AD = update_avedist!(AVEDIST, AD, assignments, costs, counts)

        # compute std dev of each component in each cluster
        sd = Array{D}(undef, d, NROWS)
        update_sd!(X, assignments, centers, counts, sd)

        # discard small clusters
        centers = discard_clusters!(X, centers, counts, θn)

        # split or lump
        if size(centers, 2) == NROWS
            if mod(i, 2) != 0 || i == iter
                centers = lump_clusters!(centers, counts, L, θc)
            elseif mod(i, 2) == 0
                centers = split_clusters!(X, centers, counts, sd, k, AVEDIST, AD, θn, θe)
            end
        end
    end

    return ClusteringResult(centers, assignments, costs, counts, sd)
end

"""
    update_assignments!(dist, assignments, costs, counts) -> None

    Update `assignments` in place using `dist`. Also updates `costs`
    and `counts`.

    # Arguments
    -        `dist`: array of the distance of each point to each cluster
    - `assignments`: vector of length `n` with the assigned cluster
    -       `costs`: vector with the cost to assign data point `i` to
                        the nearest cluster center
    -      `counts`: array of the number of patterns assigned to each cluster
"""
function update_assignments!(
    dist::Matrix{<:Real},       # in
    assignments::Vector{Int},   # out
    costs::Vector{<:Real},      # out
    counts::Vector{Int},
)        # out
    k, n = size(dist)

    fill!(counts, 0)
    fill!(costs, 0)

    for j = 1:n
        c, a = findmin(view(dist, :, j))
        assignments[j] = a
        costs[j] += c
        counts[a] += 1
    end
end

"""
    update_centers!(X, assignments, centers, counts) -> None

    Calculate new means and update `centers` in place using `assignments`.

    # Arguments
    -           `X`: array of the distance of each point to each cluster
    - `assignments`: vector of length `n` with the assigned cluster
    -     `centers`: array of cluster centers
    -      `counts`: array of the number of patterns assigned to each cluster
"""
function update_centers!(
    X::AbstractMatrix{<:Real},                  # in
    assignments::Vector{Int},                   # in
    centers::AbstractMatrix{<:AbstractFloat},   # out
    counts::Vector{Int},
)                        # in
    d, n = size(X)
    k = size(centers, 2)

    fill!(centers, 0)

    # for each data point
    for j = 1:n
        cj = assignments[j]
        # for each dimension
        for i = 1:d
            centers[i, cj] += X[i, j]
        end
    end

    # for each cluster
    for j = 1:k
        # for each dimension
        for i = 1:d
            centers[i, j] /= counts[j]
        end
    end
end

"""
    update_avedist!(AVEDIST, AD, assignments, costs, counts) -> Float64

    Calculate the average distance (AVEDIST) of each cluster to each of it's member
    patterns and the overall average distance (AD) of clusters to their members.

    # Arguments
    -     `AVEDIST`: vector with the average distance of a cluster center to each of its members
    -          `AD`: the overall average distance of cluster centers to their members
    - `assignments`: vector of length `n` with the assigned cluster
    -       `costs`: vector with the cost to assign data point `i` to
                        the nearest cluster center
    -      `counts`: array of the number of patterns assigned to each cluster
"""
function update_avedist!(
    AVEDIST::Vector{Float64},   # in/out
    AD::Float64,                # out
    assignments::Vector{Int},   # in
    costs::Vector{<:Real},      # in
    counts::Vector{Int},
)        # in
    n = size(costs, 1)
    k = size(counts, 1)

    for j = 1:k
        cost = 0
        for i = 1:n
            if assignments[i] == j
                cost += costs[i]
            end
        end
        AVEDIST[j] = cost / counts[j]
        AD += AVEDIST[j]
    end

    AD /= k
    return AD
end

"""
    update_sd!(X, assignments, centers, counts, cd) -> None

    Calculate new means and update `centers` in place using `assignments`.

    # Arguments
    -           `X`: array of the distance of each point to each cluster
    - `assignments`: vector of length `n` with the assigned cluster
    -     `centers`: array of cluster centers
    -      `counts`: array of the number of patterns assigned to each cluster
    -          `sd`: array of the standard deviation of each cluster 
                        in each dimension
"""
function update_sd!(
    X::AbstractMatrix{<:Real},
    assignments::Vector{Int},
    centers::AbstractMatrix{<:AbstractFloat},
    counts::Vector{Int},
    sd::AbstractMatrix{<:AbstractFloat},
)
    d, n = size(X)
    k = size(centers, 2)

    fill!(sd, 0)

    # for each data point
    for j = 1:n
        cj = assignments[j]
        # for each dimension
        for i = 1:d
            sd[i, cj] += (X[i, j] - centers[cj])^2
        end
    end

    # for each cluster
    for j = 1:k
        # for each dimension
        for i = 1:d
            sd[i, j] = sqrt(sd[i, j] / counts[j])
        end
    end
end
