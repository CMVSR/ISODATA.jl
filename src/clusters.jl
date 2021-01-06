
"""
    discard(to_delete, centers, [...]) -> centers::AbstractMatrix{<:AbstractFloat}

    Discard all cluster centers in `centers` based on indices in `to_delete`.

    # Arguments
    - `to_delete`: vector of indices to delete
    -   `centers`: array of cluster centers
"""
function discard(to_delete::Vector{Int}, centers::AbstractMatrix{<:AbstractFloat})
    unique!(to_delete)
    sort!(to_delete)
    l = size(to_delete, 1)
    k = size(centers, 2)

    if l > 0 && k > l
        temp = centers[:, setdiff(1:end, to_delete)]
        return temp
    end

    return centers
end

"""
    discard_clusters!(X, centers, counts, θn) -> None

    Discard all cluster centers that meet the criteria and update `centers`
    in-place. 

    The criteria for deleting clusters are as follows:
        1. If the number of member patterns in a cluster is less than (θn*n)
    Discard `if(1)`

    # Arguments
    -          `X`: input data matrix
    -    `centers`: array of cluster centers
    -     `counts`: array of the number of patterns assigned to each cluster
    -         `θn`: minimum number of cluster members (as percentage)
"""
function discard_clusters!(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    counts::Vector{Int},
    θn::Float64,
)
    d, n = size(X)
    k = size(centers, 2)

    to_delete = Vector{Int}()
    for j = 1:k
        if (counts[j] < (θn * n))
            push!(to_delete, j)
        end
    end

    # if clusters are to be deleted, set flag
    flag = !isempty(to_delete)

    if flag
        centers = discard(to_delete, centers)
        return centers, flag
    else
        return centers, flag
    end
end

"""
    split!(to_split, centers, sd) -> None

    Split all cluster centers in `centers` based on indices in `to_split`
    and update `centers` in-place.

    To split a cluster, find the dimension which contributes the most to its
        variance (highest standard deviation) and create two copies of the
        cluster. In one of the copies, add 1 to that dimension, and in the 
        other, subtract 1 from that dimension.

    # Arguments
    - `to_split`: vector of indices to split
    -  `centers`: array of cluster centers
    -       `sd`: array of the standard deviation of each cluster 
                    in each dimension
"""
function split!(
    to_split::Vector{Int},
    centers::AbstractMatrix{<:AbstractFloat},
    sd::AbstractMatrix{<:AbstractFloat},
)
    unique!(to_split)
    l = size(to_split, 1)
    to_delete = Vector{Int}()

    if l > 0
        for j = 1:l
            k = to_split[j]

            # create a copy
            c = view(centers, :, k)

            # find the dimension which contributes most to the variance
            p, i = findmax(view(sd, :, k))

            # create a plus copy, which has plus 1 in the p dimension 
            c_plus = copy(c)
            c_plus[i] += 1

            # create a minus copy, which has minus 1 in the p dimension 
            c_minus = copy(c)
            c_minus[i] -= 1

            push!(to_delete, k)
            centers = cat(centers, c_plus, dims = 2)
            centers = cat(centers, c_minus, dims = 2)
        end
    end

    centers = discard(to_delete, centers)
    return centers
end

"""
    split_clusters!(
        X, centers, counts, sd, maxclusters, AVEDIST, AD, θn, θe) -> None

    Split all cluster centers that meet the criteria and update `centers`
        in-place. 

    The criteria for splitting clusters are as follows:
        1. If the standard deviation of a cluster in any of its dimensions 
            exceeds θe.
        2. If the average distance of a cluster's center to its members 
            exceeds the overall average.
        3. If the number of member patterns in a cluster exceeds 2*(θn*n)+2), 
            where n is the total number of patterns in the dataset.
        4. If the number of clusters is less than half of the clusters expected 
            (based on maxclusters)
    Split `if(1 AND ((2 AND 3) OR 4)`

    # Arguments
    -           `X`: input data matrix
    -     `centers`: array of cluster centers
    -      `counts`: array of the number of patterns assigned to each cluster
    -          `sd`: vector of the standard deviation of each cluster 
                        in each dimension
    - `maxclusters`: user defined parameter for the maximum number of clusters to create
    -     `AVEDIST`: vector with the average distance of a cluster center to each of its members
    -          `AD`: the overall average distance of cluster centers to their members
    -          `θn`: minimum number of cluster members (as percentage)
    -          `θe`: maximum standard deviation allowed for a cluster
"""
function split_clusters!(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    counts::Vector{Int},
    sd::AbstractMatrix{<:AbstractFloat},
    maxclusters::Integer,
    AVEDIST::Vector{Float64},
    AD::Float64,
    θn::Float64,
    θe::Float64,
)
    d, n = size(X)
    k = size(centers, 2)

    to_split = Vector{Int}()
    for j = 1:k
        for i = 1:d
            if sd[i, j] > θe
                if (
                    (AVEDIST[j] > AD && counts[j] > 2 * (θn * n) + 2) ||
                    k <= 0.5 * maxclusters
                )
                    push!(to_split, j)
                end
            end
        end
    end

    centers = split!(to_split, centers, sd)
    return centers
end

"""
    lump!(to_lump, centers, counts) -> None

    Lump all pairs of cluster centers in `centers` based on indices in 
    `to_lump` and update `centers` in-place.

    To lump two clusters, create a new cluster based off the weighted centers
    of the clusters to merge. The weights are equal to the number of member
    patterns for each cluster.

    -  `to_lump`: vector of indices to split
    -  `centers`: array of cluster centers
    -   `counts`: array of the number of patterns assigned to each cluster
"""
function lump!(
    to_lump::Vector{Int},
    centers::AbstractMatrix{<:AbstractFloat},
    counts::Vector{Int},
)
    l = size(to_lump, 1)
    to_delete = Vector{Int}()

    if l > 0
        for j in range(1, l, step = 2)
            # create a copy of each cluster to merge
            i1 = to_lump[j]
            i2 = to_lump[j+1]
            c1 = view(centers, :, i1)
            c2 = view(centers, :, i2)

            # empty cluster based on c1
            p = copy(c1)
            fill!(p, 0)

            d = size(centers, 1)
            for i = 1:d
                p[d] = (
                    1 / (counts[i1] + counts[i2]) * (counts[i1] * c1[i]) +
                    counts[i2] * c2[d]
                )
            end

            push!(to_delete, i1)
            push!(to_delete, i2)
            centers = cat(centers, p, dims = 2)
        end
    end

    centers = discard(to_delete, centers)
    return centers
end

"""
    lump_clusters!(centers, counts, L, θc) -> None

    Lump all cluster centers that meet the criteria and update `centers`
        in-place. 

    The criteria for lumping two clusters are as follows:
        1. If the distance between two clusters is less than θc
        2. If the clusters have not already been assigned to lump
        3. If the number of clusters to lump does not exceed L
    Lump `if(1 AND 2 AND 3)`

    # Arguments
    -     `centers`: array of cluster centers
    -      `counts`: array of the number of patterns assigned to each cluster
    -           `L`: maximum number of clusters to lump
    -          `θc`: maximum standard deviation allowed for a cluster
"""
function lump_clusters!(
    centers::AbstractMatrix{<:AbstractFloat},
    counts::Vector{Int},
    L::Integer,
    θc::Float64,
)
    k = size(centers, 2)
    dist = pairwise(Euclidean(), centers, centers)

    to_lump = Vector{Int}()
    while k > L
        for j = 1:k
            for i = 1:k
                if (i != j && dist[i, j] < θc)
                    if (i ∉ to_lump && j ∉ to_lump)
                        push!(to_lump, i)
                        push!(to_lump, j)
                        k -= 2 # two fewer clusters available to lump
                    end
                end
            end
        end
        break
    end

    centers = lump!(to_lump, centers, counts)
    return centers
end
