
function kmeans(
    X::AbstractMatrix{<:Real},
    k::Integer;
    T::Float64 = 0.98,
    iseeds = randomcenters
)
    d, n = size(X)

    # assign initial cluster centers
    centers = iseeds(X, k)
    assignments = Vector{Int}(undef, n)

    dist = pairwise(Euclidean(), centers, X, dims = 2)
    D = eltype(dist)

    costs = Vector{D}(undef, n)
    counts = Vector{Int}(undef, k)
    sd = Array{D}(undef, d, k)
    iter = 0

    while true
        iter += 1
        if iter > 1000
            return ClusteringResult(centers, assignments, costs, counts, sd, iter)
        end

        dist = pairwise(Euclidean(), centers, X, dims = 2)
        prev_assignments = deepcopy(assignments)
        update_assignments!(dist, assignments, costs, counts)
        update_centers!(X, assignments, centers, counts)

        # check convergence threshold
        updates = assignments .== prev_assignments
        if (sum(updates)/length(updates)) > T
            return ClusteringResult(centers, assignments, costs, counts, sd, iter)
        end

        # compute std dev of each component in each cluster
        sd = Array{D}(undef, d, k)
        update_sd!(X, assignments, centers, counts, sd)
    end
end