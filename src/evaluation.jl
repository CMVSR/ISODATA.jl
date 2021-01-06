
function clusterfit(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat}
)
    d, n = size(X)

    assignments = Vector{Int}(undef, n)
    dist = pairwise(Euclidean(), centers, X, dims = 2)
    D = eltype(dist)

    costs = Vector{D}(undef, n)
    counts = Vector{Int}(undef, size(centers, 2))

    update_assignments!(dist, assignments, costs, counts)

    return assignments
end
