"""
    Discard clusters that meet the criteria
"""
function discard_clusters(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        counts::Vector{Int},
        θn::Float64)
    d, n = size(X)
    k = size(centers, 2)

    to_delete = Vector{Integer}()
    for j in 1:k
        if counts[j]/d < (θn*n)
            for i in 1:d
                push!(to_delete, j)
            end
        end
    end

    if size(to_delete, 1) > 0
        sort!(to_delete)
        deleteat!(items, to_delete)
    end
end

function split_clusters(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        counts::Vector{Int},
        assignments::Vector{Int},
        sd::AbstractMatrix{<:AbstractFloat},
        maxclusters:Integer,
        AVEDIST::Vector{Float64},
        AD::Float64,
        θn::Float64,
        θe::Float64)
    d, n = size(X)
    k = size(centers, 2)

    to_split = Vector{Integer}()
    for j in 1:k
        for i in 1:d
            if sd[i, j] > θe
                if ((AVEDIST[j] > AD && counts[j]>2*(θn*n)+2)
                    || k <= 0.5*maxclusters)
                c, a = findmin(view(dist, :, j))
                push!(to_split, j)
                end
            end
        end
    end

    l = size(to_split)
    if l > 0
        for j in 1:l
            # create a copy
            c = view(centers, :, j)

            # find the dimension which contributes most to the variance
            p, i = findmax(view(sd, :, j))

            # create a plus copy, which has plus 1 in the p dimension 
            c_plus = view(c)
            c_plus[i] += 1

            # create a minus copy, which has minus 1 in the p dimension 
            c_minus = view(c)
            c_minus[i] -= 1

            deleteat!(centers, j)
            push!(centers, c_plus)
            push!(centers, c_minus)
        end
    end
end

function lump_clusters(centers::AbstractMatrix{<:AbstractFloat},
                       counts::Vector{Int},
                       assignments::Vector{Int},
                       L::Integer,
                       θc::Float64)
    k = size(centers, 2)

    to_lump = Vector{Integer}()
    for j in 1:k
    end
end