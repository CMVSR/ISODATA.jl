
const _default_θn_bias = Float64(0.35)
const _default_θe_bias = Float64(0.35)
const _default_θc_bias = Float64(0.35)

struct Summary
    label::Float64
    means::Vector{Float64}
    sd::Vector{Float64}
    members::Int
end

function summarize(data)
    # assumes dimensions by samples format
    d, n = size(data)

    # assumes last row is class label
    k = size(unique(data[end, :]), 1)

    res = Vector{Summary}(undef, k)

    for i in 1:k
        inds = findall(x -> x == Float64(i), data[end, :])
        temp = data[:, inds]

        sd = Vector{Float64}(undef, d-1)
        means = Vector{Float64}(undef, d-1)

        for j in 1:d-1
            means[j] = sum(temp[j, :])/size(temp, 2)
            sd[j] = sqrt(sum((temp[j, :] .- means[j]).^2)/n)
        end

        res[i] = Summary(Float64(i), means, sd, size(temp, 2))
    end

    return res
end

"""
    iso_param(data, summaries) -> Float64, Float64, Float64

    Given data and the summarized data (based on class label),
    attempt to generate ISODATA parameters based on a bias 
    percentage.
"""
function iso_param(
	data,
	summaries::Array{Summary};
	_θn_bias::Float64=_default_θn_bias,
	_θe_bias::Float64=_default_θe_bias,
	_θc_bias::Float64=_default_θc_bias
)
	d, n = size(data)
	m = size(summaries, 1)
	
	θn, θe, θc = 0.0, 0.0, 0.0
	
	# find θn, minimum number of cluster members
	min_members = 0 
	total_members = 0
	for i = 1:m
		if min_members < summaries[i].members
			min_members = summaries[i].members
		end
		total_members += summaries[i].members
	end
	
	# θn is _θn_bias% lower than min_members/total_members
	θn = (min_members/total_members)*(1-_θn_bias)
	
	# find θe, maximum standard deviation allowed for a cluster
	max_sd = 0.0
	for i = 1:m
		for j = 1:d
			if summaries[i].sd[j] > max_sd
				max_sd = summaries[i].sd[j]
			end
		end
	end
	
	# θe is _θe_bias% higher than max_sd
	θe = max_sd*(1+_θe_bias)
	
	# find θc, minimum distance between clusters
	means = Array{Float64}(undef, d, m)
	for i = 1:m
		means[:, i] = summaries[i].means
	end
	
	dists = pairwise(Euclidean(), means, means)
	min_dist = maximum(dists)

	for i = 1:m
		for j = 1:m
			if (dists[i, j] < min_dist && i != j)
				min_dist = dists[i, j]
			end
		end
	end
	
	# θc is _θc_bias% lower than min_dist
	θc = min_dist*(1-_θc_bias)
	
	return θn, θe, θc
end