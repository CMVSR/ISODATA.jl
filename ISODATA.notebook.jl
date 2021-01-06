### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a6abf980-2042-11eb-326f-67792f308488
begin
	using Revise
	
	using BenchmarkTools
	using Clustering
	using CSV
	using Distances
	using DataFrames
	using ISODATA
	using MLDataUtils
	using MLLabelUtils
	using StatsBase
end

# ╔═╡ c4a2ee6c-4f8d-11eb-24f5-e53ed7aa6789
begin
	# load iris dataset
	idf = CSV.read("data/iris.data", DataFrame, header=false)
	iris_raw = convert(Matrix, idf)
	iX, iY = iris_raw[:, 1:end-1], iris_raw[:, end]
	iY = convertlabel(
		MLLabelUtils.LabelEnc.Indices,
		iY,
		labelenc(iY))
	iX = Array{Float64}(transpose(iX))
	iris = cat(iX, transpose(iY), dims=1)
end

# ╔═╡ 7d36be2c-4f70-11eb-3b81-2b78885c3771
begin
	# load glass dataset
	gdf = CSV.read("data/glass.data", DataFrame, header=false)
	glass_raw = convert(Matrix, gdf[:, 2:end])
	gX, gY = glass_raw[:, 1:end-1], glass_raw[:, end]
	gY = convertlabel(
		MLLabelUtils.LabelEnc.Indices,
		gY,
		labelenc(gY))
	gX = Array{Float64}(transpose(gX))
	glass = cat(gX, transpose(gY), dims=1)
end

# ╔═╡ b2ce85e2-4f8e-11eb-3c52-0fce3a112046
begin
	# load wine dataset
	wdf = CSV.read("data/wine.data", DataFrame, header=false)
	wine_raw = convert(Matrix, wdf)
	wX, wY = wine_raw[:, 2:end], wine_raw[:, 1]
	wine = cat(transpose(wX), transpose(wY), dims=1)
end

# ╔═╡ affc89c4-4f64-11eb-29f8-15a7c437fe24
struct Summary
	label::Float64
	means::Vector{Float64}
	sd::Vector{Float64}
	members::Int
end

# ╔═╡ f277c7dc-4f64-11eb-147d-95a2b10bab9f
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

# ╔═╡ f800520a-4f64-11eb-2c36-85794e9f2a12
summaries = summarize(iris)

# ╔═╡ e1a8995c-4f66-11eb-0d59-392ffc2743f9
function iso_param(
	data,
	summaries::Vector{Summary};
	_θn_bias::Float64=0.35,
	_θe_bias::Float64=0.35,
	_θc_bias::Float64=0.35
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

# ╔═╡ 21a507b6-4f67-11eb-26a5-57d9727165ac
iso_param(iris[1:end-1, :], summaries)

# ╔═╡ 3a6fc37c-4f6b-11eb-2947-470ea113fdde
begin
	θn, θe, θc = iso_param(iris[1:end-1, :], summaries)
	base_conv(iris[1:end-1, :], 3, θn=θn, θe=θe, θc=θc)
end

# ╔═╡ 666a9e28-4df1-11eb-108b-fd83c3af1a27
struct ExperimentResult
	iter::Float64
	centers::Float64
	adj_idx::Float64
	rand_idx::Float64
	mirkin_idx::Float64
	hubert_idx::Float64
end

# ╔═╡ 601e9d26-4dec-11eb-0998-27a195aa4026
# experiment1: default isodata params
function experiment1(data, num_trials, iseeds)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	for i in 1:num_trials
		# train test split
		train, test = splitobs(shuffleobs(data), at = 0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		# calculate isodata parameters
		
		# run isodata
		res = base_conv(train_x, 3, θn=0.25, θe=0.12, θc=1.5, iseeds=iseeds)
		
		# fit centers to test data
		fit_y = clusterfit(test_x, res.centers)
		
		# cluster analysis
		adj_idx, rand_idx, mirkin_idx, hubert_idx = randindex(test_y, fit_y)
		num_centers = size(res.centers, 2)
		
		# save results
		results[i] = ExperimentResult(
			res.iter, num_centers, adj_idx, rand_idx, mirkin_idx, hubert_idx)
	end
	
	iters = 0.0
	centers = 0.0
	adjs = 0.0
	rands = 0.0
	mirkins = 0.0
	huberts = 0.0
	for i in 1:num_trials
		iters += results[i].iter
		centers += results[i].centers
		adjs += results[i].adj_idx
		rands += results[i].rand_idx
		mirkins += results[i].mirkin_idx
		huberts += results[i].hubert_idx
	end
	
	avg_iters = iters/num_trials
	avg_centers = centers/num_trials
	avg_adjs = adjs/num_trials
	avg_rands = rands/num_trials
	avg_mirkins = mirkins/num_trials
	avg_huberts = huberts/num_trials
	return ExperimentResult(
		avg_iters, avg_centers, avg_adjs, avg_rands, avg_mirkins, avg_huberts)
end

# ╔═╡ eb969ec6-4de2-11eb-216d-f9e02d81ac11
experiment1(iris, 10, isodata_s)

# ╔═╡ a5734878-4df5-11eb-2ac6-8fa0741d90d9
experiment1(iris, 10, kmppcenters)

# ╔═╡ b153a462-4df5-11eb-0cfc-6f4e735ec560
experiment1(iris, 10, randomcenters)

# ╔═╡ 7a1ba842-4f6b-11eb-222b-2b41308b3d35
# experiment2: computed isodata params
function experiment2(
	data,
	k,
	num_trials,
	iseeds; 
	_θn_bias::Float64=0.35,
	_θe_bias::Float64=0.35,
	_θc_bias::Float64=0.35
)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	for i in 1:num_trials
		# train test split
		train, test = splitobs(shuffleobs(data), at = 0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		# normalize
		# train_z = fit(ZScoreTransform, train_x, dims=2)
		# train_x = StatsBase.transform(train_z, train_x)
		# test_x = StatsBase.transform(train_z, test_x)
		
		# calculate isodata parameters
		summaries = summarize(train)
		θn, θe, θc = iso_param(data[1:end-1, :], summaries)
		
		# run isodata
		res = base_conv(train_x, k, θn=θn, θe=θe, θc=θc, iseeds=iseeds)
		
		# fit centers to test data
		fit_y = clusterfit(test_x, res.centers)
		
		# cluster analysis
		adj_idx, rand_idx, mirkin_idx, hubert_idx = randindex(test_y, fit_y)
		num_centers = size(res.centers, 2)
		
		# save results
		results[i] = ExperimentResult(
			res.iter, num_centers, adj_idx, rand_idx, mirkin_idx, hubert_idx)
	end
	
	iters = 0.0
	centers = 0.0
	adjs = 0.0
	rands = 0.0
	mirkins = 0.0
	huberts = 0.0
	for i in 1:num_trials
		iters += results[i].iter
		centers += results[i].centers
		adjs += results[i].adj_idx
		rands += results[i].rand_idx
		mirkins += results[i].mirkin_idx
		huberts += results[i].hubert_idx
	end
	
	avg_iters = iters/num_trials
	avg_centers = centers/num_trials
	avg_adjs = adjs/num_trials
	avg_rands = rands/num_trials
	avg_mirkins = mirkins/num_trials
	avg_huberts = huberts/num_trials
	return ExperimentResult(
		avg_iters, avg_centers, avg_adjs, avg_rands, avg_mirkins, avg_huberts)
end

# ╔═╡ c43a6cda-4f6b-11eb-3480-910efacac2e6
experiment2(iris, 3, 10, isodata_s)

# ╔═╡ 06f60a46-4f6c-11eb-141f-6d444b5d56a7
experiment2(iris, 3, 10, kmppcenters)

# ╔═╡ 4d3eb44e-4f6c-11eb-19e5-55f8d6eb03ad
experiment2(iris, 3, 10, randomcenters)

# ╔═╡ dd40e1d6-4f71-11eb-026c-6fc896e10fcd
#experiment2(glass, 6, 10, isodata_s)

# ╔═╡ 3452e904-4f72-11eb-288c-4d397ede82ab
#experiment2(glass, 6, 10, kmppcenters)

# ╔═╡ 2f78af8a-4f73-11eb-227e-8733704a70d3
#experiment2(glass, 6, 10, randomcenters)

# ╔═╡ 4b011c6c-4f8f-11eb-0c84-1d844e23378b
"EXPERIMENT 2 - WINE"

# ╔═╡ 2efc4230-4f8f-11eb-2b8e-85ee78f4ebaf
experiment2(wine, 3, 10, isodata_s)

# ╔═╡ 3c6826a0-4f8f-11eb-3912-81f43868d9f3
experiment2(wine, 3, 10, kmppcenters)

# ╔═╡ 44fbc7fe-4f8f-11eb-1003-bb016d171cc9
experiment2(wine, 3, 10, randomcenters)

# ╔═╡ 226e09e6-4f93-11eb-2fc9-d15f03f16ddb
# kmeans with new cluster center
function experiment3(
	data,
	k,
	num_trials,
	iseeds; 
	_θn_bias::Float64=0.35,
	_θe_bias::Float64=0.35,
	_θc_bias::Float64=0.35
)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	for i in 1:num_trials
		# train test split
		train, test = splitobs(shuffleobs(data), at = 0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		# normalize
		train_z = fit(ZScoreTransform, train_x, dims=2)
		train_x = StatsBase.transform(train_z, train_x)
		test_x = StatsBase.transform(train_z, test_x)
		
		# run isodata
		res = ISODATA.kmeans(train_x, k, iseeds=iseeds)
		
		# fit centers to test data
		fit_y = clusterfit(test_x, res.centers)
		
		# cluster analysis
		adj_idx, rand_idx, mirkin_idx, hubert_idx = randindex(test_y, fit_y)
		num_centers = size(res.centers, 2)
		
		# save results
		results[i] = ExperimentResult(
			res.iter, num_centers, adj_idx, rand_idx, mirkin_idx, hubert_idx)
	end
	
	iters = 0.0
	centers = 0.0
	adjs = 0.0
	rands = 0.0
	mirkins = 0.0
	huberts = 0.0
	for i in 1:num_trials
		iters += results[i].iter
		centers += results[i].centers
		adjs += results[i].adj_idx
		rands += results[i].rand_idx
		mirkins += results[i].mirkin_idx
		huberts += results[i].hubert_idx
	end
	
	avg_iters = iters/num_trials
	avg_centers = centers/num_trials
	avg_adjs = adjs/num_trials
	avg_rands = rands/num_trials
	avg_mirkins = mirkins/num_trials
	avg_huberts = huberts/num_trials
	return ExperimentResult(
		avg_iters, avg_centers, avg_adjs, avg_rands, avg_mirkins, avg_huberts)
end

# ╔═╡ 505daeb0-4f93-11eb-1671-e361fdb821e9
experiment3(iris, 3, 1000, isodata_s)

# ╔═╡ 60697ad2-4f93-11eb-017f-69c9c730506f
experiment3(iris, 3, 1000, kmppcenters)

# ╔═╡ 60fd52de-4f93-11eb-2f43-e797797924af
experiment3(iris, 3, 1000, randomcenters)

# ╔═╡ 850b91f4-4f93-11eb-2447-071b5f7d8cd1
experiment3(wine, 3, 1000, isodata_s)

# ╔═╡ 84d58f32-4f93-11eb-2224-478298799589
experiment3(wine, 3, 1000, kmppcenters)

# ╔═╡ 84554f52-4f93-11eb-296a-c567ad3db9c7
experiment3(wine, 3, 1000, randomcenters)

# ╔═╡ 6166c486-4f94-11eb-368c-bf989822fd98
begin
	train, test = splitobs(shuffleobs(iris), at = 0.8)
	train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
	test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
	
	train_z = fit(ZScoreTransform, train_x, dims=2)
	train_x = StatsBase.transform(train_z, train_x)
	test_x = StatsBase.transform(train_z, test_x)
	
	ISODATA.kmeans(train_x, 3, iseeds=isodata_s)
end

# ╔═╡ Cell order:
# ╠═a6abf980-2042-11eb-326f-67792f308488
# ╠═c4a2ee6c-4f8d-11eb-24f5-e53ed7aa6789
# ╠═7d36be2c-4f70-11eb-3b81-2b78885c3771
# ╠═b2ce85e2-4f8e-11eb-3c52-0fce3a112046
# ╠═affc89c4-4f64-11eb-29f8-15a7c437fe24
# ╠═f277c7dc-4f64-11eb-147d-95a2b10bab9f
# ╠═f800520a-4f64-11eb-2c36-85794e9f2a12
# ╠═e1a8995c-4f66-11eb-0d59-392ffc2743f9
# ╠═21a507b6-4f67-11eb-26a5-57d9727165ac
# ╠═3a6fc37c-4f6b-11eb-2947-470ea113fdde
# ╠═666a9e28-4df1-11eb-108b-fd83c3af1a27
# ╠═601e9d26-4dec-11eb-0998-27a195aa4026
# ╠═eb969ec6-4de2-11eb-216d-f9e02d81ac11
# ╠═a5734878-4df5-11eb-2ac6-8fa0741d90d9
# ╠═b153a462-4df5-11eb-0cfc-6f4e735ec560
# ╠═7a1ba842-4f6b-11eb-222b-2b41308b3d35
# ╠═c43a6cda-4f6b-11eb-3480-910efacac2e6
# ╠═06f60a46-4f6c-11eb-141f-6d444b5d56a7
# ╠═4d3eb44e-4f6c-11eb-19e5-55f8d6eb03ad
# ╠═dd40e1d6-4f71-11eb-026c-6fc896e10fcd
# ╠═3452e904-4f72-11eb-288c-4d397ede82ab
# ╠═2f78af8a-4f73-11eb-227e-8733704a70d3
# ╟─4b011c6c-4f8f-11eb-0c84-1d844e23378b
# ╠═2efc4230-4f8f-11eb-2b8e-85ee78f4ebaf
# ╠═3c6826a0-4f8f-11eb-3912-81f43868d9f3
# ╠═44fbc7fe-4f8f-11eb-1003-bb016d171cc9
# ╠═226e09e6-4f93-11eb-2fc9-d15f03f16ddb
# ╠═505daeb0-4f93-11eb-1671-e361fdb821e9
# ╠═60697ad2-4f93-11eb-017f-69c9c730506f
# ╠═60fd52de-4f93-11eb-2f43-e797797924af
# ╠═850b91f4-4f93-11eb-2447-071b5f7d8cd1
# ╠═84d58f32-4f93-11eb-2224-478298799589
# ╠═84554f52-4f93-11eb-296a-c567ad3db9c7
# ╠═6166c486-4f94-11eb-368c-bf989822fd98
