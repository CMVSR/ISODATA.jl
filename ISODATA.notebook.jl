### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a6abf980-2042-11eb-326f-67792f308488
begin
	using Revise
	
	# dataset
	using DataFrames
	using CSV
	
	using BenchmarkTools
	using Clustering
	using ISODATA
	using MLDataUtils
	using MLLabelUtils
	using StatsBase
end

# ╔═╡ f277c7dc-4f64-11eb-147d-95a2b10bab9f


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

# ╔═╡ a1777880-5065-11eb-197d-0dba960318ed
begin
	si = summarize(iris)
	si, iso_param(iris[1:end-1, :], si)
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
	for i = 5:7
		inds = findall(x -> x == i, gY)
		gY[inds] .= i - 1
	end
	gX = Array{Float64}(transpose(gX))
	glass = cat(gX, transpose(gY), dims=1)
end

# ╔═╡ a3565518-5065-11eb-1b6f-27696c4519a6
begin
	sg = summarize(glass)
	sg, iso_param(glass[1:end-1, :], sg)
end

# ╔═╡ b2ce85e2-4f8e-11eb-3c52-0fce3a112046
begin
	# load wine dataset
	wdf = CSV.read("data/wine.data", DataFrame, header=false)
	wine_raw = convert(Matrix, wdf)
	wX, wY = wine_raw[:, 2:end], wine_raw[:, 1]
	wine = cat(transpose(wX), transpose(wY), dims=1)
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

# ╔═╡ 7a1ba842-4f6b-11eb-222b-2b41308b3d35
# experiment2: computed isodata params
function experiment2(
	data,
	k,
	num_trials,
	iseeds;
	normalize=false,
	_θn_bias::Float64=0.20,
	_θe_bias::Float64=0.20,
	_θc_bias::Float64=0.20
)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	for i in 1:num_trials
		# train test split
		train, test = splitobs(shuffleobs(data), at = 0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		if normalize
			train_z = fit(ZScoreTransform, train_x, dims=2)
			train_x = StatsBase.transform(train_z, train_x)
			test_x = StatsBase.transform(train_z, test_x)
		end
		
		# calculate isodata parameters
		summaries = summarize(train)
		θn, θe, θc = iso_param(
			train_x,
			summaries,
			_θn_bias=_θn_bias,
			_θe_bias=_θe_bias,
			_θc_bias=_θc_bias)
		
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

# ╔═╡ ebed5ca6-5068-11eb-36f1-51e85780d0f1
"EXPERIMENT 2: ISODATA - IRIS"

# ╔═╡ c43a6cda-4f6b-11eb-3480-910efacac2e6
experiment2(iris, 3, 1000, isodata_s)

# ╔═╡ 06f60a46-4f6c-11eb-141f-6d444b5d56a7
experiment2(iris, 3, 1000, kmppcenters)

# ╔═╡ 4d3eb44e-4f6c-11eb-19e5-55f8d6eb03ad
experiment2(iris, 3, 1000, randomcenters)

# ╔═╡ e20c5c64-5068-11eb-3a40-d5c965657350
"EXPERIMENT 2: ISODATA - GLASS"

# ╔═╡ dd40e1d6-4f71-11eb-026c-6fc896e10fcd
experiment2(glass, 6, 1000, isodata_s, normalize=false)

# ╔═╡ 3452e904-4f72-11eb-288c-4d397ede82ab
experiment2(glass, 6, 1000, kmppcenters, normalize=false)

# ╔═╡ 2f78af8a-4f73-11eb-227e-8733704a70d3
experiment2(glass, 6, 1000, randomcenters, normalize=false)

# ╔═╡ 4b011c6c-4f8f-11eb-0c84-1d844e23378b
"EXPERIMENT 2: ISODATA - WINE"

# ╔═╡ 2efc4230-4f8f-11eb-2b8e-85ee78f4ebaf
experiment2(wine, 3, 1000, isodata_s)

# ╔═╡ 3c6826a0-4f8f-11eb-3912-81f43868d9f3
experiment2(wine, 3, 1000, kmppcenters)

# ╔═╡ 44fbc7fe-4f8f-11eb-1003-bb016d171cc9
experiment2(wine, 3, 1000, randomcenters)

# ╔═╡ 226e09e6-4f93-11eb-2fc9-d15f03f16ddb
# kmeans with new cluster center
function experiment3(
	data,
	k,
	num_trials,
	iseeds
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

# ╔═╡ f81844c8-5068-11eb-1a4a-c3563bd09221
"EXPERIMENT 3: KMEANS - IRIS"

# ╔═╡ 505daeb0-4f93-11eb-1671-e361fdb821e9
experiment3(iris, 3, 1000, isodata_s)

# ╔═╡ 60697ad2-4f93-11eb-017f-69c9c730506f
experiment3(iris, 3, 1000, kmppcenters)

# ╔═╡ 60fd52de-4f93-11eb-2f43-e797797924af
experiment3(iris, 3, 1000, randomcenters)

# ╔═╡ f9a7d948-5068-11eb-069f-27055b9d6bf7
"EXPERIMENT 3: KMEANS - GLASS"

# ╔═╡ 5601a86c-5068-11eb-2981-65b6287e8ce9
experiment3(glass, 6, 1000, isodata_s)

# ╔═╡ 5e47cb8e-5068-11eb-0459-ff6e67a2fcd4
experiment3(glass, 6, 1000, kmppcenters)

# ╔═╡ 5e332274-5068-11eb-2f5c-1f89f5d6da9e
experiment3(glass, 6, 1000, randomcenters)

# ╔═╡ fad939fe-5068-11eb-26d9-351ad4e57198
"EXPERIMENT 3: KMEANS - WINE"

# ╔═╡ 850b91f4-4f93-11eb-2447-071b5f7d8cd1
experiment3(wine, 3, 1000, isodata_s)

# ╔═╡ 84d58f32-4f93-11eb-2224-478298799589
experiment3(wine, 3, 1000, kmppcenters)

# ╔═╡ 84554f52-4f93-11eb-296a-c567ad3db9c7
experiment3(wine, 3, 1000, randomcenters)

# ╔═╡ Cell order:
# ╠═a6abf980-2042-11eb-326f-67792f308488
# ╟─f277c7dc-4f64-11eb-147d-95a2b10bab9f
# ╠═c4a2ee6c-4f8d-11eb-24f5-e53ed7aa6789
# ╠═a1777880-5065-11eb-197d-0dba960318ed
# ╠═7d36be2c-4f70-11eb-3b81-2b78885c3771
# ╠═a3565518-5065-11eb-1b6f-27696c4519a6
# ╠═b2ce85e2-4f8e-11eb-3c52-0fce3a112046
# ╠═666a9e28-4df1-11eb-108b-fd83c3af1a27
# ╠═7a1ba842-4f6b-11eb-222b-2b41308b3d35
# ╟─ebed5ca6-5068-11eb-36f1-51e85780d0f1
# ╠═c43a6cda-4f6b-11eb-3480-910efacac2e6
# ╠═06f60a46-4f6c-11eb-141f-6d444b5d56a7
# ╠═4d3eb44e-4f6c-11eb-19e5-55f8d6eb03ad
# ╟─e20c5c64-5068-11eb-3a40-d5c965657350
# ╠═dd40e1d6-4f71-11eb-026c-6fc896e10fcd
# ╠═3452e904-4f72-11eb-288c-4d397ede82ab
# ╠═2f78af8a-4f73-11eb-227e-8733704a70d3
# ╟─4b011c6c-4f8f-11eb-0c84-1d844e23378b
# ╠═2efc4230-4f8f-11eb-2b8e-85ee78f4ebaf
# ╠═3c6826a0-4f8f-11eb-3912-81f43868d9f3
# ╠═44fbc7fe-4f8f-11eb-1003-bb016d171cc9
# ╠═226e09e6-4f93-11eb-2fc9-d15f03f16ddb
# ╟─f81844c8-5068-11eb-1a4a-c3563bd09221
# ╠═505daeb0-4f93-11eb-1671-e361fdb821e9
# ╠═60697ad2-4f93-11eb-017f-69c9c730506f
# ╠═60fd52de-4f93-11eb-2f43-e797797924af
# ╟─f9a7d948-5068-11eb-069f-27055b9d6bf7
# ╠═5601a86c-5068-11eb-2981-65b6287e8ce9
# ╠═5e47cb8e-5068-11eb-0459-ff6e67a2fcd4
# ╠═5e332274-5068-11eb-2f5c-1f89f5d6da9e
# ╟─fad939fe-5068-11eb-26d9-351ad4e57198
# ╠═850b91f4-4f93-11eb-2447-071b5f7d8cd1
# ╠═84d58f32-4f93-11eb-2224-478298799589
# ╠═84554f52-4f93-11eb-296a-c567ad3db9c7
