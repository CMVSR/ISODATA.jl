### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 0b8acfe4-5073-11eb-1630-33963891fa65
begin
	using Revise
	
	using MAT
	
	using Clustering
	using ISODATA
	using MLDataUtils
	using MultivariateStats
	using Plots
	using StatsBase
end

# ╔═╡ 9d50e9dc-5112-11eb-261e-fb54e7238993
gradient = cgrad([:black, :purple, :red, :yellow])

# ╔═╡ 24a82cb0-5073-11eb-198e-a9de05e0c7a3
begin
	dfile = matopen("data/SalinasA.mat")
	Xraw = read(dfile)["salinasA"]
	close(dfile)
	Xraw
end

# ╔═╡ 103c6df2-5077-11eb-1ee3-137d3269b6eb
heatmap(Xraw[:, :, 4])

# ╔═╡ a688fe58-5073-11eb-2968-bd2d796d1f2f
begin
	tfile = matopen("data/SalinasA_gt.mat")
	Yraw = read(tfile)["salinasA_gt"]
	close(tfile)
	Yraw
end

# ╔═╡ b58883cc-5075-11eb-0c36-87c4d7f584f3
heatmap(Yraw, c=gradient)

# ╔═╡ 02200e30-5074-11eb-18c1-bbbd91537f47
X = reshape(deepcopy(Xraw), 7138, 224)'

# ╔═╡ 97b50c02-5074-11eb-1974-47e0028a67bb
begin
	Y = reshape(deepcopy(Yraw), 7138)
	for i in 10:14
		Y[findall(x -> x == Float64(i), Y)] .= i - 8.0
	end
	Y = round.(Int, Y)
end

# ╔═╡ efe37d6e-50f6-11eb-1e0d-c3b097dac6d6
heatmap(reshape(Y, 83, 86), c=gradient)

# ╔═╡ 59fa6a82-510d-11eb-06a9-c729dc0b2920
struct ExperimentResult
	iter::Float64
	centers::Float64
	adj_idx::Float64
	rand_idx::Float64
	mirkin_idx::Float64
	hubert_idx::Float64
end

# ╔═╡ 32c3f282-5110-11eb-050b-e3dc9548a51e
function avg_result(results::Vector{ExperimentResult})
	n = size(results, 1)
	
	iters = 0.0
	centers = 0.0
	adjs = 0.0
	rands = 0.0
	mirkins = 0.0
	huberts = 0.0
	for i in 1:n
		iters += results[i].iter
		centers += results[i].centers
		adjs += results[i].adj_idx
		rands += results[i].rand_idx
		mirkins += results[i].mirkin_idx
		huberts += results[i].hubert_idx
	end
	
	avg_iters = iters/n
	avg_centers = centers/n
	avg_adjs = adjs/n
	avg_rands = rands/n
	avg_mirkins = mirkins/n
	avg_huberts = huberts/n
	return ExperimentResult(
		avg_iters, avg_centers, avg_adjs, avg_rands, avg_mirkins, avg_huberts)
end

# ╔═╡ 1db73a8c-510d-11eb-089b-7b92b54526e5
# kmeans on salinasA
function experiment4(
	data,
	k,
	num_trials,
	iseeds;
	normalize=true
)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	best_score = 0.0
	best_centers = Array{Float64}
	best_idx = 0
	
	for i in 1:num_trials
		train, test = splitobs(shuffleobs(data), at=0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		if normalize
			train_z = fit(ZScoreTransform, train_x, dims=2)
			train_x = StatsBase.transform(train_z, train_x)
			test_x = StatsBase.transform(train_z, test_x)
		end
		
		# run kmeans
		res = ISODATA.kmeans(train_x, 6, iseeds=iseeds)
		
		# fit centers to test data
		fit_y = clusterfit(test_x, res.centers)
		
		# cluster analysis
		adj_idx, rand_idx, mirkin_idx, hubert_idx = randindex(test_y, fit_y)
		num_centers = size(res.centers, 2)
		
		results[i] = ExperimentResult(
			res.iter, num_centers, adj_idx, rand_idx, mirkin_idx, hubert_idx)
		
		if adj_idx > best_score
			best_score = adj_idx
			best_centers = res.centers
			best_idx = i
		end
	end
	
	return results[best_idx], best_centers, avg_result(results)
end

# ╔═╡ 415ae730-50f5-11eb-03e3-6365247ed76b
begin
	inds = findall(x -> x != 0.0, Y)
	
	M = fit(PCA, X; maxoutdim=3)
	Xout = transform(M, X)
	
	full = transpose(cat(Xout', Y, dims=2))
	filtered = full[:, inds]
end

# ╔═╡ f65644f8-5114-11eb-2fd9-09e75a642aca
transpose(cat(X', Y, dims=2))

# ╔═╡ f9a6065e-510d-11eb-2a36-e72f2d507a66
rkmeans_res, rkmeans_centers, rkmeans_avg = experiment4(
	filtered, 6, 100, randomcenters)

# ╔═╡ 226e3d78-510f-11eb-2d8d-155aabe1918d
begin
	rkmeans_pred = clusterfit(filtered[1:end-1, :], rkmeans_centers)
	rkmeans_Y = deepcopy(Y)
	rkmeans_Y[inds] .= rkmeans_pred
	heatmap(reshape(rkmeans_Y, 83, 86), c=gradient)
end

# ╔═╡ afa88d40-510f-11eb-327f-ab765d0a3c8b
kkmeans_res, kkmeans_centers, kkmeans_avg = experiment4(
	filtered, 6, 100, kmppcenters)

# ╔═╡ baae817c-510f-11eb-372b-05a0b920d188
begin
	kkmeans_pred = clusterfit(filtered[1:end-1, :], kkmeans_centers)
	kkmeans_Y = deepcopy(Y)
	kkmeans_Y[inds] .= kkmeans_pred
	heatmap(reshape(kkmeans_Y, 83, 86), c=gradient)
end

# ╔═╡ e8d00128-510f-11eb-359e-9964c959ce74
ikmeans_res, ikmeans_centers, ikmeans_avg = experiment4(
	filtered, 6, 100, isodata_s, normalize=true)

# ╔═╡ f17da848-510f-11eb-04b6-7d129dec43fb
begin
	ikmeans_pred = clusterfit(filtered[1:end-1, :], ikmeans_centers)
	ikmeans_Y = deepcopy(Y)
	ikmeans_Y[inds] .= ikmeans_pred
	heatmap(reshape(ikmeans_Y, 83, 86), c=gradient)
end

# ╔═╡ b159bdf4-5110-11eb-00a0-1bece66bc22b
# isodata on salinasA
function experiment5(
	data,
	k,
	num_trials,
	iseeds;
	normalize=true,
	_θn_bias::Float64=0.20,
	_θe_bias::Float64=0.20,
	_θc_bias::Float64=0.20
)
	results = Vector{ExperimentResult}(undef, num_trials)
	
	best_score = 0.0
	best_centers = Array{Float64}
	best_idx = 0
	
	for i in 1:num_trials
		train, test = splitobs(shuffleobs(data), at=0.8)
		train_x, train_y = train[1:end-1, :], round.(Int, train[end, :])
		test_x, test_y = test[1:end-1, :], round.(Int, test[end, :])
		
		if normalize
			train_z = fit(ZScoreTransform, train_x, dims=2)
			train_x = StatsBase.transform(train_z, train_x)
			test_x = StatsBase.transform(train_z, test_x)
		end
		
		# calculate isodata params
		summaries = summarize(train)
		θn, θe, θc = iso_param(
			train_x,
			summaries,
			_θn_bias=_θn_bias,
			_θe_bias=_θe_bias,
			_θc_bias=_θc_bias)
		
		# run kmeans
		res = base_conv(train_x, k, θn=θn, θe=θe, θc=θc, iseeds=iseeds)
		
		# fit centers to test data
		fit_y = clusterfit(test_x, res.centers)
		
		# cluster analysis
		adj_idx, rand_idx, mirkin_idx, hubert_idx = randindex(test_y, fit_y)
		num_centers = size(res.centers, 2)
		
		results[i] = ExperimentResult(
			res.iter, num_centers, adj_idx, rand_idx, mirkin_idx, hubert_idx)
		
		if adj_idx > best_score
			best_score = adj_idx
			best_centers = res.centers
			best_idx = i
		end
	end
	
	return results[best_idx], best_centers, avg_result(results)
end

# ╔═╡ 5b33ed70-5111-11eb-0795-015a8472911b
risodata_res, risodata_centers, risodata_avg = experiment5(
	filtered, 6, 1000, randomcenters, normalize=true,
	_θn_bias=0.0,
	_θe_bias=0.0,
	_θc_bias=0.0)

# ╔═╡ 02b383c0-5116-11eb-2748-d371c5aa42b6
begin
	risodata_pred = clusterfit(filtered[1:end-1, :], risodata_centers)
	risodata_Y = deepcopy(Y)
	risodata_Y[inds] .= risodata_pred
	heatmap(reshape(risodata_Y, 83, 86), c=gradient)
end

# ╔═╡ 9bc5aa2e-5111-11eb-06d7-b3cb072d95cd
kisodata_res, kisodata_centers, kisodata_avg = experiment5(
	filtered, 6, 1000, kmppcenters, normalize=true,
	_θn_bias=0.0,
	_θe_bias=0.0,
	_θc_bias=0.0)

# ╔═╡ aae23512-5115-11eb-0b01-4764155fb2d1
begin
	kisodata_pred = clusterfit(filtered[1:end-1, :], kisodata_centers)
	kisodata_Y = deepcopy(Y)
	kisodata_Y[inds] .= kisodata_pred
	heatmap(reshape(kisodata_Y, 83, 86), c=gradient)
end

# ╔═╡ a66f7ffc-5111-11eb-12a4-91874657e647
iisodata_res, iisodata_centers, iisodata_avg = experiment5(
	filtered, 6, 1000, isodata_s, normalize=true,
	_θn_bias=0.0,
	_θe_bias=0.0,
	_θc_bias=0.0)

# ╔═╡ d1d323d6-5113-11eb-3835-c13bc51b8e13
begin
	iisodata_pred = clusterfit(filtered[1:end-1, :], iisodata_centers)
	iisodata_Y = deepcopy(Y)
	iisodata_Y[inds] .= iisodata_pred
	heatmap(reshape(iisodata_Y, 83, 86), c=gradient)
end

# ╔═╡ Cell order:
# ╠═0b8acfe4-5073-11eb-1630-33963891fa65
# ╠═9d50e9dc-5112-11eb-261e-fb54e7238993
# ╠═24a82cb0-5073-11eb-198e-a9de05e0c7a3
# ╠═103c6df2-5077-11eb-1ee3-137d3269b6eb
# ╠═a688fe58-5073-11eb-2968-bd2d796d1f2f
# ╠═b58883cc-5075-11eb-0c36-87c4d7f584f3
# ╠═02200e30-5074-11eb-18c1-bbbd91537f47
# ╠═97b50c02-5074-11eb-1974-47e0028a67bb
# ╠═efe37d6e-50f6-11eb-1e0d-c3b097dac6d6
# ╠═59fa6a82-510d-11eb-06a9-c729dc0b2920
# ╠═32c3f282-5110-11eb-050b-e3dc9548a51e
# ╠═1db73a8c-510d-11eb-089b-7b92b54526e5
# ╠═415ae730-50f5-11eb-03e3-6365247ed76b
# ╠═f65644f8-5114-11eb-2fd9-09e75a642aca
# ╠═f9a6065e-510d-11eb-2a36-e72f2d507a66
# ╠═226e3d78-510f-11eb-2d8d-155aabe1918d
# ╠═afa88d40-510f-11eb-327f-ab765d0a3c8b
# ╠═baae817c-510f-11eb-372b-05a0b920d188
# ╠═e8d00128-510f-11eb-359e-9964c959ce74
# ╠═f17da848-510f-11eb-04b6-7d129dec43fb
# ╠═b159bdf4-5110-11eb-00a0-1bece66bc22b
# ╠═5b33ed70-5111-11eb-0795-015a8472911b
# ╠═02b383c0-5116-11eb-2748-d371c5aa42b6
# ╠═9bc5aa2e-5111-11eb-06d7-b3cb072d95cd
# ╠═aae23512-5115-11eb-0b01-4764155fb2d1
# ╠═a66f7ffc-5111-11eb-12a4-91874657e647
# ╠═d1d323d6-5113-11eb-3835-c13bc51b8e13
