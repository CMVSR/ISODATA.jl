### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ a6abf980-2042-11eb-326f-67792f308488
begin
	using Revise
	
	using BenchmarkTools
	using Clustering
	using ISODATA
	using MLDataUtils
	using MLLabelUtils
	using StatsBase
	using AssignmentSolver
end

# ╔═╡ 7ae1b520-2043-11eb-18a0-ffbfd39df927
function cluster_acc(y_true, y_pred)
    w = counts(y_true, y_pred)
    cost = maximum(w) .- w
    reward = Float64.(reward2cost(cost))
    auctionsol, lambda = auction_assignment(reward)
    compute_objective(auctionsol, cost)
    ind = auctionsol.r2c

    n = size(ind, 1)
    x = 0
    for i in 1:n
        x += w[i, ind[i]]
    end

    return (x * 1.0/(size(y_true, 1))) * 100.0
end

# ╔═╡ e1ca4c6c-2042-11eb-04a7-51562484df32
begin
	X, Y, fnames = load_iris()
	Y = convertlabel(MLLabelUtils.LabelEnc.Indices{Int64,3}(), Y, labelenc(Y))
	X, Y, fnames
end

# ╔═╡ f89da308-2042-11eb-3802-652dd4864c58
res = base(X, 3, θn=0.25, θe=0.12, θc=1.5, iter=15)

# ╔═╡ 8a3e0028-2043-11eb-0bd5-7fd1a9a9a9f2
acc = cluster_acc(Y, res.assignments)

# ╔═╡ Cell order:
# ╠═a6abf980-2042-11eb-326f-67792f308488
# ╠═7ae1b520-2043-11eb-18a0-ffbfd39df927
# ╠═e1ca4c6c-2042-11eb-04a7-51562484df32
# ╠═f89da308-2042-11eb-3802-652dd4864c58
# ╠═8a3e0028-2043-11eb-0bd5-7fd1a9a9a9f2
