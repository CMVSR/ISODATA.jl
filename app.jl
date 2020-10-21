using Revise

using BenchmarkTools
using Clustering
using ISODATA
using MLDataUtils
using MLLabelUtils
using StatsBase
using AssignmentSolver

X, Y, fnames = load_iris();
Y = convertlabel(MLLabelUtils.LabelEnc.Indices{Int64,3}(), Y, labelenc(Y))
# X = zscore(X)
# res = base(X, 3, θn=0.25, θe=0.12, θc=1.5, iter=15)
# println(unique(res.assignments))
# println(randindex(res.assignments, Y))

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
a = [0, 0, 1, 0, 2, 2]
b = [1, 1, 2, 2, 0, 0]
print(cluster_acc(a, b))
#  print(cluster_acc(Y, res.assignments))
# runs = 1000
# results = []
# for i in 1:runs
#     res = base(X, 3, θn=0.25, θe=0.12, θc=1.5, iter=15)
#     acc = cluster_acc(Y, res.assignments)
#     push!(results, acc)
#     #push!(results, randindex(res.assignments, Y)[1])
# end

# avg_accuracy = mean(results)
# min_accuracy = minimum(results)
# max_accuracy = maximum(results)
# println("ISODATA:")
# println(avg_accuracy)
# println(min_accuracy)
# println(max_accuracy)
# println("Average number of clusters: ")

# results = []
# for i in 1:runs
#     res = kmeans(X, 3, init=:rand)
#     # res = kmeans(X, 3)
#     acc = cluster_acc(Y, res.assignments)
#     push!(results, acc)
#     # push!(results, randindex(res.assignments, Y)[2])
# end
# avg_accuracy = mean(results)
# min_accuracy = minimum(results)
# max_accuracy = maximum(results)
# println("Kmeans:")
# println(avg_accuracy)
# println(min_accuracy)
# println(max_accuracy)