using Revise

using ISODATA
using MLDataUtils
using StatsBase

X, Y, fnames = load_iris();
X = zscore(X)

res = base(X, 3)
println(res.centers)
println(res.assignments)
# println(res.costs)
println(res.counts)
println(res.sd)
