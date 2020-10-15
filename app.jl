using ISODATA
using MLDataUtils

X, Y, fnames = load_iris();

res = base(X, 3)
# println(res.centers)
# println(res.assignments)
# println(res.costs)
println(res.counts)
