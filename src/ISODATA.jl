module ISODATA

using StatsBase
using Distances

# source files
include("base.jl")
include("centers.jl")
include("clusters.jl")
include("evaluation.jl")
include("kmeans.jl")
include("params.jl")

export base, base_conv
export randomcenters, kmppcenters, isodata_s
export clusterfit
export kmeans
export summarize, iso_param

end
