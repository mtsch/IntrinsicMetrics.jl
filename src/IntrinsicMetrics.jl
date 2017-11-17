module IntrinsicMetrics

using NearestNeighbors
using Distances
using LightGraphs, SimpleWeightedGraphs
using RecipesBase

include("manifolds.jl")
include("intrinsic_metric.jl")
include("plotting.jl")

# package code goes here
pts = rand(AsymTorus(10, 5, 3.), 1000, noise = 1)

end # module
