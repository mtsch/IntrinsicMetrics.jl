module IntrinsicMetrics

using NearestNeighbors
using Distances
using LightGraphs, SimpleWeightedGraphs
using RecipesBase

include("intrinsic_metric.jl")
include("plotting.jl")

export IntrinsicMetric, ambinet_metric, incidence_graph, points

end # module
