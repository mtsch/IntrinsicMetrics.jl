module IntrinsicMetrics

using NearestNeighbors
using Distances
using LightGraphs, SimpleWeightedGraphs
using RecipesBase

include("intrinsic_metric.jl")
include("plotting.jl")

export IntrinsicMetric,
    ambient_metric, incidence_graph, points, n_points, ambient_dim

end # module
