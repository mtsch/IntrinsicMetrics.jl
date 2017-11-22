module IntrinsicMetrics

using NearestNeighbors
using Distances
using LightGraphs, SimpleWeightedGraphs
using RecipesBase

include("intrinsic_metric.jl")
include("plotting.jl")

export IntrinsicMetric,
    ambientmetric, adjgraph, points, npoints, ambientdim, points3d,
    evaluate, pairwise!, pairwise, colwise!, colwise,
    Euclidean, Minkowski, Cityblock, Chebyshev

end
