struct IntrinsicMetric{I<:Integer, T<:Real, M<:Metric} <: Metric
    graph::SimpleWeightedGraph{I, T}
    points::Matrix{T}
    kdtree::KDTree{<:AbstractArray{T}, M, T}
end

function IntrinsicMetric(pts, nn; metric = Euclidean())
    kdtree = KDTree(pts, metric, reorder = false)
    n = size(pts, 2)

    # Temporary Graph is used to efficiently initialize incidence matrix.
    # TODO: figure out how to build sparse matrix from schratch?
    # TODO: add a threshold for maximum edge length or something similar?
    g_tmp = Graph(n)
    neighs, dists = knn(kdtree, pts, nn, false)
    for (i, ns) in enumerate(neighs)
        for j in ns
            add_edge!(g_tmp, i, j)
        end
    end
    g = SimpleWeightedGraph(g_tmp)
    for (i, ns, ds) in zip(1:n, neighs, dists)
        for (j, d) in zip(ns, ds)
            i == j && continue
            g.weights[i, j] = d
            g.weights[j, i] = d
        end
    end

    IntrinsicMetric(g, pts, kdtree)
end

#=
# Constructor
function IntrinsicMetric(pts, min_degree; metric = Euclidean())
    kdtree = KDTree(pts, metric, reorder = false)
    n = length(pts)

    function estimate_δ(pts, kdtree, min_degree)
        k = min_degree + 2
        idxs, dsts = knn(kdtree, pts, k, false)
        maximum(maximum.(dsts))
    end

    δ = estimate_δ(pts, kdtree, min_degree)

    g_tmp = Graph(n)
    for (i, ns) in enumerate(inrange(kdtree, pts, δ))
        g_tmp.fadjlist[i] = filter(j -> j != i, ns)
        g_tmp.ne += length(ns) - 1
    end
    g = SimpleWeightedGraph(g_tmp)
    for (i, ns) in enumerate(inrange(kdtree, pts, δ))
        for j in ns
            i == j && continue
            d = evaluate(metric, pts[i], pts[j])
            g.weights[j, i] = d
            g.weights[i, j] = d
        end
    end

    IntrinsicMetric(g, pts, kdtree)
end
=#

function Distances.evaluate(m::IntrinsicMetric, p1, p2)
    p1 == p2 && return 0

    nearest1 = knn(m.kdtree, p1, 1)
    nearest2 = knn(m.kdtree, p2, 1)

    total_dist = first(nearest1[2]) + first(nearest2[2])

    i1 = first(nearest1[1])
    i2 = first(nearest2[1])

    total_dist + dijkstra_shortest_paths(m.graph, i1).dists[i2]

end
