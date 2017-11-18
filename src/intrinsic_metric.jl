struct IntrinsicMetric{T<:Real, M<:Metric} <: Metric
    graph::SimpleWeightedGraph{<:Integer, T}
    points::Matrix{T}
    kdtree::KDTree{<:AbstractArray{T}, M, T}
    metric::M
end

ambient_metric(in::IntrinsicMetric) = in.metric
incidence_graph(in::IntrinsicMetric) = in.graph
points(in::IntrinsicMetric) = in.points
n_points(in::IntrinsicMetric) = size(in.points, 2)
ambient_dim(in::IntrinsicMetric) = size(in.points, 1)

function IntrinsicMetric(pts, nn; metric = Euclidean())
    kdtree = KDTree(pts, metric, reorder = false)
    n = size(pts, 2)

    # Temporary Graph is used to efficiently initialize incidence matrix.
    # TODO: Figure out how to build sparse matrix from schratch?
    # TODO: Add a threshold for maximum edge length or something similar?
    g_tmp = Graph(n)
    neighs, dists = knn(kdtree, pts, nn, false)
    for (i, ns) in enumerate(neighs)
        for j in ns
            i == j && continue
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

    IntrinsicMetric(g, pts, kdtree, metric)
end

function Distances.evaluate(m::IntrinsicMetric, p1, p2)
    p1 == p2 && return 0

    nearest1 = knn(m.kdtree, p1, 1)
    nearest2 = knn(m.kdtree, p2, 1)

    total_dist = first(nearest1[2]) + first(nearest2[2])

    i1 = first(nearest1[1])
    i2 = first(nearest2[1])

    total_dist + dijkstra_shortest_paths(m.graph, i1).dists[i2]

end

# TODO: colwise: each col with each other col
# TODO: pairwise: with one argument

function Distances.pairwise(in::IntrinsicMetric,
                            a::AbstractMatrix, b::AbstractMatrix)
    res = zeros(promote_type(eltype(a), eltype(b)), size(a, 2), size(b, 2))
    pairwise!(res, in, a, b)
end

function Distances.pairwise!(res::AbstractMatrix, in::IntrinsicMetric,
                             a::AbstractMatrix, b::AbstractMatrix)
    size(a, 1) == size(b, 1) ||
        throw(DimensionMismatch("The numbers of rows in a and b must match."))

    a_idx, a_dst = map.(first, knn(in.kdtree, a, 1))
    b_idx, b_dst = map.(first, knn(in.kdtree, b, 1))

    n = size(a, 2)
    m = size(b, 2)

    for i in 1:n
        dsts = dijkstra_shortest_paths(in.graph, a_idx[i]).dists[b_idx]
        for j in 1:m
            a[:, i] ≈ b[:, j] && continue
            res[i, j] = min(a_dst[i] + b_dst[j] + dsts[j])
        end
    end

    res
end

function Distances.pairwise(in::IntrinsicMetric, a::AbstractMatrix)
    res = zeros(eltype(a), size(a, 2), size(a, 2))
    pairwise!(res, in, a)
end

function Distances.pairwise!(res::AbstractMatrix, in::IntrinsicMetric,
                             a::AbstractMatrix)
    a_idx, a_dst = map.(first, knn(in.kdtree, a, 1))
    n = size(a, 2)

    for i in 1:n
        dsts = dijkstra_shortest_paths(in.graph, a_idx[i]).dists[a_idx]
        for j in 1:i-1
            a[:, i] ≈ a[:, j] && continue
            d = min(a_dst[i] + a_dst[j] + dsts[j])
            res[i, j] = d
            res[j, i] = d
        end
    end

    res
end
