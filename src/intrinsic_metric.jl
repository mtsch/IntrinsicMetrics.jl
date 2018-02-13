struct IntrinsicMetric{T, D, M<:Metric,
                       V<:SVector{D, T}, N<:NearestNeighbors.NNTree{V, M}} <: Metric
    points ::Vector{V}
    radius ::T
    metric ::M
    graph  ::SimpleWeightedGraph{Int, T}
    tree   ::N
end

adjgraph(im::IntrinsicMetric) = im.graph
points(im::IntrinsicMetric) = im.points
npoints(im::IntrinsicMetric) = length(im.points)
ambientdim(im::IntrinsicMetric{T, D}) where {T, D} = D
ambientmetric(im::IntrinsicMetric) = im.metric

function Base.show(io::IO, im::IntrinsicMetric{T, D, M, V, N}) where {T, D, M, V, N}
    println(io, "IntrinsicMetric{$T, $D, $M, $V, $N}")
    println(io, "  Number of points: $(length(im.points))")
    println(io, "  Dimensions: $D")
    println(io, "  Radius: $(im.radius)")
    println(io, "  Metric: $(im.metric)")
    print(io, "  Graph: $(im.graph)")
end

function IntrinsicMetric(data::AbstractMatrix{T}, r;
                         metric = Euclidean()) where T
    dim = size(data, 1)
    pts = reinterpret(SVector{dim, T}, data, (length(data) ÷ dim, ))
    IntrinsicMetric(pts, r, metric = metric)
end

function IntrinsicMetric{T, D, M<:Metric}(pts::AbstractVector{SVector{D, T}}, r;
                                          metric::M = Euclidean())
    tree = BallTree(pts, metric, reorder = false)

    neighs = inrange(tree, pts, r, false)
    is = Int[]; js = Int[]; ds = T[]
    for (i, ns) in enumerate(neighs)
        for j in ns
            i < j || continue
            append!(is, [i, j])
            append!(js, [j, i])
            d = evaluate(metric, pts[i], pts[j])
            append!(ds, [d, d])
        end
    end
    n = length(pts)
    g = SimpleWeightedGraph(sparse(is, js, ds, n, n))

    IntrinsicMetric{T, D, M, eltype(pts), typeof(tree)}(pts, T(r), metric, g, tree)
end

function Distances.evaluate(im::IntrinsicMetric, p1, p2)
    length(p1) == length(p2) == ambientdim(im) ||
        throw(DimensionMismatch("Incompatible dimensions!"))
    p1 == p2 && return 0

    nearest1 = knn(im.tree, p1, 1)
    nearest2 = knn(im.tree, p2, 1)

    total_dist = first(nearest1[2]) + first(nearest2[2])

    i1 = first(nearest1[1])
    i2 = first(nearest2[1])

    total_dist + dijkstra_shortest_paths(im.graph, i1).dists[i2]
end

# TODO: colwise: each col with each other col?

function Distances.pairwise(im::IntrinsicMetric,
                            a::AbstractMatrix, b::AbstractMatrix)
    res = zeros(promote_type(eltype(a), eltype(b)), size(a, 2), size(b, 2))
    pairwise!(res, im, a, b)
end

function Distances.pairwise!(res::AbstractMatrix, im::IntrinsicMetric,
                             a::AbstractMatrix, b::AbstractMatrix)
    n = size(a, 2)
    m = size(b, 2)
    size(a, 1) == size(b, 1) == ambientdim(im) ||
        throw(DimensionMismatch("Incompatible dimensions!"))
    size(res) == (n, m) ||
        throw(DimensionMismatch("Output matrix should be $n×$m, " *
                                "but is $(size(res, 1))×(size(res, 2))"))

    a_idx, a_dst = map.(first, knn(im.tree, a, 1))
    b_idx, b_dst = map.(first, knn(im.tree, b, 1))

    for i in 1:n
        dsts = dijkstra_shortest_paths(im.graph, a_idx[i]).dists[b_idx]
        for j in 1:m
            a[:, i] ≈ b[:, j] && continue
            res[i, j] = min(a_dst[i] + b_dst[j] + dsts[j])
        end
    end

    res
end

function Distances.pairwise(im::IntrinsicMetric, a::AbstractMatrix)
    res = zeros(eltype(a), size(a, 2), size(a, 2))
    pairwise!(res, im, a)
end

function Distances.pairwise!(res::AbstractMatrix, im::IntrinsicMetric,
                             a::AbstractMatrix)
    n = size(a, 2)
    size(a, 1) == ambientdim(im) ||
        throw(DimensionMismatch("Incompatible dimensions!"))
    size(res) == (n, n) ||
        throw(DimensionMismatch("Output matrix should be $n×$n, " *
                                "but is $(size(res, 1))×(size(res, 2))"))

    a_idx, a_dst = map.(first, knn(im.tree, a, 1))

    for i in 1:n
        dsts = dijkstra_shortest_paths(im.graph, a_idx[i]).dists[a_idx]
        for j in 1:i-1
            a[:, i] ≈ a[:, j] && continue
            d = min(a_dst[i] + a_dst[j] + dsts[j])
            res[i, j] = d
            res[j, i] = d
        end
    end

    res
end

function Distances.pairwise!(res::AbstractMatrix{T},
                             im::IntrinsicMetric{T, D}) where {T, D}
    pts = reinterpret(T, points(im), (D, npoints(im)))
    pairwise!(res, im, pts)
end

function Distances.pairwise(im::IntrinsicMetric{T, D}) where {T, D}
    pts = reinterpret(T, points(im), (D, npoints(im)))
    pairwise(im, pts)
end
