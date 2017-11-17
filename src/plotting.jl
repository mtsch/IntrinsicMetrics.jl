@userplot Points3d

@recipe function f(p::Points3d)
    if length(p.args) != 1 || !(typeof(p.args[1]) <: AbstractMatrix)
        error("points3d is expecting a single matrix argument. Got: $(typeof(h.args))")
    end

    pts = p.args[1]
    seriestype := :scatter
    markersize --> 0.5
    pts[1, :], pts[2, :], pts[3, :]
end

@recipe function f(m::IntrinsicMetric)
    legend := false
    pts = m.points
    g = m.graph

    @series begin
        seriestype := :scatter
        markersize --> 0.5
        label := :vertex
        pts[1, :], pts[2, :], pts[3, :]
    end

    xs = Segments()
    ys = Segments()
    zs = Segments()
    for e in edges(g)
        p1 = pts[:, src(e)]
        p2 = pts[:, dst(e)]
        push!(xs, [p1[1], p2[1]])
        push!(ys, [p1[2], p2[2]])
        push!(zs, [p1[3], p2[3]])
    end

    @series begin
        markersize := :none
        seriestype := :path3d
        linewidth --> 1
        label := :edge
        xs.pts, ys.pts, zs.pts
    end
end
