@userplot Points3d

@recipe function f(p::Points3d)
    if length(p.args) != 1 || !(typeof(p.args[1]) <: AbstractMatrix)
        error("points3d is expecting a single matrix argument. Got: $(typeof(p.args))")
    end

    pts = p.args[1]
    seriestype := :scatter
    markersize --> 0.5
    pts[1, :], pts[2, :], pts[3, :]
end

@recipe function f(im::IntrinsicMetric)
    ambientdim(im) == 3 || error("Only 3d points are supported!")

    legend := false
    pts = points(im)
    g = adjgraph(im)

    @series begin
        seriestype := :scatter
        markersize --> 0.5
        label := :vertex
        pts[1, :], pts[2, :], pts[3, :]
    end

    xs = []; ys = []; zs = []
    for e in edges(g)
        p1 = pts[:, src(e)]
        p2 = pts[:, dst(e)]
        append!(xs, [p1[1], p2[1], NaN])
        append!(ys, [p1[2], p2[2], NaN])
        append!(zs, [p1[3], p2[3], NaN])
    end
    # Remove last NaN
    pop!(xs); pop!(ys); pop!(zs)

    @series begin
        markersize := :none
        seriestype := :path3d
        linewidth --> 1
        label := :edge
        xs, ys, zs
    end
end
