@userplot Points3d

@recipe function f(p::Points3d)
    if length(p.args) != 1 || !(typeof(first(p.args)) <: AbstractMatrix)
        error("points3d is expecting a single matrix argument. " *
              "Got: $(typeof(p.args))")
    end

    pts = first(p.args)
    seriestype := :scatter
    markersize --> 0.5
    pts[1, :], pts[2, :], pts[3, :]
end

@recipe function f(im::IntrinsicMetric{T}) where T
    2 ≤ ambientdim(im) ≤ 3 || error("Can't plot $(ambientdim(im))d points!")
    flat = ambientdim(im) == 2

    legend := false
    pts = points(im)
    g = adjgraph(im)

    @series begin
        seriestype := :scatter
        markersize --> 0.5
        label := :vertex
        if flat
            pts[1, :], pts[2, :]
        else
            pts[1, :], pts[2, :], pts[3, :]
        end
    end

    xs = T[]; ys = T[]; zs = T[]
    for e in edges(g)
        p1 = pts[:, src(e)]
        p2 = pts[:, dst(e)]
        append!(xs, [p1[1], p2[1], NaN])
        append!(ys, [p1[2], p2[2], NaN])
        !flat && append!(zs, [p1[3], p2[3], NaN])
    end

    @series begin
        markersize := :none
        seriestype := :path
        linewidth --> 1
        label := :edge
        if flat
            xs, ys
        else
            xs, ys, zs
        end
    end
end
