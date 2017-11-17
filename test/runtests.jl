using IntrinsicMetrics
using Base.Test
using Distances
using LightGraphs, SimpleWeightedGraphs
include("manifolds.jl")

# IDEAS:
# pairwise on a full graph should be the same as euclidean
# distance between opposite points in sphere should be π
# pairwise pts pts must be symmetric

sphere1000 = rand(Sphere(1), 1000)
torus1000  = rand(Torus(1, 3), 1000)
atorus1000 = rand(AsymTorus(5, 2, 1), 1000)
circle1000 = rand(Circle(1), 1000)

testsets1000 = [sphere1000, torus1000, atorus1000, circle1000]

function isdistancematrix(m; approx = false)
    if m ≉ m'
        println("Not symmetric!")
        return false
    elseif any(m .< 0)
        println("Not positive!")
        return false
    elseif any(diag(m) .!= 0)
        println("Diagonal not zero!")
        return false
    end
    true
end

@testset "Sampling sanity checks" begin
    for ts in testsets1000[1:3]
        @test size(ts) == (3, 1000)
    end
    @test size(testsets1000[4]) == (2, 1000)
end

@testset "IntrinsicMetric" begin
    @testset "Type" begin
        for ts in testsets1000, metric in [Euclidean, Cityblock]
            @test typeof(IntrinsicMetric(ts, 10, metric = metric())) ==
                IntrinsicMetric{Float64, metric}
        end
    end

    @testset "Graph properties" begin
        for ts in testsets1000, nn in [10, 30, 60, 90]
            im = IntrinsicMetric(ts, nn)
            @test nv(im.graph) == 1000
            @test ne(im.graph) ≤  1000 * nn
            @test isdistancematrix(im.graph.weights)
        end
    end
end

@testset "Distance evaluation" begin
    @testset "evaluate" begin
        for ts in testsets1000, nn in [10, 30, 60, 90]
            n = size(ts, 2)
            im = IntrinsicMetric(ts, nn)
            d1 = Float64[]
            d2 = Float64[]
            d3 = Float64[]
            ds = Float64[]
            # Pick 100 points and check distance properties.
            for _ in 1:100
                p1 = ts[:, rand(1:n)]
                p2 = ts[:, rand(1:n)]
                p3 = ts[:, rand(1:n)]
                push!(d1, evaluate(im, p1, p2))
                push!(ds, evaluate(im, p2, p1))
                push!(d2, evaluate(im, p2, p3))
                push!(d3, evaluate(im, p1, p3))
            end
            # Symmetry.
            @test all(d1 .≈ ds)
            # Non-negativity.
            @test all(d1 .≥ 0)
            # Triangle inequality.
            @test all(d3 .≤ d1 .+ d2 .+ sqrt(eps(Float64)))
        end

        # Going around sphere should be about π.
        im = IntrinsicMetric(hcat(sphere1000, [0, 0, 1], [0, 0,-1]), 30)
        @test evaluate(im, [0, 0,-1], [0, 0, 1]) ≈ π     atol=0.1
        @test evaluate(im, [0, 0,-2], [0, 0, 1]) ≈ (1+π) atol=0.1
        @test evaluate(im, [0, 0,-1], [0, 0, 2]) ≈ (1+π) atol=0.1
        @test evaluate(im, [0, 0,-2], [0, 0, 2]) ≈ (2+π) atol=0.1

        # Same for circle.
        im = IntrinsicMetric(circle1000, 30)
        @test evaluate(im, [0,-1], [ 0, 1]) ≈ π atol=0.01
        @test evaluate(im, [1, 0], [-1, 0]) ≈ π atol=0.01
    end

    @testset "pairwise" begin
        @testset "one" begin
            for ts in testsets1000, nn in [10, 60]
                im = IntrinsicMetric(ts, nn)

                # Points from sample.
                pw1 = pairwise(im, ts, ts)
                pw2 = pairwise(im, ts)
                @test isdistancematrix(pw1)
                @test isdistancematrix(pw2)
                @test pw1 ≈ pw2
                @test size(pw1) == size(pw2) == (1000, 1000)

                # Random points.
                pts = (rand(size(ts, 1), 100) - 0.5) * 3
                pw3 = pairwise(im, pts, pts)
                pw4 = pairwise(im, pts)
                @test isdistancematrix(pw3)
                @test isdistancematrix(pw4)
                @test pw3 ≈ pw4
                @test size(pw3) == size(pw4) == (100, 100)
            end
        end

        @testset "two" begin
            for ts in testsets1000, nn in [10, 60]
                im = IntrinsicMetric(ts, nn)
                pts1 = (rand(size(ts, 1), 100) - 0.5) * 3
                pts2 = (rand(size(ts, 1), 1000) - 0.5) * 3

                pw1 = pairwise(im, pts1, pts2)
                pw2 = pairwise(im, pts2, pts1)

                @test pw1' ≈ pw2
                @test all(pw1 .≥ 0)
                @test size(pw1) == (100, 1000)
                @test_throws DimensionMismatch pairwise(im, rand(3, 100), rand(2, 100))
            end
        end
    end
end
