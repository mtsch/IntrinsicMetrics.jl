println("Loading libs...")
using IntrinsicMetrics
using Base.Test
using Distances
using LightGraphs, SimpleWeightedGraphs
using Plots; plotly()
using RecipesBase
include("manifolds.jl")

println("Preparing tests...")
n = 1000
repeats_small = 10

# Randomly generate datasets by sampling points from manifolds.
sphere = rand(Sphere(1), n)
torus  = rand(Torus(1, 3), n)
atorus = rand(AsymTorus(5, 2, 1), n)
circle = rand(Circle(1), n)

# Datasets used in testing.
testsets = [sphere, torus, atorus, circle]

# Nearest neighbors values used in testing.
nearestneighn = [10, 100]

# Metrics used in testing.
metrics = [Euclidean(), Minkowski(0.1)]

function isdistancematrix(m; report = true)
    if m ≉ m'
        report && println("Not symmetric!")
        return false
    elseif any(m .< 0)
        report && println("Not positive!")
        return false
    elseif any(diag(m) .≠ 0)
        report && println("Diagonal not zero!")
        return false
    end
    true
end

println("Running tests...")

@testset "sanity checks" begin
    # Check sampling.
    for testset in testsets[1:3]
        @test size(testset) == (3, n)
    end
    @test size(testsets[4]) == (2, n)

    # Check isdistancematrix.
    for _ in 1:repeats_small
        rmat = rand(n, n) - 0.5
        @test !isdistancematrix(rmat, report = false)
        symrmat = full(Symmetric(rmat))
        @test !isdistancematrix(symrmat, report = false)
        for i in 1:n
            symrmat[i, i] = 0
        end
        @test !isdistancematrix(symrmat, report = false)
        @test isdistancematrix(abs.(symrmat))
    end
end

@testset "IntrinsicMetric internals" begin
    @testset "ambient metric" begin
        for testset in testsets, nn in nearestneighn, metric in metrics
            in = IntrinsicMetric(testset, 30, metric = metric)
            @test ambient_metric(in) == metric
            @test typeof(in) == IntrinsicMetric{Float64, typeof(metric)}
       end
    end

    @testset "points" begin
        for testset in testsets, nn in nearestneighn, metric in metrics
            in = IntrinsicMetric(testset, nn, metric = metric)
            @test points(in) == testset
            @test size(testset) == (ambient_dim(in), n_points(in))
        end
    end

    @testset "graph properties" begin
        for testset in testsets, nn in nearestneighn, metric in metrics
            in = IntrinsicMetric(testset, nn, metric = metric)
            @test nv(incidence_graph(in)) == n_points(in)
            @test ne(incidence_graph(in)) ≤  n_points(in) * nn
            @test isdistancematrix(incidence_graph(in).weights)
        end
    end
end

@testset "evaluate" begin
    for testset in testsets, nn in nearestneighn, metric in metrics
        in = IntrinsicMetric(testset, nn, metric = metric)
        d1 = Float64[]
        d2 = Float64[]
        d3 = Float64[]
        ds = Float64[]
        # Pick 100 points and check distance properties.
        for _ in 1:100
            p1 = testset[:, rand(1:n)]
            p2 = testset[:, rand(1:n)]
            p3 = testset[:, rand(1:n)]
            push!(d1, evaluate(in, p1, p2))
            push!(ds, evaluate(in, p2, p1))
            push!(d2, evaluate(in, p2, p3))
            push!(d3, evaluate(in, p1, p3))
        end
        # Symmetry.
        @test d1 ≈ ds
        # Non-negativity.
        @test all(d1 .≥ 0)
        # Triangle inequality.
        @test all(d3 .≤ d1 .+ d2 .+ sqrt(eps(Float64)))
    end

    # Going around sphere should be about π.
    in = IntrinsicMetric(hcat(sphere, [0, 0, 1], [0, 0,-1]), 30)
    @test evaluate(in, [0, 0,-1], [0, 0, 1]) ≈ π     atol=0.1
    @test evaluate(in, [0, 0,-2], [0, 0, 1]) ≈ (1+π) atol=0.1
    @test evaluate(in, [0, 0,-1], [0, 0, 2]) ≈ (1+π) atol=0.1
    @test evaluate(in, [0, 0,-2], [0, 0, 2]) ≈ (2+π) atol=0.1

    # Same for circle.
    in = IntrinsicMetric(circle, 30)
    @test evaluate(in, [0,-1], [ 0, 1]) ≈ π atol=0.01
    @test evaluate(in, [1, 0], [-1, 0]) ≈ π atol=0.01
end

@testset "pairwise - one matrix" begin
    for testset in testsets, nn in nearestneighn, metric in metrics
        in = IntrinsicMetric(testset, nn, metric = metric)

        # Points from sample.
        pw1 = pairwise(in, testset, testset)
        pw2 = pairwise(in, testset)
        @test isdistancematrix(pw1)
        @test isdistancematrix(pw2)
        @test pw1 ≈ pw2
        @test size(pw1) == size(pw2) == (n, n)

        # Random points.
        pts = (rand(size(testset, 1), 100) - 0.5) * 3
        pw3 = pairwise(in, pts, pts)
        pw4 = pairwise(in, pts)
        @test isdistancematrix(pw3)
        @test isdistancematrix(pw4)
        @test pw3 ≈ pw4
        @test size(pw3) == size(pw4) == (100, 100)

        d1 = Float64[]
        d2 = Float64[]
        d3 = Float64[]
        for _ in 1:repeats_small
            i = rand(1:n)
            j = rand(1:n)
            k = rand(1:n)
            push!(d1, pw2[i, j])
            push!(d2, pw2[j, k])
            push!(d3, pw2[i, k])
        end
        # Triangle inequality (symmetry and non-negativity were tested).
        @test all(d3 .≤ d1 .+ d2 .+ sqrt(eps(Float64)))
    end
end

@testset "pairwise - two matrices" begin
    for testset in testsets, nn in nearestneighn, metric in metrics
        in = IntrinsicMetric(testset, nn, metric = metric)
        pts1 = (rand(size(testset, 1), 100) - 0.5) * 3
        pts2 = (rand(size(testset, 1), 1000) - 0.5) * 3

        pw1 = pairwise(in, pts1, pts2)
        pw2 = pairwise(in, pts2, pts1)

        @test pw1' ≈ pw2
        @test all(pw1 .≥ 0)
        @test size(pw1) == (100, 1000)
        @test_throws DimensionMismatch pairwise(in, rand(3, 100),
                                                rand(2, 100))
    end
end

@testset "pairwise - full graph" begin
    for _ in 1:repeats_small, metric in [Chebyshev(), Cityblock(),
                                         Euclidean(), Minkowski(0.5)]
        pts = rand(5, n)
        in = IntrinsicMetric(pts, n, metric = metric)

        @test pairwise(metric, pts) ≈ pairwise(metric, pts)
    end
end

@testset "plotting does not crash" begin
    for _ in 1:repeats_small
        @test points3d(rand(3, 1000)) ≠ nothing
    end
    # TODO: 2d plotting
    for testset in testsets[1:3], nn in nearestneighn, metric in metrics
        @test plot(IntrinsicMetric(testset, nn, metric = metric)) ≠ nothing
    end
end
