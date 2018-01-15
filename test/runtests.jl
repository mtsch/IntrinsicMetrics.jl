using IntrinsicMetrics
using Base.Test
using LightGraphs, SimpleWeightedGraphs
using Plots; plotly()
include("manifolds.jl")

n = 200
repeats_small = 10

# Randomly generate datasets by sampling points from manifolds.
sphere = rand(Sphere(1), n)
atorus = rand(AsymTorus(5, 2, 1), n)
circle = rand(Circle(1), n)
cloud9 = rand(9, n)

# Datasets used in testing.
testsets = [sphere, atorus, circle, cloud9]

# Radius values used in testing.
radii = [0.5, 0.7]

# Metrics used in testing.
metrics = [Euclidean(), Minkowski(3), Cityblock(), Chebyshev()]

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

@testset "sanity checks" begin
    # Check sampling.
    @test size(testsets[1]) == (3, n)
    @test size(testsets[2]) == (3, n)
    @test size(testsets[3]) == (2, n)
    @test size(testsets[4]) == (9, n)

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
        for testset in testsets, r in radii, metric in metrics
            im = IntrinsicMetric(testset, r, metric = metric)
            @test ambientmetric(im) == metric
            @test isa(im, IntrinsicMetric{Float64, typeof(metric)})
       end
    end

    @testset "points" begin
        for testset in testsets, r in radii, metric in metrics
            im = IntrinsicMetric(testset, r, metric = metric)
            @test points(im) == testset
            @test size(testset) == (ambientdim(im), npoints(im))
        end
    end

    @testset "graph properties" begin
        for testset in testsets, r in radii, metric in metrics
            im = IntrinsicMetric(testset, r, metric = metric)
            @test nv(adjgraph(im)) == npoints(im)
            @test isdistancematrix(adjgraph(im).weights)
        end
    end

    @testset "float types" begin
        for T in [Float16, Float32, Float64]
            testset = rand(Sphere(T(1.0)), n)
            im = IntrinsicMetric(testset, 30)
            @test eltype(testset) == T
            @test isa(im, IntrinsicMetric{T, Euclidean})
            @test isa(evaluate(im, [1.,0.,0.], [-1.,0.,0.]), T)
        end
    end
end

@testset "evaluate" begin
    for testset in testsets, r in radii, metric in metrics
        im = IntrinsicMetric(testset, r, metric = metric)
        d1 = Float64[]
        d2 = Float64[]
        d3 = Float64[]
        ds = Float64[]
        # Pick 100 points and check distance properties.
        for _ in 1:100
            p1 = testset[:, rand(1:n)]
            p2 = testset[:, rand(1:n)]
            p3 = testset[:, rand(1:n)]
            push!(d1, evaluate(im, p1, p2))
            push!(ds, evaluate(im, p2, p1))
            push!(d2, evaluate(im, p2, p3))
            push!(d3, evaluate(im, p1, p3))
        end
        # Symmetry.
        @test d1 ≈ ds
        # Non-negativity.
        @test all(d1 .≥ 0)
        # Triangle inequality.
        @test all(d3 .≤ d1 .+ d2 .+ sqrt(eps(Float64)))
    end

    # Going around sphere should be about π.
    im = IntrinsicMetric(hcat(rand(Sphere(1), 1000), [0, 0, 1], [0, 0,-1]), 0.5)
    @test evaluate(im, [0, 0,-1], [0, 0, 1]) ≈ π     atol=0.1
    @test evaluate(im, [0, 0,-2], [0, 0, 1]) ≈ (1+π) atol=0.1
    @test evaluate(im, [0, 0,-1], [0, 0, 2]) ≈ (1+π) atol=0.1
    @test evaluate(im, [0, 0,-2], [0, 0, 2]) ≈ (2+π) atol=0.1

    # Same for circle.
    im = IntrinsicMetric(rand(Circle(1), 1000), 0.1)
    @test evaluate(im, [0,-1], [ 0, 1]) ≈ π atol=0.01
    @test evaluate(im, [1, 0], [-1, 0]) ≈ π atol=0.01
end

@testset "pairwise - size errors" begin
    for testset in testsets, r in radii, metric in metrics
        im = IntrinsicMetric(testset, r, metric = metric)
        d  = size(testset, 1)
        dm = DimensionMismatch

        @test_throws dm evaluate(im, rand(d-1, 1), rand(d,   1))
        @test_throws dm evaluate(im, rand(d,   1), rand(d-1, 1))
        @test_throws dm evaluate(im, rand(d+1, 1), rand(d+1, 1))

        @test_throws dm pairwise(im, rand(d,   10), rand(d+1, 11))
        @test_throws dm pairwise(im, rand(d-1, 10), rand(d-1, 11))

        @test_throws dm pairwise!(zeros(5, 5), im, rand(d-1, 5), rand(d,   5))
        @test_throws dm pairwise!(zeros(5, 5), im, rand(d-1, 5), rand(d-1, 5))
        @test_throws dm pairwise!(zeros(6, 5), im, rand(d,   5), rand(d,   5))
        @test_throws dm pairwise!(zeros(5, 6), im, rand(d,   5), rand(d,   5))

        @test_throws dm pairwise(im, rand(d+1, 10))

        @test_throws dm pairwise!(zeros(10, 11), im, rand(d,   10))
        @test_throws dm pairwise!(zeros(11, 11), im, rand(d+1, 11))

        @test_throws dm pairwise!(zeros(10, 10), im)
    end
end

@testset "pairwise - one matrix" begin
    for testset in testsets, r in radii, metric in metrics
        im = IntrinsicMetric(testset, r, metric = metric)

        # Points from sample.
        pw1 = pairwise(im, testset, testset)
        pw2 = pairwise(im, testset)
        pw3 = pairwise(im)
        @test isdistancematrix(pw1)
        @test isdistancematrix(pw2)
        @test isdistancematrix(pw3)
        @test pw1 ≈ pw2 ≈ pw3
        @test size(pw1) == size(pw2) == size(pw3) == (n, n)

        # Random points.
        pts = (rand(size(testset, 1), 100) - 0.5) * 3
        pw4 = pairwise(im, pts, pts)
        pw5 = pairwise(im, pts)
        @test isdistancematrix(pw4)
        @test isdistancematrix(pw5)
        @test pw4 ≈ pw5
        @test size(pw4) == size(pw5) == (100, 100)

        d1 = Float64[]; d2 = Float64[]; d3 = Float64[]
        for _ in 1:100
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
    for testset in testsets, r in radii, metric in metrics
        im = IntrinsicMetric(testset, r, metric = metric)
        pts1 = (rand(size(testset, 1), 100) - 0.5) * 3
        pts2 = (rand(size(testset, 1), 1000) - 0.5) * 3

        pw1 = pairwise(im, pts1, pts2)
        pw2 = pairwise(im, pts2, pts1)

        @test pw1' ≈ pw2
        @test all(pw1 .≥ 0)
        @test size(pw1) == (100, 1000)
    end
end

@testset "pairwise - full graph" begin
    for _ in 1:repeats_small, metric in metrics
        pts = rand(5, n)
        im = IntrinsicMetric(pts, 10, metric = metric)

        @test adjgraph(im).weights ≈ pairwise(im, pts) ≈ pairwise(metric, pts)
    end
end

@testset "plotting does not crash" begin
    for _ in 1:repeats_small
        @test points3d(rand(3, 1000)) ≠ nothing
        @test_throws ErrorException points3d(rand(1000), rand(1000), rand(1000))
    end
    r = radii[1]
    for testset in testsets[1:3], metric in metrics
        @test plot(IntrinsicMetric(testset, r, metric = metric)) ≠ nothing
    end
    @test_throws ErrorException plot(IntrinsicMetric(testsets[4], r))
end
