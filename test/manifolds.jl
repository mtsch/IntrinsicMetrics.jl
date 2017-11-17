# Sampling from manifolds. Currently used for testing only.
abstract type Manifold{Int, T<:AbstractFloat} end

# ............................................................................ #
struct Torus{T} <: Manifold{3, T}
    R::T
    r::T
    center::Tuple{T, T, T}
end

function Torus(R, r)
    args = convert.(AbstractFloat, (R, r))
    z = zero(eltype(args))
    Torus(args..., (z, z, z))
end

(t::Torus)(θ, φ) =
    ((t.R + t.r*cos(θ)) * cos(φ), (t.R + t.r*cos(θ)) * sin(φ), t.r * sin(θ)) .+
    t.center

# ............................................................................ #
struct AsymTorus{T} <: Manifold{3, T}
    R::T
    r_large::T
    r_small::T
    center::Tuple{T, T, T}
end

function AsymTorus(R, r1, r2)
    args = convert.(AbstractFloat, (R, r1, r2))
    z = zero(eltype(args))
    AsymTorus(args..., (z, z, z))
end

function (at::AsymTorus)(θ, φ)
    R = at.R
    r_large = at.r_large
    r_small = at.r_small

    r = (r_large - r_small) * (cos(φ) + 1)/2 + r_small
    ((R + r*cos(θ)) * cos(φ), (R + r*cos(θ)) * sin(φ), r*sin(θ)) .+ at.center
end

# ............................................................................ #
struct Sphere{T} <: Manifold{3, T}
    r::T
    center::Tuple{T, T, T}
end

function Sphere(r)
    arg = convert(AbstractFloat, r)
    z = zero(typeof(arg))
    Sphere(arg, (z, z, z))
end

(s::Sphere)(θ, φ) =
    (s.r*cos(θ) * sin(φ), s.r*sin(θ) * sin(φ), s.r*cos(φ)) .+ s.center

# ............................................................................ #
struct Circle{T} <: Manifold{2, T}
    r::T
    center::Tuple{T, T}
end

function Circle(r)
    arg = convert(AbstractFloat, r)
    z = zero(typeof(arg))
    Circle(arg, (z, z))
end

(c::Circle)(θ) = (c.r * cos(θ), c.r * sin(θ)) .+ c.center

# ............................................................................ #
# Uniformly sampling n points from a manifold returning D by n matrix.
# TODO: Circular noise? Overenginnered as it is?
function Base.rand(m::Manifold{D, T}, n=1; noise=0) where {D, T}
    res = zeros(T, D, n)
    for i in 1:n
        pars = 2π .* rand(D-1)
        res[:, i] .= m(pars...) .+ (rand(D) - 0.5) * noise
    end
    res
end
