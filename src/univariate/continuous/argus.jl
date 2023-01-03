using SpecialFunctions

struct Argus{T<:Real} <: ContinuousUnivariateDistribution
    p::T
    c::T
    Argus{T}(p::T, c::T) where {T<:Real} = new{T}(p, c)
end

function Argus(p::T, c::T; check_args::Bool=true) where {T<:Real}
    @check_args Argus (p > 0) (c > 0)
    Argus{T}(p, c)
end
Argus(p::Real, c::Real; check_args::Bool=true) = Argus(promote(p, c)...; check_args=check_args)
Argus() = Argus(1, 1; check_args=false)

@distr_support Argus 0 d.c

#### Conversions
#
Base.convert(::Type{Argus{T}}, d::Argus) where {T<:Real} = Argus{T}(T(d.p), T(d.c))
Base.convert(::Type{Argus{T}}, d::Argus{T}) where {T<:Real} = d

#### Parameters

params(d::Argus) = (d.p, d.c)
partype(::Argus{T}) where {T<:Real} = T

#### Helper functions

Ψ(p::Real) = normcdf(p) - p * normpdf(p) - 1/2
xval(d::Argus, x::Real) = 1-(x/d.c)^2

#### Statistics

mean(d::Argus, x::Real) = d.c * √(π/8) * d.p * exp(-d.p^2/4) * besseli(1, (d.p^2/4)) / Ψ(d.p)
mode(d::Argus, x::Real) = d.c / √2 / d.p * √((d.p^2 - 2) + √(d.p^4 + 4))
var(d::Argus, x::Real) = d.c^2 * (1 - 3/d.p^2 + d.p*normpdf(d.p)/Ψ(d.p)) - mean(d, x)^2

#### Evaluation

function pdf(d::Argus, x::Real)
    d.p^3/sqrt(2π)/Ψ(d.p) * x/d.c^2 * √xval(d, x) * exp(-d.p^2 * xval(d, x)/2)
end

function logpdf(d::Argus, x::Real)
    log(pdf(d, x))
end

cdf(d::Argus, x::Real) = 1 - Ψ(d.p * √xval(d, x)) / Ψ(d.p)

