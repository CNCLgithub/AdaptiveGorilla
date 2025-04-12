export von_mises,
    VonMises


struct VonMises <: Gen.Distribution{Float64} end

const von_mises = VonMises()

function Gen.random(::VonMises, mu::Float64, k::Float64)
    d = Distributions.VonMises(mu, k)
    rand(d)
end

function Gen.logpdf(::VonMises, x::Float64, mu::Float64, k::Float64)
    d = Distributions.VonMises(mu, k)
    Distributions.logpdf(d, x)
end

(::VonMises)(mu, k) = Gen.random(von_mises, mu, k)

Gen.has_output_grad(::VonMises) = false
Gen.logpdf_grad(::VonMises, value::Set, args...) = (nothing,)
