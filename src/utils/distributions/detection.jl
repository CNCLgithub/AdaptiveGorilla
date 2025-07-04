export Detection,
    intensity,
    position,
    detect,
    DetectionRFS,
    detect_mixture

struct Detection
    "X location"
    x::Float64
    "Y location"
    y::Float64
    "Color intensity [1, 2]"
    i::Float64
end
intensity(x::Detection) = x.i
position(x::Detection) = SVector{2, Float64}(x.x, x.y)

import Base.isapprox

function Base.isapprox(x::Detection, y::Detection)
    Base.isapprox(x.x, y.x) &&
        Base.isapprox(x.y, y.y) &&
        Base.isapprox(x.i, y.i)
end


const DetectionRFS = RFGM(MRFS{Detection}(), (100, 10.0))

struct DetectionRV <: Gen.Distribution{Detection} end

const detect = DetectionRV()

function Gen.random(::DetectionRV, mu_pos::SVector{2, Float64},
                    var_pos::Float64, mu_material::Float64, var_material::Float64)
    x,y = Gen.random(broadcasted_normal, mu_pos, var_pos)
    i = Gen.random(normal, mu_material, var_material) # REVIEW: reimplement with Beta distribution
    Detection(x, y, i)
end

function Gen.logpdf(::DetectionRV, x::Detection, mu_pos::SVector{2, Float64},
                    var_pos::Float64, mu_material::Float64, var_material::Float64)
    px, py = position(x)
    intx = intensity(x)
    pdf_int =
        Gen.logpdf(normal, intx, mu_material, var_material)
    Gen.logpdf(normal, px, mu_pos[1], var_pos) +
        Gen.logpdf(normal, py, mu_pos[2], var_pos) +
        pdf_int
end

(::DetectionRV)(mp, vp, mm, vm) = Gen.random(detect, mp, vp, mm, vm)

Gen.has_output_grad(::DetectionRV) = false
Gen.logpdf_grad(::DetectionRV, value, args...) = (nothing,)

detect_mixture = HomogeneousMixture(detect, [0, 0, 0, 0])

using Luxor: sethue, box, Point, setopacity
import MOTCore.paint

function MOTCore.paint(p::ObjectPainter,  obs::AbstractVector{T}
                       ) where {T<:Detection}
    for i = eachindex(obs)
        d = obs[i]
        sethue(0.2, 0.2, 0.2)
        setopacity(1.0)
        box(Point(d.x, -d.y), 20.0, 20.0,
            action = :stroke)
        c = d.i == 1.0 ? 0.99 : 0.01
        sethue(c, c, c)
        setopacity(0.7)
        box(Point(d.x, -d.y), 20.0, 20.0,
            action = :fill)
    end
    return nothing
end
