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
    "Color intensity [0, 1]"
    i::Float64
end
intensity(x::Detection) = x.i
position(x::Detection) = SVector{2, Float64}(x.x, x.y)


const DetectionRFS = RFGM(MRFS{Detection}(), (200, 10.0))

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
    Gen.logpdf(broadcasted_normal, position(x), mu_pos, var_pos) +
        Gen.logpdf(normal, intensity(x), mu_material, var_material) # REVIEW: reimplement with Beta distribution
end

(::DetectionRV)(mp, vp, mm, vm) = Gen.random(detect, mp, vp, mm, vm)

Gen.has_output_grad(::DetectionRV) = false
Gen.logpdf_grad(::DetectionRV, value, args...) = (nothing,)

detect_mixture = HomogeneousMixture(detect, [0, 0, 0, 0])
