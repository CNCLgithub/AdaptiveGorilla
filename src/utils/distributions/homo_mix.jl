struct DetectionMixture <: Gen.Distribution{Detection}
end

const detect_mixture = DetectionMixture()

function Gen.random(::DetectionMixture, mu_pos::SVector{2, Float64},
                    var_pos::Float64, light_prop::Float64, var_material::Float64)
    x,y = Gen.random(broadcasted_normal, mu_pos, var_pos)
    mat = Gen.random(bernoulli, light_prop) ? 1.0 : 2.0
    i = Gen.random(normal, mat, var_material) # REVIEW: reimplement with Beta distribution
    Detection(x, y, i)
end

function Gen.logpdf(::DetectionMixture, x::Detection, mu_pos::SVector{2, Float64},
                    var_pos::Float64, light_prop::Float64, var_material::Float64)
    px, py = position(x)
    intx = intensity(x)
    int = logsumexp(
        # Light component
        log(light_prop) + Gen.logpdf(normal, intx, 1.0, var_material),
        # Dark component
        log(1.0 - light_prop) + Gen.logpdf(normal, intx, 2.0, var_material)
    )

    loc = Gen.logpdf(normal, px, mu_pos[1], var_pos) +
        Gen.logpdf(normal, py, mu_pos[2], var_pos)

    int + loc
end

(::DetectionMixture)(mp, vp, mm, vm) = Gen.random(detect, mp, vp, mm, vm)

Gen.has_output_grad(::DetectionMixture) = false
Gen.logpdf_grad(::DetectionMixture, value, args...) = (nothing,)
