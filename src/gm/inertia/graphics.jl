"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    ns = length(singles)
    ne = length(ensembles)
    # Return a collection of RandomFiniteElements
    es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne)
    # add the individual object representations
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    penalty = wm.single_cpoisson_log_penalty
    single_var = wm.single_size * single_noise
    @inbounds for i in 1:ns
        single = singles[i]
        pos = get_pos(single)
        args = (pos,
                single_var,
                Float64(Int(single.mat)),
                material_noise)
        es[i] = CPoissonElement{Detection}(detect, args, penalty)
    end
    # the ensembles
    @inbounds for i = 1:ne
        @unpack matws, rate, pos, var = ensembles[i]
        light_w = clamp(matws[1], 0.0, 1.0)
        mix_args = (pos,            # Avg. 2D position
                    var,            # Ensemble spread
                    light_w,        # Proportion light
                    material_noise) # variance for color
        es[ns + i] =
            PoissonElement{Detection}(rate, detect_mixture, mix_args)
    end
    return es
end
