"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    ns = length(singles)
    ne = length(ensembles)
    # es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne + 1)
    es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne)
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    # add the single object representations
    # REVIEW: what about all singles, but miss gorilla?
    @inbounds for i in 1:ns
        single = singles[i]
        pos = get_pos(single)
        args = (pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        # es[i] = CPoissonElement{Detection}(detect, args)
        es[i] = CPoissonElement{Detection}(detect, args)
        # es[i] = LogBernElement{Detection}(wm.single_rfs_logweight, detect, args)
        # es[i] = NegBinomElement{Detection}(2, 0.4, detect, args)
    end
    # the ensemble
    @inbounds for i = 1:ne
        @unpack matws, rate, pos, var = ensembles[i]
        # Clamped to possibly explain both materials
        light_w = clamp(matws[1], 0.0, 1.0)
        varw = var #* single_noise
        mix_args = (pos,
                    varw,
                    light_w,
                    material_noise)
        es[ns + i] =
            PoissonElement{Detection}(rate, detect_mixture, mix_args)
    end
    # catch all ensemble
    # NOTE: This is needed in case of all individuals
    # catch_all_args = (S2V([0., 0.]), 1000.0, 0.5, material_noise)
    # es[end] = LogBernElement{Detection}(-.0001, detect_mixture, catch_all_args)
    return es
end
