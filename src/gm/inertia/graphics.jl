"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    ns = length(singles)
    ne = length(ensembles)
    es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne + 1)
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    # add the single object representations
    # REVIEW: what about all singles, but miss gorilla?
    @inbounds for i in 1:ns
        single = singles[i]
        pos = get_pos(single)
        args = (pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        es[i] = LogBernElement{Detection}(wm.single_rfs_logweight, detect, args)
    end
    # the ensemble
    @inbounds for i = 1:ne
        @unpack matws, rate, pos, var = ensembles[i]
        # Clamped to possibly explain light gorilla
        matws = clamp.(matws, 0.01, 0.99)
        varw = var * single_noise
        mix_args = (matws,
                    Fill(pos, 2),
                    Fill(varw, 2),
                    [1.0, 2.0],
                    Fill(material_noise, 2))
        es[ns + i] =
            PoissonElement{Detection}(rate, detect_mixture, mix_args)
    end
    # catch all ensemble
    # NOTE: This is needed in case of all individuals
    es[end] =
        PoissonElement{Detection}(0.1, detect,
                                  (S2V([0., 0.]), 1000.0, 0.5, 10.0))
    return es
end
