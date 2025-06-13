"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    ns = length(singles)
    ne = length(ensembles)
    es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne)
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    # add the single object representations
    @inbounds for i in 1:ns
        single = singles[i]
        pos = get_pos(single)
        args = (pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        # bw = inbounds(pos, bbmin, bbmax) ? 0.95 : 0.05
        # NOTE: Manualy set so logpdf(1) > logpdf(0)
        es[i] = NegBinomElement{Detection}(2, 0.4, detect, args)
        # es[i] = PoissonElement{Detection}(1.0, detect, args)
        # es[i] = BernoulliElement{Detection}(0.99, detect, args)
    end
    # the ensemble
    @inbounds for i = 1:ne
        @unpack matws, rate, pos, var = ensembles[i]
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
    # es[end] =
    #     PoissonElement{Detection}(0.1, detect,
    #                               (S2V([0., 0.]), 1000.0, 0.5, 10.0))
    return es
end
