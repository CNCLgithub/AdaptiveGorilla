struct HomoMixture{T} <: Gen.Distribution{T}
    base_dist::Gen.Distribution{T}
    dims::Vector{Int}
end

(dist::HomoMixture)(args...) = random(dist, args...)

Gen.has_output_grad(dist::HomoMixture) = has_output_grad(dist.base_dist)
Gen.has_argument_grads(dist::HomoMixture) = (true, has_argument_grads(dist.base_dist)...)
Gen.is_discrete(dist::HomoMixture) = is_discrete(dist.base_dist)

function args_for_component(dist::HomoMixture, k::Int, args)
    # returns a generator
    return (arg[fill(Colon(), dim)..., k]
            for (arg, dim) in zip(args, dist.dims))
end

function Gen.random(dist::HomoMixture, weights, args...)
    k = categorical(weights)
    return random(dist.base_dist, args_for_component(dist, k, args)...)
end

function Gen.logpdf(dist::HomoMixture, x, weights, args...)
    K = length(weights)
    ld = -Inf
    @inbounds for k = 1:K
        _ld = Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...)
        ld = logsumexp(ld, _ld + log(weights[k]))
    end
    return ld
    # log_densities = Vector{Float64}(undef, K)
    # @inbounds for k = 1:K
    #     log_densities[k] = Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...)
    #     log_densities[k] += log(weights[k])
    # end
    # # log_densities = [Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]
    # # log_densities = log_densities .+ log.(weights)
    # return logsumexp(log_densities)
end

function Gen.logpdf_grad(dist::HomoMixture, x, weights, args...)
    K = length(weights)
    log_densities = [Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]
    log_weighted_densities = log_densities .+ log.(weights)
    relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

    # log_grads[k] contains the gradients for the k'th component
    log_grads = [Gen.logpdf_grad(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]

    # compute gradient with respect to x
    log_grads_x = [log_grad[1] for log_grad in log_grads]
    x_grad = if has_output_grad(dist.base_dist)
        sum(log_grads_x .* relative_weighted_densities)
    else
        nothing
    end

    # compute gradients with respect to the weights
    weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

    # compute gradients with respect to each argument
    arg_grads = Any[]
    for (i, (has_grad, arg, dim)) in enumerate(zip(has_argument_grads(dist)[2:end], args, dist.dims))
        if has_grad
            if dim == 0
                grads = [log_grad[i+1] for log_grad in log_grads]
                grad_weights = relative_weighted_densities
            else
                grads = cat(
                    [log_grad[i+1] for log_grad in log_grads]...,
                    dims=dist.dims[i]+1)
                grad_weights = reshape(
                    relative_weighted_densities,
                    (1 for d in 1:dist.dims[i])..., K)
            end
            push!(arg_grads, grads .* grad_weights)
        else
            push!(arg_grads, nothing)
        end
    end

    return (x_grad, weights_grad, arg_grads...)
end
