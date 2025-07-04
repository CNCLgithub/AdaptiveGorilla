export softmax,
    softmax!


isbetween(x::Real, low::Real, high::Real) = (x >= low) && (x <= high)

"""
$(TYPEDSIGNATURES)

Computes softmax of an array, with temperature `t`.
"""
function softmax(x::Array{Float64}, t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}, t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

function inbounds(pos::S2V, bbmin::S2V, bbmax::S2V)
    x, y = pos
    bbmin[1] <= x &&
        x <= bbmax[1] &&
        bbmin[2] <= y &&
        y <= bbmax[2]
end


# adapted from
# https://github.com/JeffreySarnoff/AngleBetweenVectors.jl/blob/01c91199fef4d4e8b9178c81662c98ea7f3868d6/src/AngleBetweenVectors.jl#L28
function vec2_angle(a::SVector{2, Float64}, b::SVector{2, Float64})
    a = normalize(a)
    b = normalize(b)
    y = a - b
    x = a + b

    a = 2 * atan(norm(y), norm(x))

    !(signbit(a) || signbit(pi - a)) ? a : (signbit(a) ? 0.0 : pi)
end

function sosq(x::S2V, y::S2V)
    sum(@. (x - y)^(2))
end

function sigmoid(x::Float64, x0::Float64 = 0., m::Float64=1.0)
    x = abs(x) # assume always positive 
    1 / (1 + exp(-m*x + x0))
end

function sigmoid_grad(x::Float64, x0::Float64 = 0., m::Float64=1.0)
    y = sigmoid(x, x0, m)
    y * (1 - y)
end

function fast_sigmoid(x::Real)
    x = abs(x) + 0.001
    (1/x) / (1 + (1/x))
end

function fast_sigmoid_grad(x::Real)
    x = abs(x)
    1 / (x + 1)^2
end

function l2log(x::Vector{T}) where {T<:Real}
    logsqs = T(-Inf)
    n = length(x)
    @inbounds for i = 1:n
        logsqs = logsumexp(2 * x[i], logsqs)
    end
    return logsqs
end


# NOTE: from: https://www.geeksforgeeks.org/program-calculate-value-ncr/
function ncr(n::Int, r::Int)
    # If r is greater than n, return 0
    (r > n) && return 0
    # If r is 0 or equal to n, return 1
    (r == 0 || n == r) && return 1
    # Initialize the logarithmic sum to 0
    res = 0.0
    # Calculate the logarithmic sum of 
    # the numerator and denominator using loop
    for i = 0:(r-1)
        # Add the logarithm of (n-i) 
        # and subtract the logarithm of (i+1)
        res += log(n-i) - log(i+1)
    end
    # Convert logarithmic sum back to a normal number
    round(Int, exp(res))
end

# NOTE: from: https://stackoverflow.com/a/794
function combination(n::Int, p::Int, x::Int)
    r = k = 0
    c = Vector{Int}(undef, p)
    @assert x <= ncr(n, p) "$(x)th lexical index DNE!"
    for i = 0:(p-2)
        c[i+1] = (i != 0) ? c[i] : 0
        while true # do-while in julia
            c[i+1] += 1
            r = ncr(n - c[i+1], p - (i+1))
            k = k + r
            (k < x) || break
        end
        k = k - r
    end
    c[p] = c[p-1] + x - k
    return c
end

function comb_index(n::Int, s::Vector{Int})
    k = length(s)
    index = 0
    j = 1;
    for i = 0:(k-1)
        while j < s[i+1]
            index += ncr(n - j, k - i - 1);
            j += 1
        end
    end
    return index
end

const NEGLN2 = -log(2)

"""
Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
See [Maechler2012accurate] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function log1mexp(x::Float64)
    x = min(x, 0.0)
    x  > NEGLN2 ? log(-expm1(x)) : log1p(-exp(x))
end

#from https://discuss.pytorch.org/t/how-to-calculate-log-1-softmax-x-numerically-stably/169007/11
function log1msumexp(x::Vector{Float64})
    xi = argmax(x)
    xm = x[xi]
    xbar = x .- xm
    lse = logsumexp(xbar)
    xbarexp = exp.(xbar)
    sumexp = sum(xbarexp) .- xbarexp
    sumexp[xi] = 1.0
    log1msm = log.(sumexp)
    xbar[xi] = -Inf
    log1msm[xi] = logsumexp(xbar)
    log1msm .-= lse
    return log1msm
end
