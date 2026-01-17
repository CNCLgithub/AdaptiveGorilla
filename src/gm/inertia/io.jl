export load_wm_from_toml


function initial_state(wm::InertiaWM, state::SchollState, target_count::Int = 4,
                       swap_colors::Bool = false)
    initial_state(wm, map(x -> x.pos, state.objects),
                  target_count, swap_colors)
end
function initial_state(wm::InertiaWM, positions, target_count::Int = 4,
                       swap_colors::Bool = false)
    n = Int64(wm.object_rate)
    singles = Vector{InertiaSingle}(undef, n)
    for i = 1:n
        x, y = positions[i]
        color = xor(swap_colors, i <= target_count) ? Light : Dark
        singles[i] = InertiaSingle(color, S2V(x, y), S2V(0., 0.))
    end
    InertiaState(singles, InertiaEnsemble[])
end

_kernel_prefix(t::Int, i::Int) = :kernel => t => :xs => i

function write_obs_mask!(
    cm::ChoiceMap, wm::InertiaWM,
    t::Int,
    i::Int,
    xy,
    mat::Material;
    prefix = _kernel_prefix
    )
    x, y = xy
    intensity = mat == Light ? 1.0 : 2.0
    cm[prefix(t, i)] = Detection(x, y, intensity)
    return nothing
end

function load_wm_from_toml(path::String; kwargs...)
    toml = TOML.parsefile(path)
    stub = toml["WorldModel"]
    parts = Dict()
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    specified = load_inner(parts, stub["params"])
    protocol(; merge(specified, kwargs)...)
end

function print_granularity_schema(state::InertiaState)
    ns = length(state.singles)
    ne = length(state.ensembles)
    c = object_count(state)
    println("Granularity: $(ns) singles; $(ne) ensembles; $(c) total")
    ndark = count(x -> material(x) == Dark, state.singles)
    println("\tSingles: $(ndark) Dark | $(ns-ndark) Light")
    println("\tEnsembles: $(map(e -> (rate(e), e.matws[1]), state.ensembles))")
    return nothing
end


function pretty_state(state::InertiaState)
    print_granularity_schema(state)
    ns = length(state.singles)
    ne = length(state.ensembles)
    c = object_count(state)
    for i = 1:ns
        o = state.singles[i]
        println("  Single $(i): $(material(o)), p=$(get_pos(o)), v=$(get_vel(o))")
    end
    for i = 1:ne
        o = state.ensembles[i]
        println("  Ensemble $(i): $(rate(o)), p=$(get_pos(o)), v=$(get_vel(o))")
    end
    return nothing
end

import Printf

import Base.show

function Base.show(io::IO, ::MIME"text/plain", s::InertiaSingle)
    m = material(s)
    px, py = get_pos(s)
    vx, vy = get_vel(s)
    a = get_avel(s)
    fmt = Printf.Format(
        raw"⚛ {m = %s, p=[%3.1f,%3.1f], v=[%3.1f,%3.1f], θ=%3.1f}"
    )
    s = Printf.format(fmt, m, px, py, vx, vy, a)
    println(io, s)
end

function Base.show(io::IO, ::MIME"text/plain", e::InertiaEnsemble)
    r = rate(e)
    m = materials(e)[1]
    px, py = get_pos(e)
    vx, vy = get_vel(e)
    v = get_var(e)
    fmt = Printf.Format(
        raw"𝛌[%d]:{m=%3.1f, p=[%3.1f,%3.1f], v=[%3.1f,%3.1f], σ=%3.1f}"
    )
    s = Printf.format(fmt, r, m, px, py, vx, vy, v)
    println(io, s)
end
