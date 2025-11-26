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
        # TODO: Check velocity
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

function load_wm_from_toml(path::String)
    toml = TOML.parsefile(path)
    stub = toml["WorldModel"]
    parts = Dict()
    protocol = getfield(AdaptiveGorilla, Symbol(stub["protocol"]))
    kwargs = load_inner(parts, stub["params"])
    protocol(; kwargs...)
end
