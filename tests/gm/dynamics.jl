using Gen
using AdaptiveGorilla
using AdaptiveGorilla: inertia_init,
    birth_single,
    inertia_force,
    inertia_kernel,
    wm_inertia

function test_force()
    wm = InertiaWM()
    single = birth_single(wm)
    tr, w = Gen.generate(inertia_force, (wm, single),
                         choicemap(:is_stable => false)
                         )
    display(Gen.get_choices(tr))
    @show w
end

# test_force()

function test_kernel()
    wm = InertiaWM()
    state = inertia_init(wm)
    cm = choicemap()
    cm[:birth => :pregnant] = true
    tr, w = Gen.generate(inertia_kernel, (1, state, wm),
                         cm
                         )
    display(Gen.get_choices(tr))
    @show w
    @show length(state.singles)
    @show length(get_retval(tr).singles)
end

test_kernel()


function test_wm()
    wm = InertiaWM()
    state = inertia_init(wm)
    cm = choicemap()
    addr = :kernel => 2 => :birth => :pregnant
    cm[addr] = true
    tr, w = Gen.generate(wm_inertia, (2, wm), cm)
    cs = Gen.get_choices(tr)
    display(cs)
    display(cm)
    @show cs[addr]
    display(get_submap(cs, :kernel => 2 => :birth))
    @show w
end
test_wm()
