using MOTCore
using AdaptiveGorilla

dataset = "pilot"
ntrials = 10
duration = 40

function main()
    wm = SchollWM(n_dots = 8,
                  dot_radius = 10.0,
                  )

    _, states = wm_scholl(duration, wm)
    render_scene(wm, states, "state")
end;


main();
