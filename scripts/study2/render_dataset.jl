using JSON3
using MOTCore
using MOTCore: state_from_positions, render_scene


wm = SchollWM(;
              n_dots=8,
              dot_radius = 5.0,
              area_width = 720.0,
              area_height =480.0,
              vel=3.0,
              vel_step = 0.75,
              vel_prob = 0.15
)

targets = [true, true, true, true, false, false, false, false]


function main()
    dname = "target_ensemble"
    version = "2025-06-09_W96KtK"
    dpath = "/spaths/datasets/$(dname)/$(version)"
    local dataset
    open("$(dpath)/dataset.json", "r") do f
        dataset = JSON3.read(f)
    end

    outpath = "$(dpath)/renders"
    isdir(outpath) || mkdir(outpath)

    manifest = dataset["manifest"]
    trials = dataset["trials"]

    ntrials = length(dataset)
    Threads.@threads for i = 1:manifest["ntrials"]
        positions = trials[i]["positions"]
        states = state_from_positions(wm, positions, targets)
        scene_path = "$(outpath)/$(i)"
        isdir(scene_path) || mkdir(scene_path)
        render_scene(wm, states, scene_path)
    end

    return nothing
end

main();
