using JSON3
using MOTCore
using AdaptiveGorilla

dataset = "pilot"
ntrials = 10
duration = 100
out_dir = "/spaths/datasets"
out_path = "$(out_dir)/$(dataset).json"

function main()
    # TODO: pin wm parameters
    wm = SchollWM(n_dots = 8,
                  dot_radius = 10.0,
                  )
    data = Dict()
    data[:trials] = [gen_trial(wm, duration) for _ = 1:ntrials]
    data[:manifest] = Dict(:ntrials => ntrials,
                           :duration => duration)
    open(out_path, "w") do io
        JSON3.write(io, data)
    end
end;


main();
