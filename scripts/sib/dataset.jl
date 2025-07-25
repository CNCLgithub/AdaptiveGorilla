using Gen
using JSON3
using MOTCore
using StaticArrays

const S2V = SVector{2, Float64}

dataset = "sib"
nscenes = 7 # unique scenes
# Each scene will have 4 conditions:
# Gorilla color x parent color
# light gorilla splitting from target
# light gorilla splitting from distractor
# dark gorilla splitting from target
# dark gorilla splitting from distractor
fps = 24
duration = 10 # seconds
frames = fps * duration
out_dir = "/spaths/datasets"
out_path = "$(out_dir)/$(dataset).json"

function gen_scene(wm::SchollWM, nframes::Int64)
    _, states = wm_scholl(nframes, wm)
    steps = Vector{Vector{S2V}}(undef,nframes)
    @inbounds for j in 1:nframes
        state = states[j]
        objects = state.objects
        no = length(objects)
        step = Vector{S2V}(undef, no)
        for k in 1:no
            obj = objects[k]
            step[k] = obj.pos
        end
        steps[j] = step
    end
    gorilla = Dict(
        # onset time
        :frame =>
            uniform_discrete(round(Int64, 0.5 * nframes),
                             round(Int64, 0.8 * nframes)),
        # random target
        :parent =>
            uniform_discrete(1, round(Int64, 0.5 * wm.n_dots)),
        # speed to move across
        :speedx => normal(wm.vel, 1.0),
        :speedy => normal(wm.vel, 1.0),
    )
    Dict(
        :positions => steps,
        :gorilla =>  gorilla,
    )
end

function main()
    # TODO: pin wm parameters
    wm = SchollWM(
        ;
        n_dots = 8,
        dot_radius = 10.0,
        area_width = 720.0,
        area_height = 480.0,
        vel=4.0,
        vel_min = 2.0,
        vel_max = 5.5,
        vel_step = 0.20,
        vel_prob = 0.20
    )
    data = Dict()
    data[:trials] = [gen_scene(wm, frames) for _ = 1:nscenes]
    data[:manifest] = Dict(:ntrials => ntrials,
                           :duration => duration,
                           :fps => fps,
                           :frames => frames)
    open(out_path, "w") do io
        JSON3.write(io, data)
    end
end;


main();
