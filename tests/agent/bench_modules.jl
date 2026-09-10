# ------------------------------------------------------------------------------
# Module-level timing benchmark for the study3 (LoadCurve) models.
#
# NOT an official test. Sequentially runs each model for several steps and
# attributes wall time + allocations to each cognitive module using
# TimerOutputs, so the cost decomposition of e.g. MO vs. JA can be compared
# on identical observation sequences.
#
# Usage:
#   julia --project=. tests/agent/bench_modules.jl [model ...]
#
#   Default models: ["mo", "ja"]
#   Datasets are expected under DPATH (override with env var AG_DATASET).
# ------------------------------------------------------------------------------

using TimerOutputs
using Printf: @sprintf
using AdaptiveGorilla
import AdaptiveGorilla as AG

TO = TimerOutput()   # global timer; snapshot per model with copy(TO)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REPO_ROOT   = "/project"                 # tests/ -> repo root
MODEL_DIR   = joinpath(REPO_ROOT, "scripts", "study3", "params")
WM_TOML     = joinpath(MODEL_DIR, "wm.toml")

MODELS      = isempty(ARGS) ? ["mo", "ja"] : ARGS
DATASET     = get(ENV, "AG_DATASET", "study3")
DPATH       = get(ENV, "AG_DPATH", "/spaths/datasets/$(DATASET)/dataset.json")

FRAMES      = 240
WARMUP      = 40        # frames before measurement begins
TRIAL_IDX   = 1
NTARGET     = 4
NDISTRACTOR = 8

# -----------------------------------------------------------------------------
# Experiment construction (mirrors scripts/study3/run_model.jl)
# -----------------------------------------------------------------------------
wm = load_wm_from_toml(WM_TOML; object_rate = Float64(NTARGET + NDISTRACTOR))
exp = LoadCurve(wm, DPATH, TRIAL_IDX, FRAMES, NTARGET, NDISTRACTOR)

@assert isfile(DPATH) "Dataset not found: $DPATH"

# -----------------------------------------------------------------------------
# Timed stepping: each module is attributed inside the same sequential run,
# because module state mutates in place across frames (no isolation possible).
# -----------------------------------------------------------------------------
function run_model_timed!(exp, model::String; warmup::Int = WARMUP)
    agent = load_agent(joinpath(MODEL_DIR, "$(model).toml"), exp.init_query)
    to = TimerOutput()   # one per model: reset_timer! would clobber shared data

    # Warm-up: lets Adaptive-Compute / MO regranularization engage before
    # the timed region. Untimed; module states still evolve normally.
    for t = 1:warmup
        obs = get_obs(exp, t)
        AG.module_step!(agent.perception, t, obs)
        AG.module_step!(agent.attention,  t, agent.perception)
        AG.module_step!(agent.planning,   t, agent.attention, agent.perception)
        AG.module_step!(agent.memory,     t, agent.perception)
    end

    reset_timer!(TO)
    GC.gc()
    for t = (warmup + 1):(FRAMES - 1)
        obs = get_obs(exp, t)
        @timeit TO "perception" AG.module_step!(agent.perception, t, obs)
        @timeit TO "attention"  AG.module_step!(agent.attention,  t, agent.perception)
        @timeit TO "planning"   AG.module_step!(agent.planning,   t, agent.attention, agent.perception)
        @timeit TO "memory"     AG.module_step!(agent.memory,     t, agent.perception)
    end
    return deepcopy(TO)
end

results = Dict{String, TimerOutput}()
for model in MODELS
    @info "Benchmarking model: $(model)"
    results[model] = run_model_timed!(exp, model)
end

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
for model in MODELS
    println("\n================ $(model) (post-warmup) ================")
    print(results[model])
end

# Per-frame averages, for cross-model comparison
if length(MODELS) > 1
    n_frames = FRAMES - 1 - WARMUP
    println("\n================ Per-frame averages =================")
    for model in MODELS
        d = TimerOutputs.todict(results[model])
        for name in ("perception", "attention", "planning", "memory")
            entry = d[name]
            time_ns     = entry["time"]       # nanoseconds (cumulative)
            alloc_bytes = entry["allocated"]  # bytes (cumulative)
            println(@sprintf("%-4s %-11s  %8.3f ms/frame  %10.1f KB/frame",
                             model, name,
                             time_ns / n_frames * 1e-6,
                             alloc_bytes / n_frames / 1024))
        end
    end
end
