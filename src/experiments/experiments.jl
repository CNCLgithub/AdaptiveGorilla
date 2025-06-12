export Experiment, get_obs,
    render_agent_state, run_analyses

abstract type Experiment end

"""
Runs the agent on the next tick of the experiment and
returns some measurements of the agent's interal state.
"""
function step_agent! end

include("most.jl")
include("target_ensemble.jl")
# include("safari.jl")
