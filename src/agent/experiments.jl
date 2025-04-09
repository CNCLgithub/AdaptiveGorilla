function planner_expectation(pm::MentalModule{T}) where {T<:CollisionCounter}
    planner, state = mparse(pm)
    state.expectation
end


function run_analyses(exp::Gorillas, agent::Agent)
    gorilla_p = estimate_marginal(agent.perception,
                                  detect_gorilla, ())
    col_p = planner_expectation(agent.planning)
    Dict(:gorilla_p => gorilla_p,
         :collision_p => col_p)
end
