
function run_analyses(exp::Gorillas, agent::Agent)
    gorilla_p = estimate_marginal(agent.perception,
                                  detect_gorilla, ())
    Dict(:gorilla_p => gorilla_p)
end
