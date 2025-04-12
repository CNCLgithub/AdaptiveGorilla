export IncrementalQuery,
    increment

struct IncrementalQuery <: Gen_Compose.Query
    model::Gen.GenerativeFunction
    constraints::Gen.ChoiceMap
    args::Tuple
    argdiffs::Tuple
    step::Int
end

function increment(q::IncrementalQuery,
                   cm::Gen.ChoiceMap,
                   new_args::Tuple)
    args = step_increment_args(new_args, q.args, q.argdiffs)
    IncrementalQuery(
                     q.model,
                     cm,
                     args,
                     q.argdiffs,
                     q.step + 1
                    )
end

function step_increment_args(next_args::Tuple, args::Tuple, argdiffs::Tuple)
    new_args = collect(args)
    ndiff = count(x -> isa(x, UnknownChange), argdiffs)
    @assert length(next_args) == ndiff "Length missmatch for diff args"
    c = 1
    @inbounds for i = 1:length(args)
        if isa(argdiffs[i], UnknownChange)
            new_args[i] = next_args[c]
            c += 1
        end
    end
    return Tuple(new_args)
end
