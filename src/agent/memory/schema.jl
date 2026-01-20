using SHA
using UUIDs
using Printf: @sprintf
using UnicodePlots: barplot

"""
    LogEntry

Keeps track of transformations in the registry.
"""
struct LogEntry
    prev_schema_id::UInt64
    next_schema_id::UInt64
    time::Int
    operation::Symbol
end

"""
    GranularitySpec
    
A representation in your domain. Each has a unique identity.
Can be either:
- Atomic: an original individual element
- Ensemble: result of merging other representations
"""
abstract type GranularitySpec end

struct AtomicRep <: GranularitySpec
    id::UUID
    index::Int  # Original position in the system

    function AtomicRep(index::Int)
        new(uuid4(), index)
    end
end

struct EnsembleRep <: GranularitySpec
    id::UUID
    size::Int
    components::Set{UUID}  # IDs of representations that were merged to create this
    function EnsembleRep(components::Set{UUID})
        n = length(components)
        n < 2 && error("Ensemble must have at least 2 components")
        new(uuid4(), n, components)
    end
    function EnsembleRep(n::Int)
        n < 2 && error("Ensemble must have at least 2 components")
        new(uuid4(), n, Set{UUID}())
    end
    function EnsembleRep(n::Int, components::Set{UUID})
        lc = length(components)
        if lc !== n && !isempty(components)
            error("Number of components must either match `n` or be empty")
        end
        new(uuid4(), n, components)
    end
end

Base.hash(r::AtomicRep) = hash(r.id)
Base.hash(r::EnsembleRep) = hash(r.id)
Base.:(==)(r1::AtomicRep, r2::AtomicRep) = r1.id == r2.id
Base.:(==)(r1::EnsembleRep, r2::EnsembleRep) = r1.id == r2.id
Base.:(==)(r1::GranularitySpec, r2::GranularitySpec) = false

is_ambiguous(r::EnsembleRep) = isempty(r.components)

function merge_reps(a::AtomicRep, b::AtomicRep)
    EnsembleRep(Set{UUID}((a.id, b.id)))
end

function merge_reps(a::AtomicRep, b::EnsembleRep)
    components = is_ambiguous(b) ? b.components :
        union(Set{UUID}([a.id]), b.components)
    EnsembleRep(b.size + 1, components)
end

merge_reps(a::EnsembleRep, b::AtomicRep) = merge_reps(b, a)

function merge_reps(a::EnsembleRep, b::EnsembleRep)
    is_ambiguous(a) || is_ambiguous(b) ?
        EnsembleRep(a.size + b.size, Set{UUID}()) :
        EnsembleRep(union(a.components, b.components))
end

"""
    GranularitySchema
    
A schema is a specific set of representations that collectively describe the system.
Each schema has a unique identity based on the exact set of representation IDs.
"""
struct GranularitySchema
    id::UInt64
    representations::Vector{UUID}  # Ordered list of representation IDs
    n::Int  # Number of original atomic elements
    natomic::Int
    function GranularitySchema(reps::Vector{UUID}, n::Int, natomic::Int)
        id = _gschema_hash(reps, n)
        new(id, reps, n, natomic)
    end
end

function _gschema_hash(uuids::Vector{UUID}, n::Int)
    nr = length(uuids)
    id = hash(n)
    @inbounds for i = 1:nr
        id = hash(uuids[i], id)
    end
    return id
end

natomic(gs::GranularitySchema) = gs.natomic
nrep(gs::GranularitySchema) = length(gs.representations)
nensemble(gs::GranularitySchema) = nrep(gs) - natomic(gs)

"""
    SchemaRegistry
    
Tracks schemas (which are sets of representations) and their evolution.
"""
mutable struct SchemaRegistry 
    "Storage of representation formats"
    reps::Dict{UUID, GranularitySpec}
    "Hashed storage of schemas"
    schemas::Dict{UInt64, GranularitySchema}
    "GranularitySpec lineage"
    parents::Dict{UUID, UUID}
    "Log of operations in registry"
    transitions::Vector{LogEntry}
    
    "Initialize a registry with `n` atomic representations"
    function SchemaRegistry(n::Int)
        reps = Dict{UUID, GranularitySpec}()
        repv = Vector{UUID}(undef, n)
        for i = 1:n
            rep = AtomicRep(i)
            repv[i] = rep.id
            reps[rep.id] = rep
        end
        schema = GranularitySchema(repv, n, n)
        new(reps,
            Dict{UInt64, GranularitySchema}(schema.id => schema),
            Dict{UUID, UUID}(),
            LogEntry[])
    end
end

function is_atomic(registry::SchemaRegistry, rep_id::UUID)
    rep = registry.reps[rep_id]
    isa(rep, AtomicRep)
end

function count_atomic(registry::SchemaRegistry, schema_id::UInt64)
    count_atomic(registry, registry.schemas[schema_id].representations)
end

function count_atomic(registry::SchemaRegistry, reps::Vector{UUID})
    c = 0
    for rep_uuid = reps
        rep = registry.reps[rep_uuid]
        if isa(rep, AtomicRep)
            c += 1
        end
    end
    return c
end

function register_schema!(registry::SchemaRegistry, schema::GranularitySchema)
    for rep_id = schema.representations
        if !haskey(registry.reps, rep_id)
            error("Attempted to register schema $(schema.id) with orphan rep $(rep_id)")
        end
    end
    registry.schemas[schema.id] = schema
    return nothing
end

function merge_in_schema!(registry::SchemaRegistry, 
                          current_schema_id::UInt64,
                          pair_idx::Int,
                          time::Int)
    schema = registry.schemas[current_schema_id]
    nr = length(schema.representations)
    a, b = combination(nr, 2, pair_idx)
    merge_in_schema!(registry, current_schema_id, a, b, time)
end

"""
Merge representations within a schema to create a new schema.
"""
function merge_in_schema!(registry::SchemaRegistry, 
                          current_schema_id::UInt64,
                          a_idx::Int,
                          b_idx::Int,
                          time::Int)

    a_idx, b_idx = minmax(a_idx, b_idx)
    # Get representation IDs to merge
    g = registry.schemas[current_schema_id]
    a_uuid = g.representations[a_idx]
    b_uuid = g.representations[b_idx]
    
    local ensemble_id::UUID
    if (haskey(registry.parents, a_uuid) &&
        haskey(registry.parents, b_uuid) &&
        registry.parents[a_uuid] == registry.parents[b_uuid])
        # Check if representations share a parent in the registry
        ensemble_id = registry.parents[a_uuid]
    else
        # Create and register merged ensemble
        a = registry.reps[a_uuid]
        b = registry.reps[b_uuid]
        ensemble = merge_reps(a, b)
        registry.reps[ensemble.id] = ensemble 
        ensemble_id = ensemble.id
    end
    
    # Create new schema with merged representation
    g = registry.schemas[current_schema_id]
    nr = nrep(g)
    new_reps = Vector{UUID}(undef, nr - 1)
    c = 1
    for i = 1:nr
        if i == a_idx || i == b_idx
            continue
        else
            new_reps[c] = g.representations[i]
            c += 1
        end
    end
    new_reps[end] = ensemble_id

    new_schema =
        GranularitySchema(new_reps, g.n, count_atomic(registry, new_reps))
    register_schema!(registry, new_schema)
    
    # Record transition
    push!(registry.transitions,
          LogEntry(current_schema_id,
                   new_schema.id,
                   time, :merge))
    
    return new_schema.id
end

"""
Split an ensemble in a schema to create a new schema.
This is STOCHASTIC and creates new representations.
"""
function split_in_schema!(registry::SchemaRegistry,
                          current_schema_id::UInt64,
                          index_to_split::Int,
                          time::Int)
    current_schema = registry.schemas[current_schema_id]
    ensemble_id = current_schema.representations[index_to_split]
    
    # Verify it's an ensemble
    if !(registry.reps[ensemble_id] isa EnsembleRep)
        describe_schema(registry, current_schema_id)
        msg = "Attempting to split Atomic at $(index_to_split), id: $(ensemble_id)"
        msg *= "\n === REGISTRY TRANSACTIONS === \n"
        msg *= format_registry_transitions(registry)
        error(msg)
    end
    
    # Perform split, registering new reps
    individual_id, remaining_id, reduced =
        split_ensemble!(registry, ensemble_id)
    
    # Create new schema with split representations
    nr = nrep(current_schema)
    ns = natomic(current_schema)
    new_reps = Vector{UUID}(undef, nr + 1)

    # copy over atomics
    for i = 1:ns
        new_reps[i] = current_schema.representations[i]
    end

    # copy over ensembles
    ensemble_shift = ifelse(reduced, 2, 1)
    c = ns + ensemble_shift + 1
    for i = (ns + 1):nr
        if reduced && i == index_to_split
            c -= 1
        else
            new_reps[c] = current_schema.representations[i]
        end
        c += 1
    end
    # new reps
    new_reps[ns + 1] = individual_id
    remaining_index = reduced ? ns + 2 : index_to_split + 1
    new_reps[remaining_index] = remaining_id

    new_schema =
        GranularitySchema(new_reps, current_schema.n, ns + ensemble_shift)
    register_schema!(registry, new_schema)

    # Record transition
    push!(registry.transitions,
          LogEntry(current_schema_id, new_schema.id, time, :split))
    
    return new_schema.id
end


"""
Split an ensemble representation.
Returns: (individual_id, remaining_ensemble_id)

The only guarantee: merging the results recovers the original ensemble.
"""
function split_ensemble!(registry::SchemaRegistry, 
                         ensemble_id::UUID)

    ensemble = registry.reps[ensemble_id]
    if !(ensemble isa EnsembleRep)
        error("Cannot split Atomic with ID $(ensemble_id)")
    end

    # Create two NEW representations with fresh identities
    # These are NOT any of the original components
    individual = AtomicRep(-1)  # -1 indicates derived, not original
    registry.reps[individual.id] = individual

    reduced = ensemble.size <= 2 # splitting ensemble into two atomics
    local remaining_id::UUID
    # Create and register the other representation
    if reduced
        other = AtomicRep(-1)  # -1 indicates derived, not original
        registry.reps[other.id] = other
        remaining_id = other.id
    else
        e =  EnsembleRep(ensemble.size - 1) # Will have empty components
        registry.reps[e.id] = e
        remaining_id = e.id
    end

    # Record lineage
    registry.parents[individual.id] = ensemble_id
    registry.parents[remaining_id] = ensemble_id
    
    (individual.id, remaining_id, reduced)
end

"""
Get representation details for a schema.
"""
function describe_schema(registry::SchemaRegistry, schema_id::UInt64)
    schema = registry.schemas[schema_id]
    println("=== Schema $(schema_id) ===")
    print("[Meta data: n => $(schema.n) | natomic => $(natomic(schema)) |")
    print(" natomic (registry) => ")
    print(count_atomic(registry, schema.representations))
    println(" ]")

    
    println("[Representation IDs]:")
    for (i, rep_id) in enumerate(schema.representations)

        println("  <$(i)>: " * pretty_rep(registry, rep_id))
    end
end

function get_initial_schema(registry::SchemaRegistry)
     first(keys(registry.schemas))
end

function format_registry_transitions(registry::SchemaRegistry)
    msg = ""
    for entry in registry.transitions
        msg *= "Time $(entry.time): $(string(entry.prev_schema_id)[1:8]) "
        msg *= "=> $(entry.operation) => $(string(entry.next_schema_id)[1:8])\n"
    end
    return msg
end

# Example usage demonstrating the path-dependent nature
function example_usage()
    println("=== Path-Dependent GranularitySpec Tracking ===\n")
    
    registry = SchemaRegistry(5)
    
    # Get initial schema [a, b, c]
    schema1 = get_initial_schema(registry)
    println("Schema 1: Initial state")
    describe_schema(registry, schema1)
    
    # Merge all to create ensemble [d]
    schema2 = merge_in_schema!(registry, schema1, 1, 2, 1)
    println("\nSchema 2: After merging all elements")
    describe_schema(registry, schema2)
    
    # Split to get [f, g] - these are NEW representations, not a,b,c
    schema3 = split_in_schema!(registry, schema2, 4, 2)
    println("\nSchema 3: After stochastic split")
    describe_schema(registry, schema3)
    
    # Merge back - recovers the ensemble involutively
    schema4 = merge_in_schema!(registry, schema3, 4, 5, 3)
    println("\nSchema 4: After merging split results")
    describe_schema(registry, schema4)
    
    println("\n=== Hashes (for timeseries indexing) ===")
    for (i, sid) in enumerate([schema1, schema2, schema3, schema4])
        println("Schema $i: $(sid)")
    end
    
    # println("\n=== Timeseries Data ===")
    # for (schema_id, data_points) in registry.timeseries
    #     if !isempty(data_points)
    #         println("Schema $(string(schema_id)[1:8])...: $(length(data_points)) data points")
    #     end
    # end

    println("\n=== Log Data ===")
    println(format_registry_transitions(registry))

    return registry
end

# registry = example_usage()

# Exended API

function init_schema(registry::SchemaRegistry)
    if length(registry.schemas) !== 1
        error("Registry already diverged")
    end
    first(collect(keys(registry.schemas)))
end

function is_valid_schema(registry::SchemaRegistry,
                         t::InertiaTrace,
                         schema_id::UInt64)
    g = registry.schemas[schema_id]
    check = Int(object_count(t)) == g.n &&
        natomic(g) == single_count(t) == count_atomic(registry, schema_id)

    return check
end

function ammend_schema!(registry::SchemaRegistry,
                        t::InertiaTrace,
                        schema_id::UInt64)
    g = registry.schemas[schema_id]
    oc = Int(object_count(t))
    new_schema_id = if g.n == oc - 1
        # Birth - one more single
        schema_birth_at!(registry, t, schema_id)
    elseif g.n == oc + 1
        # Death - one less single
        schema_death_at!(registry, t, schema_id)
    else
        print_granularity_schema(t)
        describe_schema(registry, schema_id)
        error("Unrecognized schema divergence")
    end

    if !is_valid_schema(registry, t, new_schema_id)
        print_granularity_schema(t)
        describe_schema(registry, schema_id)
        describe_schema(registry, new_schema_id)
        error("Schema ammendment failed")
    end
    
    return new_schema_id
end


function schema_death_at!(registry::SchemaRegistry,
                          t::InertiaTrace,
                          schema_id::UInt64)
    nsingle = single_count(t)
    death_idx = find_death_idx(t)
    g = registry.schemas[schema_id]
    if death_idx === 0
        # Sometimes, the MAP changes and never had a birth
        # In this case, we can remove the last atomic rep.
        death_idx = natomic(g)
        # print_granularity_schema(t)
        # describe_schema(registry, schema_id)
        # display(get_choices(t))
        # error("No death found in trace")
    end
    nreps = length(g.representations)
    reps = Vector{UUID}(undef, nreps - 1)
    for i = 1:nreps
        rep_id = g.representations[i]
        if i < death_idx
            reps[i] = rep_id
        elseif i > death_idx
            reps[i-1] = rep_id
        end
    end
    new_schema = GranularitySchema(reps, g.n - 1, nsingle)
    register_schema!(registry, new_schema)

    return new_schema.id
end

function find_death_idx(trace::InertiaTrace)
    t, wm, istate = get_args(trace)
    death_idx = 0
    for k = t:-1:1
        # death occured here
        if trace[:kernel => k => :bd => :i] == 3
            death_idx = trace[:kernel => k => :bd => :switch => :dead]
            break
        end
    end
    return death_idx
end

function schema_birth_at!(registry::SchemaRegistry,
                          t::InertiaTrace,
                          schema_id::UInt64)

    birth_idx = single_count(t)
    g = registry.schemas[schema_id]
    nreps = length(g.representations)
    reps = Vector{UUID}(undef, nreps + 1)
    for i = 1:nreps
        rep_id = g.representations[i]
        shift = ifelse(i < birth_idx, 0, 1)
        reps[i + shift] = rep_id
    end
    # Generate and register new atomic
    birthed = AtomicRep(g.n + 1) # REVIEW: could be ambiguous
    reps[birth_idx] = birthed.id
    registry.reps[birthed.id] = birthed
    
    # Initialize and register new schema
    new_schema =
        GranularitySchema(reps, Int(object_count(t)), birth_idx)
    register_schema!(registry, new_schema) 

    return new_schema.id
end

function transform_schema!(registry::SchemaRegistry,
                           schema_id::UInt64,
                           time::Int,
                           tr::InertiaTrace,
                           cm::ChoiceMap)

    local new_schema_id::UInt64

    # No change
    new_schema_id =
        if cm[:s0 => :nsm] == 1
            schema_id
        elseif cm[:s0 => :nsm] == 2
            # Split k'th ensemble
            split_idx = single_count(tr) + cm[:s0 => :state => :idx]
            split_in_schema!(registry, schema_id, split_idx, time)
        else
            # Merge a and b
            pair_idx = cm[:s0 => :state => :pair]
            merge_in_schema!(registry, schema_id, pair_idx, time)
        end

    return new_schema_id
end

function guess_schema_set(tr::InertiaTrace)
    sc = single_count(tr)
    sc != representation_count(tr) &&
        error("Could not guess schema from InertiaTrace")
    return sc
end


function memory_schema_set(perception::MentalModule{<:HyperFilter})
    visp, visstate = mparse(perception)
    chain = visstate.chains[1]
    trace = retrieve_map(chain)
    guess_schema_set(trace)
end

function reconstitute_deltas(delta_integral::Dict{UUID, Float64},
                             registry::SchemaRegistry,
                             schema_id::UInt64)

    schema = registry.schemas[schema_id]
    n = nrep(schema)
    v = Vector{Float64}(undef, n)
    for (i, rep_id) = enumerate(schema.representations)
        v[i] = get(delta_integral, rep_id, -Inf)
    end
    return v
end

function accumulate_deltas!(rep_scores::Dict{UUID, Float64},
                            registry::SchemaRegistry,
                            schema_id::UInt64,
                            deltas::Vector{Float64})
    schema = registry.schemas[schema_id]
    for (i, rep_id) = enumerate(schema.representations)
        delta = deltas[i]
        rep_scores[rep_id] =
            logsumexp(delta, get(rep_scores, rep_id, -Inf))
    end
    return nothing
end

function pretty_rep(registry::SchemaRegistry, rep_id::UUID)
    rep = registry.reps[rep_id]
    short_id = string(rep_id)[1:8]
    name = if rep isa AtomicRep
        if rep.index >= 0
            @sprintf "⚛ (#%2d) | %s" rep.index short_id
        else
            parent = string(registry.parents[rep_id])[1:8]
            @sprintf "⚛ (  ?) | %s | 🔗 %s" short_id parent
        end
    else
        @sprintf "𝛌 ( %2d) | %s" rep.size short_id
    end
end


function plot_rep_weights(registry::SchemaRegistry,
                          rep_scores::Dict{UUID, Float64},
                          k = 8)

    rep_scores = filter(x -> !isinf(x.second), rep_scores)
    names = collect(keys(rep_scores))
    xs = collect(values(rep_scores))
    cap = maximum(xs)
    if length(rep_scores) > k
        inds = partialsortperm(xs, 1:k; rev = true)
        names = names[inds]
        xs = xs[inds]
    end
    xs .-= cap
    xs .*= -1
    clamp!(xs, 0.0, 1E4)
    names = map(names) do rep_id
        pretty_rep(registry, rep_id)
    end
    display(barplot(names, xs, xlabel = "- ℧", title = "=== Rep Scores ==="))
    return nothing
end

function plot_schema_scores(schema_map::Vector{UInt64},
                            schema_scores::Vector{Float64})
    display(barplot(schema_map, -1 .* schema_scores, xlabel = "- ℧",
                    title = "=== Schema Scores ==="))
    return nothing
end
