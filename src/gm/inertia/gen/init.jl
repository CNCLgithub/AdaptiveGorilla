################################################################################
# Object state prior
################################################################################
@gen static function state_prior(wm::InertiaWM)

    xs, ys = object_bounds(wm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0., 2*pi), :ang)
    mag = @trace(normal(wm.vel, 1e-2), :std)

    pos = S2V(x, y)
    vel = S2V(mag*cos(ang), mag*sin(ang))

    result::Tuple{S2V, S2V} = (pos, vel)
    return result
end

################################################################################
# Birth
################################################################################

@gen static function birth_single(wm::InertiaWM, rate::Float64)
    ms = materials(wm)
    nm = length(ms)
    mws = Fill(1.0 / nm, nm)
    midx = @trace(categorical(mws), :material)
    material = ms[midx]
    loc, vel = @trace(state_prior(wm), :state)
    baby::InertiaObject = InertiaSingle(material, loc, vel)
    return baby
end

@gen static function birth_ensemble(wm::InertiaWM, rate::Float64)
    # assuming | materials | = 2
    # NOTE: could replace with dirichlet
    plight ~ beta(0.5, 0.5)
    mws = S2V(plight, 1.0 - plight)
    loc, vel = @trace(state_prior(wm), :state)
    var ~ inv_gamma(wm.ensemble_shape, wm.ensemble_scale)
    ensemble::InertiaObject =
        InertiaEnsemble(rate, mws, loc, var, vel)
    return ensemble
end

################################################################################
# Prior over initial state
################################################################################

# @gen static function inertia_init(wm::InertiaWM)
#     n ~ poisson(wm.object_rate) # total number of objects
#     ns ~ binom(n, wm.irate) # number of individuals
#     singles ~ Map(birth_single)(Fill(wm, ns))
#     rem = n - ns
#     # the number of ensembles
#     ne ~ poisson(1)
#     # the size of each ensemble
#     _ne = max(min(n - ns, ne), 1.0)
#     ws ~ dirichlet(_ne, 1.0)
#     ensembles ~ Map(birth_ensemble)(ws, _ne)
#     state::InertiaState = InertiaState(wm, singles, ensembles)
#     return state
# end




struct ProdNode
    split::Float64
    cardinality::Int
    wm::InertiaWM
end

cardinality(x::ProdNode) = x.cardinality
split_weight(x::ProdNode) = cardinality(x) == 1 ? 0.0 : x.split
wm(x::ProdNode) = x.wm

function split_node(x::ProdNode, ratio::Float64)
    c = cardinality(x)
    result = Vector{ProdNode}(undef, 2)
    k1 = max(floor(Int64, c * ratio), 1)
    k2 = max(1, c - k1)
    result[1] = ProdNode(x.split, k1, x.wm)
    result[2] = ProdNode(x.split, k2, x.wm)
    return result
end

struct AggNode
    pnode::ProdNode
    objects::Vector{InertiaObject}
end

objects(x::AggNode) = x.objects

function extract_objects(v::Vector{AggNode})
    reduce(vcat, map(objects, v))
end

function InertiaState(wm::InertiaWM, anode::AggNode)
    objs = objects(anode)
    InertiaState(filter(x -> isa(x, InertiaSingle), objs),
                 filter(x -> isa(x, InertiaEnsemble), objs))
end

@gen static function inertia_prod(x::ProdNode)
    split ~ bernoulli(split_weight(x))
    ratio ~ beta(0.5, 0.5)
    children::Vector{ProdNode} = split ? split_node(x, ratio) : ProdNode[]
    result = Production(x, children)
    return result
end

const object_switch = Switch(birth_single, birth_ensemble)

@gen function inertia_agg(x::ProdNode, children::Vector{AggNode})
    local objs::Vector{InertiaObject}
    if isempty(children)
        c = cardinality(x)
        idx = min(c, 2)
        obj ~ object_switch(idx, wm(x), float(c))
        objs = [obj]
    else
        objs = extract_objects(children)
    end

    agg::AggNode = AggNode(x, objs)
    return agg
end

const recurse_prior = Recurse(inertia_prod,
                              inertia_agg,
                              2, # one split at a time
                              ProdNode,# U (production to children)
                              ProdNode,# V (production to aggregation)
                              AggNode) # W (aggregation to parents)

@gen static function inertia_init(wm::InertiaWM)
    n ~ poisson(wm.object_rate) # total number of objects
    objects ~ recurse_prior(ProdNode(0.5, n, wm), 1)
    state::InertiaState = InertiaState(wm, objects)
    return state
end
