# Benchmark: mRFS cost of iso elements vs. poisson elements over masks
#
# NOT an official test. Exploratory benchmark comparing the runtime / memory
# of evaluating `DetectionRFS` (mRFS over detection masks) with:
#   (1) 8 isomorphic elements (CIsoElement, one per object)
#   (2) 2-3 poisson elements (CPoissonElement, ensembles)
#
# Hypothesis: configuration (2) is slower than (1), despite having fewer
# elements, because poisson cardinalities blow up the partition enumeration.
#
# Run:  julia --project=. tests/bench_rfs_granularity.jl
# Note: BenchmarkTools must be in the project environment:
#   julia --project=. -e 'using Pkg; Pkg.add("BenchmarkTools")'

using AdaptiveGorilla
import AdaptiveGorilla as AG
using Gen
using GenRFS
using BenchmarkTools
using Statistics
using LinearAlgebra

using StaticArrays: SVector

# ---------------------------------------------------------------------------
# World-model parameters (defaults matching InertiaWM)
# ---------------------------------------------------------------------------
single_noise      = 10.0
single_size       = 1.0
material_noise    = 0.5
penalty           = -3000.0

STEPS  = 100   # partition enumeration steps (RFGM estimator arg)
TEMP   = 1.0

# element geometry: 8 objects along a horizontal band
POS = [SVector(-70.0 + 20.0 * (i - 1), 0.0) for i in 1:32]  # sized for the n-sweep
SINGLE_VAR = single_size * single_noise

# ---------------------------------------------------------------------------

"""`n` isomorphic elements (one per object position)."""
function iso_elements(n::Int)
    [CIsoElement{AG.Detection}(AG.detect,
                               (POS[i], SINGLE_VAR,
                                Float64(Int(AG.Light)), material_noise),
                               penalty)
     for i in 1:n]
end

"""2-3 poisson elements covering `n` detections.

Detections are partitioned evenly among `split` ensembles; rates match the
partition sizes.  Variance uses the ensemble-spread semantics
(VAR_POS * separation), like `apply_merge`.
"""
function poisson_elements_n(split::Int, n::Int)
    @assert split in (2, 3)
    # even partition of 1:n into `split` contiguous chunks
    base = n ÷ split; rem = n % split
    starts = vcat(0, accumulate(+, [rem >= j ? base + 1 : base for j in 1:split]))
    ranges = [(starts[j] + 1):starts[j + 1] for j in 1:split]
    rates = Float64[length(r) for r in ranges]
    es = Vector{RandomFiniteElement{AG.Detection}}(undef, split)
    for (j, r) in enumerate(rates)
        # center of this ensemble's members; spread ~ separation of members
        members = ranges[j]
        c = sum(POS[members]) / length(members)
        var = AG.VAR_POS * maximum(norm(POS[i] - c) for i in members)
        es[j] = CPoissonElement{AG.Detection}(r, AG.detect_mixture,
                                              (c, var, 0.5, material_noise),
                                              penalty)
    end
    es
end

# ---------------------------------------------------------------------------
# Reference observations.
#
# NOTE: DetectionRFS is an RFGM (a generative function), not a Distribution.
# It is called as `xs ~ DetectionRFS(es)` inside the model; scoring an
# observed set is done with Gen.generate + a choicemap of the xs, whose
# trace score is the mRFS log-likelihood (see GenRFS/src/rfgm.jl).
#
# For the benchmark we hold the OBSERVATION FIXED: the same 8 detections
# (one per object position) are scored under every element configuration,
# so all three timings address the same inference problem.
# ---------------------------------------------------------------------------
xs_fixed = [AG.Detection(POS[i][1], POS[i][2], 2.0) for i in 1:8]

"""Score observed xs under the mRFS over elements es."""
function score_obs(es, xs)
    choices = choicemap()
    for (i, x) in enumerate(xs)
        choices[i] = x
    end
    tr, _ = Gen.generate(AG.DetectionRFS, (es,), choices)
    Gen.get_score(tr)
end

"""Fresh sample from the RFGM."""
sample_rfgm(es) = Gen.get_retval(Gen.simulate(AG.DetectionRFS, (es,)))

iso_es  = iso_elements(8)
poi2_es = poisson_elements_n(2, 8)
poi3_es = poisson_elements_n(3, 8)

# sanity: all configurations must give finite score on the SAME observations
@assert isfinite(score_obs(iso_es,  xs_fixed))
@assert isfinite(score_obs(poi2_es, xs_fixed))
@assert isfinite(score_obs(poi3_es, xs_fixed))

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
# Scoring the SAME fixed observation under each configuration
SUITE = BenchmarkGroup()
SUITE["score"]["iso-8"]      = @benchmarkable score_obs($iso_es,  $xs_fixed)
SUITE["score"]["poisson-2"]  = @benchmarkable score_obs($poi2_es, $xs_fixed)
SUITE["score"]["poisson-3"]  = @benchmarkable score_obs($poi3_es, $xs_fixed)

# Fresh sampling of an observation (cost of the generative step)
SUITE["sample"]["iso-8"]     = @benchmarkable sample_rfgm($iso_es)
SUITE["sample"]["poisson-2"] = @benchmarkable sample_rfgm($poi2_es)
SUITE["sample"]["poisson-3"] = @benchmarkable sample_rfgm($poi3_es)

tune!(SUITE)
results = run(SUITE, verbose = true, samples = 100)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
using Printf

@printf("\n================ Summary (median) ================\n")
@printf("%-14s %12s %12s %12s %12s\n",
        "config", "time µs", "gc-time µs", "allocs", "bytes")
for k in ["iso-8", "poisson-2", "poisson-3"]
    trial = results["score"][k]
    est = median(trial)          # Trial -> TrialEstimate
    @printf("%-14s %12.2f %12.2f %12d %12d\n",
            k,
            BenchmarkTools.time(est) / 1e3,
            BenchmarkTools.gctime(est) / 1e3,
            BenchmarkTools.allocs(est), BenchmarkTools.memory(est))
end

# Hypothesis check: log-ratio of medians (positive => poisson slower)
for p in ["poisson-2", "poisson-3"]
    ratio = BenchmarkTools.time(median(results["score"][p])) /
            BenchmarkTools.time(median(results["score"]["iso-8"]))
    println("$p / iso-8 score-time ratio = $(round(ratio, digits=2))x")
end

# ---------------------------------------------------------------------------
# Element-count sweep: separate fixed per-call overhead from per-element
# marginal cost. If time = a + b*n fits, `a` is fixed overhead (trace /
# choicemap machinery) and `b` the per-element cost.
# ---------------------------------------------------------------------------
SWEEP_N = [1, 2, 4, 8, 16, 32]

SUITE_SWEEP = BenchmarkGroup()
for n in SWEEP_N
    SUITE_SWEEP["iso"]["n=$(n)"] =
        @benchmarkable score_obs(iso_elements(n), xs_fixed[1:n])
end
for n in SWEEP_N
    for split in (2, 3)
        SUITE_SWEEP["poisson-$(split)"]["n=$(n)"] =
            @benchmarkable score_obs(poisson_elements_n(split, $n), xs_fixed[$(1:n)])
    end
end

tune!(SUITE_SWEEP)
sweep = run(SUITE_SWEEP, verbose = false, samples = 50)

println("\n================ Element-count sweep (median) ================")
for kind in ["iso", "poisson-2", "poisson-3"]
    for n in SWEEP_N
        est = median(sweep[kind]["n=$(n)"])
        @printf("%-12s n=%3d  %10.2f µs  %8d bytes\n",
                kind, n, BenchmarkTools.time(est) / 1e3, BenchmarkTools.memory(est))
    end
end
