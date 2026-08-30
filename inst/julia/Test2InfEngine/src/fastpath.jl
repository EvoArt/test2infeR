# Fast path for the five sql-e2e bundle models: pre-tuned dense HMC.
#
# ## What this is
#
# The same Turing model, the same Mooncake gradient (with the hand-written
# `seq_loglik` rule from `seq_loglik_rule.jl`), sampled with **HMC at a fixed
# trajectory length and a dense mass matrix adapted offline** instead of NUTS
# adapting a diagonal metric from scratch on every run.
#
# Nothing here is a separate implementation of the model. The fast path is a
# sampler configuration, so a fit through it and a fit through NUTS are the same
# posterior by construction — only the route differs.
#
# ## Why the dense metric is the whole point
#
# rw1 sets `gamma_y = sigma_g * cumsum(raw)_y`, so every year effect shares a
# factor with `sigma_g`. That is a funnel, and it is exactly the off-diagonal
# structure a diagonal metric cannot represent — a diagonal metric can rescale
# axes but not rotate them. Measured on the real cohort by `tune/tune_hmc.jl`
# (1000 adapt + 1000 draws, sampling time only, which is what a pre-tuned run
# actually pays):
#
#   metric     sample time   min ESS   min ESS/s
#   diagonal    244-317 s    336-407    1.21-1.67
#   dense        47- 53 s    532-659   11.2 -12.3     ~8x
#
# Dense steps at eps ~0.29-0.36 against diagonal's ~0.047, at tree depth 4.0
# against 6.3 — roughly a sixth of the gradient evaluations per draw.
#
# Adapting a dense metric costs ~1.7x more warmup, which is why comparing on
# TOTAL wall time makes it look like a wash. That comparison is the wrong one
# here: warmup is paid once, offline, and shipped in `pretuned.jl`. Judge this
# path on post-warmup ESS/second.
#
# ## Scope, deliberately narrow
#
# The pre-tuned constants are specific to the five bundle variants AND to the
# cohort they were tuned on. `fastpath_pretuned` matches on the model variant
# and checks the parameter count; it cannot detect that you have pointed it at
# different data. Wrong data under a fixed metric samples BADLY rather than
# failing, so the fallback to NUTS is the safe default everywhere else, and
# `run_hmm_inference` only reaches for this path when explicitly asked.

# AdvancedHMC comes via Turing rather than as a direct dependency: Turing
# already depends on it, so this avoids adding a Project.toml entry (and the
# manifest churn that install_local causes) for something already loaded.
const AdvancedHMC = Turing.Inference.AdvancedHMC
const DenseEuclideanMetric = AdvancedHMC.DenseEuclideanMetric
using LinearAlgebra: Symmetric, cholesky, isposdef

include("pretuned.jl")

"""
Which of the five bundle variants a fit corresponds to, or `nothing`.

Identified by the test mask and whether Se/Sp are inferred — the two things
that actually change the parameter vector. Anything outside this set has no
tuned constants and must go through NUTS.
"""
function fastpath_variant(test_mask::AbstractVector{Bool}, infer_sesp::Bool)
    all_tests = all(test_mask)
    culture_only = test_mask == Bool[false, false, true, false, false, false]
    # DPP is column 4.
    no_dpp = test_mask == Bool[true, true, true, false, true, true]

    if all_tests
        return infer_sesp ? "all_inferred" : "all_fixed"
    elseif culture_only && !infer_sesp
        return "culture_only_fixed"
    elseif no_dpp
        return infer_sesp ? "no_dpp_inferred" : "no_dpp_fixed"
    else
        return nothing
    end
end

"""
    fastpath_pretuned(name, n_params, year_process)

The tuned constants for a variant, or `nothing` if there are none that match.

The parameter count is checked because it is cheap and catches the obvious
mismatch (a different number of years in the data). It is NOT a check that the
data is the cohort these were tuned on — nothing here can detect that.
"""
function fastpath_pretuned(name::AbstractString, n_params::Int, year_process::Symbol)
    year_process === :rw1 || return nothing   # tuned under rw1 only
    for p in PRETUNED_ALL
        if p.name == name
            p.n == n_params || return nothing
            p.year_process == 1 || return nothing
            return p
        end
    end
    return nothing
end

"""
    fastpath_permutation(vi, layout_order)

Map from the tuned metric's parameter order to the model's varinfo order.

`pretuned.jl` was generated against a flat layout
(alpha, log_sigma_g, gamma_raw, pi1_0, pi1_mult, [logit_Se, logit_Sp]).
Turing's `VarInfo` orders parameters by declaration in the `@model`, which is
not the same. A mass matrix applied under the wrong ordering is not an error —
it samples badly and silently — so this is derived from the varinfo rather than
assumed, and the caller must treat a `nothing` return as "do not use the metric".
"""
function fastpath_permutation(model, n_years::Int, S::Int, infer_sesp::Bool)
    # The two orderings are NOT the same when Se/Sp are inferred, and the
    # difference is silent: a mass matrix under the wrong ordering samples
    # badly rather than erroring. Measured on DynamicPPL 0.41.8:
    #
    #   fixed Se/Sp    alpha 1:4  sigma_g 5  gamma_raw 6:10  pi1_0 11  pi1_mult 12
    #   inferred       alpha 1:4  sigma_g 5  gamma_raw 6:10  Se 11:16  Sp 17:22
    #                  pi1_0 23  pi1_mult 24
    #
    # `pretuned.jl` puts Se/Sp LAST, after pi1_mult. So the fixed case is the
    # identity and the inferred case is a genuine permutation. Rather than
    # hardcode either, read the ranges from the model itself: the layout is a
    # DynamicPPL implementation detail and has changed before.
    vi = try
        DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)
    catch
        return nothing
    end
    ranges = try
        ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, vi)
        DynamicPPL.get_all_ranges_and_transforms(ldf)
    catch
        return nothing
    end

    # Flat-layout (tuned metric) offsets.
    i = 1
    want = Dict{Symbol,UnitRange{Int}}()
    want[:alpha]     = i:(i + S - 1);        i += S
    want[:sigma_g]   = i:i;                  i += 1
    want[:gamma_raw] = i:(i + n_years - 1);  i += n_years
    want[:pi1_0]     = i:i;                  i += 1
    want[:pi1_mult]  = i:i;                  i += 1
    if infer_sesp
        want[:Se] = i:(i + 5); i += 6
        want[:Sp] = i:(i + 5); i += 6
    end
    n = i - 1

    perm = zeros(Int, n)
    for (vn, rt) in pairs(ranges)
        sym = Symbol(split(string(vn), '[')[1])
        haskey(want, sym) || return nothing
        w = want[sym]
        r = rt.range
        length(r) == length(w) || return nothing
        for (a, b) in zip(r, w)
            (1 <= a <= n) || return nothing
            perm[a] = b
        end
    end
    any(iszero, perm) && return nothing
    sort(perm) == collect(1:n) || return nothing
    perm
end

"""
    fastpath_sample(model, pretuned, perm, n_draws, n_adapt, n_chains, seed, adtype)

Sample `model` with fixed-L HMC under the pre-tuned dense metric.

`perm` reorders the tuned inverse mass matrix into the model's own parameter
order. Warmup is still run — `n_adapt` steps of the same sampler — because the
tuned constants fix the geometry, not the starting point: the chain still has to
reach the typical set. It is short compared with NUTS' adaptation because
nothing is being learned.
"""
function fastpath_sample(model, pt::PretunedHMC, perm::Vector{Int},
                         n_draws::Int, n_adapt::Int, n_chains::Int,
                         seed::Int, adtype)
    # Guard the silent failure: applying NO permutation to a 70-parameter
    # inferred model is not an error anywhere downstream, it just samples badly.
    # So the sizes must agree before anything is indexed.
    length(perm) == size(pt.inv_mass, 1) ||
        error("permutation length $(length(perm)) does not match the tuned " *
              "metric's $(size(pt.inv_mass, 1)) parameters")
    sort(perm) == collect(1:length(perm)) ||
        error("permutation is not a permutation of 1:$(length(perm))")

    M = pt.inv_mass[perm, perm]
    # A permuted covariance must stay symmetric positive definite; if it does
    # not, the ordering is wrong and sampling would silently degrade.
    maximum(abs.(M .- M')) < 1e-10 ||
        error("permuted inverse mass matrix is not symmetric")
    isposdef(Symmetric(M)) ||
        error("permuted inverse mass matrix is not positive definite -- " *
              "the parameter ordering does not match the tuned metric")

    # AdvancedHMC.HMC(n_leapfrog; integrator, metric) is the constructor that
    # accepts a metric INSTANCE. `Turing.HMC` takes `metricT` (a type) and so
    # cannot carry a pre-tuned matrix, and `AdvancedHMC.HMC(eps, L)` takes no
    # metric at all -- the step size rides in the integrator instead.
    metric = DenseEuclideanMetric(Matrix(Symmetric(M)))
    spl = AdvancedHMC.HMC(pt.L; integrator=AdvancedHMC.Leapfrog(pt.eps),
                          metric=metric)

    backend = n_chains == 1 ? MCMCSerial() : MCMCThreads()
    Random.seed!(seed)
    # No adaptation: the metric and step size are fixed, so the only reason to
    # burn draws up front is to reach the typical set. discard_initial does that
    # without letting an adaptor touch the tuned constants.
    sample(model, externalsampler(spl; adtype=adtype), backend, n_draws, n_chains;
           discard_initial=n_adapt, progress=false, check_model=false)
end
