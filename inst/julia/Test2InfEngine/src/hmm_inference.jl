using Distributions, DensityInterface
using HiddenMarkovModels, Turing, StaticArrays
using LogExpFunctions: logistic, log1pexp
using Random, Statistics, LinearAlgebra
using Serialization: serialize, deserialize
using ADTypes
# Mooncake is a full dependency and is loaded HERE, at module load, not lazily
# via Base.require. That is what lets the precompile workload build a Mooncake
# gradient: DifferentiationInterface only wires up its Mooncake extension if
# Mooncake was loaded before the gradient is prepared, so loading it up front is
# the requirement for precompiling the reverse-mode path, not an obstacle to it.
#
# One forward-mode and one reverse-mode engine only, so `using` stays bounded.
# Mooncake is the reverse choice: on the compacted likelihood it is the fastest
# backend measured (6.1ms per gradient against Enzyme's 13.2ms and ForwardDiff's
# 19.8ms). See gridded/HMM_PERFORMANCE.md.
using Mooncake

const DynamicPPL = Turing.DynamicPPL
const var"@varname" = DynamicPPL.var"@varname"

# ForwardDiff remains the default for MAP/MLE, where the problem is small and
# forward mode wins. NUTS/HMC should be asked for "mooncake": it is ~3x faster
# on the gradient at this parameter count.
const adtype = AutoForwardDiff()

function ad_backend(name::AbstractString)
    name == "forwarddiff" && return AutoForwardDiff()
    name == "mooncake" || error("ad must be \"forwarddiff\" or \"mooncake\"")
    AutoMooncake(; config=nothing)
end

clamp_prob(x; eps=1e-9) = clamp(x, eps, 1 - eps)

season_of(t::Int, S=4) = (t - 1) % S + 1
year_of(t::Int, S=4) = (t - 1) ÷ S + 1

struct Control
    t::Int
end

const YEAR_PROCESSES = (:iid, :rw1, :rw2, :ar1, :shrunk, :none)

struct DiagnosticHMM{T,V1<:AbstractVector{T},V2<:AbstractVector{T},V3<:AbstractVector{T},V4<:AbstractVector{T}} <: AbstractHMM
    π1        :: T
    alpha     :: V1
    gamma     :: V2
    Se        :: V3
    Sp        :: V4
    S         :: Int
end
Base.length(::DiagnosticHMM) = 2

HiddenMarkovModels.initialization(h::DiagnosticHMM) = SVector(1 - h.π1, h.π1)

function HiddenMarkovModels.transition_matrix(h::DiagnosticHMM, c::Control)
    lam = clamp_prob(logistic(h.alpha[season_of(c.t, h.S)] + h.gamma[year_of(c.t, h.S)]))
    return SMatrix{2,2}(1 - lam, zero(lam), lam, one(lam))
end

# Se and Sp carry their own vector types rather than being pinned to Vector:
# under some AD backends the model's Se/Sp arrive as views, and requiring a
# concrete Vector here makes construction throw a MethodError.
struct TBEmission{T,V1<:AbstractVector{T},V2<:AbstractVector{T}} <: Distribution{Multivariate, Discrete}
    Se       :: V1
    Sp       :: V2
    infected :: Bool
end

DensityInterface.DensityKind(::TBEmission) = HasLogDensity()

# `x` may be longer than the test panel: when a badger is captured more than
# once in the same time-step, the extra captures are appended to the same
# observation vector. `mod1` reuses each test's Se/Sp for those repeats, so they
# contribute additional likelihood terms instead of overwriting one another.
function DensityInterface.logdensityof(d::TBEmission, x::AbstractVector)
    ll = zero(eltype(d.Se))
    K = length(d.Se)
    @inbounds for k in eachindex(x)
        xk = x[k]
        isnan(xk) && continue
        kk = mod1(k, K)
        ll += d.infected ? (xk == 1 ? log(d.Se[kk]) : log(1 - d.Se[kk])) :
                           (xk == 1 ? log(1 - d.Sp[kk]) : log(d.Sp[kk]))
    end
    return ll
end

function HiddenMarkovModels.obs_distributions(h::DiagnosticHMM, ::Control)
    SVector(TBEmission(h.Se, h.Sp, false), TBEmission(h.Se, h.Sp, true))
end

function HiddenMarkovModels._forward_digest_observation!(
    current_state_marginals::AbstractVector{<:Real},
    current_obs_likelihoods::AbstractVector{<:Real},
    h::DiagnosticHMM,
    obs,
    c::Control;
    error_if_not_finite::Bool,
)
    logb1 = zero(eltype(h.Se))
    logb2 = zero(eltype(h.Se))
    K = length(h.Se)
    @inbounds for k in eachindex(obs)
        v = obs[k]
        isnan(v) && continue
        kk = mod1(k, K)
        pU = v == 1.0 ? (1 - h.Sp[kk]) : h.Sp[kk]
        pI = v == 1.0 ? h.Se[kk] : (1 - h.Se[kk])
        logb1 += log(clamp_prob(pU))
        logb2 += log(clamp_prob(pI))
    end

    logm = max(logb1, logb2)
    b1 = exp(logb1 - logm)
    b2 = exp(logb2 - logm)

    # Two-state model, so these indices are always in range. Worth ~24% to
    # ForwardDiff on this path; nothing to Enzyme.
    @inbounds begin
        a1 = current_state_marginals[1] * b1
        a2 = current_state_marginals[2] * b2
        cscale = inv(a1 + a2)

        current_state_marginals[1] = a1 * cscale
        current_state_marginals[2] = a2 * cscale
        current_obs_likelihoods[1] = b1
        current_obs_likelihoods[2] = b2
    end

    logL = -log(cscale) + logm
    return cscale, logL
end

# Hand-rolled forward pass for one badger.
#
# This replaces `logdensityof(::DiagnosticHMM, ...)` in the model. It computes
# exactly the same quantity (verified bit-identical), but carries two scalars
# through the recursion instead of going via HiddenMarkovModels: that library
# calls `obs_distributions` once per timepoint, which builds two `TBEmission`
# structs holding vector references, so it cannot stack-allocate them (~2 MB of
# garbage per log-density call). Measured on the real data: 11.7ms per gradient
# against 57.6ms, both under Enzyme. See gridded/HMM_PERFORMANCE.md.
#
# The DiagnosticHMM path is kept because the posterior-decoding code
# (forward_backward) still uses it; only the likelihood is replaced.
# Compacted observations: 79% of the observation cells on this grid are NaN and
# 51% of timepoints hold no test at all, so testing isnan per cell spends four
# fifths of the emission loop doing nothing, on a data-dependent branch.
#
# Flattening once, up front, into
#   vals[p] - the 0/1 result       tidx[p] - which test it belongs to
#   ptr[j]  - where timepoint j's results start
# turns the inner loop into a straight run over only the results that exist. An
# empty timepoint becomes an empty range and costs nothing beyond the state
# update.
function compact_observations(obs_seq, n_tests::Int)
    ptr = Vector{Int}(undef, length(obs_seq) + 1)
    vals = Int8[]; tidx = Int32[]
    ptr[1] = 1
    for (j, o) in enumerate(obs_seq)
        for k in eachindex(o)
            v = o[k]
            isnan(v) && continue
            push!(vals, v == 1 ? Int8(1) : Int8(0))
            push!(tidx, Int32(mod1(k, n_tests)))
        end
        ptr[j+1] = length(vals) + 1
    end
    (ptr=ptr, vals=vals, tidx=tidx)
end

# Hand-rolled forward pass for one badger.
#
# Replaces `logdensityof(::DiagnosticHMM, ...)` in the model. Same quantity
# (verified bit-identical), but two scalars through the recursion instead of
# going via HiddenMarkovModels, which calls obs_distributions once per timepoint
# and builds two TBEmission structs holding vector references (~2 MB of garbage
# per log-density call).
#
# Measured on the real data, gradient: 57.6ms via HiddenMarkovModels, 11.7ms
# scalar, 6.1ms with compacted observations. See gridded/HMM_PERFORMANCE.md.
#
# The DiagnosticHMM path is kept: forward_backward still uses it for posterior
# decoding. Only the likelihood is replaced.
function seq_loglik(ptr, vals, tidx, times, pi1, alpha, gamma, Se, Sp, S::Int)
    n = length(times)
    T = eltype(alpha)
    aU = one(T) - pi1
    aI = pi1
    ll = zero(T)
    @inbounds for j in 1:n
        if j > 1
            t = times[j]
            lam = clamp_prob(logistic(alpha[season_of(t, S)] + gamma[year_of(t, S)]))
            aI = aI + aU * lam
            aU = aU * (one(T) - lam)
        end
        lo = ptr[j]; hi = ptr[j+1] - 1
        if hi >= lo
            bU = one(T); bI = one(T)
            for p in lo:hi
                k = tidx[p]
                if vals[p] == 1
                    bU *= one(T) - Sp[k]
                    bI *= Se[k]
                else
                    bU *= Sp[k]
                    bI *= one(T) - Se[k]
                end
            end
            aU *= bU; aI *= bI
            c = aU + aI
            ll += log(c)
            aU /= c; aI /= c
        end
    end
    return ll
end

const SE_FIXED_DEFAULT = [0.407, 0.407, 0.100, 0.650, 0.809, 0.492]
const SP_FIXED_DEFAULT = [0.943, 0.943, 0.999, 0.943, 0.936, 0.931]
const SE_CI_DEFAULT = [(0.370, 0.530), (0.370, 0.530), (0.025, 0.373),
                       (0.502, 0.798), (0.640, 0.901), (0.441, 0.544)]
const SP_CI_DEFAULT = [(0.890, 0.980), (0.890, 0.980), (0.939, 0.999),
                       (0.881, 0.999), (0.621, 0.987), (0.622, 0.986)]

clamp01(x; eps=1e-4) = clamp(x, eps, 1 - eps)
sd_from_ci(ci) = max((clamp01(ci[2]) - clamp01(ci[1])) / 3.92, 1e-3)
beta_from_moments(μ, σ) = (c = μ * (1 - μ) / σ^2 - 1; Beta(max(μ * c, 0.5), max((1 - μ) * c, 0.5)))

function default_priors_from_table1()
    se_priors = [beta_from_moments(clamp01(SE_FIXED_DEFAULT[k]), sd_from_ci(SE_CI_DEFAULT[k])) for k in eachindex(SE_FIXED_DEFAULT)]
    sp_priors = [beta_from_moments(clamp01(SP_FIXED_DEFAULT[k]), sd_from_ci(SP_CI_DEFAULT[k])) for k in eachindex(SP_FIXED_DEFAULT)]
    se_prior_mean = Float64.(mean.(se_priors))
    sp_prior_mean = Float64.(mean.(sp_priors))
    return se_priors, sp_priors, se_prior_mean, sp_prior_mean
end

function priors_from_config(se_prior_mean::AbstractVector, sp_prior_mean::AbstractVector,
                            se_prior_ci::AbstractMatrix, sp_prior_ci::AbstractMatrix)
    if size(se_prior_ci, 1) != length(se_prior_mean) || size(se_prior_ci, 2) != 2
        error("se_prior_ci must be an n_tests x 2 matrix")
    end
    if size(sp_prior_ci, 1) != length(sp_prior_mean) || size(sp_prior_ci, 2) != 2
        error("sp_prior_ci must be an n_tests x 2 matrix")
    end
    se_priors = [beta_from_moments(clamp01(se_prior_mean[k]), sd_from_ci((se_prior_ci[k, 1], se_prior_ci[k, 2]))) for k in eachindex(se_prior_mean)]
    sp_priors = [beta_from_moments(clamp01(sp_prior_mean[k]), sd_from_ci((sp_prior_ci[k, 1], sp_prior_ci[k, 2]))) for k in eachindex(sp_prior_mean)]
    se_priors, sp_priors
end

const HAZARD_PRIOR_MEAN = -3.0
const HAZARD_PRIOR_SD   = 1.5

const PENALTY_WEIGHT = 50.0
const PENALTY_SCALE  = 0.02

build_gamma(raw::AbstractVector, sigma_g, process::Symbol, rho) = begin
    if process === :none
        return zero.(raw)
    elseif process === :iid || process === :shrunk
        return sigma_g .* raw
    elseif process === :rw1
        return sigma_g .* cumsum(raw)
    elseif process === :rw2
        return sigma_g .* cumsum(cumsum(raw))
    elseif process === :ar1
        out = similar(raw)
        out[1] = sigma_g * raw[1] / sqrt(max(1 - rho^2, 1e-8))
        for y in 2:length(raw)
            out[y] = rho * out[y - 1] + sigma_g * raw[y]
        end
        return out
    else
        error("unknown process $process")
    end
end

sigma_prior(process::Symbol) =
    process === :shrunk ? truncated(Normal(0, 0.10); lower=0) :
    process === :rw1    ? truncated(Normal(0, 0.05); lower=0) :
    process === :rw2    ? truncated(Normal(0, 0.01); lower=0) :
                          truncated(Normal(0, 0.30); lower=0)

@model function hmm_model(obs_seq, ctrl_seq, seq_ends, S, n_years, se_priors, sp_priors,
                          entry_year, times_seq, obs_ptr, obs_vals, obs_tidx;
                          process::Symbol=:rw1,
                          hazard_mean::Float64=HAZARD_PRIOR_MEAN,
                          hazard_sd::Float64=HAZARD_PRIOR_SD,
                          penalty::Bool=true,
                          se_fixed=nothing,
                          sp_fixed=nothing)
    alpha ~ MvNormal(fill(hazard_mean, S), hazard_sd^2 * I(S))
    sigma_g ~ sigma_prior(process)
    gamma_raw ~ MvNormal(zeros(n_years), I(n_years))
    rho = 0.0
    if process === :ar1
        rho ~ Beta(5, 2)
    end
    gamma = build_gamma(gamma_raw, sigma_g, process, rho)

    if se_fixed === nothing
        Se ~ arraydist(se_priors)
    else
        Se = eltype(alpha).(se_fixed)
    end
    if sp_fixed === nothing
        Sp ~ arraydist(sp_priors)
    else
        Sp = eltype(alpha).(sp_fixed)
    end
    # Probability of already being infected at first capture. A badger entering
    # in a high-incidence year is more likely to arrive infected, so this is
    # tied to the year effect rather than held constant across the study.
    pi1_0 ~ Normal(-1.7, 1.0)
    pi1_mult ~ Normal(1.0, 1.0)
    pi1_vec = clamp_prob.(logistic.(pi1_0 .+ pi1_mult .* gamma))

    if penalty
        for k in eachindex(Se)
            Turing.@addlogprob! -PENALTY_WEIGHT * log1pexp(-(Se[k] + Sp[k] - 1.0) / PENALTY_SCALE)
        end
    end

    # Each badger starts at the pi1 of the year it entered, so the likelihood
    # is accumulated per badger rather than in one batched call.
    ll = zero(eltype(alpha))
    start = 1
    for (i, e) in enumerate(seq_ends)
        ll += seq_loglik(view(obs_ptr, start:(e+1)), obs_vals, obs_tidx,
                         view(times_seq, start:e),
                         pi1_vec[entry_year[i]], alpha, gamma, Se, Sp, S)
        start = e + 1
    end
    Turing.@addlogprob! ll
end

# A badger can be captured more than once within one time-step. `repeat_captures`
# decides what happens to the extras:
#
#   :stack - every capture contributes its own emission terms, so repeat
#            positives multiply the evidence (the cumulative treatment used by
#            the reference PPV method).
#   :pool  - captures are merged into one observation per time-step (positive if
#            any capture was positive), so repeats do not compound.
#   :last  - keep only the last capture, discarding the rest. Legacy behaviour,
#            retained so earlier fits reproduce exactly.
#
# Every branch must return a concrete Vector{Float64}: these vectors are pushed
# into a Vector{Vector{Float64}} and differentiated through, and an abstract
# element type here surfaces as a hard crash rather than a type error.
function build_gridded_sequences(individuals; n_tests::Int=6,
                                 repeat_captures::Symbol=:stack)
    repeat_captures in (:stack, :pool, :last) ||
        error("repeat_captures must be :stack, :pool or :last")

    map(individuals) do b
        t0, t1 = minimum(b.times), maximum(b.times)
        grid = collect(t0:t1)

        lookup = Dict{Int, Vector{Int}}()
        for (j, t) in enumerate(b.times)
            push!(get!(lookup, t, Int[]), j)
        end

        obs = Vector{Vector{Float64}}(undef, length(grid))
        for (i, t) in enumerate(grid)
            js = get(lookup, t, Int[])
            obs[i] = if isempty(js)
                fill(NaN, n_tests)
            elseif length(js) == 1
                Float64.(b.obs[js[1]])
            elseif repeat_captures === :last
                Float64.(b.obs[js[end]])
            elseif repeat_captures === :pool
                pooled = fill(NaN, n_tests)
                for k in 1:n_tests
                    for j in js
                        v = b.obs[j][k]
                        isnan(v) && continue
                        pooled[k] = isnan(pooled[k]) ? v : max(pooled[k], v)
                    end
                end
                pooled
            else  # :stack
                stacked = Vector{Float64}(undef, n_tests * length(js))
                for (n, j) in enumerate(js)
                    off = (n - 1) * n_tests
                    for k in 1:n_tests
                        stacked[off + k] = b.obs[j][k]
                    end
                end
                stacked
            end
        end

        captured = [haskey(lookup, t) for t in grid]
        (id=b.id, times=grid, obs=obs, captured=captured)
    end
end

function pack_sequences(individuals)
    obs_seq = Vector{Vector{Float64}}()
    ctrl_seq = Vector{Control}()
    seq_ends = Int[]
    for b in individuals
        for j in eachindex(b.times)
            push!(obs_seq, b.obs[j])
            push!(ctrl_seq, Control(b.times[j]))
        end
        push!(seq_ends, length(obs_seq))
    end
    return obs_seq, ctrl_seq, seq_ends
end

function extract_params(result; n_tests::Int, numSeasons::Int, n_years::Int,
                        process::Symbol=:rw1, se_fixed=nothing, sp_fixed=nothing)
    p = result.params
    alpha = Float64[p[@varname(alpha[s])] for s in 1:numSeasons]
    sigma_g = Float64(p[@varname(sigma_g)])
    gamma_raw = Float64[p[@varname(gamma_raw[y])] for y in 1:n_years]
    rho = process === :ar1 ? Float64(p[@varname(rho)]) : 0.0
    gamma = Float64.(build_gamma(gamma_raw, sigma_g, process, rho))
    Se = se_fixed === nothing ? Float64[p[@varname(Se[k])] for k in 1:n_tests] : Float64.(se_fixed)
    Sp = sp_fixed === nothing ? Float64[p[@varname(Sp[k])] for k in 1:n_tests] : Float64.(sp_fixed)
    pi1_vec = clamp_prob.(logistic.(Float64(p[@varname(pi1_0)]) .+
                                    Float64(p[@varname(pi1_mult)]) .* gamma))
    pi1 = pi1_vec[1]
    (pi1=pi1, pi1_vec=pi1_vec, alpha=alpha, gamma=gamma, Se=Se, Sp=Sp, sigma_g=sigma_g, rho=rho)
end

function p_inf_over_time_pointestimate(individuals, P, numSeasons::Int)
    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    for b in individuals
        ey = year_of(minimum(b.times), numSeasons)
        hmm = DiagnosticHMM(P.pi1_vec[ey], P.alpha, P.gamma, P.Se, P.Sp, numSeasons)
        ctrl_b = [Control(t) for t in b.times]
        gam, _ = forward_backward(hmm, b.obs, ctrl_b)
        results[b.id] = gam[2, :]
        times[b.id] = b.times
    end
    (p_inf=results, times=times)
end

# Sample whole trajectories rather than reading off marginals.
#
# The marginals from forward_backward are P(infected at t) one timepoint at a
# time. They cannot be treated as independent coin flips: the chain is
# absorbing, so a badger infected at t is infected at every later t, and a
# badger caught several times in one year contributes the same latent state
# repeatedly. Averaging independent Bernoullis over captures therefore gives
# intervals that are far too narrow.
#
# Because U -> I is absorbing, a trajectory is determined entirely by WHEN the
# badger became infected, so the posterior over trajectories is a distribution
# over that single switch point. P(switch at k) is proportional to the
# marginal increment, which makes sampling exact and cheap.
function sample_trajectory!(out::AbstractVector{Bool}, gam::AbstractVector{Float64},
                            rng::AbstractRNG)
    n = length(gam)
    # Increments of the marginal give the probability of first becoming
    # infected at each step; the leading value covers "already infected".
    u = rand(rng)
    acc = gam[1]
    if u <= acc
        fill!(out, true)
        return out
    end
    for k in 2:n
        inc = max(gam[k] - gam[k - 1], 0.0)
        acc += inc
        if u <= acc
            @inbounds for j in 1:n
                out[j] = j >= k
            end
            return out
        end
    end
    fill!(out, false)
    return out
end

# Prevalence per sampled trajectory set, at fixed parameters. Returns a
# n_times x n_reps matrix for each estimand, so the caller can take quantiles.
function prevalence_trajectory_draws(individuals, p_inf, times, n_reps::Int, seed::Int;
                                    keep_infection_times::Bool=true)
    rng = MersenneTwister(seed)
    by_id = Dict(b.id => b for b in individuals)
    all_times = sort(unique(vcat(values(times)...)))
    tindex = Dict(t => i for (i, t) in enumerate(all_times))

    grid_sum = zeros(length(all_times), n_reps)
    grid_n = zeros(Int, length(all_times))
    cap_sum = zeros(length(all_times), n_reps)
    cap_n = zeros(Int, length(all_times))

    for b in individuals
        gam = p_inf[b.id]
        tv = times[b.id]
        traj = Vector{Bool}(undef, length(tv))
        for r in 1:n_reps
            sample_trajectory!(traj, gam, rng)
            for (i, t) in enumerate(tv)
                ti = tindex[t]
                grid_sum[ti, r] += traj[i]
                b.captured[i] && (cap_sum[ti, r] += traj[i])
            end
        end
        for (i, t) in enumerate(tv)
            ti = tindex[t]
            grid_n[ti] += 1
            b.captured[i] && (cap_n[ti] += 1)
        end
    end

    # The chain is absorbing, so a trajectory is fully described by the
    # timepoint at which the badger became infected (0 = never infected within
    # its record). Storing that instead of the whole path loses nothing.
    inf_ids = Int[]
    inf_draw = Int[]
    inf_time = Int[]
    if keep_infection_times
        rng2 = MersenneTwister(seed)
        for b in individuals
            gam = p_inf[b.id]; tv = times[b.id]
            traj = Vector{Bool}(undef, length(tv))
            for r in 1:n_reps
                sample_trajectory!(traj, gam, rng2)
                k = findfirst(traj)
                push!(inf_ids, b.id); push!(inf_draw, r)
                push!(inf_time, k === nothing ? 0 : tv[k])
            end
        end
    end

    grid = fill(NaN, length(all_times), n_reps)
    cap = fill(NaN, length(all_times), n_reps)
    for ti in 1:length(all_times)
        grid_n[ti] > 0 && (grid[ti, :] .= grid_sum[ti, :] ./ grid_n[ti])
        cap_n[ti] > 0 && (cap[ti, :] .= cap_sum[ti, :] ./ cap_n[ti])
    end
    (times=all_times, grid=grid, capture=cap,
     infection_id=inf_ids, infection_draw=inf_draw, infection_time=inf_time)
end

struct DrawArrays
    pi1 :: Matrix{Float64}   # n_years x n_samps
    alpha :: Matrix{Float64}
    gamma :: Matrix{Float64}
    Se :: Matrix{Float64}
    Sp :: Matrix{Float64}
end

function extract_all_draws(chain; n_tests::Int, S::Int, n_years::Int,
                           process::Symbol=:rw1, se_fixed=nothing, sp_fixed=nothing)
    b0 = Float64.(vec(chain[@varname(pi1_0)]))
    bm = Float64.(vec(chain[@varname(pi1_mult)]))
    n_samps = length(b0)
    sg = Float64.(vec(chain[@varname(sigma_g)]))
    rho = if process === :ar1
        Float64.(vec(chain[@varname(rho)]))
    else
        zeros(n_samps)
    end

    alpha = Matrix{Float64}(undef, S, n_samps)
    for s in 1:S
        alpha[s, :] .= Float64.(vec(chain[@varname(alpha[s])]))
    end

    gamma_raw = Matrix{Float64}(undef, n_years, n_samps)
    for y in 1:n_years
        gamma_raw[y, :] .= Float64.(vec(chain[@varname(gamma_raw[y])]))
    end

    gamma = Matrix{Float64}(undef, n_years, n_samps)
    for i in 1:n_samps
        gamma[:, i] .= Float64.(build_gamma(gamma_raw[:, i], sg[i], process, rho[i]))
    end

    Se = Matrix{Float64}(undef, n_tests, n_samps)
    Sp = Matrix{Float64}(undef, n_tests, n_samps)
    if se_fixed === nothing
        for k in 1:n_tests
            Se[k, :] .= Float64.(vec(chain[@varname(Se[k])]))
            Sp[k, :] .= Float64.(vec(chain[@varname(Sp[k])]))
        end
    else
        for k in 1:n_tests
            Se[k, :] .= se_fixed[k]
            Sp[k, :] .= sp_fixed[k]
        end
    end

    pi1 = Matrix{Float64}(undef, n_years, n_samps)
    for i in 1:n_samps
        pi1[:, i] .= clamp_prob.(logistic.(b0[i] .+ bm[i] .* gamma[:, i]))
    end

    DrawArrays(pi1, alpha, gamma, Se, Sp)
end

# How many draws the posterior-mean p_inf is averaged over when the caller has
# asked for no per-draw trajectory output. Bounded so that this cost does not
# grow with chain length; see the note in p_inf_over_time_nuts.
const DEFAULT_MEAN_DRAWS = 200

@inline function hmm_for_draw(d::DrawArrays, i::Int, S::Int, entry_year::Int)
    DiagnosticHMM(d.pi1[entry_year, i], d.alpha[:, i], d.gamma[:, i],
                  d.Se[:, i], d.Sp[:, i], S)
end

function p_inf_over_time_nuts(individuals, chain; n_tests::Int, numSeasons::Int, n_years::Int,
                              process::Symbol=:rw1, se_fixed=nothing, sp_fixed=nothing,
                              keep_draws::Bool=false, max_draws::Int=0)
    draws = extract_all_draws(chain; n_tests=n_tests, S=numSeasons, n_years=n_years,
                              process=process, se_fixed=se_fixed, sp_fixed=sp_fixed)
    n_total = size(draws.pi1, 2)
    # Reconstructing trajectories costs one forward-backward pass per
    # (badger x draw), so it dominates runtime on a long chain. Thinning to an
    # evenly spaced subsample keeps the posterior spread while decoupling this
    # cost from how long the chain was run.
    # max_draws == 0 asks for no per-draw output, but the mean p_inf is still
    # needed, so it is averaged over a bounded subsample rather than every draw.
    n_keep = max_draws > 0 ? max_draws : DEFAULT_MEAN_DRAWS
    keep = n_keep < n_total ?
           round.(Int, range(1, n_total; length=n_keep)) : collect(1:n_total)
    n_samps = length(keep)

    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    # Per-draw trajectories, kept only when the caller wants posterior intervals:
    # this is n_badgers x n_times x n_draws, so it is not free.
    per_draw = keep_draws ? Dict{Int, Matrix{Float64}}() : nothing
    for b in individuals
        ctrl_b = [Control(t) for t in b.times]
        ey = year_of(minimum(b.times), numSeasons)
        acc = zeros(length(b.times))
        mat = keep_draws ? Matrix{Float64}(undef, length(b.times), n_samps) : nothing
        for (j, i) in enumerate(keep)
            hmm = hmm_for_draw(draws, i, numSeasons, ey)
            gam, _ = forward_backward(hmm, b.obs, ctrl_b)
            acc .+= gam[2, :]
            keep_draws && (mat[:, j] .= gam[2, :])
        end
        results[b.id] = acc ./ n_samps
        times[b.id] = b.times
        keep_draws && (per_draw[b.id] = mat)
    end
    (p_inf=results, times=times, p_inf_draws=per_draw)
end

# Prevalence for every posterior draw, so the caller can take quantiles.
# Returns a n_times x n_draws matrix for each estimand.
function prevalence_draws(individuals, p_inf_draws, times)
    by_id = Dict(b.id => b for b in individuals)
    all_times = sort(unique(vcat(values(times)...)))
    n_draws = size(first(values(p_inf_draws)), 2)

    grid = fill(NaN, length(all_times), n_draws)
    cap = fill(NaN, length(all_times), n_draws)

    for (ti, t) in enumerate(all_times)
        gsum = zeros(n_draws); gn = 0
        csum = zeros(n_draws); cn = 0
        for (id, tv) in times
            idx = findfirst(==(t), tv)
            idx === nothing && continue
            row = @view p_inf_draws[id][idx, :]
            gsum .+= row; gn += 1
            if by_id[id].captured[idx]
                csum .+= row; cn += 1
            end
        end
        gn > 0 && (grid[ti, :] .= gsum ./ gn)
        cn > 0 && (cap[ti, :] .= csum ./ cn)
    end

    (times=all_times, grid=grid, capture=cap)
end

# Infection time per (badger, draw) for the NUTS path. Each draw's trajectory is
# already a full path, so the infection time is the first timepoint at which it
# crosses one half; 0 means never, within that badger's record.
function infection_times_from_draws(individuals, p_inf_draws, times)
    ids = Int[]; draws = Int[]; tms = Int[]
    for b in individuals
        m = p_inf_draws[b.id]
        tv = times[b.id]
        for r in 1:size(m, 2)
            k = findfirst(>=(0.5), @view m[:, r])
            push!(ids, b.id); push!(draws, r)
            push!(tms, k === nothing ? 0 : tv[k])
        end
    end
    (id=ids, draw=draws, time=tms)
end

function quantile_rows(m::Matrix{Float64}, q::Float64)
    [all(isnan, @view m[i, :]) ? NaN : quantile(collect(skipmissing(@view m[i, :])), q)
     for i in 1:size(m, 1)]
end

function prevalence_two_ways(individuals, p_inf, times)
    by_id = Dict(b.id => b for b in individuals)
    all_times = sort(unique(vcat(values(times)...)))

    grid_prop = Float64[]
    grid_tot = Float64[]
    cap_prop = Float64[]
    cap_tot = Float64[]

    for t in all_times
        gv = Float64[]
        cv = Float64[]
        for (id, tv) in times
            idx = findfirst(==(t), tv)
            idx === nothing && continue
            p = p_inf[id][idx]
            push!(gv, p)
            by_id[id].captured[idx] && push!(cv, p)
        end
        push!(grid_prop, isempty(gv) ? NaN : mean(gv))
        push!(grid_tot, isempty(gv) ? 0.0 : sum(gv))
        push!(cap_prop, isempty(cv) ? NaN : mean(cv))
        push!(cap_tot, isempty(cv) ? 0.0 : sum(cv))
    end

    (times=all_times,
     grid_proportion=grid_prop,
     grid_total=grid_tot,
     capture_proportion=cap_prop,
     capture_total=cap_tot)
end

function create_infection_matrix(p_inf_over_time, times, ids)
    all_times = sort(unique(vcat(values(times)...)))
    n_times = length(all_times)
    n_ids = length(ids)
    inf_matrix = fill(NaN, n_times, n_ids)

    for (col_idx, id) in enumerate(ids)
        t_vec = times[id]
        p_vec = p_inf_over_time[id]
        for (row_idx, t) in enumerate(all_times)
            idx = findfirst(==(t), t_vec)
            if idx !== nothing
                inf_matrix[row_idx, col_idx] = p_vec[idx]
            end
        end
    end

    (matrix=inf_matrix, times=all_times, ids=ids)
end

function annual_hazard(alpha::AbstractVector, gamma::AbstractVector, y::Int, S::Int)
    1 - prod(1 - clamp_prob(logistic(alpha[s] + gamma[y])) for s in 1:S)
end

function build_individual_sequences(test_mat::Matrix{Float64}; repeat_captures::Symbol=:stack)
    if size(test_mat, 2) >= 8
        time_vec = Int.(round.(test_mat[:, 1]))
        id_vec = Int.(round.(test_mat[:, 2]))
        obs_mat = test_mat[:, (end-5):end]
        n_tests = size(obs_mat, 2)

        ids = sort(unique(id_vec))
        base = map(ids) do id
            id_rows = findall(==(id), id_vec)
            id_times = time_vec[id_rows]
            ord = sortperm(id_times)
            rows = id_rows[ord]
            times = Int.(id_times[ord])
            obs = [Float64[obs_mat[r, k] for k in 1:n_tests] for r in rows]
            (id=id, times=times, obs=obs)
        end
        return build_gridded_sequences(base; n_tests=n_tests, repeat_captures=repeat_captures)
    end

    base = [(id=i, times=[1], obs=[test_mat[i, :]]) for i in 1:size(test_mat, 1)]
    return build_gridded_sequences(base; n_tests=size(test_mat, 2), repeat_captures=repeat_captures)
end

function run_hmm_inference(test_mat::Matrix{Float64}, method::String,
                           nuts_samples::Int, target_acc::Float64, seed::Int;
                           year_process::String="rw1",
                           test_mask::AbstractVector{Bool}=trues(6),
                           se_fixed::AbstractVector{<:Real}=Float64[],
                           sp_fixed::AbstractVector{<:Real}=Float64[],
                           se_prior_mean::AbstractVector{<:Real}=Float64[],
                           sp_prior_mean::AbstractVector{<:Real}=Float64[],
                           se_prior_ci::AbstractMatrix{<:Real}=zeros(0, 0),
                           sp_prior_ci::AbstractMatrix{<:Real}=zeros(0, 0),
                           hazard_mean::Float64=HAZARD_PRIOR_MEAN,
                           hazard_sd::Float64=HAZARD_PRIOR_SD,
                             penalty::Bool=true,
                           start_year::Int=0,
                           repeat_captures::String="stack",
                           traj_draws::Int=500,
                           chain_cache::String="",
                           reuse_chain::Bool=false,
                           ad_name::String="")
    Random.seed!(seed)
    # Never mutate the caller's matrix: both the sentinel replacement below and
    # the test masking further down write NaN into it, so a caller reusing one
    # matrix across several fits would have earlier masks leak into later ones.
    test_mat = copy(test_mat)
    test_mat[test_mat .== -10.0] .= NaN

    numSeasons = 4

    if length(test_mask) != 6
        error("test_mask must have length 6")
    end
    if !any(test_mask)
        error("test_mask selects no tests")
    end

    proc = Symbol(year_process)
    proc in YEAR_PROCESSES || error("Unknown year_process $year_process")

    function apply_test_mask!(tm::Matrix{Float64}, mask::AbstractVector{Bool})
        size(tm, 2) < 6 && return tm
        test_cols = (size(tm, 2) - 5):size(tm, 2)
        for (j, keep) in enumerate(mask)
            keep && continue
            tm[:, test_cols[j]] .= NaN
        end
        tm
    end
    apply_test_mask!(test_mat, test_mask)

    rc = Symbol(repeat_captures)
    rc in (:stack, :pool, :last) ||
        error("repeat_captures must be \"stack\", \"pool\" or \"last\"")
    individuals = build_individual_sequences(test_mat; repeat_captures=rc)
    # Stacked repeats make some observation vectors longer than the panel, so
    # take the panel width from the input matrix.
    n_tests = size(test_mat, 2) >= 8 ? 6 : size(test_mat, 2)
    n_timepoints = maximum(vcat([b.times for b in individuals]...))
    n_years = year_of(n_timepoints, numSeasons)

    se_priors_default, sp_priors_default, se_prior_default_mean, sp_prior_default_mean = default_priors_from_table1()

    se_prior_mean_use = isempty(se_prior_mean) ? copy(se_prior_default_mean) : Float64.(se_prior_mean)
    sp_prior_mean_use = isempty(sp_prior_mean) ? copy(sp_prior_default_mean) : Float64.(sp_prior_mean)

    if length(se_prior_mean_use) != n_tests || length(sp_prior_mean_use) != n_tests
        error("prior means must have length $n_tests")
    end

    if size(se_prior_ci, 1) == n_tests && size(se_prior_ci, 2) == 2 &&
       size(sp_prior_ci, 1) == n_tests && size(sp_prior_ci, 2) == 2
        se_priors, sp_priors = priors_from_config(se_prior_mean_use, sp_prior_mean_use,
                                                  Float64.(se_prior_ci), Float64.(sp_prior_ci))
    else
        se_priors = se_priors_default
        sp_priors = sp_priors_default
    end

    use_fixed = !isempty(se_fixed) || !isempty(sp_fixed)
    if use_fixed
        (length(se_fixed) == n_tests && length(sp_fixed) == n_tests) ||
            error("When fixed Se/Sp are supplied they must both be length $n_tests")
    end

    se_fixed_use = use_fixed ? Float64.(se_fixed) : nothing
    sp_fixed_use = use_fixed ? Float64.(sp_fixed) : nothing

    obs_seq, ctrl_seq, seq_ends = pack_sequences(individuals)
    entry_year = [year_of(minimum(b.times), numSeasons) for b in individuals]
    # The hand-rolled likelihood indexes timepoints directly rather than going
    # through Control structs.
    times_seq = [c.t for c in ctrl_seq]
    cobs = compact_observations(obs_seq, n_tests)
    # `obs_seq`/`ctrl_seq` are NOT passed to the model: since the compaction the
    # likelihood reads the CSR arrays (ptr/vals/tidx) and the model body never
    # touches them. Handing them over anyway costs 48.8x on the gradient.
    #
    # `obs_seq` is a Vector{Vector{Float64}} with one inner vector per capture
    # (29,737 on the real cohort). Mooncake builds a tangent for EVERY inner
    # vector and zeroes the whole structure before each reverse sweep, so
    # `set_to_zero_maybe!!` alone was 98.9% of gradient time. Measured, full
    # cohort, rw1, Se/Sp fixed:
    #
    #   with    obs_seq   primal 2.30 ms   gradient 509.5 ms   ratio 221.6x
    #   without obs_seq   primal 2.30 ms   gradient  10.4 ms   ratio   4.6x
    #
    # 4.6x is a normal reverse-mode ratio; 221x was this dead argument. It also
    # explains why the gradient appeared to scale quadratically in cohort size
    # while the primal stayed linear: the tangent structure grows with the data.
    #
    # Empty stand-ins keep the model's arity and its `@model` signature intact.
    model = hmm_model(Vector{Float64}[], Control[], seq_ends, numSeasons, n_years, se_priors, sp_priors,
                      entry_year, times_seq, cobs.ptr, cobs.vals, cobs.tidx;
                      process=proc,
                      hazard_mean=hazard_mean,
                      hazard_sd=hazard_sd,
                      penalty=penalty,
                      se_fixed=se_fixed_use,
                      sp_fixed=sp_fixed_use)

    ids = [b.id for b in individuals]

    result = if method == "nuts" || method == "hmc"
        # ForwardDiff unless told otherwise; pass ad_name="mooncake" for the
        # faster reverse-mode path, having loaded Mooncake first.
        ad = ad_backend(isempty(ad_name) ? "forwarddiff" : ad_name)
        n_chains = parse(Int, get(ENV, "TEST2INFER_NUTS_CHAINS", "4"))
        backend = n_chains == 1 ? MCMCSerial() : MCMCThreads()
        # method="hmc" asks for the pre-tuned fast path: fixed-L HMC under a
        # dense metric adapted offline for this exact variant. Opt-in, because
        # the constants are specific to the five sql-e2e variants AND to the
        # cohort they were tuned on -- wrong data under a fixed metric samples
        # badly rather than failing. No match => fall back to NUTS.
        pretuned = nothing
        perm = nothing
        if method == "hmc"
            variant = fastpath_variant(test_mask, !use_fixed)
            if variant !== nothing
                D = length(DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)[:])
                pretuned = fastpath_pretuned(variant, D, proc)
                if pretuned !== nothing
                    perm = fastpath_permutation(model, n_years, numSeasons, !use_fixed)
                    perm === nothing && (pretuned = nothing)
                end
            end
            pretuned === nothing &&
                @info "no pre-tuned constants match this model; using NUTS"
        end

        chain = if reuse_chain && !isempty(chain_cache) && isfile(chain_cache)
            deserialize(chain_cache)
        elseif pretuned !== nothing
            fastpath_sample(model, pretuned, perm, nuts_samples,
                            min(200, nuts_samples), n_chains, seed, ad)
        else
            sample(model, NUTS(target_acc; adtype=ad), backend, nuts_samples, n_chains;
                   progress=false, check_model=false)
        end

        # Cache the chain before any post-processing. Sampling is the expensive,
        # unrepeatable part; rebuilding trajectories from it is cheap to rerun.
        # A bug in the latter must never destroy the former.
        if !isempty(chain_cache) && !(reuse_chain && isfile(chain_cache))
            mkpath(dirname(chain_cache))
            serialize(chain_cache, chain)
        end

        p_inf_over_time_data = p_inf_over_time_nuts(individuals, chain;
                                                    n_tests=n_tests,
                                                    numSeasons=numSeasons,
                                                    n_years=n_years,
                                                    process=proc,
                                                    se_fixed=se_fixed_use,
                                                    sp_fixed=sp_fixed_use,
                                                    keep_draws=traj_draws > 0,
                                                    max_draws=traj_draws)
        prevalence = prevalence_two_ways(individuals, p_inf_over_time_data.p_inf, p_inf_over_time_data.times)
        ntraj = traj_draws > 0 ?
                infection_times_from_draws(individuals,
                                           p_inf_over_time_data.p_inf_draws,
                                           p_inf_over_time_data.times) :
                (id=Int[], draw=Int[], time=Int[])
        pdraws = traj_draws > 0 ?
                 prevalence_draws(individuals, p_inf_over_time_data.p_inf_draws,
                                  p_inf_over_time_data.times) :
                 (grid=fill(NaN, length(prevalence.times), 1),
                  capture=fill(NaN, length(prevalence.times), 1))
        inf_matrix = create_infection_matrix(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, ids)
        p_inf_last = Dict(b.id => p_inf_over_time_data.p_inf[b.id][findlast(b.captured)] for b in individuals)

        # When Se/Sp are fixed the model never samples them, so they are not in
        # the chain: report the supplied values back.
        se_est = use_fixed ? copy(se_fixed_use) :
                 [mean(vec(chain[@varname(Se[k])])) for k in 1:n_tests]
        sp_est = use_fixed ? copy(sp_fixed_use) :
                 [mean(vec(chain[@varname(Sp[k])])) for k in 1:n_tests]
        sigma_g_est = mean(vec(chain[@varname(sigma_g)]))
        gamma_raw_est = [mean(vec(chain[@varname(gamma_raw[y])])) for y in 1:n_years]
        rho_est = proc === :ar1 ? mean(vec(chain[@varname(rho)])) : 0.0
        gamma_est = build_gamma(gamma_raw_est, sigma_g_est, proc, rho_est)
        year_index = collect(1:n_years)
        # start_year == 0 means the caller did not supply a calendar origin, so
        # there is nothing to report; leave the column missing rather than
        # passing the year index off as a year.
        calendar_year = start_year == 0 ? fill(-1, n_years) : start_year .+ (year_index .- 1)
        annual_h = [annual_hazard([mean(vec(chain[@varname(alpha[s])])) for s in 1:numSeasons], gamma_est, y, numSeasons) for y in 1:n_years]

        (p_inf_last=p_inf_last,
         p_inf_over_time=p_inf_over_time_data.p_inf,
         times=p_inf_over_time_data.times,
         prevalence_times=prevalence.times,
         prevalence_proportion=prevalence.grid_proportion,
         prevalence_total=prevalence.grid_total,
         prevalence_grid_proportion=prevalence.grid_proportion,
         prevalence_grid_total=prevalence.grid_total,
         prevalence_capture_proportion=prevalence.capture_proportion,
         prevalence_capture_total=prevalence.capture_total,
         prevalence_grid_lower=quantile_rows(pdraws.grid, 0.025),
         prevalence_grid_upper=quantile_rows(pdraws.grid, 0.975),
         prevalence_capture_lower=quantile_rows(pdraws.capture, 0.025),
         prevalence_capture_upper=quantile_rows(pdraws.capture, 0.975),
         prevalence_grid_draws=pdraws.grid,
         prevalence_capture_draws=pdraws.capture,
         trajectory_id=ntraj.id,
         trajectory_draw=ntraj.draw,
         trajectory_infection_time=ntraj.time,
         ids=ids,
         infection_matrix=inf_matrix.matrix,
         infection_matrix_times=inf_matrix.times,
         Se=se_est,
         Sp=sp_est,
         pi1_vec=clamp_prob.(logistic.(mean(vec(chain[@varname(pi1_0)])) .+
                                       mean(vec(chain[@varname(pi1_mult)])) .* gamma_est)),
         year_effect_year_index=year_index,
         year_effect_calendar_year=calendar_year,
         year_effect_gamma=gamma_est,
         year_effect_annual_hazard=annual_h,
         test_mask=collect(test_mask),
         settings=(method=method,
                   year_process=String(proc),
                   tests=findall(identity, test_mask),
                   fixed_sesp=use_fixed,
                   penalty=penalty,
                   hazard_mean=hazard_mean,
                   hazard_sd=hazard_sd,
                   start_year=start_year,
                   repeat_captures=String(rc)))

    elseif method == "map" || method == "mle"
        est = method == "map" ?
            maximum_a_posteriori(model; adtype=adtype, check_model=false) :
            maximum_likelihood(model; adtype=adtype, check_model=false)

        P = extract_params(est;
                           n_tests=n_tests,
                           numSeasons=numSeasons,
                           n_years=n_years,
                           process=proc,
                           se_fixed=se_fixed_use,
                           sp_fixed=sp_fixed_use)
        p_inf_over_time_data = p_inf_over_time_pointestimate(individuals, P, numSeasons)
        prevalence = prevalence_two_ways(individuals, p_inf_over_time_data.p_inf, p_inf_over_time_data.times)
        tdraws = traj_draws > 0 ?
                 prevalence_trajectory_draws(individuals, p_inf_over_time_data.p_inf,
                                             p_inf_over_time_data.times,
                                             traj_draws, seed) :
                 (grid=fill(NaN, length(prevalence.times), 1),
                  capture=fill(NaN, length(prevalence.times), 1),
                  infection_id=Int[], infection_draw=Int[], infection_time=Int[])
        inf_matrix = create_infection_matrix(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, ids)
        p_inf_last = Dict(b.id => p_inf_over_time_data.p_inf[b.id][findlast(b.captured)] for b in individuals)

        year_index = collect(1:n_years)
        # start_year == 0 means the caller did not supply a calendar origin, so
        # there is nothing to report; leave the column missing rather than
        # passing the year index off as a year.
        calendar_year = start_year == 0 ? fill(-1, n_years) : start_year .+ (year_index .- 1)
        annual_h = [annual_hazard(P.alpha, P.gamma, y, numSeasons) for y in 1:n_years]

        (p_inf_last=p_inf_last,
         p_inf_over_time=p_inf_over_time_data.p_inf,
         times=p_inf_over_time_data.times,
         prevalence_times=prevalence.times,
         prevalence_proportion=prevalence.grid_proportion,
         prevalence_total=prevalence.grid_total,
         prevalence_grid_proportion=prevalence.grid_proportion,
         prevalence_grid_total=prevalence.grid_total,
         prevalence_capture_proportion=prevalence.capture_proportion,
         prevalence_capture_total=prevalence.capture_total,
         prevalence_grid_lower=quantile_rows(tdraws.grid, 0.025),
         prevalence_grid_upper=quantile_rows(tdraws.grid, 0.975),
         prevalence_capture_lower=quantile_rows(tdraws.capture, 0.025),
         prevalence_capture_upper=quantile_rows(tdraws.capture, 0.975),
         prevalence_grid_draws=tdraws.grid,
         prevalence_capture_draws=tdraws.capture,
         trajectory_id=tdraws.infection_id,
         trajectory_draw=tdraws.infection_draw,
         trajectory_infection_time=tdraws.infection_time,
         ids=ids,
         infection_matrix=inf_matrix.matrix,
         infection_matrix_times=inf_matrix.times,
         Se=P.Se,
         Sp=P.Sp,
         pi1_vec=P.pi1_vec,
         year_effect_year_index=year_index,
         year_effect_calendar_year=calendar_year,
         year_effect_gamma=P.gamma,
         year_effect_annual_hazard=annual_h,
         test_mask=collect(test_mask),
         settings=(method=method,
                   year_process=String(proc),
                   tests=findall(identity, test_mask),
                   fixed_sesp=use_fixed,
                   penalty=penalty,
                   hazard_mean=hazard_mean,
                   hazard_sd=hazard_sd,
                   start_year=start_year,
                   repeat_captures=String(rc)))
    else
        error("Unknown method: $method")
    end

    return result
end
