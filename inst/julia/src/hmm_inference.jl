using DataFrames, Distributions, DensityInterface
using HiddenMarkovModels, Turing, StaticArrays
using LogExpFunctions: logistic, log1pexp
using Random, Statistics, CSV, LinearAlgebra
using JLD2
using ADTypes

const DynamicPPL = Turing.DynamicPPL
const var"@varname" = DynamicPPL.var"@varname"

const adtype = AutoForwardDiff()
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

struct TBEmission{T} <: Distribution{Multivariate, Discrete}
    Se       :: Vector{T}
    Sp       :: Vector{T}
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

    a1 = current_state_marginals[1] * b1
    a2 = current_state_marginals[2] * b2
    cscale = inv(a1 + a2)

    current_state_marginals[1] = a1 * cscale
    current_state_marginals[2] = a2 * cscale
    current_obs_likelihoods[1] = b1
    current_obs_likelihoods[2] = b2

    logL = -log(cscale) + logm
    return cscale, logL
end

const TEST_NAMES = ["ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak"]
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

@model function hmm_model(obs_seq, ctrl_seq, seq_ends, S, n_years, se_priors, sp_priors;
                          process::Symbol=:rw1,
                          hazard_mean::Float64=HAZARD_PRIOR_MEAN,
                          hazard_sd::Float64=HAZARD_PRIOR_SD,
                          pi1_a::Float64=1.0,
                          pi1_b::Float64=5.0,
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
    pi1 ~ Beta(pi1_a, pi1_b)

    if penalty
        for k in eachindex(Se)
            Turing.@addlogprob! -PENALTY_WEIGHT * log1pexp(-(Se[k] + Sp[k] - 1.0) / PENALTY_SCALE)
        end
    end

    hmm = DiagnosticHMM(pi1, alpha, gamma, Se, Sp, S)
    Turing.@addlogprob! logdensityof(hmm, obs_seq, ctrl_seq; seq_ends)
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
    pi1 = Float64(p[@varname(pi1)])
    (pi1=pi1, alpha=alpha, gamma=gamma, Se=Se, Sp=Sp, sigma_g=sigma_g, rho=rho)
end

function p_inf_over_time_pointestimate(individuals, P, numSeasons::Int)
    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    hmm = DiagnosticHMM(P.pi1, P.alpha, P.gamma, P.Se, P.Sp, numSeasons)
    for b in individuals
        ctrl_b = [Control(t) for t in b.times]
        gamma, _ = forward_backward(hmm, b.obs, ctrl_b)
        results[b.id] = gamma[2, :]
        times[b.id] = b.times
    end
    (p_inf=results, times=times)
end

struct DrawArrays
    pi1 :: Vector{Float64}
    alpha :: Matrix{Float64}
    gamma :: Matrix{Float64}
    Se :: Matrix{Float64}
    Sp :: Matrix{Float64}
end

function extract_all_draws(chain; n_tests::Int, S::Int, n_years::Int,
                           process::Symbol=:rw1, se_fixed=nothing, sp_fixed=nothing)
    pi1 = Float64.(vec(chain[@varname(pi1)]))
    n_samps = length(pi1)
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

    DrawArrays(pi1, alpha, gamma, Se, Sp)
end

@inline function hmm_for_draw(d::DrawArrays, i::Int, S::Int)
    DiagnosticHMM(d.pi1[i], d.alpha[:, i], d.gamma[:, i], d.Se[:, i], d.Sp[:, i], S)
end

function p_inf_over_time_nuts(individuals, chain; n_tests::Int, numSeasons::Int, n_years::Int,
                              process::Symbol=:rw1, se_fixed=nothing, sp_fixed=nothing)
    draws = extract_all_draws(chain; n_tests=n_tests, S=numSeasons, n_years=n_years,
                              process=process, se_fixed=se_fixed, sp_fixed=sp_fixed)
    n_samps = length(draws.pi1)

    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    for b in individuals
        ctrl_b = [Control(t) for t in b.times]
        acc = zeros(length(b.times))
        for i in 1:n_samps
            hmm = hmm_for_draw(draws, i, numSeasons)
            gamma, _ = forward_backward(hmm, b.obs, ctrl_b)
            acc .+= gamma[2, :]
        end
        results[b.id] = acc ./ n_samps
        times[b.id] = b.times
    end
    (p_inf=results, times=times)
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

# Posterior-draw version of the degeneracy check. Same rule as
# `check_mode_params`: only the assays actually used carry information.
function check_mode(chain; n_tests::Int=6, sp_prior_mean::AbstractVector=SP_FIXED_DEFAULT,
                    used::AbstractVector{Bool}=trues(n_tests),
                    prevalence_mean::Real=NaN)
    idx = findall(used)
    isempty(idx) && error("check_mode: no assays marked as used")

    Se = [vec(chain[@varname(Se[k])]) for k in 1:n_tests]
    Sp = [vec(chain[@varname(Sp[k])]) for k in 1:n_tests]
    pi1 = vec(chain[@varname(pi1)])

    youden = [Se[k] .+ Sp[k] .- 1.0 for k in 1:n_tests]
    frac_bad = [mean(youden[k] .< 0) for k in 1:n_tests]
    se_mean = [mean(Se[k]) for k in 1:n_tests]

    all_valid = all(frac_bad[idx] .== 0)
    se_collapsed = se_mean[idx] .< (1 .- sp_prior_mean[idx] .+ 0.05)
    high_state = mean(pi1) > 0.5 ||
                 (isfinite(prevalence_mean) && prevalence_mean > 0.5)
    mirror_like = any(se_collapsed) && high_state
    implausible_prevalence = isfinite(prevalence_mean) && prevalence_mean > 0.5

    (ok = all_valid && !mirror_like && !implausible_prevalence,
     frac_draws_below_youden0 = frac_bad,
     se_posterior_mean = se_mean,
     sp_posterior_mean = [mean(Sp[k]) for k in 1:n_tests],
     min_youden_used = minimum(mean.(youden[idx])),
     used = collect(used),
     used_tests = TEST_NAMES[idx],
     n_used = length(idx),
     pi1_mean = mean(pi1),
     prevalence_mean = prevalence_mean,
     mirror_like = mirror_like,
     implausible_prevalence = implausible_prevalence)
end

# Degeneracy check for point estimates.
#
# `used` is the assay mask. Masked assays keep prior-driven Se/Sp that carry no
# information about this fit, so averaging over all six hides exactly the
# failure this is meant to catch: with a single observed assay, prevalence and
# Se are not jointly identified, and the optimiser can land on the mirror
# solution (Se -> 0, prevalence -> 1). Judge only the assays that were used.
function check_mode_params(P; sp_prior_mean::AbstractVector=SP_FIXED_DEFAULT,
                           used::AbstractVector{Bool}=trues(length(P.Se)),
                           prevalence_mean::Real=NaN)
    idx = findall(used)
    isempty(idx) && error("check_mode_params: no assays marked as used")

    youden = P.Se .+ P.Sp .- 1.0
    youden_used = youden[idx]
    all_valid = all(youden_used .> 0)

    # Mirror solution: used assays have Se at or below their false-positive
    # rate, i.e. the "positive" label has flipped meaning.
    se_collapsed = P.Se[idx] .< (1 .- sp_prior_mean[idx] .+ 0.05)
    high_state = P.pi1 > 0.5 ||
                 (isfinite(prevalence_mean) && prevalence_mean > 0.5)
    mirror_like = any(se_collapsed) && high_state

    # A single-assay panel cannot separate prevalence from Se at all, so an
    # implausibly high prevalence is itself the symptom.
    implausible_prevalence = isfinite(prevalence_mean) && prevalence_mean > 0.5

    (ok = all_valid && !mirror_like && !implausible_prevalence,
     youden = youden,
     youden_used = youden_used,
     min_youden_used = minimum(youden_used),
     used = collect(used),
     used_tests = TEST_NAMES[idx],
     n_used = length(idx),
     prevalence_mean = prevalence_mean,
     mirror_like = mirror_like,
     implausible_prevalence = implausible_prevalence)
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
                           pi1_a::Float64=1.0,
                           pi1_b::Float64=5.0,
                           penalty::Bool=true,
                           start_year::Int=0,
                           repeat_captures::String="stack")
    Random.seed!(seed)
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
    model = hmm_model(obs_seq, ctrl_seq, seq_ends, numSeasons, n_years, se_priors, sp_priors;
                      process=proc,
                      hazard_mean=hazard_mean,
                      hazard_sd=hazard_sd,
                      pi1_a=pi1_a,
                      pi1_b=pi1_b,
                      penalty=penalty,
                      se_fixed=se_fixed_use,
                      sp_fixed=sp_fixed_use)

    ids = [b.id for b in individuals]

    result = if method == "nuts"
        if use_fixed
            error("NUTS with fixed Se/Sp is not supported in this interface; use MAP or MLE")
        end
        n_chains = parse(Int, get(ENV, "TEST2INFER_NUTS_CHAINS", "4"))
        backend = n_chains == 1 ? MCMCSerial() : MCMCThreads()
        chain = sample(model, NUTS(target_acc; adtype=adtype), backend, nuts_samples, n_chains;
                       progress=false, check_model=false)

        chain_cache_path = get(ENV, "TEST2INFER_NUTS_CHAIN_PATH", "")
        if !isempty(chain_cache_path)
            jldsave(chain_cache_path; chain)
        end

        p_inf_over_time_data = p_inf_over_time_nuts(individuals, chain;
                                                    n_tests=n_tests,
                                                    numSeasons=numSeasons,
                                                    n_years=n_years,
                                                    process=proc,
                                                    se_fixed=se_fixed_use,
                                                    sp_fixed=sp_fixed_use)
        prevalence = prevalence_two_ways(individuals, p_inf_over_time_data.p_inf, p_inf_over_time_data.times)
        mode_report = check_mode(chain; n_tests=n_tests, sp_prior_mean=sp_prior_mean_use,
                                 used=collect(test_mask),
                                 prevalence_mean=mean(filter(isfinite, prevalence.grid_proportion)))
        inf_matrix = create_infection_matrix(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, ids)
        p_inf_last = Dict(b.id => p_inf_over_time_data.p_inf[b.id][findlast(b.captured)] for b in individuals)

        se_est = [mean(vec(chain[@varname(Se[k])])) for k in 1:n_tests]
        sp_est = [mean(vec(chain[@varname(Sp[k])])) for k in 1:n_tests]
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
         ids=ids,
         infection_matrix=inf_matrix.matrix,
         infection_matrix_times=inf_matrix.times,
         mode_report=mode_report,
         Se=se_est,
         Sp=sp_est,
         year_effect_year_index=year_index,
         year_effect_calendar_year=calendar_year,
         year_effect_gamma=gamma_est,
         year_effect_annual_hazard=annual_h,
         test_mask=collect(test_mask),
         settings=(method=method,
                   year_process=String(proc),
                   tests=TEST_NAMES[findall(identity, test_mask)],
                   fixed_sesp=use_fixed,
                   penalty=penalty,
                   hazard_mean=hazard_mean,
                   hazard_sd=hazard_sd,
                   pi1_a=pi1_a,
                   pi1_b=pi1_b,
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
        mode_report = check_mode_params(P; sp_prior_mean=sp_prior_mean_use,
                                        used=collect(test_mask),
                                        prevalence_mean=mean(filter(isfinite, prevalence.grid_proportion)))
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
         ids=ids,
         infection_matrix=inf_matrix.matrix,
         infection_matrix_times=inf_matrix.times,
         mode_report=mode_report,
         Se=P.Se,
         Sp=P.Sp,
         year_effect_year_index=year_index,
         year_effect_calendar_year=calendar_year,
         year_effect_gamma=P.gamma,
         year_effect_annual_hazard=annual_h,
         test_mask=collect(test_mask),
         settings=(method=method,
                   year_process=String(proc),
                   tests=TEST_NAMES[findall(identity, test_mask)],
                   fixed_sesp=use_fixed,
                   penalty=penalty,
                   hazard_mean=hazard_mean,
                   hazard_sd=hazard_sd,
                   pi1_a=pi1_a,
                   pi1_b=pi1_b,
                   start_year=start_year,
                   repeat_captures=String(rc)))
    else
        error("Unknown method: $method")
    end

    return result
end
