using DataFrames, Distributions, DensityInterface
using HiddenMarkovModels, Turing, ProgressMeter, StaticArrays
using LogExpFunctions: logistic
using Random, Statistics, CSV, LinearAlgebra
using JLD2
const DynamicPPL = Turing.DynamicPPL
const var"@varname" = DynamicPPL.var"@varname"
using ADTypes

adtype = AutoForwardDiff()
clamp_prob(x; eps=1e-9) = clamp(x, eps, 1 - eps)

# Time grid helpers
season_of(t::Int, S=4) = (t - 1) % S + 1
year_of(t::Int, S=4) = (t - 1) ÷ S + 1

# Per-observation control: global time index + this badger's sex code
struct Control
    t::Int
    sex_code::Float64   # 1.0 = female, 0.0 = male
end

# HMM struct
struct BadgerHMM{T,V1<:AbstractVector{T},V2<:AbstractVector{T},V3<:AbstractVector{T},V4<:AbstractVector{T}} <: AbstractHMM
    π1        :: T
    alpha     :: V1     # length S (season fixed effects)
    gamma     :: V2     # length n_years (year effects, already scaled by sigma_g)
    delta_sex :: T       # 0.0 for no-sex model
    Se        :: V3
    Sp        :: V4
    S         :: Int
end
Base.length(::BadgerHMM) = 2

HiddenMarkovModels.initialization(h::BadgerHMM) = SVector(1 - h.π1, h.π1)

function HiddenMarkovModels.transition_matrix(h::BadgerHMM, c::Control)
    lam = clamp_prob(logistic(h.alpha[season_of(c.t, h.S)] + h.gamma[year_of(c.t, h.S)] + h.delta_sex * c.sex_code))
    return SMatrix{2,2}(1 - lam, zero(lam), lam, one(lam))
end

# Emission distribution
struct TBEmission{T} <: Distribution{Multivariate, Discrete}
    Se       :: Vector{T}
    Sp       :: Vector{T}
    infected :: Bool
end
DensityInterface.DensityKind(::TBEmission) = HasLogDensity()
function DensityInterface.logdensityof(d::TBEmission, x::AbstractVector)
    ll = zero(eltype(d.Se))
    @inbounds for k in eachindex(x)
        isnan(x[k]) && continue
        ll += d.infected ? (x[k]==1 ? log(d.Se[k]) : log(1-d.Se[k])) :
                           (x[k]==1 ? log(1-d.Sp[k]) : log(d.Sp[k]))
    end
    return ll
end
function HiddenMarkovModels.obs_distributions(h::BadgerHMM, ::Control)
    SVector(TBEmission(h.Se, h.Sp, false), TBEmission(h.Se, h.Sp, true))
end

# Custom _forward_digest_observation! for optimization
function HiddenMarkovModels._forward_digest_observation!(
    current_state_marginals::AbstractVector{<:Real},
    current_obs_likelihoods::AbstractVector{<:Real},
    h::BadgerHMM,
    obs,
    c::Control;
    error_if_not_finite::Bool,
)
    logb1 = zero(eltype(h.Se))   # uninfected
    logb2 = zero(eltype(h.Se))   # infected
    @inbounds for k in eachindex(obs)
        v = obs[k]
        isnan(v) && continue
        pU = v == 1.0 ? (1 - h.Sp[k]) : h.Sp[k]
        pI = v == 1.0 ? h.Se[k] : (1 - h.Se[k])
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

# Weakly-informative Se/Sp priors per test
const SE_PRIOR_MEANS = [0.60, 0.70, 0.06, 0.70, 0.44, 0.56]
const SE_PRIOR_STDS  = [0.15, 0.15, 0.05, 0.15, 0.15, 0.15]
const SP_PRIOR_MEANS = [0.95, 0.95, 1.00, 0.95, 0.96, 0.98]
const SP_PRIOR_STDS  = [0.05, 0.05, 0.01, 0.05, 0.03, 0.03]

beta_from_moments(μ, σ) = (c = μ*(1-μ)/σ^2 - 1; Beta(max(μ*c, 0.5), max((1-μ)*c, 0.5)))
se_priors = beta_from_moments.(SE_PRIOR_MEANS, SE_PRIOR_STDS)
sp_priors = beta_from_moments.(SP_PRIOR_MEANS, SP_PRIOR_STDS)

# Turing model
@model function badger_model(obs_seq, ctrl_seq, seq_ends, S, n_years, se_priors, sp_priors,
                              ::Val{with_sex}=Val(false)) where {with_sex}
    alpha   ~ MvNormal(zeros(S), I(S))
    sigma_g ~ truncated(Normal(0, 0.5); lower=0)
    gamma_raw ~ MvNormal(zeros(n_years), I(n_years))
    gamma = sigma_g * gamma_raw

    if with_sex
        delta_sex ~ Normal(0.0, 1.0)
    else
        delta_sex = zero(sigma_g)
    end

    Se ~ arraydist(se_priors)
    Sp ~ arraydist(sp_priors)
    pi1 ~ Beta(1.0, 5.0)

    hmm = BadgerHMM(pi1, alpha, gamma, delta_sex, Se, Sp, S)
    Turing.@addlogprob! logdensityof(hmm, obs_seq, ctrl_seq; seq_ends)
end

# Extract parameters from result
function extract_params(result; with_sex::Bool, n_tests::Int, numSeasons::Int, n_years::Int)
    p = result.params
    alpha = [p[@varname(alpha[s])] for s in 1:numSeasons]
    sigma_g = p[@varname(sigma_g)]
    gamma = sigma_g .* [p[@varname(gamma_raw[y])] for y in 1:n_years]
    Se = [p[@varname(Se[k])] for k in 1:n_tests]
    Sp = [p[@varname(Sp[k])] for k in 1:n_tests]
    pi1 = p[@varname(pi1)]
    delta_sex = with_sex ? p[@varname(delta_sex)] : 0.0
    (pi1=pi1, alpha=alpha, gamma=gamma, delta_sex=delta_sex, Se=Se, Sp=Sp)
end

# Posterior infection probability for point estimate (MAP/MLE)
function p_inf_last_pointestimate(badgers, P, numSeasons::Int)
    results = Dict{Int, Float64}()
    for b in badgers
        hmm = BadgerHMM(P.pi1, P.alpha, P.gamma, P.delta_sex, P.Se, P.Sp, numSeasons)
        ctrl_b = [Control(t, Float64(b.female)) for t in b.times]
        gamma, _ = forward_backward(hmm, b.obs, ctrl_b)
        results[b.id] = gamma[2, end]
    end
    results
end

# Infection probability over time for point estimate (MAP/MLE)
function p_inf_over_time_pointestimate(badgers, P, numSeasons::Int)
    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    for b in badgers
        hmm = BadgerHMM(P.pi1, P.alpha, P.gamma, P.delta_sex, P.Se, P.Sp, numSeasons)
        ctrl_b = [Control(t, Float64(b.female)) for t in b.times]
        gamma, _ = forward_backward(hmm, b.obs, ctrl_b)
        results[b.id] = gamma[2, :]
        times[b.id] = b.times
    end
    (p_inf=results, times=times)
end

# Posterior infection probability for NUTS
function p_inf_last_nuts(badgers, chain; with_sex::Bool, n_tests::Int, numSeasons::Int, n_years::Int)
    n_samps = length(vec(chain[@varname(pi1)]))
    pi1s = vec(chain[@varname(pi1)])
    sigma_gs = vec(chain[@varname(sigma_g)])
    alphas = [vec(chain[@varname(alpha[s])]) for s in 1:numSeasons]
    gamma_raws = [vec(chain[@varname(gamma_raw[y])]) for y in 1:n_years]
    delta_sexs = with_sex ? vec(chain[@varname(delta_sex)]) : zeros(n_samps)
    Ses = [vec(chain[@varname(Se[k])]) for k in 1:n_tests]
    Sps = [vec(chain[@varname(Sp[k])]) for k in 1:n_tests]

    results = Dict{Int, Float64}()
    for b in badgers
        ctrl_b = [Control(t, Float64(b.female)) for t in b.times]
        samples = [begin
            alpha = Float64[alphas[s][i] for s in 1:numSeasons]
            gamma = Float64[sigma_gs[i] * gamma_raws[y][i] for y in 1:n_years]
            delta = Float64(delta_sexs[i])
            Se = Float64[Ses[k][i] for k in 1:n_tests]
            Sp = Float64[Sps[k][i] for k in 1:n_tests]
            hmm = BadgerHMM(Float64(pi1s[i]), alpha, gamma, delta, Se, Sp, numSeasons)
            gam, _ = forward_backward(hmm, b.obs, ctrl_b)
            gam[2, end]
        end for i in 1:n_samps]
        results[b.id] = mean(samples)
    end
    results
end

# Infection probability over time for NUTS
function p_inf_over_time_nuts(badgers, chain; with_sex::Bool, n_tests::Int, numSeasons::Int, n_years::Int)
    n_samps = length(vec(chain[@varname(pi1)]))
    pi1s = vec(chain[@varname(pi1)])
    sigma_gs = vec(chain[@varname(sigma_g)])
    alphas = [vec(chain[@varname(alpha[s])]) for s in 1:numSeasons]
    gamma_raws = [vec(chain[@varname(gamma_raw[y])]) for y in 1:n_years]
    delta_sexs = with_sex ? vec(chain[@varname(delta_sex)]) : zeros(n_samps)
    Ses = [vec(chain[@varname(Se[k])]) for k in 1:n_tests]
    Sps = [vec(chain[@varname(Sp[k])]) for k in 1:n_tests]

    results = Dict{Int, Vector{Float64}}()
    times = Dict{Int, Vector{Int}}()
    for b in badgers
        ctrl_b = [Control(t, Float64(b.female)) for t in b.times]
        samples = [begin
            alpha = Float64[alphas[s][i] for s in 1:numSeasons]
            gamma = Float64[sigma_gs[i] * gamma_raws[y][i] for y in 1:n_years]
            delta = Float64(delta_sexs[i])
            Se = Float64[Ses[k][i] for k in 1:n_tests]
            Sp = Float64[Sps[k][i] for k in 1:n_tests]
            hmm = BadgerHMM(Float64(pi1s[i]), alpha, gamma, delta, Se, Sp, numSeasons)
            gam, _ = forward_backward(hmm, b.obs, ctrl_b)
            gam[2, :]
        end for i in 1:n_samps]
        # Average across samples for each time point
        n_times = length(b.times)
        mean_p = [mean([samples[i][t] for i in 1:n_samps]) for t in 1:n_times]
        results[b.id] = mean_p
        times[b.id] = b.times
    end
    (p_inf=results, times=times)
end

# Calculate prevalence over time from individual infection probabilities
function calculate_prevalence(p_inf_over_time, times, n_individuals)
    # Get all unique time points
    all_times = sort(unique(vcat(values(times)...)))
    
    # Calculate proportion and total infected at each time point
    proportion = Float64[]
    total = Int[]
    
    for t in all_times
        infected_count = 0
        total_count = 0
        for (id, t_vec) in times
            idx = findfirst(==(t), t_vec)
            if idx !== nothing
                infected_count += (p_inf_over_time[id][idx] >= 0.5 ? 1 : 0)
                total_count += 1
            end
        end
        if total_count > 0
            push!(proportion, infected_count / total_count)
            push!(total, infected_count)
        else
            push!(proportion, NaN)
            push!(total, 0)
        end
    end
    
    (times=all_times, proportion=proportion, total=total)
end

# Main inference function
function run_hmm_inference(test_mat::Matrix{Float64}, method::String, 
                          with_sex::Bool, nuts_samples::Int, 
                          target_acc::Float64, seed::Int)
    Random.seed!(seed)
    
    # Parse test matrix: assume rows are (time, id, tests...)
    # Simplified: assume single time point per individual for now
    n_individuals = size(test_mat, 1)
    n_tests = size(test_mat, 2)
    n_timepoints = 1
    numSeasons = 4
    n_years = 1
    
    # Create badger data structure
    badgers = [(id=i, times=[1], obs=[test_mat[i, :]], female=0) for i in 1:n_individuals]
    
    # Pack sequences
    obs_seq = Vector{Vector{Float64}}()
    ctrl_seq = Vector{Control}()
    seq_ends = Int[]
    for b in badgers
        for j in eachindex(b.times)
            push!(obs_seq, b.obs[j])
            push!(ctrl_seq, Control(b.times[j], Float64(b.female)))
        end
        push!(seq_ends, length(obs_seq))
    end
    
    # Create model
    model = badger_model(obs_seq, ctrl_seq, seq_ends, numSeasons, n_years, se_priors, sp_priors, Val(with_sex))
    
    result = if method == "nuts"
        # Fit MAP first for initialization
        map_ = maximum_a_posteriori(model; adtype=adtype, check_model=false)
        # Initialize NUTS from MAP to avoid label switching
        init_params = map_.params
        chain = sample(model, NUTS(target_acc; adtype=adtype), nuts_samples; progress=false, check_model=false,
                       initial_params=DynamicPPL.InitFromParams(init_params))
        p_inf_last = p_inf_last_nuts(badgers, chain; with_sex=with_sex, n_tests=n_tests, numSeasons=numSeasons, n_years=n_years)
        p_inf_over_time_data = p_inf_over_time_nuts(badgers, chain; with_sex=with_sex, n_tests=n_tests, numSeasons=numSeasons, n_years=n_years)
        prevalence = calculate_prevalence(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, n_individuals)
        (p_inf_last=p_inf_last, p_inf_over_time=p_inf_over_time_data.p_inf, times=p_inf_over_time_data.times, 
         prevalence_times=prevalence.times, prevalence_proportion=prevalence.proportion, prevalence_total=prevalence.total,
         ids=[b.id for b in badgers])
    elseif method == "map"
        map_ = maximum_a_posteriori(model; adtype=adtype, check_model=false)
        P = extract_params(map_; with_sex=with_sex, n_tests=n_tests, numSeasons=numSeasons, n_years=n_years)
        p_inf_last = p_inf_last_pointestimate(badgers, P, numSeasons)
        p_inf_over_time_data = p_inf_over_time_pointestimate(badgers, P, numSeasons)
        prevalence = calculate_prevalence(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, n_individuals)
        (p_inf_last=p_inf_last, p_inf_over_time=p_inf_over_time_data.p_inf, times=p_inf_over_time_data.times,
         prevalence_times=prevalence.times, prevalence_proportion=prevalence.proportion, prevalence_total=prevalence.total,
         ids=[b.id for b in badgers])
    elseif method == "mle"
        mle = maximum_likelihood(model; adtype=adtype, check_model=false)
        P = extract_params(mle; with_sex=with_sex, n_tests=n_tests, numSeasons=numSeasons, n_years=n_years)
        p_inf_last = p_inf_last_pointestimate(badgers, P, numSeasons)
        p_inf_over_time_data = p_inf_over_time_pointestimate(badgers, P, numSeasons)
        prevalence = calculate_prevalence(p_inf_over_time_data.p_inf, p_inf_over_time_data.times, n_individuals)
        (p_inf_last=p_inf_last, p_inf_over_time=p_inf_over_time_data.p_inf, times=p_inf_over_time_data.times,
         prevalence_times=prevalence.times, prevalence_proportion=prevalence.proportion, prevalence_total=prevalence.total,
         ids=[b.id for b in badgers])
    else
        error("Unknown method: $method")
    end
    
    return result
end
