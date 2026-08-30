"""
    Test2InfEngine

HMM inference engine for longitudinal diagnostic test data, backing the
`test2infeR` R package.

This exists as a package rather than a script that R `include()`s so that the
model's compiled code can be cached. Loading the engine used to pay ~40 s of
JIT compilation on the first `run_hmm_inference` call of every R session; the
`@compile_workload` block at the bottom of this file moves that cost to package
precompilation, which happens once at install time.

`run_hmm_inference` is exported, so `using Test2InfEngine` puts it in `Main`
and the R side can keep calling it as a bare name.
"""
module Test2InfEngine

export run_hmm_inference

include("hmm_inference.jl")
# Pre-tuned dense HMC constants for the five sql-e2e bundle variants, plus the
# variant lookup. Data and dispatch only, no AD dependency.
include("fastpath.jl")

using PrecompileTools: @setup_workload, @compile_workload

# Run a miniature fit of each method so that the model, its AD tape and the HMM
# forward pass are all compiled into the precompile cache. The data is tiny and
# the sample counts are the smallest that still exercise the real code paths --
# this block runs at precompile time, so it trades install time for the startup
# time of every later session.
@setup_workload begin
    n_ind, n_per = 6, 4
    rows = Vector{Float64}[]
    for id in 1:n_ind, t in 1:n_per
        # Deterministic pattern rather than rand(): precompilation should not
        # depend on RNG state, and the values only need to be valid, not random.
        push!(rows, vcat(Float64[t, id, 1.0],
                         Float64[(id + t + k) % 2 for k in 1:6]))
    end
    test_mat = Matrix(reduce(hcat, rows)')

    # NUTS defaults to 4 threaded chains. Force a single serial chain here:
    # precompilation must not depend on the runner's thread count, and the
    # per-chain code is the part worth caching either way.
    prev_chains = get(ENV, "TEST2INFER_NUTS_CHAINS", nothing)
    ENV["TEST2INFER_NUTS_CHAINS"] = "1"

    @compile_workload begin
        run_hmm_inference(copy(test_mat), "map", 2, 0.8, 1)
        run_hmm_inference(copy(test_mat), "mle", 2, 0.8, 1)
        run_hmm_inference(copy(test_mat), "nuts", 2, 0.8, 1)
        # The reverse-mode path, compiled here rather than on the first real
        # call of every session. Mooncake is a full dependency and is loaded at
        # module load, so DifferentiationInterface's Mooncake extension is
        # already wired up by the time this prepares a gradient -- which is the
        # whole reason the dependency is not lazy.
        #
        # This is the expensive one to JIT: building the Mooncake tape for the
        # HMM forward pass dominates the first Mooncake fit in a fresh session.
        run_hmm_inference(copy(test_mat), "nuts", 2, 0.8, 1; ad_name="mooncake")
    end

    if prev_chains === nothing
        delete!(ENV, "TEST2INFER_NUTS_CHAINS")
    else
        ENV["TEST2INFER_NUTS_CHAINS"] = prev_chains
    end
end

end # module
