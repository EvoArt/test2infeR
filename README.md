# test2infeR

Hidden Markov Model (HMM) inference engine for diagnostic testing data.

## Installation

Install from GitHub:

```r
# install.packages("devtools")
devtools::install_github("EvoArt/test2infeR")
```

## Usage

```r
library(test2infeR)

# Explicitly set up the Julia engine (downloads Julia if needed)
setup_hmm_engine(install_julia = TRUE)

# Synthetic data for quick checks / CI
d <- simulate_badger_data(n_badgers = 80, n_years = 8, seed = 123)
test_mat <- as.matrix(d[, c(
  "time", "id", "group",
  "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak"
)])

# MAP is default, iid year effect is default
fit <- hmm_inference(test_mat)

# Label the year effect with real calendar years (time == 1 is start_year Q1)
fit_dated <- hmm_inference(test_mat, start_year = 2006)

# Panel masking + fixed Se/Sp
fit_fixed_cgi <- hmm_inference(
  test_mat,
  tests = c("Culture", "IGRA", "StatPak"),
  se_fixed = c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492),
  sp_fixed = c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)
)

# Alternate year process
fit_rw1 <- hmm_inference(test_mat, year_process = "rw1")

# Outputs
fit$individual
fit$prevalence_grid
fit$sesp
fit$year_effect
```

## Methods

- `method = "map"`: Maximum a Posteriori estimation
- `method = "mle"`: Maximum Likelihood estimation
- `method = "nuts"`: No-U-Turn Sampler (full Bayesian inference)

## Main options

- `year_process`: `"iid"` (default), `"rw1"`, `"rw2"`, `"ar1"`, `"shrunk"`, `"none"`.
  The default is the exchangeable year effect `gamma_y = sigma_g * z_y` with
  `sigma_g ~ Normal+(0, 0.3)` - the process used by the validated reference
  model. The alternatives smooth the hazard path but leave per-badger infection
  probabilities essentially unchanged (r >= 0.994 across all variants), so keep
  the default unless you specifically want a smooth hazard.
- `repeat_captures`: what to do when an animal is captured more than once
  within a single time-step. `"stack"` (default) lets every capture contribute
  its own emission terms, so repeat results compound; `"pool"` merges them into
  one observation per time-step; `"last"` keeps only the final capture and
  discards the rest. **`"last"` was the behaviour before 0.3.0** and is retained
  only to reproduce earlier fits - it loses data silently.
- `tests`: select assay panel by names, indices, or a logical mask
- `se_fixed`/`sp_fixed`: fix Se/Sp to supplied values (MAP/MLE)
- `se_prior_mean`, `sp_prior_mean`, `se_prior_ci`, `sp_prior_ci`: override default Table-1 priors
- `start_year`: calendar year of `time == 1`, used to label `year_effect$calendar_year`

## Output

Returns a list containing:

1. **`individual`**: Individual-level results
   - `id`: Individual identifier
   - `p_infected_last`: Posterior probability of infection at last observation
   - `method`: Inference method used

2. **`p_inf_over_time`**: Infection probabilities over time for each individual
   - `id`: Individual identifier
   - `time`: Time point
   - `p_infected`: Posterior probability of infection at that time

3. **`prevalence_grid`** / **`prevalence_capture`**: Population-level prevalence over time
   - `time`: Time point
   - `proportion_infected`: Proportion infected
   - `total_infected`: Total number of infected individuals

4. **`sesp`**: Fitted Se/Sp and Youden index by test
   - `used`: whether the assay was in the panel. **Masked assays keep their
     prior means** - those rows are not estimates from this fit and must not be
     read as results.

5. **`year_effect`**: Year-level gamma path and implied annual hazard.
   `calendar_year` is `NA` unless `start_year` was supplied.

6. **`settings`**: Resolved run settings (panel, process, fixed/inferred Se/Sp)

7. **`mode_report`**: Identifiability check. **Judged only on the assays
   actually used**, since masked assays carry prior-driven Se/Sp that say
   nothing about the fit. `ok = FALSE` means the fit is on a ridge - its
   prevalence is not an estimate. See the identifiability note below.

## Implementation

The Julia engine uses a 2-state absorbing HMM (Uninfected -> Infected):
- Per-season latent grid between first and last capture per badger
- Seasonal hazards + configurable year process (`iid` default)
- Test-specific Se/Sp with Table-1 priors by default
- Optional fixed Se/Sp regime
- Forward-backward smoothing for individual-level posterior probabilities

## Identifiability

With a **single observed assay and inferred Se/Sp, prevalence and sensitivity
are not jointly identified.** The posterior is bimodal: one mode has high Se and
low prevalence, the mirror mode has Se collapsing toward zero and prevalence
toward one. Which one an optimiser reaches depends only on its starting point,
so neither is a result.

`mode_report` catches this - it flags a used assay whose Se has fallen to or
below its false-positive rate, and any fit whose mean prevalence exceeds 50%.
Always check `fit$mode_report$ok` before reading a prevalence estimate from a
reduced panel. To use a single assay, fix Se/Sp rather than inferring them.

## Notes

- First setup may download Julia and precompile dependencies (can take several minutes)
- Subsequent runs reuse cached Julia artifacts
- For a complete synthetic install-to-output run, see `inst/scripts/e2e_synthetic_panel_analysis.R`
