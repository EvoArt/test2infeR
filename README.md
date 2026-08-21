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

# test_mat: one row per capture. Columns are time, id, group, then the six
# test results (1 positive, 0 negative, NA not run).
fit <- hmm_inference(test_mat)

# A subset of the tests, by column
fit_subset <- hmm_inference(test_mat, tests = c(3, 5, 6))

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

- `year_process`: `"rw1"` (default), `"iid"`, `"rw2"`, `"ar1"`, `"shrunk"`, `"none"`.
- `repeat_captures`: what to do when an animal is captured more than once
  within a single time-step. `"stack"` (default) lets every capture contribute
  its own emission terms, so repeat results compound; `"pool"` merges them into
  one observation per time-step; `"last"` keeps only the final capture and
  discards the rest (the behaviour before 0.3.0, kept only to reproduce
  earlier fits).
- `tests`: select the panel by column index (e.g. `c(3, 5, 6)`) or a length-6 logical mask
- `se_fixed`/`sp_fixed`: fix Se/Sp to supplied values (MAP/MLE)
- `se_prior_mean`, `sp_prior_mean`, `se_prior_ci`, `sp_prior_ci`: override the default Se/Sp priors
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

4. **`sesp`**: Fitted Se/Sp and Youden index by test, labelled `test_1`..`test_K`
   - `used`: whether the test was in the panel. Tests left out keep their prior
     means, so those rows are not estimates.

5. **`year_effect`**: Year-level gamma path and implied annual hazard.
   `calendar_year` is `NA` unless `start_year` was supplied.

6. **`pi1_by_year`**: Probability an individual is already infected when first
   seen, by entry year. Derived from the year effect, so it varies with it.

7. **`settings`**: Resolved run settings (panel, process, fixed/inferred Se/Sp)


## Notes

- First setup may download Julia and precompile dependencies (can take several minutes)
- Subsequent runs reuse cached Julia artifacts
- For a complete synthetic install-to-output run, see `inst/scripts/e2e_synthetic_panel_analysis.R`
