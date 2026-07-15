# test2infeR

R interface to a Julia-based Hidden Markov Model (HMM) inference engine for diagnostic testing data.

## Installation

Install from GitHub:

```r
# install.packages("devtools")
devtools::install_github("EvoArt/test2infeR")
```

## Usage

```r
library(test2infeR)

# Set up the Julia engine (downloads Julia if needed)
setup_hmm_engine()

# Create a test matrix (individuals x tests)
# Each row is an individual, each column is a test result (0/1 or NA for missing)
test_mat <- matrix(c(
  1, 0, 1, 0, 1, 0,  # Individual 1: mixed results
  0, 0, 0, 0, 0, 0,  # Individual 2: all negative
  1, 1, 1, 1, 1, 1   # Individual 3: all positive
), nrow = 3, byrow = TRUE)

# Run inference
result <- hmm_inference(test_mat, method = "map")

# View individual-level results
print(result$individual)

# View infection probabilities over time
print(result$p_inf_over_time)

# View prevalence over time
print(result$prevalence)
```

## Methods

- `method = "nuts"`: No-U-Turn Sampler (full Bayesian inference)
- `method = "map"`: Maximum a Posteriori estimation
- `method = "mle"`: Maximum Likelihood estimation

## Output

Returns a list with three data frames:

1. **`individual`**: Individual-level results
   - `id`: Individual identifier
   - `p_infected_last`: Posterior probability of infection at last observation
   - `method`: Inference method used

2. **`p_inf_over_time`**: Infection probabilities over time for each individual
   - `id`: Individual identifier
   - `time`: Time point
   - `p_infected`: Posterior probability of infection at that time

3. **`prevalence`**: Population-level prevalence over time
   - `time`: Time point
   - `proportion_infected`: Proportion of population infected (treating last test as exit)
   - `total_infected`: Total number of infected individuals

## Implementation

The Julia engine uses the same HMM implementation as the RData2 model:
- 2-state HMM (Uninfected → Infected, absorbing)
- Seasonal and year effects on transition probabilities
- Optional sex effect
- Test-specific sensitivity and specificity with weakly informative priors
- Forward-backward smoothing for individual-level posterior probabilities
- NUTS initialized from MAP to avoid label switching

## Notes

- The package automatically handles Julia installation via JuliaCall
- First run will download Julia and install dependencies (may take several minutes)
- Subsequent runs will reuse the cached Julia installation
- No data files are included in the package - users must provide their own test matrices
