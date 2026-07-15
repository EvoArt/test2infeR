# Minimal inference tests for CI
# Tests MAP, MLE, and NUTS on a small synthetic dataset

test_that("Julia engine can be set up", {
  skip_on_cran()
  expect_silent(setup_hmm_engine())
  expect_true(hmm_engine_loaded())
})

test_that("MAP inference works on synthetic data", {
  skip_on_cran()
  
  # Small synthetic test matrix (3 individuals, 6 tests)
  test_mat <- matrix(c(
    1, 0, 1, 0, 1, 0,  # Individual 1: mixed results
    0, 0, 0, 0, 0, 0,  # Individual 2: all negative
    1, 1, 1, 1, 1, 1   # Individual 3: all positive
  ), nrow = 3, byrow = TRUE)
  
  result <- hmm_inference(test_mat, method = "map", nuts_samples = 100, seed = 123)
  
  # Check result structure
  expect_true("individual" %in% names(result))
  expect_true("p_inf_over_time" %in% names(result))
  expect_true("prevalence" %in% names(result))
  
  # Check individual results
  expect_equal(nrow(result$individual), 3)
  expect_true(all(result$individual$p_infected_last >= 0))
  expect_true(all(result$individual$p_infected_last <= 1))
  
  # Check prevalence results
  expect_true("time" %in% names(result$prevalence))
  expect_true("proportion_infected" %in% names(result$prevalence))
  expect_true("total_infected" %in% names(result$prevalence))
})

test_that("MLE inference works on synthetic data", {
  skip_on_cran()
  
  # Small synthetic test matrix
  test_mat <- matrix(c(
    1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1
  ), nrow = 3, byrow = TRUE)
  
  result <- hmm_inference(test_mat, method = "mle", nuts_samples = 100, seed = 123)
  
  # Check result structure
  expect_true("individual" %in% names(result))
  expect_equal(nrow(result$individual), 3)
  expect_true(all(result$individual$p_infected_last >= 0))
  expect_true(all(result$individual$p_infected_last <= 1))
})

test_that("NUTS inference works on synthetic data", {
  skip_on_cran()
  
  # Small synthetic test matrix
  test_mat <- matrix(c(
    1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1
  ), nrow = 3, byrow = TRUE)
  
  # Use very few samples for CI speed
  result <- hmm_inference(test_mat, method = "nuts", nuts_samples = 50, seed = 123)
  
  # Check result structure
  expect_true("individual" %in% names(result))
  expect_equal(nrow(result$individual), 3)
  expect_true(all(result$individual$p_infected_last >= 0))
  expect_true(all(result$individual$p_infected_last <= 1))
})

test_that("Inference handles edge cases", {
  skip_on_cran()
  
  # All negative
  test_mat_all_neg <- matrix(c(0, 0, 0, 0, 0, 0), nrow = 1, byrow = TRUE)
  result <- hmm_inference(test_mat_all_neg, method = "map", nuts_samples = 100, seed = 123)
  expect_true(result$individual$p_infected_last < 0.5)
  
  # All positive
  test_mat_all_pos <- matrix(c(1, 1, 1, 1, 1, 1), nrow = 1, byrow = TRUE)
  result <- hmm_inference(test_mat_all_pos, method = "map", nuts_samples = 100, seed = 123)
  expect_true(result$individual$p_infected_last > 0.5)
})
