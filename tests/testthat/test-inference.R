test_that("Julia engine can be set up", {
  skip_on_cran()
  skip_without_engine()
  expect_true(hmm_engine_loaded())
})

test_that("synthetic data helper returns expected columns", {
  d <- simulate_test_data(n_individuals = 20, n_periods = 3, seed = 1)
  expect_true(all(c("time", "id", "group", paste0("test_", 1:6)) %in% names(d)))
  expect_gt(nrow(d), 0)
})

test_that("MAP inference works with defaults and returns expanded outputs", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_test_data(n_individuals = 24, n_periods = 4, seed = 11)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  result <- hmm_inference(test_mat, method = "map", seed = 123)

  expect_true(all(c("individual", "prevalence_grid", "prevalence_capture", "sesp", "year_effect", "settings") %in% names(result)))
  expect_gt(nrow(result$individual), 0)
  expect_true(all(result$individual$p_infected_last >= 0 & result$individual$p_infected_last <= 1))
  expect_equal(nrow(result$sesp), 6)
  expect_true(all(c("test", "used", "Se", "Sp", "Youden") %in% names(result$sesp)))
  expect_equal(result$settings$year_process, "rw1")
  expect_true(all(result$sesp$used))
})

test_that("panel masking and fixed Se/Sp are respected", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_test_data(n_individuals = 22, n_periods = 4, seed = 12)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  se_fix <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
  sp_fix <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)

  fit <- hmm_inference(
    test_mat,
    method = "map",
    tests = c(3, 5, 6),
    year_process = "iid",
    se_fixed = se_fix,
    sp_fixed = sp_fix,
    seed = 99
  )

  expect_equal(as.integer(fit$settings$tests), c(3L, 5L, 6L))
  expect_true(isTRUE(fit$settings$fixed_sesp))
  expect_equal(round(fit$sesp$Se, 6), round(se_fix, 6))
  expect_equal(round(fit$sesp$Sp, 6), round(sp_fix, 6))
})

test_that("NUTS path still works on tiny synthetic problem", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_test_data(n_individuals = 12, n_periods = 2, seed = 13)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  fit <- hmm_inference(test_mat, method = "nuts", nuts_samples = 30, seed = 321)
  expect_true("individual" %in% names(fit))
  expect_gt(nrow(fit$individual), 0)
})


test_that("start_year labels the year_effect table", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_test_data(n_individuals = 20, n_periods = 4, seed = 21)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  no_year <- hmm_inference(test_mat, method = "map", seed = 5)
  expect_true(all(is.na(no_year$year_effect$calendar_year)))

  with_year <- hmm_inference(test_mat, method = "map", start_year = 2006, seed = 5)
  expect_equal(
    with_year$year_effect$calendar_year,
    2006L + with_year$year_effect$year_index - 1L
  )
})

test_that("a single-test panel is selected by column index", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_test_data(n_individuals = 24, n_periods = 4, seed = 31)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  fit <- hmm_inference(test_mat, method = "map", tests = 3, seed = 7)

  expect_equal(as.integer(fit$settings$tests), 3L)
  expect_equal(which(fit$sesp$used), 3L)
  expect_equal(fit$sesp$test, paste0("test_", 1:6))
})

test_that("test names are rejected -- tests are positional", {
  expect_error(
    hmm_inference(matrix(0, 4, 6), method = "map", tests = "test_3"),
    "logical mask or numeric indices"
  )
})

test_that("NUTS runs with fixed Se/Sp", {
  # This used to be rejected outright. The only real obstacle was that the
  # post-processing read Se/Sp out of the chain, where fixed values are never
  # sampled; it now reports the supplied values instead.
  skip_on_cran()
  skip_without_engine()
  se_fix <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
  sp_fix <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)
  d <- simulate_test_data(n_individuals = 12, n_periods = 2, seed = 13)
  test_mat <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])
  fit <- hmm_inference(test_mat, method = "nuts", nuts_samples = 30, seed = 321,
                       se_fixed = se_fix, sp_fixed = sp_fix)
  expect_gt(nrow(fit$individual), 0)
  expect_equal(fit$sesp$Se, se_fix, tolerance = 1e-8)
  expect_equal(fit$sesp$Sp, sp_fix, tolerance = 1e-8)
})

test_that("repeat captures in one time-step are stacked by default", {
  skip_on_cran()
  skip_without_engine()

  d <- simulate_test_data(n_individuals = 20, n_periods = 4, seed = 41)
  cols <- c("time", "id", "group", paste0("test_", 1:6))

  # Give one individual a second capture in a time-step it already occupies, and
  # make the two disagree, so the three rules cannot coincide by accident.
  extra <- d[1, ]
  extra$test_3 <- 1
  d2 <- rbind(d, extra)
  d2 <- d2[order(d2$id, d2$time), ]
  tm <- as.matrix(d2[, cols])

  fit <- hmm_inference(tm, method = "map", seed = 3)
  expect_equal(fit$settings$repeat_captures, "stack")

  pooled  <- hmm_inference(tm, method = "map", repeat_captures = "pool", seed = 3)
  dropped <- hmm_inference(tm, method = "map", repeat_captures = "last", seed = 3)

  expect_equal(pooled$settings$repeat_captures, "pool")
  expect_equal(dropped$settings$repeat_captures, "last")

  # The disagreeing repeat must actually change the answer, otherwise this
  # test would pass even if the extra capture were silently discarded.
  expect_false(isTRUE(all.equal(fit$individual$p_infected_last,
                                dropped$individual$p_infected_last)))
})

test_that("an unknown repeat_captures rule is rejected", {
  expect_error(
    hmm_inference(matrix(0, 4, 6), method = "map", repeat_captures = "nonsense"),
    "should be one of"
  )
})

test_that("initial infection probability varies with entry year", {
  skip_on_cran()
  skip_without_engine()

  d <- simulate_test_data(n_individuals = 40, n_periods = 6, seed = 51)
  tm <- as.matrix(d[, c("time", "id", "group", paste0("test_", 1:6))])

  fit <- hmm_inference(tm, method = "map", start_year = 2000)

  # pi1 is tied to the year effect, so it must not be one constant value.
  expect_gt(diff(range(fit$pi1_by_year$pi1)), 0)
  expect_true(all(fit$pi1_by_year$pi1 > 0 & fit$pi1_by_year$pi1 < 1))
  expect_equal(nrow(fit$pi1_by_year), nrow(fit$year_effect))
})
