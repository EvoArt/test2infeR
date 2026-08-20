test_that("Julia engine can be set up", {
  skip_on_cran()
  skip_without_engine()
  expect_true(hmm_engine_loaded())
})

test_that("synthetic data helper returns expected columns", {
  d <- simulate_badger_data(n_badgers = 20, n_years = 3, seed = 1)
  expect_true(all(c(
    "time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT",
    "Culture", "DPP", "IGRA", "StatPak", "calendar_year", "quarter"
  ) %in% names(d)))
  expect_gt(nrow(d), 0)
})

test_that("MAP inference works with defaults and returns expanded outputs", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_badger_data(n_badgers = 24, n_years = 4, seed = 11)
  test_mat <- as.matrix(d[, c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")])

  result <- hmm_inference(test_mat, method = "map", seed = 123)

  expect_true(all(c("individual", "prevalence_grid", "prevalence_capture", "sesp", "year_effect", "settings") %in% names(result)))
  expect_gt(nrow(result$individual), 0)
  expect_true(all(result$individual$p_infected_last >= 0 & result$individual$p_infected_last <= 1))
  expect_equal(nrow(result$sesp), 6)
  expect_true(all(c("test", "used", "Se", "Sp", "Youden") %in% names(result$sesp)))
  expect_equal(result$settings$year_process, "iid")
  expect_true(all(result$sesp$used))
})

test_that("panel masking and fixed Se/Sp are respected", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_badger_data(n_badgers = 22, n_years = 4, seed = 12)
  test_mat <- as.matrix(d[, c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")])

  se_fix <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
  sp_fix <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)

  fit <- hmm_inference(
    test_mat,
    method = "map",
    tests = c("Culture", "IGRA", "StatPak"),
    year_process = "iid",
    se_fixed = se_fix,
    sp_fixed = sp_fix,
    seed = 99
  )

  expect_equal(fit$settings$tests, c("Culture", "IGRA", "StatPak"))
  expect_true(isTRUE(fit$settings$fixed_sesp))
  expect_equal(round(fit$sesp$Se, 6), round(se_fix, 6))
  expect_equal(round(fit$sesp$Sp, 6), round(sp_fix, 6))
})

test_that("NUTS path still works on tiny synthetic problem", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_badger_data(n_badgers = 12, n_years = 2, seed = 13)
  test_mat <- as.matrix(d[, c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")])

  fit <- hmm_inference(test_mat, method = "nuts", nuts_samples = 30, seed = 321)
  expect_true("individual" %in% names(fit))
  expect_gt(nrow(fit$individual), 0)
})


test_that("start_year labels the year_effect table", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_badger_data(n_badgers = 20, n_years = 4, seed = 21)
  test_mat <- as.matrix(d[, c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")])

  no_year <- hmm_inference(test_mat, method = "map", seed = 5)
  expect_true(all(is.na(no_year$year_effect$calendar_year)))

  with_year <- hmm_inference(test_mat, method = "map", start_year = 2006, seed = 5)
  expect_equal(
    with_year$year_effect$calendar_year,
    2006L + with_year$year_effect$year_index - 1L
  )
})

test_that("mode check judges only the assays actually used", {
  skip_on_cran()
  skip_without_engine()
  d <- simulate_badger_data(n_badgers = 24, n_years = 4, seed = 31)
  test_mat <- as.matrix(d[, c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")])

  fit <- hmm_inference(test_mat, method = "map", tests = c("Culture"), seed = 7)

  expect_equal(as.character(fit$mode_report$used_tests), "Culture")
  expect_equal(as.integer(fit$mode_report$n_used), 1L)
  # min_youden_used must come from Culture alone, not the prior-driven rest.
  expect_equal(
    as.numeric(fit$mode_report$min_youden_used),
    fit$sesp$Youden[fit$sesp$test == "Culture"],
    tolerance = 1e-8
  )
})

test_that("NUTS with fixed Se/Sp is rejected in R, before Julia runs", {
  se_fix <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
  sp_fix <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)
  expect_error(
    hmm_inference(matrix(0, 4, 6), method = "nuts", se_fixed = se_fix, sp_fixed = sp_fix),
    "not supported"
  )
})

test_that("repeat captures in one time-step are stacked by default", {
  skip_on_cran()
  skip_without_engine()

  d <- simulate_badger_data(n_badgers = 20, n_years = 4, seed = 41)
  cols <- c("time", "id", "group", "ELISA_BELT", "ELISA_INDIRECT",
            "Culture", "DPP", "IGRA", "StatPak")

  # Give one badger a second capture in a time-step it already occupies, and
  # make the two disagree, so the three rules cannot coincide by accident.
  extra <- d[1, ]
  extra$Culture <- 1
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
