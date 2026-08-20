#' Simulate longitudinal diagnostic test data
#'
#' Generates capture rows in the shape [hmm_inference()] expects: `time`, `id`,
#' `group`, then one column per test.
#'
#' @param n_individuals Number of individuals.
#' @param n_periods Number of periods (4 time-steps each).
#' @param capture_prob Per-step probability of observing an individual.
#' @param n_tests Number of tests.
#' @param se,sp Test sensitivities and specificities, length `n_tests`.
#' @param missing_prob Probability that a given test is not run at a capture.
#' @param seed Random seed.
#' @return A data frame with `time`, `id`, `group` and `test_1`..`test_n`.
#' @export
simulate_test_data <- function(n_individuals = 120,
                               n_periods = 12,
                               capture_prob = 0.45,
                               n_tests = 6,
                               se = c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492),
                               sp = c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931),
                               missing_prob = 0.2,
                               seed = 123) {
  set.seed(seed)

  if (length(se) != n_tests || length(sp) != n_tests) {
    stop("`se` and `sp` must have length `n_tests`.")
  }

  n_steps <- n_periods * 4
  test_names <- paste0("test_", seq_len(n_tests))

  alpha <- stats::qlogis(c(0.04, 0.09, 0.02, 0.01))
  gamma <- cumsum(stats::rnorm(n_periods, mean = 0, sd = 0.03))

  rows <- vector("list", n_individuals)

  for (id in seq_len(n_individuals)) {
    infected <- FALSE
    caps <- vector("list", n_steps)
    n_caps <- 0L

    for (t in seq_len(n_steps)) {
      y <- ((t - 1L) %/% 4L) + 1L
      s <- ((t - 1L) %% 4L) + 1L

      if (!infected) {
        infected <- stats::runif(1) < stats::plogis(alpha[s] + gamma[y])
      }
      if (stats::runif(1) > capture_prob) next

      n_caps <- n_caps + 1L
      obs <- numeric(n_tests)
      for (k in seq_len(n_tests)) {
        p_pos <- if (infected) se[k] else (1 - sp[k])
        obs[k] <- as.numeric(stats::runif(1) < p_pos)
        if (stats::runif(1) < missing_prob) obs[k] <- NA_real_
      }

      caps[[n_caps]] <- c(list(time = t, id = id, group = 1),
                          stats::setNames(as.list(obs), test_names))
    }

    # Every individual needs at least one capture to appear on the grid.
    if (n_caps == 0L) {
      caps[[1]] <- c(list(time = sample.int(n_steps, 1), id = id, group = 1),
                     stats::setNames(as.list(rep(0, n_tests)), test_names))
      n_caps <- 1L
    }

    rows[[id]] <- do.call(rbind, lapply(caps[seq_len(n_caps)], as.data.frame))
  }

  out <- do.call(rbind, rows)
  out[order(out$id, out$time), , drop = FALSE]
}
