#' Simulate synthetic badger diagnostic capture data
#'
#' Generates longitudinal capture rows in the same shape expected by
#' [hmm_inference()]: `time`, `id`, `group`, and the six test columns.
#'
#' @param n_badgers Number of badgers.
#' @param n_years Number of calendar years (4 seasons per year).
#' @param capture_prob Per-season capture probability for each badger.
#' @param start_year Calendar start year. This also drives the assay-era
#'   structure: ELISA_BELT/IGRA/StatPak exist from 2006, DPP from 2014, and
#'   ELISA_INDIRECT before 2006. The default sits in the modern era so that
#'   all of Culture, IGRA and StatPak are actually observed.
#' @param seed Random seed.
#' @return A data frame with columns `time`, `id`, `group`,
#'   `ELISA_BELT`, `ELISA_INDIRECT`, `Culture`, `DPP`, `IGRA`, `StatPak`,
#'   `calendar_year`, and `quarter`.
#' @export
simulate_badger_data <- function(n_badgers = 120,
                                 n_years = 12,
                                 capture_prob = 0.45,
                                 start_year = 2006,
                                 seed = 123) {
  set.seed(seed)

  n_seasons <- n_years * 4
  test_names <- c("ELISA_BELT", "ELISA_INDIRECT", "Culture", "DPP", "IGRA", "StatPak")

  se <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
  sp <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)

  alpha <- stats::qlogis(c(0.04, 0.09, 0.02, 0.01))
  gamma <- cumsum(stats::rnorm(n_years, mean = 0, sd = 0.03))

  rows <- vector("list", n_badgers)

  for (id in seq_len(n_badgers)) {
    infected <- FALSE
    cap_rows <- vector("list", n_seasons)
    n_caps <- 0L

    for (t in seq_len(n_seasons)) {
      y <- ((t - 1L) %/% 4L) + 1L
      s <- ((t - 1L) %% 4L) + 1L

      if (!infected) {
        lam <- stats::plogis(alpha[s] + gamma[y])
        infected <- stats::runif(1) < lam
      }

      if (stats::runif(1) > capture_prob) next

      n_caps <- n_caps + 1L
      obs <- numeric(6)
      for (k in seq_len(6)) {
        p_pos <- if (infected) se[k] else (1 - sp[k])
        obs[k] <- as.numeric(stats::runif(1) < p_pos)
      }

      # Light assay-era structure for realism.
      year_cal <- start_year + (t - 1L) %/% 4L
      if (year_cal >= 2006) obs[1] <- NA_real_ else obs[2] <- NA_real_
      if (year_cal < 2014) obs[4] <- NA_real_
      if (year_cal < 2006 || year_cal > 2015) obs[6] <- NA_real_
      if (year_cal < 2006) obs[5] <- NA_real_

      cap_rows[[n_caps]] <- data.frame(
        time = t,
        id = id,
        group = 1,
        ELISA_BELT = obs[1],
        ELISA_INDIRECT = obs[2],
        Culture = obs[3],
        DPP = obs[4],
        IGRA = obs[5],
        StatPak = obs[6],
        stringsAsFactors = FALSE
      )
    }

    if (n_caps == 0L) {
      t <- sample.int(n_seasons, 1)
      year_cal <- start_year + (t - 1L) %/% 4L
      obs <- rep(0, 6)
      if (year_cal >= 2006) obs[1] <- NA_real_ else obs[2] <- NA_real_
      if (year_cal < 2014) obs[4] <- NA_real_
      if (year_cal < 2006 || year_cal > 2015) obs[6] <- NA_real_
      if (year_cal < 2006) obs[5] <- NA_real_
      cap_rows[[1]] <- data.frame(
        time = t,
        id = id,
        group = 1,
        ELISA_BELT = obs[1],
        ELISA_INDIRECT = obs[2],
        Culture = obs[3],
        DPP = obs[4],
        IGRA = obs[5],
        StatPak = obs[6],
        stringsAsFactors = FALSE
      )
      n_caps <- 1L
    }

    rows[[id]] <- do.call(rbind, cap_rows[seq_len(n_caps)])
  }

  out <- do.call(rbind, rows)
  out <- out[order(out$id, out$time), , drop = FALSE]

  out$calendar_year <- start_year + (out$time - 1L) %/% 4L
  out$quarter <- ((out$time - 1L) %% 4L) + 1L

  out[, c("time", "id", "group", test_names, "calendar_year", "quarter")]
}
