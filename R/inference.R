#' Run HMM inference on diagnostic test data
#'
#' @param test_mat Test matrix (individuals x tests)
#' @param method Inference method: "map", "mle", or "nuts"
#' @param nuts_samples Number of NUTS samples (for method="nuts")
#' @param traj_draws Trajectories sampled per badger for prevalence intervals; 0 skips them
#' @param chain_cache Path to save the NUTS chain to, before post-processing
#' @param reuse_chain Load the cached chain instead of sampling again
#' @param ad AD backend for NUTS: "forwarddiff" (default) or "enzyme". Enzyme is
#'   roughly 1.6x faster but requires the caller to have run
#'   `JuliaCall::julia_command("using Enzyme;")` first, because
#'   DifferentiationInterface only wires up its Enzyme extension if Enzyme is
#'   loaded before it prepares a gradient.
#' @param target_acc Target acceptance rate for NUTS
#' @param year_process Year-effect process: `"rw1"` (default), `"iid"`,
#'   `"rw2"`, `"ar1"`, `"shrunk"` or `"none"`.
#' @param tests Tests to use: a length-6 logical mask or column indices in 1:6.
#' @param se_fixed Optional fixed sensitivities (length 6). If supplied, `sp_fixed` must also be supplied.
#' @param sp_fixed Optional fixed specificities (length 6). If supplied, `se_fixed` must also be supplied.
#' @param se_prior_mean Optional Se prior means (length 6).
#' @param sp_prior_mean Optional Sp prior means (length 6).
#' @param se_prior_ci Optional Se prior CIs as 6x2 matrix/list (lo, hi per test).
#' @param sp_prior_ci Optional Sp prior CIs as 6x2 matrix/list (lo, hi per test).
#' @param hazard_mean Mean for seasonal hazard prior.
#' @param hazard_sd SD for seasonal hazard prior.
#' @param penalty Optional logical to enforce Se+Sp>1 soft constraint. If NULL,
#'   defaults to TRUE when Se/Sp are inferred and FALSE when fixed.
#' @param repeat_captures Handling of repeat captures within one time-step:
#'   `"stack"` (default) keeps each as a separate observation, `"pool"` merges
#'   them into one, `"last"` keeps only the final capture.
#' @param start_year Calendar year of `time == 1`. `NULL` (default) leaves
#'   `year_effect$calendar_year` as `NA`.
#' @param seed Random seed
#' @return List with individual infection probabilities and prevalence over time
#' @export
hmm_inference <- function(test_mat, method = c("map", "mle", "nuts"),
                         nuts_samples = 1000,
                         target_acc = 0.65,
                         year_process = c("rw1", "iid", "rw2", "ar1", "shrunk", "none"),
                         tests = seq_len(6),
                         se_fixed = NULL,
                         sp_fixed = NULL,
                         se_prior_mean = NULL,
                         sp_prior_mean = NULL,
                         se_prior_ci = NULL,
                         sp_prior_ci = NULL,
                         traj_draws = 500,
                         chain_cache = NULL,
                         reuse_chain = FALSE,
                         ad = c("forwarddiff", "enzyme"),
                         hazard_mean = -3.0,
                         hazard_sd = 1.5,
                         penalty = NULL,
                         repeat_captures = c("stack", "pool", "last"),
                         start_year = NULL,
                         seed = 123) {
  method <- match.arg(method)
  year_process <- match.arg(year_process)
  repeat_captures <- match.arg(repeat_captures)

  # Tests are identified by column position, not by name: the engine has no
  # knowledge of what any particular assay is.
  test_labels <- paste0("test_", seq_len(6))

  resolve_test_mask <- function(tests_arg) {
    if (is.logical(tests_arg)) {
      if (length(tests_arg) != 6) stop("`tests` logical vector must be length 6.")
      return(as.logical(tests_arg))
    }
    if (is.numeric(tests_arg)) {
      idx <- as.integer(tests_arg)
      if (any(is.na(idx)) || any(idx < 1 | idx > 6)) {
        stop("`tests` numeric indices must be in 1:6.")
      }
      m <- rep(FALSE, 6)
      m[unique(idx)] <- TRUE
      return(m)
    }
    stop("`tests` must be a length-6 logical mask or numeric indices in 1:6.")
  }

  normalize_len6 <- function(x, name) {
    if (is.null(x)) return(NULL)
    x <- as.numeric(x)
    if (length(x) != 6 || any(!is.finite(x))) {
      stop("`", name, "` must be a numeric vector of length 6.")
    }
    x
  }

  normalize_ci_6x2 <- function(x, name) {
    if (is.null(x)) return(NULL)
    m <- as.matrix(x)
    if (!is.numeric(m) || !all(dim(m) == c(6, 2))) {
      stop("`", name, "` must be numeric with shape 6x2.")
    }
    m
  }

  test_mask <- resolve_test_mask(tests)
  if (!any(test_mask)) stop("`tests` selects no assays.")

  se_fixed <- normalize_len6(se_fixed, "se_fixed")
  sp_fixed <- normalize_len6(sp_fixed, "sp_fixed")
  if (xor(is.null(se_fixed), is.null(sp_fixed))) {
    stop("Provide both `se_fixed` and `sp_fixed`, or neither.")
  }

  se_prior_mean <- normalize_len6(se_prior_mean, "se_prior_mean")
  sp_prior_mean <- normalize_len6(sp_prior_mean, "sp_prior_mean")
  se_prior_ci <- normalize_ci_6x2(se_prior_ci, "se_prior_ci")
  sp_prior_ci <- normalize_ci_6x2(sp_prior_ci, "sp_prior_ci")

  if (is.null(penalty)) {
    penalty <- is.null(se_fixed)
  }
  penalty <- isTRUE(penalty)
  
  if (is.data.frame(test_mat)) {
    test_mat <- as.matrix(test_mat)
  }

  if (!is.matrix(test_mat)) {
    stop("`test_mat` must be a matrix or data frame.")
  }

  if (ncol(test_mat) < 6) {
    stop("`test_mat` must have at least 6 test columns.")
  }

  if (ncol(test_mat) == 7) {
    test_mat <- test_mat[, 2:7, drop = FALSE]
  }

  if (ncol(test_mat) >= 8) {
    test_cols <- (ncol(test_mat) - 5):ncol(test_mat)
    drop_cols <- test_cols[!test_mask]
    if (length(drop_cols) > 0) test_mat[, drop_cols] <- NA_real_
  } else {
    drop_cols <- which(!test_mask)
    if (length(drop_cols) > 0) test_mat[, drop_cols] <- NA_real_
  }

  storage.mode(test_mat) <- "double"

  # All argument validation happens above, so a bad call fails the same way
  # whether or not the Julia engine has been set up.
  ensure_engine()

  JuliaCall::julia_assign("r_test_mat", test_mat)
  # Trailing ";" keeps JuliaCall from echoing the whole matrix / result object
  # into the console, which otherwise buries every log this produces.
  JuliaCall::julia_command("j_test_mat = Float64.(coalesce.(Matrix(r_test_mat), NaN));")

  JuliaCall::julia_assign("j_nuts_samples", as.integer(nuts_samples))
  JuliaCall::julia_assign("j_target_acc", as.numeric(target_acc))
  JuliaCall::julia_assign("j_seed", as.integer(seed))
  JuliaCall::julia_assign("j_method", method)
  JuliaCall::julia_assign("j_year_process", year_process)
  JuliaCall::julia_assign("j_test_mask", as.logical(test_mask))
  JuliaCall::julia_assign("j_se_fixed", if (is.null(se_fixed)) numeric(0) else as.numeric(se_fixed))
  JuliaCall::julia_assign("j_sp_fixed", if (is.null(sp_fixed)) numeric(0) else as.numeric(sp_fixed))
  JuliaCall::julia_assign("j_se_prior_mean", if (is.null(se_prior_mean)) numeric(0) else as.numeric(se_prior_mean))
  JuliaCall::julia_assign("j_sp_prior_mean", if (is.null(sp_prior_mean)) numeric(0) else as.numeric(sp_prior_mean))
  JuliaCall::julia_assign("j_se_prior_ci", if (is.null(se_prior_ci)) matrix(numeric(0), nrow = 0, ncol = 0) else se_prior_ci)
  JuliaCall::julia_assign("j_sp_prior_ci", if (is.null(sp_prior_ci)) matrix(numeric(0), nrow = 0, ncol = 0) else sp_prior_ci)
  JuliaCall::julia_assign("j_hazard_mean", as.numeric(hazard_mean))
  JuliaCall::julia_assign("j_hazard_sd", as.numeric(hazard_sd))
  JuliaCall::julia_assign("j_penalty", as.logical(penalty))
  JuliaCall::julia_assign("j_start_year", as.integer(if (is.null(start_year)) 0L else start_year))
  JuliaCall::julia_assign("j_repeat_captures", repeat_captures)
  JuliaCall::julia_assign("j_traj_draws", as.integer(traj_draws))
  JuliaCall::julia_assign("j_reuse_chain", as.logical(reuse_chain))
  JuliaCall::julia_assign("j_ad", match.arg(ad))
  JuliaCall::julia_assign("j_chain_cache",
                          if (is.null(chain_cache)) "" else normalizePath(chain_cache, mustWork = FALSE))
  
  JuliaCall::julia_command(paste0(
    "result = run_hmm_inference(",
    "j_test_mat, j_method, j_nuts_samples, j_target_acc, j_seed; ",
    "year_process=j_year_process, test_mask=j_test_mask, ",
    "se_fixed=j_se_fixed, sp_fixed=j_sp_fixed, ",
    "se_prior_mean=j_se_prior_mean, sp_prior_mean=j_sp_prior_mean, ",
    "se_prior_ci=j_se_prior_ci, sp_prior_ci=j_sp_prior_ci, ",
    "hazard_mean=j_hazard_mean, hazard_sd=j_hazard_sd, ",
    "penalty=j_penalty, start_year=j_start_year, ",
    "repeat_captures=j_repeat_captures, traj_draws=j_traj_draws, ",
    "chain_cache=j_chain_cache, reuse_chain=j_reuse_chain, ad_name=j_ad",
    ");"
  ))
  
  p_inf_last <- JuliaCall::julia_eval("result.p_inf_last")
  ids <- JuliaCall::julia_eval("result.ids")
  ids <- as.numeric(ids)

  p_inf_last_vec <- if (is.list(p_inf_last)) {
    vals <- suppressWarnings(as.numeric(unlist(p_inf_last, use.names = TRUE)))
    nms <- names(unlist(p_inf_last, use.names = TRUE))
    if (!is.null(nms) && length(nms) == length(vals)) {
      lookup <- vals
      names(lookup) <- nms
      out <- as.numeric(lookup[as.character(ids)])
      if (length(out) == length(ids) && any(!is.na(out))) out else vals
    } else {
      vals
    }
  } else {
    as.numeric(p_inf_last)
  }
  if (length(p_inf_last_vec) != length(ids)) {
    p_inf_last_vec <- rep_len(p_inf_last_vec, length(ids))
  }

  p_inf_over_time <- JuliaCall::julia_eval("result.p_inf_over_time")
  times <- JuliaCall::julia_eval("result.times")

  prevalence_times <- JuliaCall::julia_eval("result.prevalence_times")
  prevalence_proportion <- JuliaCall::julia_eval("result.prevalence_proportion")
  prevalence_total <- JuliaCall::julia_eval("result.prevalence_total")
  prevalence_grid_proportion <- JuliaCall::julia_eval("haskey(result, :prevalence_grid_proportion) ? result.prevalence_grid_proportion : result.prevalence_proportion")
  prevalence_grid_total <- JuliaCall::julia_eval("haskey(result, :prevalence_grid_total) ? result.prevalence_grid_total : result.prevalence_total")
  prevalence_capture_proportion <- JuliaCall::julia_eval("haskey(result, :prevalence_capture_proportion) ? result.prevalence_capture_proportion : result.prevalence_proportion")
  prevalence_capture_total <- JuliaCall::julia_eval("haskey(result, :prevalence_capture_total) ? result.prevalence_capture_total : result.prevalence_total")

  has_q <- JuliaCall::julia_eval("haskey(result, :prevalence_grid_lower)")
  qget <- function(key) if (has_q) JuliaCall::julia_eval(paste0("result.", key)) else NA_real_
  prevalence_grid_lower <- qget("prevalence_grid_lower")
  prevalence_grid_upper <- qget("prevalence_grid_upper")
  prevalence_capture_lower <- qget("prevalence_capture_lower")
  prevalence_capture_upper <- qget("prevalence_capture_upper")

  prevalence_draws <- if (has_q)
    JuliaCall::julia_eval("result.prevalence_capture_draws") else NULL

  trajectories <- if (JuliaCall::julia_eval(
        "haskey(result, :trajectory_id) && !isempty(result.trajectory_id)"))
    data.frame(
      id = JuliaCall::julia_eval("result.trajectory_id"),
      draw = JuliaCall::julia_eval("result.trajectory_draw"),
      infection_time = JuliaCall::julia_eval("result.trajectory_infection_time")
    ) else NULL

  infection_matrix <- JuliaCall::julia_eval("result.infection_matrix")
  infection_matrix_times <- JuliaCall::julia_eval("result.infection_matrix_times")

  se_vals <- JuliaCall::julia_eval("haskey(result, :Se) ? result.Se : Float64[]")
  sp_vals <- JuliaCall::julia_eval("haskey(result, :Sp) ? result.Sp : Float64[]")
  year_idx <- JuliaCall::julia_eval("haskey(result, :year_effect_year_index) ? result.year_effect_year_index : Int[]")
  year_calendar <- JuliaCall::julia_eval("haskey(result, :year_effect_calendar_year) ? result.year_effect_calendar_year : Int[]")
  year_gamma <- JuliaCall::julia_eval("haskey(result, :year_effect_gamma) ? result.year_effect_gamma : Float64[]")
  pi1_vals <- JuliaCall::julia_eval("haskey(result, :pi1_vec) ? collect(result.pi1_vec) : Float64[]")
  year_hazard <- JuliaCall::julia_eval("haskey(result, :year_effect_annual_hazard) ? result.year_effect_annual_hazard : Float64[]")
  settings <- JuliaCall::julia_eval("haskey(result, :settings) ? result.settings : nothing")
  
  individual_df <- data.frame(
    id = ids,
    p_infected_last = p_inf_last_vec,
    method = method,
    stringsAsFactors = FALSE
  )

  p_inf_over_time_list <- lapply(ids, function(id) {
    key <- as.character(id)
    tvals <- times[[key]]
    pvals <- p_inf_over_time[[key]]
    if (is.null(tvals) || is.null(pvals)) return(NULL)
    tvals <- as.numeric(tvals)
    pvals <- as.numeric(pvals)
    n <- min(length(tvals), length(pvals))
    if (n == 0) return(NULL)
    data.frame(
      id = id,
      time = tvals[seq_len(n)],
      p_infected = pvals[seq_len(n)],
      stringsAsFactors = FALSE
    )
  })
  p_inf_over_time_list <- Filter(Negate(is.null), p_inf_over_time_list)
  p_inf_over_time_df <- if (length(p_inf_over_time_list) > 0) {
    do.call(rbind, p_inf_over_time_list)
  } else {
    data.frame(id = numeric(0), time = numeric(0), p_infected = numeric(0), stringsAsFactors = FALSE)
  }

  prevalence_df <- data.frame(
    time = prevalence_times,
    proportion_infected = prevalence_proportion,
    total_infected = prevalence_total,
    stringsAsFactors = FALSE
  )

  prevalence_grid_df <- data.frame(
    time = prevalence_times,
    proportion_infected = prevalence_grid_proportion,
    total_infected = prevalence_grid_total,
    lower = prevalence_grid_lower,
    upper = prevalence_grid_upper,
    stringsAsFactors = FALSE
  )

  prevalence_capture_df <- data.frame(
    time = prevalence_times,
    proportion_infected = prevalence_capture_proportion,
    total_infected = prevalence_capture_total,
    lower = prevalence_capture_lower,
    upper = prevalence_capture_upper,
    stringsAsFactors = FALSE
  )

  infection_matrix_mat <- matrix(infection_matrix, nrow = length(infection_matrix_times))
  rownames(infection_matrix_mat) <- infection_matrix_times
  colnames(infection_matrix_mat) <- ids

  # `used` matters: masked assays keep their prior means, which are not
  # estimates from this fit. Reading Se/Sp for an unused assay as a result is
  # the mistake this column exists to prevent.
  sesp_df <- data.frame(
    test = test_labels,
    used = as.logical(test_mask),
    Se = as.numeric(se_vals),
    Sp = as.numeric(sp_vals),
    Youden = as.numeric(se_vals) + as.numeric(sp_vals) - 1,
    stringsAsFactors = FALSE
  )

  year_calendar <- as.integer(year_calendar)
  year_calendar[year_calendar < 0] <- NA_integer_

  year_effect_df <- data.frame(
    year_index = as.integer(year_idx),
    calendar_year = year_calendar,
    gamma = as.numeric(year_gamma),
    annual_hazard = as.numeric(year_hazard),
    stringsAsFactors = FALSE
  )

  pi1_by_year_df <- data.frame(
    year_index = as.integer(year_idx),
    calendar_year = year_calendar,
    pi1 = as.numeric(pi1_vals),
    stringsAsFactors = FALSE
  )

  settings_list <- if (is.null(settings)) {
    list(
      method = method,
      year_process = year_process,
      tests = which(test_mask),
      penalty = penalty,
      repeat_captures = repeat_captures,
      start_year = start_year
    )
  } else {
    settings
  }
  
  list(
    individual = individual_df,
    p_inf_over_time = p_inf_over_time_df,
    prevalence = prevalence_df,
    prevalence_grid = prevalence_grid_df,
    prevalence_draws = prevalence_draws,
    trajectories = trajectories,
    prevalence_capture = prevalence_capture_df,
    infection_matrix = infection_matrix_mat,
    sesp = sesp_df,
    year_effect = year_effect_df,
    pi1_by_year = pi1_by_year_df,
    settings = settings_list,
    method = method
  )
}
