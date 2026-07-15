#' Run HMM inference on diagnostic test data
#'
#' @param test_mat Test matrix (individuals x tests)
#' @param method Inference method: "nuts", "map", or "mle"
#' @param nuts_samples Number of NUTS samples (for method="nuts")
#' @param target_acc Target acceptance rate for NUTS
#' @param seed Random seed
#' @return List with individual infection probabilities and prevalence over time
#' @export
hmm_inference <- function(test_mat, method = c("nuts", "map", "mle"),
                         nuts_samples = 1000,
                         target_acc = 0.65, seed = 123) {
  method <- match.arg(method)
  ensure_engine()
  
  # Convert test_mat to appropriate format for Julia
  if (is.data.frame(test_mat)) {
    # Assume data frame has columns: id, time, test1, test2, ...
    test_mat <- as.matrix(test_mat[, -c(1, 2)])  # Remove id and time columns
  }
  
  # Convert to Julia array
  JuliaCall::julia_assign("r_test_mat", test_mat)
  JuliaCall::julia_command("j_test_mat = Matrix{Float64}(r_test_mat)")
  
  # Set parameters
  JuliaCall::julia_assign("j_nuts_samples", as.integer(nuts_samples))
  JuliaCall::julia_assign("j_target_acc", as.numeric(target_acc))
  JuliaCall::julia_assign("j_seed", as.integer(seed))
  JuliaCall::julia_assign("j_method", method)
  
  # Call Julia inference function
  JuliaCall::julia_command("result = run_hmm_inference(j_test_mat, j_method, j_nuts_samples, j_target_acc, j_seed)")
  
  # Extract individual-level results
  p_inf_last <- JuliaCall::julia_eval("result.p_inf_last")
  ids <- JuliaCall::julia_eval("result.ids")
  
  # Extract infection probabilities over time
  p_inf_over_time <- JuliaCall::julia_eval("result.p_inf_over_time")
  times <- JuliaCall::julia_eval("result.times")
  
  # Extract prevalence over time
  prevalence_times <- JuliaCall::julia_eval("result.prevalence_times")
  prevalence_proportion <- JuliaCall::julia_eval("result.prevalence_proportion")
  prevalence_total <- JuliaCall::julia_eval("result.prevalence_total")
  
  # Extract infection matrix (timepoint x id)
  infection_matrix <- JuliaCall::julia_eval("result.infection_matrix")
  infection_matrix_times <- JuliaCall::julia_eval("result.infection_matrix_times")
  
  # Convert individual infection probabilities to data frame
  individual_df <- data.frame(
    id = ids,
    p_infected_last = p_inf_last,
    method = method,
    stringsAsFactors = FALSE
  )
  
  # Convert infection probabilities over time to list of data frames
  p_inf_over_time_list <- lapply(ids, function(id) {
    data.frame(
      id = id,
      time = times[[as.character(id)]],
      p_infected = p_inf_over_time[[as.character(id)]],
      stringsAsFactors = FALSE
    )
  })
  p_inf_over_time_df <- do.call(rbind, p_inf_over_time_list)
  
  # Convert prevalence to data frame
  prevalence_df <- data.frame(
    time = prevalence_times,
    proportion_infected = prevalence_proportion,
    total_infected = prevalence_total,
    stringsAsFactors = FALSE
  )
  
  # Convert infection matrix to matrix with proper names
  infection_matrix_mat <- matrix(infection_matrix, nrow = length(infection_matrix_times))
  rownames(infection_matrix_mat) <- infection_matrix_times
  colnames(infection_matrix_mat) <- ids
  
  # Return as list
  list(
    individual = individual_df,
    p_inf_over_time = p_inf_over_time_df,
    prevalence = prevalence_df,
    infection_matrix = infection_matrix_mat,
    method = method
  )
}
