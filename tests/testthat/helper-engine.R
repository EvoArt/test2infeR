# Engine-dependent tests need a working Julia install AND a JuliaCall that can
# complete its own setup. On CI `NOT_CRAN=true` is set, so skip_on_cran() does
# NOT skip; without this helper the first engine test fails, the engine is left
# unloaded, and every later test errors in ensure_engine() as a cascade.
#
# Setup is attempted once per session and the outcome cached, so a broken
# environment costs one attempt rather than one per test.

.engine_status <- local({
  cached <- NULL
  function() {
    if (!is.null(cached)) return(cached)
    cached <<- tryCatch({
      test2infeR::setup_hmm_engine(install_julia = FALSE)
      if (isTRUE(test2infeR::hmm_engine_loaded())) {
        list(ok = TRUE, msg = "")
      } else {
        list(ok = FALSE, msg = "Julia engine did not load")
      }
    }, error = function(e) {
      list(ok = FALSE, msg = conditionMessage(e))
    })
    cached
  }
})

skip_without_engine <- function() {
  st <- .engine_status()
  if (!st$ok) {
    testthat::skip(paste("Julia engine unavailable:", st$msg))
  }
  invisible(TRUE)
}
