# Julia bootstrap: install/locate Julia, activate the bundled engine project,
# and load the engine package.
#
# The engine lives in inst/julia/Test2InfEngine as a real Julia package rather
# than a loose script. That is what lets Julia cache its compiled code: the
# package carries a PrecompileTools workload, so the model, its AD tape and the
# HMM forward pass are compiled once at install time instead of on the first
# run_hmm_inference() call of every R session.

.hmm_state <- new.env(parent = emptyenv())
.hmm_state$setup_done <- FALSE
.hmm_state$engine_loaded <- FALSE
.hmm_state$src_dir <- NULL

#' Locate the bundled Julia engine source directory.
#'
#' Returns the engine package's `src` directory. Only used when
#' [setup_hmm_engine()] is asked to load loose sources via `src_dir=`.
#' @keywords internal
default_engine_src_dir <- function() {
  candidates <- c(
    system.file("julia", "Test2InfEngine", "src", package = "test2infeR"),
    file.path(getwd(), "inst", "julia", "Test2InfEngine", "src")
  )
  for (cand in candidates) {
    if (nzchar(cand) && dir.exists(cand)) return(normalizePath(cand))
  }
  NULL
}

#' Locate the bundled Julia project directory (containing Project.toml).
#' @keywords internal
default_engine_project_dir <- function() {
  candidates <- c(
    system.file("julia", package = "test2infeR"),
    file.path(getwd(), "inst", "julia")
  )
  for (cand in candidates) {
    if (nzchar(cand) && file.exists(file.path(cand, "Project.toml"))) {
      return(normalizePath(cand))
    }
  }
  NULL
}

#' Install/locate Julia, activate the bundled engine project, and load it.
#'
#' One-time setup. The first call with no local Julia install will download
#' and cache one via `JuliaCall::julia_setup(installJulia = TRUE)`; later
#' sessions reuse it. Run this once per R session before [hmm_inference()].
#'
#' The engine is loaded as the precompiled `Test2InfEngine` Julia package. The
#' first call after installation (or after a Julia/package upgrade) pays a
#' one-off precompilation cost of a minute or two; later sessions load the
#' cached code and reach the first result in seconds.
#'
#' @param src_dir Optional path to a directory of loose engine `*.jl` files to
#'   `include()` instead of loading the precompiled package. Intended for
#'   engine development only -- it bypasses the precompilation cache and so
#'   restores the slow per-session compile. Normally leave this `NULL`.
#' @param project_dir Path to the Julia project (containing `Project.toml`
#'   and the pinned `Manifest.toml`). Defaults to the bundled `inst/julia`.
#' @param force Re-run even if setup already completed this session.
#' @param install_julia Auto-download Julia if not found.
#' @param ... Passed to [JuliaCall::julia_setup()].
#' @return Invisibly, the engine source directory (or `NULL` when the
#'   precompiled package was used).
#' @export
setup_hmm_engine <- function(src_dir = NULL, project_dir = NULL,
                             force = FALSE, install_julia = TRUE, ...) {
  if (.hmm_state$setup_done && .hmm_state$engine_loaded && !force) {
    return(invisible(.hmm_state$src_dir))
  }
  if (is.null(project_dir)) project_dir <- default_engine_project_dir()
  if (is.null(project_dir) || !file.exists(file.path(project_dir, "Project.toml"))) {
    stop("Could not locate the Julia engine project directory (with Project.toml). ",
         "Pass `project_dir=` pointing at it.")
  }
  project_dir <- normalizePath(project_dir, winslash = "/")

  JuliaCall::julia_setup(installJulia = install_julia, ...)
  .hmm_state$setup_done <- TRUE

  JuliaCall::julia_command("import Pkg")
  JuliaCall::julia_command(sprintf('Pkg.activate(raw"%s")', project_dir))
  tryCatch(
    JuliaCall::julia_command("Pkg.instantiate()"),
    error = function(e) {
      JuliaCall::julia_command("Pkg.resolve()")
      JuliaCall::julia_command("Pkg.instantiate()")
    }
  )

  if (is.null(src_dir)) {
    load_engine_package()
  } else {
    src_dir <- normalizePath(src_dir, winslash = "/")
    if (!dir.exists(src_dir)) {
      stop("`src_dir` does not exist: ", src_dir)
    }
    load_engine_deps()
    load_engine(src_dir)
  }

  .hmm_state$src_dir <- src_dir
  .hmm_state$engine_loaded <- TRUE
  invisible(src_dir)
}

#' Load the precompiled engine package into Main.
#'
#' `Test2InfEngine` exports `run_hmm_inference`, so after this the entry point
#' is available in `Main` under the same bare name the loose-source path used.
#' @keywords internal
load_engine_package <- function() {
  JuliaCall::julia_command("using Test2InfEngine")
}

# Loose engine sources refer to these packages as bare globals. Only needed on
# the `src_dir=` development path; the package supplies its own imports.
.hmm_engine_using <- c(
  "using Turing", "using Distributions", "using LinearAlgebra",
  "using Random", "using Statistics", "using ForwardDiff"
)

#' @keywords internal
load_engine_deps <- function() {
  for (cmd in .hmm_engine_using) JuliaCall::julia_command(cmd)
}

#' Include every `*.jl` file in `src_dir` into Main.
#'
#' Development path only: skips the module wrapper and so also skips the
#' precompilation cache.
#' @keywords internal
load_engine <- function(src_dir) {
  cmd <- sprintf(
    'include.(filter(f -> endswith(f, ".jl") && !endswith(f, "Test2InfEngine.jl"), readdir(raw"%s"; join=true)));',
    src_dir)
  JuliaCall::julia_command(cmd)
}

#' Is the Julia engine loaded in the current session?
#'
#' Checks that the entry point is defined in Main.
#' @export
hmm_engine_loaded <- function() {
  if (!.hmm_state$setup_done) return(FALSE)
  isTRUE(JuliaCall::julia_eval("isdefined(Main, :run_hmm_inference)"))
}

#' @keywords internal
ensure_engine <- function() {
  if (!hmm_engine_loaded()) {
    stop("The Julia engine is not loaded. Call setup_hmm_engine() first.")
  }
  invisible(TRUE)
}
