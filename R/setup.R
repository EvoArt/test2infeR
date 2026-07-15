# Julia bootstrap: install/locate Julia, activate the bundled engine
# project, load its dependencies, and include the engine source.

.hmm_state <- new.env(parent = emptyenv())
.hmm_state$setup_done <- FALSE
.hmm_state$engine_loaded <- FALSE
.hmm_state$src_dir <- NULL

#' Locate the bundled Julia engine source directory.
#' @keywords internal
default_engine_src_dir <- function() {
  candidates <- c(
    system.file("julia", "src", package = "test2infeR"),
    file.path(getwd(), "inst", "julia", "src")
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
#' @param src_dir Path to the engine's `*.jl` source directory. Defaults to
#'   the bundled `inst/julia/src`.
#' @param project_dir Path to the Julia project (containing `Project.toml`
#'   and the pinned `Manifest.toml`). Defaults to the bundled `inst/julia`.
#' @param force Re-run even if setup already completed this session.
#' @param install_julia Auto-download Julia if not found.
#' @param julia_version Julia version to use (default: "1.10.9").
#' @param ... Passed to [JuliaCall::julia_setup()].
#' @return Invisibly, the normalised `src_dir`.
#' @export
setup_hmm_engine <- function(src_dir = NULL, project_dir = NULL,
                             force = FALSE, install_julia = TRUE, 
                             julia_version = "1.10.9", ...) {
  if (.hmm_state$setup_done && .hmm_state$engine_loaded && !force) {
    return(invisible(.hmm_state$src_dir))
  }
  if (is.null(src_dir)) src_dir <- default_engine_src_dir()
  if (is.null(src_dir) || !dir.exists(src_dir)) {
    stop("Could not locate the Julia engine source directory. ",
         "Pass `src_dir=` pointing at the folder of *.jl engine files.")
  }
  src_dir <- normalizePath(src_dir, winslash = "/")
  if (is.null(project_dir)) project_dir <- default_engine_project_dir()
  if (is.null(project_dir) || !file.exists(file.path(project_dir, "Project.toml"))) {
    stop("Could not locate the Julia engine project directory (with Project.toml). ",
         "Pass `project_dir=` pointing at it.")
  }
  project_dir <- normalizePath(project_dir, winslash = "/")

  JuliaCall::julia_setup(installJulia = install_julia, julia_version = julia_version, ...)
  .hmm_state$setup_done <- TRUE

  JuliaCall::julia_command("import Pkg")
  JuliaCall::julia_command(sprintf('Pkg.activate(raw"%s")', project_dir))
  JuliaCall::julia_command("Pkg.instantiate()")

  load_engine_deps()
  load_engine(src_dir)
  .hmm_state$src_dir <- src_dir
  .hmm_state$engine_loaded <- TRUE
  invisible(src_dir)
}

# Engine source files refer to these packages as bare globals
.hmm_engine_using <- c(
  "using Turing", "using Distributions", "using LinearAlgebra",
  "using Random", "using CSV", "using DataFrames", "using JLD2",
  "using Statistics", "using ForwardDiff"
)

#' @keywords internal
load_engine_deps <- function() {
  for (cmd in .hmm_engine_using) JuliaCall::julia_command(cmd)
}

#' Include every `*.jl` file in `src_dir` into Main.
#' @keywords internal
load_engine <- function(src_dir) {
  cmd <- sprintf(
    'include.(filter(contains(r"\\.jl$"), readdir(raw"%s"; join=true)))',
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
