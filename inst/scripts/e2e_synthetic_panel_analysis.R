# End-to-end synthetic check: install -> setup -> fit -> figures.
#
# Exercises the same three-model shape the real analysis uses (all-tests fixed,
# all-tests inferred, single-test fixed) on synthetic data with a known truth.
# Run by CI; also runnable by hand:
#
#   Rscript inst/scripts/e2e_synthetic_panel_analysis.R [outdir]

suppressPackageStartupMessages({
  library(test2infeR)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else file.path(tempdir(), "test2infer_e2e")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("output dir: ", outdir)
setup_hmm_engine()

START_YEAR <- 2006L  # arbitrary; only labels the year_effect table

sim <- simulate_test_data(n_individuals = 90, n_periods = 8, seed = 123)
tm <- as.matrix(sim[, c("time", "id", "group", paste0("test_", 1:6))])

ALL_TESTS <- seq_len(6)
table1_se <- c(0.407, 0.407, 0.100, 0.650, 0.809, 0.492)
table1_sp <- c(0.943, 0.943, 0.999, 0.943, 0.936, 0.931)

# Every test must be observed somewhere, or the panel fits below are not
# actually testing anything different from each other.
observed <- vapply(paste0("test_", 1:6), function(k) any(!is.na(sim[[k]])), logical(1))
if (!all(observed)) {
  stop("Some tests are never observed: ", paste(which(!observed), collapse = ", "))
}

fit_one <- function(tests, fixed, seed) {
  hmm_inference(
    tm,
    method     = "map",
    tests      = tests,
    se_fixed   = if (fixed) table1_se else NULL,
    sp_fixed   = if (fixed) table1_sp else NULL,
    start_year = START_YEAR,
    seed       = seed
  )
}

specs <- list(
  list(label = "All tests (fixed)",    tests = ALL_TESTS, fixed = TRUE,  seed = 101),
  list(label = "All tests (inferred)", tests = ALL_TESTS, fixed = FALSE, seed = 102),
  list(label = "Single test (fixed)",  tests = 3,         fixed = TRUE,  seed = 103)
)

fits <- lapply(specs, function(s) fit_one(s$tests, s$fixed, s$seed))
names(fits) <- vapply(specs, function(s) s$label, character(1))

prev <- do.call(rbind, lapply(names(fits), function(nm) {
  d <- fits[[nm]]$prevalence_grid
  d$model <- nm
  d$calendar_year <- START_YEAR + (d$time - 1L) %/% 4L
  d
}))

# --- assertions on the API contract ------------------------------------------

for (nm in names(fits)) {
  fit <- fits[[nm]]

  if (!"used" %in% names(fit$sesp)) {
    stop(nm, ": sesp is missing the `used` column")
  }
  if (!identical(which(fit$sesp$used), as.integer(fit$settings$tests))) {
    stop(nm, ": sesp$used does not match the panel the fit was given")
  }
  if (!all(fit$year_effect$calendar_year ==
           START_YEAR + fit$year_effect$year_index - 1L)) {
    stop(nm, ": year_effect$calendar_year does not follow start_year")
  }
  p <- fit$individual$p_infected_last
  if (any(!is.finite(p)) || any(p < 0 | p > 1)) {
    stop(nm, ": p_infected_last outside [0, 1]")
  }
}

# --- figures ------------------------------------------------------------------

p_prev <- ggplot(prev, aes(calendar_year, proportion_infected, colour = model)) +
  geom_line(linewidth = 1.0) +
  scale_colour_manual(values = c(
    "All tests (fixed)"    = "#2a78d6",
    "All tests (inferred)" = "#eb6834",
    "Single test (fixed)"  = "#1baf7a"
  ), name = "Model") +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal(base_size = 12) +
  labs(title = "Synthetic prevalence over time",
       subtitle = "Three-model shape, MAP, iid year effect",
       x = "Year", y = "Prevalence")

prev_path <- file.path(outdir, "synthetic_prevalence.png")
ggsave(prev_path, p_prev, width = 11, height = 6, dpi = 180, bg = "white")

m1 <- fits[["All tests (inferred)"]]$individual[, c("id", "p_infected_last")]
names(m1)[2] <- "inferred"
m2 <- fits[["All tests (fixed)"]]$individual[, c("id", "p_infected_last")]
names(m2)[2] <- "fixed"
corner <- merge(m1, m2, by = "id", all = TRUE)

p_corner <- ggplot(corner, aes(inferred, fixed)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "firebrick") +
  geom_point(alpha = 0.3, size = 1) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal(base_size = 12) +
  labs(title = "P(infected) at last capture: inferred vs fixed Se/Sp",
       x = "Inferred Se/Sp", y = "Fixed Se/Sp")

corner_path <- file.path(outdir, "synthetic_corner.png")
ggsave(corner_path, p_corner, width = 7, height = 7, dpi = 180, bg = "white")

if (!file.exists(prev_path) || !file.exists(corner_path)) {
  stop("Expected synthetic outputs were not created")
}

message("wrote: ", prev_path)
message("wrote: ", corner_path)
