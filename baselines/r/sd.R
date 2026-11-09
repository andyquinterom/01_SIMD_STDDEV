# Baseline: R, built-in `sd()`.
#
# Usage: Rscript sd.R [N]      (default N = 1,000,000,000)
# Note: `sd()` uses the sample formula (divides by N - 1); for large N the
# difference with the population formula used elsewhere is negligible.

args <- commandArgs(trailingOnly = TRUE)
N <- if (length(args) > 0) as.numeric(args[1]) else 1e9
RUNS <- 10

cat(sprintf("Generating %s random observations (double)...\n", format(N, big.mark = ",", scientific = FALSE)))
t0 <- proc.time()[["elapsed"]]
x <- runif(N)
cat(sprintf("Generation: %.3f s\n", proc.time()[["elapsed"]] - t0))

times <- numeric(RUNS)
for (i in seq_len(RUNS)) {
  t <- proc.time()[["elapsed"]]
  std <- sd(x)
  times[i] <- proc.time()[["elapsed"]] - t
  cat(sprintf("Run %d: %.3f s\n", i - 1, times[i]))
}

times <- sort(times)
cat(sprintf("Standard deviation: %.6f\n", std))
cat(sprintf("Median: %.3f s\n", times[RUNS %/% 2 + 1]))
