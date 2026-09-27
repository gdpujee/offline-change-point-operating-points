## R53: what are the *defaults* of changepoint, exactly?
##
## The Related-work sentence read "PELT ... is the default of the widely used R
## package changepoint". That is a checkable claim, and it is false: the default
## `method` of cpt.mean / cpt.meanvar / cpt.var is "AMOC" (a single change
## point); PELT is one of the methods the caller selects explicitly. The default
## `penalty` really is "MBIC" -- the half of the claim that survives.
##
## Two independent kinds of evidence are printed:
##   (1) formals(): the package's own declared defaults (metadata);
##   (2) a behavioural check on a 3-change-point series: an all-defaults call
##       returns ONE change point, the same call with method="PELT" returns
##       three. Metadata and behaviour agree, so neither is read off a summary.
##
## Run once per installed version:
##   R_LIBS=~/R_libs     Rscript experiments/changepoint_defaults.R   # 2.2.2
##   R_LIBS=~/R_libs_224 Rscript experiments/changepoint_defaults.R   # 2.2.4

library(changepoint)

cat("packageVersion:", as.character(packageVersion("changepoint")), "\n")
cat("R:", R.version.string, "\n")
cat("--- (1) declared defaults ---\n")
for (fn in c("cpt.mean", "cpt.meanvar", "cpt.var")) {
  f <- formals(get(fn))
  cat(sprintf("%-12s method.default=%-8s penalty.default=%s\n",
              fn, deparse(f$method), deparse(f$penalty)))
}

cat("--- (2) behavioural check ---\n")
set.seed(20260919)
x <- c(rnorm(60, 0, 1), rnorm(60, 4, 1), rnorm(60, 0, 1), rnorm(60, 3, 1))
cat("true change points at:", paste(c(60, 120, 180), collapse = ","), "\n")
default <- cpt.mean(x)                      # everything left at its default
pelt    <- cpt.mean(x, method = "PELT")     # PELT selected explicitly
cat("cpt.mean(x)                  -> ncpts =", ncpts(default),
    " cpts =", paste(cpts(default), collapse = ","), "\n")
cat("cpt.mean(x, method='PELT')   -> ncpts =", ncpts(pelt),
    " cpts =", paste(cpts(pelt), collapse = ","), "\n")
cat("VERDICT default is AMOC (single change):",
    ncpts(default) == 1, "\n")
