# Environment notes

- Python dependencies are specified as lower bounds in requirements.txt; it is not a lockfile.
- The analysis modules use NumPy and SciPy; figures use Matplotlib.
- experiments/run_r_reference.py invokes the benchmark's R pipeline and requires the upstream TCPDBench files plus an R installation and its required package version.
- The repository does not bundle the TCPD or TCPDBench input trees. See the README before running data-dependent steps.
