# Review of the first A100 occupied-space benchmark

Input: the original `../results.jsonl` and `../settings.json`, unchanged. There are 20 records: four sizes, each with dense, Davidson cold/warm-requested and LOBPCG cold/warm-requested. The GPU is an A100-SXM4-80GB, PyTorch 2.3.0/CUDA 12.1, float64, 5 Å Hamiltonian cutoff and 0.025 eV smearing.

## Why the original plot has only two lines

The original plot requires converged residuals, complete occupation coverage and agreement with the dense lowest eigenvalues. It also omits first-frame warm requests.

- Dense passes at all four sizes.
- Every Davidson record reaches `maxiter=100`, so no Davidson line appears.
- Cold LOBPCG passes only at 192 and 384 atoms. It hits the iteration limit before occupation coverage is complete at 768 and 1536 atoms.
- `selected_frames=1`: every warm-requested record has `warm_used=false`. These are additional cold solves, not evidence about trajectory warm-start performance.

[Filtered timing plot](timings.png) reproduces that policy. [Diagnostics](diagnostics.png) shows all timed records, including failed/incomplete runs and the warm-requested cold duplicates. Crosses indicate failures/incomplete coverage. Their runtimes are the cost of unsuccessful attempts, not comparable successful-solve speeds.

## Measured performance

Median occupied-space times in seconds (eigensolve plus occupation work):

| Atoms | Matrix dimension | Dense | Davidson, cold | LOBPCG, cold |
|---:|---:|---:|---:|---:|
| 192 | 960 | 0.0356 | 0.736, incomplete | 3.786, complete |
| 384 | 1920 | 0.0625 | 1.287, incomplete | 8.748, complete |
| 768 | 3840 | 0.2243 | 3.052, incomplete | 11.185, incomplete |
| 1536 | 7680 | 0.9839 | 9.152, incomplete | 20.099, incomplete |

The dense eigensolve alone takes 0.0182, 0.0487, 0.2110 and 0.9699 seconds. Thus the dense advantage is real for these implementations and settings; the missing sparse points do not hide a successful crossover.

The successful cold LOBPCG runs retained 853/960 and 1616/1920 states (89% and 84%), after nine expansion attempts. Their 251/289 iterations include many dense projected eigensolves and orthogonalizations. This removes much of the advantage of targeting only the occupied subspace. These state counts reflect the conservative tail bound and expansion increments; they do not establish the mathematically minimal number of required states.

The initial 16-state buffer really is thermally incomplete. At 192 atoms, its omitted occupation measured against dense is 0.168 electrons (versus the conservative estimate 8.84). At 1536 atoms it is 3.43 electrons (bound 617). The bound overestimates the error substantially, but the actual omitted population still greatly exceeds the requested 1e-6 electrons. For the initial Davidson spaces, maximum on-site density errors grow from 0.00148 to 0.00684, and total band-energy errors from -0.0337 to -0.691 eV.

These are initial algorithm implementations. The result does not establish that optimised sparse eigensolvers cannot help; it shows that these cold-start, adaptive, stringent finite-temperature solves are not competitive with the dense CUDA solver.

## Reporting bug

The old benchmark combined median runtime with convergence/work statistics from only the final repetition. At 768 atoms, warm-requested LOBPCG reports times 11.348, 11.299 and 29.453 seconds, while its final repetition reached k=3840 with about 29.313 seconds of eigensolver work. The 11.348-second median must not be interpreted as the time of that successful full-basis calculation. Earlier per-repeat outcomes were not saved and cannot be recovered from this file. The diagnostic plot marks this historical ambiguity with a diamond.

The module now saves each iterative repetition's outcome, retained count, occupation coverage and work statistics. Aggregate status is `mixed_repeats` when convergence statuses differ, and aggregate occupation coverage requires every repeat to pass. Such mixed/incomplete rows are not shown as successful timings. The existing GPU records have not been rewritten or reclassified as if those missing data were known.

## Next useful benchmark

1. Supply multiple nearby frames to measure actual previous-frame warm starts.
2. Separate fixed-subspace solver performance from discovering a thermally complete subspace. `--no-auto-expand` gives the former while still reporting truncation errors; those runs must not be presented as complete finite-temperature solutions unless the accuracy checks pass.
3. Choose occupation coverage from acceptable density/band-energy errors, rather than assuming the extremely conservative 1e-6-electron tail bound is necessary for this application. Use the dense reference to establish the accuracy/work tradeoff.
4. Benchmark shorter Hamiltonian cutoffs and the block preconditioner. Sparse structure alone is insufficient if the algorithm repeatedly diagonalizes a projected matrix approaching full size.

No new GPU solves were performed during this review. CSV and figures here were regenerated from the supplied results using:

```bash
PYTHONPATH=. python scripts/plot_occupied_profile.py examples/mldftb/profile
```
