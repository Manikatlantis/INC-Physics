# Progress Log

This file is append-only. A milestone counts as complete only when a dated entry
explicitly marks it **PASSED** and records the validator's measured criteria.

## 2026-08-18 — Project initialized

- Status: no milestones passed; M1 is in progress.
- Context: the workspace initially contained only `AGENTS.md`; the Git repository
  and required project structure are being initialized from that specification.
- Budget: day 1 of 7; estimated 7 days remaining including today.

## 2026-08-18 — M1 validation attempt 2 — FAIL (figure backend)

- Done: fixed the direct-script import path. The validator passed all 10 unit
  tests and completed all 54 Metropolis runs (18 temperatures for each of
  L=4, 8, and 12). The console measurements put the sampled L=12
  susceptibility maximum at T=2.45 on this grid.
- Metrics: each L=4 temperature used 384,000 samples, each L=8 temperature used
  64,000, and each L=12 temperature used 96,000. The process aborted after the
  final sample and before criteria evaluation, so no acceptance result or JSON
  artifact was produced.
- Diagnosis: (1) plotting selected Matplotlib's interactive macOS backend in the
  headless execution context; an isolated figure smoke test with Python's fault
  handler confirmed an abort inside `matplotlib.backends._macosx`. (2) A sampler
  or terminal-session failure was the alternative, but the crash reproduced
  without running any sampler. The cheapest first fix is to force the non-GUI
  `Agg` backend before importing `pyplot`.
- Status: **FAIL** (artifact-generation failure; full criteria not evaluated).
- Next action: force `Agg`, smoke-test figure output, and rerun the unchanged
  numeric acceptance criteria.
- Budget: day 1 of 7; estimated 7 days remaining including today.

## 2026-08-18 — M1 validation attempt 1 — FAIL (launcher)

- Done: implemented the exact L=4 oracle, Ising Hamiltonian, observables,
  checkerboard Metropolis sampler, ten core tests, and the M1 validator. The
  standalone unit-test run passed 10/10 tests in 0.120 s.
- Metrics: the milestone sweep did not start and no metrics artifact was created;
  direct execution failed with `ModuleNotFoundError: src`.
- Diagnosis: (1) the most likely cause was Python setting `sys.path[0]` to
  `scripts/` for a directly executed validator; the traceback confirmed this.
  (2) A missing/install-only package layout was the alternative, but the same
  imports worked in the standalone tests. The cheapest first fix is to add the
  resolved repository root to the validator's import path.
- Status: **FAIL** (infrastructure failure, acceptance criteria not evaluated).
- Next action: apply the import-path bootstrap and rerun the unchanged validator.
- Budget: day 1 of 7; estimated 7 days remaining including today.

## 2026-08-18 — M1 — PASSED

- Done: validated the periodic Ising Hamiltonian, full 65,536-state L=4 oracle,
  all five observables, and seeded Metropolis baselines for L=4, 8, and 12 over
  an 18-point temperature grid. The non-GUI plotting fix was smoke-tested before
  the final end-to-end run. (The attempt-2 entry appears earlier in this file
  because it was inserted at the wrong append anchor; the attempt chronology is
  1, 2, then this passing run.)
- Metrics: maximum L=4 Metropolis relative errors across every temperature were
  0.1003% for energy/site, 0.1019% for |m|, 2.7492% for susceptibility, and
  1.2070% for specific heat. The L=12 susceptibility peak was T=2.45, inside
  [2.15, 2.45]. All 10 unit tests passed.
- Artifacts: `results/m1_metrics_20260818T002648-0400.json` and
  `results/m1_observables_20260818T002648-0400.png`.
- Status: **PASSED**.
- Next action: implement M2 fixed-temperature trajectory-balance GFlowNets at
  T=3.0 and T=2.0 for L=4, then validate against the exact oracle.
- Budget: day 1 of 7 complete; estimated 6 days remaining.

## 2026-08-18 — M2 — PASSED

- Done: trained separate two-hidden-layer, 128-unit autoregressive GFlowNets at
  T=3.0 and T=2.0 with the squared trajectory-balance objective. Each model used
  8,000 CPU optimizer steps, full-support exploration, and paired spin-flip
  trajectories. Validation enumerated all 65,536 model probabilities and drew
  2,000,000 seeded empirical terminals per temperature. Four GFlowNet unit tests
  raised the complete suite to 14 passing tests.
- Metrics at T=3.0: empirical KL(exact || GFlowNet)=0.014762 nats; exact model
  KL=0.000511 nats; energy/site error=0.2222%; |m| error=0.1711%; learned
  ln Z=13.280304 versus exact 13.281033 (0.0055% error).
- Metrics at T=2.0: empirical KL(exact || GFlowNet)=0.015377 nats; exact model
  KL=0.002200 nats; energy/site error=0.6169%; |m| error=0.3366%; learned
  ln Z=17.011744 versus exact 17.105367 (0.5473% error).
- Estimator note: empirical KL uses a disclosed Jeffreys pseudocount of 0.5 per
  histogram bin so finite samples do not assign literal zero probability to
  physically nonzero rare states. Unsmooth, exactly enumerated model KL is also
  reported independently.
- Artifacts: `results/m2_metrics_20260818T012603-0400.json`,
  `results/m2_fixed_temperature_20260818T012603-0400.png`, and the two hashed
  `results/m2_model_T*_20260818T012603-0400.pt` checkpoints.
- Status: **PASSED**.
- Next action: implement and validate the single temperature-conditioned model
  for L=4 and L=8 in M3.
- Budget: day 2 of 7; estimated 5 days remaining.
