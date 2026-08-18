# AGENTS.md — GFlowNet Sampling of the 2D Ising Model Near Criticality

## Mission
Build, validate, and analyze a temperature-conditioned GFlowNet that samples spin
configurations of the 2D Ising model from the Boltzmann distribution
p(x) ∝ exp(-E(x)/T), and use the trained model to PREDICT the critical
temperature Tc two independent ways: (1) from the learned partition function
logZ(T) itself, (2) from observables computed on generated samples. This is a
physics research project for a university independent study.
Correctness against known physics is the top priority. Fabricated or
unvalidated results are failure.

## Hard constraints
- Compute: single Mac CPU only. No CUDA. Keep networks small (MLPs, <= 3 hidden
  layers, <= 256 units). GFlowNet lattice sizes: L=4 and L=8 only. MCMC may go
  to L=12. If a training run would exceed ~2 hours, shrink it and note why.

## Ground truth anchors (never violate)
- 2D square-lattice Ising, nearest-neighbor J=1, periodic boundaries, zero field.
  E(x) = -sum over neighbor pairs s_i s_j, spins in {-1, +1}.
- Exact critical temperature: Tc = 2 / ln(1 + sqrt(2)) ≈ 2.26919 (Onsager).
- For L=4 (65,536 states) compute the exact Boltzmann distribution, exact
  observables, and exact ln Z(T) by full enumeration. This is the oracle for
  everything.
- Metropolis-Hastings MCMC is the required independent baseline at every T.

## Working loop (repeat until done)
1. Read PROGRESS.md. Identify the lowest-numbered milestone not marked PASSED.
2. Implement or fix code for that milestone.
3. Run its validation script (scripts/mN_validate.py). Every milestone has
   numeric acceptance criteria; the script must print PASS or FAIL with numbers.
4. Append a dated entry to PROGRESS.md: what was done, metrics, PASS/FAIL, next
   action. Never edit past entries.
5. After each milestone PASSES, append a section to NOTES.md explaining, in
   plain language a physics undergrad can present from, the math just
   implemented: the equations, why they work, and what the key hyperparameters
   do. Include the trajectory balance derivation when M2 passes.
6. If FAIL: diagnose (list >= 2 hypotheses, test the cheapest first), fix,
   rerun. Never weaken acceptance criteria. After 3 failed distinct fixes,
   write a STUCK entry with the attempts and best hypothesis, then stop for
   human review.
7. If PASS: mark PASSED, commit, continue.

## Engineering rules
- Python 3.11+, PyTorch (CPU). torchgfn is optional; a from-scratch trajectory
  balance implementation (~200 lines) is acceptable and easier to debug.
- Repo layout: src/ (ising.py, exact.py, metropolis.py, gflownet.py,
  observables.py), scripts/, results/ (json metrics + png plots named by
  milestone and timestamp), PROGRESS.md, NOTES.md, README.md.
- Every random process takes an explicit seed. Every result file records seed,
  git commit hash, and hyperparameters.
- Unit tests for energy, neighbor indexing, and periodic boundaries FIRST. A
  sign error in E(x) invalidates everything.
- Plots: labeled axes, and a vertical line at exact Tc=2.26919 wherever T is
  on the x-axis.

## Milestones

### M1 — Exact physics core + MCMC baseline
Ising energy; full enumeration for L=4 including exact ln Z(T); Metropolis for
L in {4, 8, 12}. Observables per T: energy/site, |m|, susceptibility chi,
specific heat C, Binder cumulant U4. T grid: 1.5 to 3.2, >= 14 points, denser
in [2.0, 2.5].
ACCEPT: (a) L=4 MCMC matches exact within 1% (energy, |m|) and 5% (chi, C) at
every T. (b) chi(T) for L=12 peaks in [2.15, 2.45]. (c) Unit tests pass.

### M2 — Fixed-temperature GFlowNet vs the exact oracle
States: lattice sites in {-1, +1, unassigned}; start all unassigned; action =
assign next site in raster order; terminal = fully assigned. Reward
R(x) = exp(-E(x)/T). Train with Trajectory Balance at T=3.0 and T=2.0, L=4.
logZ is a learned scalar.
ACCEPT: (a) KL(exact || GFlowNet empirical) < 0.05 nats at both T, from
>= 200k samples over all 65,536 states. (b) Mean energy and |m| within 2% of
exact. (c) Learned logZ within 2% of exact ln Z. Log all numbers.

### M3 — Temperature-conditioned GFlowNet
One model conditioned on beta=1/T (input feature), trained with T sampled from
[1.5, 3.2]. L=4 first, then L=8.
ACCEPT: (a) L=4: KL < 0.08 at T in {1.8, 2.269, 3.0} from the single model.
(b) L=4: learned logZ(T) within 3% of exact ln Z(T) across the grid. (c) L=8:
observables match Metropolis within 3% (energy, |m|) and 10% (chi). (d) Mode
coverage at T=1.8: fraction of samples with m>0 is 0.5 ± 0.05 (both ordered
modes covered). Report all.

### M4 — Predicting Tc FROM THE MODEL ITSELF (primary result)
This is the headline: the GFlowNet's own learned structure predicts criticality.
(a) logZ route: take the conditioned model's learned logZ(beta) on a dense beta
grid; compute energy U = -d(lnZ)/d(beta) and specific heat
C = beta^2 * d^2(lnZ)/d(beta)^2 by numerical differentiation (smooth first,
e.g. spline fit). The peak of C(T) is the model's Tc prediction. Validate the
whole pipeline on exact ln Z(T) for L=4 before applying it to learned logZ.
(b) Observable route: |m|(T), chi(T), U4(T) from samples at L in {4, 8}
(+ MCMC L=12); Tc from chi peak extrapolated in 1/L and from U4 crossings.
(c) Trajectory signatures (exploratory, report whatever is found honestly):
per-step policy entropy averaged over trajectories vs T; terminal-mode
symmetry breaking (P(m>0)) vs T.
ACCEPT: (a) On exact L=4 lnZ the C-peak pipeline recovers the known
finite-size C-peak location within 2%. (b) Learned-logZ Tc prediction and
observable-route Tc both land in [2.1, 2.5] (finite-size shift is expected and
must be discussed, not hidden). (c) One summary figure: C(T) from learned logZ,
chi(T), U4(T), all with the exact Tc line.

### M5 — Report
REPORT.md: problem statement; Ising + Boltzmann background; GFlowNet
formulation (state/action space, trajectory balance, learned logZ); validation
methodology (exact oracle, MCMC baseline); results with all figures; the two
Tc predictions and how they compare; honest limitations (CPU-bound sizes,
finite-size effects, mode coverage near Tc, where MCMC still wins); follow-ups.
Every number traceable to a file in results/.

### M6 — Paper-style PDF
Convert REPORT.md into an academic-paper-styled PDF (LaTeX via pandoc or a
two-column template): abstract, intro, methods, results, discussion,
references. References limited to ones verified to exist (see citation rule).
ACCEPT: PDF builds cleanly, all figures included, no orphaned claims.

## FALLBACK (descope path if a milestone proves infeasible)
Guaranteed-deliverable path: M1 + M2 + M4 using fixed-temperature GFlowNets at
6-8 temperatures for L=4 only (skip conditioning), logZ(T) from the per-T
learned values, observable route via MCMC at L in {8, 12}. Then M5 + M6.
Record the descope decision in PROGRESS.md.

## Anti-slop rules
- Never report a metric not computed in this run. Never fill a table by guessing.
- A result contradicting known physics is a bug finding, not a discovery. Say so.
- Small and verified beats large and impressive. L=4 exactly validated beats
  L=64 unvalidated.
- Citations allowed only if verified to exist: Bengio et al. 2021 "GFlowNet
  Foundations" (arXiv:2111.09266); Malkin et al. 2022 "Trajectory Balance"
  (arXiv:2201.13259); Zhang et al. 2022 "Generative Flow Networks for Discrete
  Probabilistic Modeling" (arXiv:2202.01361); Onsager 1944 (Phys. Rev. 65, 117).
  Anything else must be checked before inclusion.
