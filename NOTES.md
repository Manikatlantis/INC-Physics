# Presentation Notes

Plain-language explanations are appended here only after their corresponding
milestone passes validation.

## M1 — Exact Ising physics and the Metropolis baseline

### What is being modeled

Each lattice site has a spin \(s_i\in\{-1,+1\}\). With nearest-neighbor coupling
\(J=1\), no external field, and periodic boundaries, the energy is

\[
E(x)=-\sum_{\langle i,j\rangle}s_i s_j.
\]

The code counts every bond once by pairing each site with its right and lower
neighbor; indices wrap at each edge. Parallel neighbors contribute \(-1\) and
antiparallel neighbors contribute \(+1\). An aligned L=4 state therefore has
32 satisfied bonds and energy \(-32\). Flipping one spin changes its four bonds,
so the local energy change is \(\Delta E=2s_i\sum_{j\in\mathrm{nn}(i)}s_j\).

At temperature \(T\), the Boltzmann probability and partition function are

\[
p_T(x)=\frac{e^{-E(x)/T}}{Z(T)},\qquad
Z(T)=\sum_x e^{-E(x)/T}.
\]

There are only \(2^{16}=65{,}536\) L=4 states, so M1 enumerates all of them.
This gives exact probabilities, \(\ln Z\), and observables against which every
approximate sampler can be checked. Numerically, \(\ln Z\) is evaluated after
subtracting the largest log-weight so exponentials do not overflow.

### The measured observables

Writing \(N=L^2\), \(M=\sum_i s_i\), and \(\beta=1/T\), the reported quantities
are

\[
u=\frac{\langle E\rangle}{N},\qquad
|m|=\frac{\langle|M|\rangle}{N},
\]

\[
\chi=\frac{\beta}{N}\left(\langle M^2\rangle-\langle|M|\rangle^2\right),
\qquad
C=\frac{\beta^2}{N}\left(\langle E^2\rangle-\langle E\rangle^2\right),
\]

\[
U_4=1-\frac{\langle M^4\rangle}{3\langle M^2\rangle^2}.
\]

The absolute-magnetization form of \(\chi\) prevents finite lattices tunneling
between the equally likely positive and negative ordered phases from looking
like a huge fluctuation. Peaks in \(\chi\) and \(C\), and crossings in \(U_4\)
for different L, become finite-size indicators of the phase transition.

### Why Metropolis sampling works

A Metropolis proposal flips one spin. If \(\Delta E\leq0\), it is accepted. If
the energy rises, it is accepted with probability

\[
A(x\rightarrow x')=e^{-\Delta E/T}.
\]

The proposal is symmetric, and the acceptance ratio is exactly the ratio of
Boltzmann weights. This gives detailed balance with \(p_T\), so after burn-in the
chain samples the desired equilibrium distribution. The implementation updates
one checkerboard sublattice at a time: sites of the same parity share no bonds on
the even lattices used here, which makes their Metropolis decisions safely
parallelizable on a CPU.

The main sampling hyperparameters have concrete roles:

- **Independent chains** reduce sensitivity to a single initial condition and
  improve coverage of both magnetization signs.
- **Burn-in sweeps** are discarded while chains relax toward equilibrium.
- **Thinning sweeps** separate recorded states and reduce autocorrelation.
- **Samples per chain** control statistical precision; heat capacity and
  susceptibility need more samples because they are variances.
- **Seed** makes every stochastic result exactly reproducible.

For the final M1 run, every temperature used 384,000 L=4 samples, 64,000 L=8
samples, and 96,000 L=12 samples. The L=4 sampler agreed with exact enumeration
well inside every required tolerance, while the L=12 susceptibility maximum on
the chosen grid occurred at T=2.45, at the upper edge of the accepted finite-size
window.

## M2 — Fixed-temperature GFlowNets and trajectory balance

### Turning spin assignment into a flow network

The initial state is a 4-by-4 lattice with every site unassigned. At step \(t\),
the policy assigns spin \(-1\) or \(+1\) to site \(t\) in raster order. After 16
steps the terminal state is a complete Ising configuration \(x\), with reward

\[
R_T(x)=e^{-E(x)/T}.
\]

The small neural policy sees assigned spins as \(\pm1\), unassigned sites as
zero, and the current step as a scalar feature. It outputs two logits, whose
softmax gives \(P_F(-1\mid s_t)\) and \(P_F(+1\mid s_t)\). M2 uses two hidden
layers of 128 units, safely inside the CPU constraint.

### Trajectory-balance derivation

Consider a construction trajectory
\(\tau=(s_0\rightarrow s_1\rightarrow\cdots\rightarrow s_n=x)\). A GFlowNet
assigns a forward probability \(P_F(s_{t+1}\mid s_t)\), a backward probability
\(P_B(s_t\mid s_{t+1})\), and a learnable source flow \(Z\). Trajectory balance
requires the total forward flow along every complete trajectory to equal its
reward-weighted reverse flow:

\[
Z\prod_{t=0}^{n-1}P_F(s_{t+1}\mid s_t)
=R(x)\prod_{t=0}^{n-1}P_B(s_t\mid s_{t+1}).
\]

Taking logarithms turns products into a stable additive residual,

\[
\delta(\tau)=\log Z
+\sum_t\log P_F(s_{t+1}\mid s_t)
-\log R(x)
-\sum_t\log P_B(s_t\mid s_{t+1}),
\]

and training minimizes

\[
\mathcal L_{\mathrm{TB}}=\mathbb E_{\tau}\!\left[\delta(\tau)^2\right].
\]

Why does this produce the Boltzmann distribution? Sum the non-log balance
equation over every trajectory ending at a particular \(x\). The backward
policy is normalized over reverse paths, so its path probabilities sum to one.
The terminal forward flow is therefore \(ZP_F(x)=R(x)\), giving

\[
P_F(x)=\frac{R(x)}{Z}.
\]

Normalizing over all terminals then forces \(Z=\sum_xR(x)\), which is exactly
the Ising partition function. Thus \(\log Z\) is not supplied by the exact
oracle; it is learned as the scalar needed to make all trajectory flows agree.

Raster order makes this case especially transparent. Every terminal \(x\) has
exactly one construction trajectory, and each state has exactly one parent, so
every backward probability is one. With \(\log R(x)=-E(x)/T\), the residual is

\[
\delta(x)=\log Z+\sum_{t=0}^{15}\log P_F(s_{t+1}\mid s_t)+\frac{E(x)}{T}.
\]

At zero residual, the summed policy log-probability equals
\(-E(x)/T-\log Z\), precisely the log Boltzmann probability.

### Training and validation choices

Each mini-batch mixes current-policy trajectories with uniformly generated
terminal trajectories. Uniform trajectories give every region of the state
space nonzero training coverage; policy trajectories focus capacity where the
model currently places mass. A 5% uniform action mixture adds further
exploration. Every sampled terminal is paired with its global spin flip. Since
zero-field Ising energy is unchanged by \(x\mapsto-x\), this prevents a spurious
preference for one ordered mode without changing the target distribution.

The important hyperparameters are:

- **Optimizer steps (8,000):** how long the balance residual is refined.
- **Batch size (1,024):** the number of complete trajectories per update.
- **Learning rate (0.002 down to 0.0002):** large enough for early progress,
  then annealed for a stable final fit.
- **Uniform fraction (0.5) and exploration (0.05):** protect state-space and
  mode coverage, especially at T=2.
- **Gradient clipping (10):** limits an unusually large early TB update without
  changing the acceptance threshold.

Validation uses two complementary KL measurements. Enumerating the
autoregressive probability of every L=4 state gives a noise-free model KL. The
required empirical KL comes from two million seeded terminal draws. Because a
finite histogram inevitably misses extremely rare states whose exact
probability is still nonzero, its reported estimator uses the standard Jeffreys
half-count,

\[
\widehat q_i=\frac{n_i+1/2}{N_{\rm sample}+K/2},\qquad
D_{\rm KL}(p\|\widehat q)=\sum_i p_i\log\frac{p_i}{\widehat q_i},
\]

with \(K=65{,}536\). The pseudocount and the unsmoothed enumerated-model KL are
both recorded so this finite-sample convention is auditable.

Both temperatures passed. The empirical KL values were 0.01476 nats at T=3 and
0.01538 nats at T=2, while the learned \(\log Z\) errors were 0.0055% and 0.547%,
respectively. The harder low-temperature model still covered both symmetry
modes because of explicit spin-flip pairing and full-support exploration.

## M3 — A single model that works across temperature

### Conditioning on inverse temperature

M2 needed a separate policy at each temperature. M3 replaces those policies
with one function \(P_F(a\mid s,\beta)\), where \(\beta=1/T\), and replaces the
single learned scalar \(\log Z\) with a small function \(\log Z_\theta(\beta)\).
The trajectory-balance residual for terminal \(x\) becomes

\[
\delta(x,\beta)=\log Z_\theta(\beta)
+\log P_F(x\mid\beta)+\beta E(x).
\]

Training samples T throughout [1.5, 3.2], so the same weights must drive this
residual toward zero for a continuum of Boltzmann distributions. Half of the
temperature draws are stratified onto T=1.8, the exact critical temperature,
and T=3.0; the other half remain continuous uniform draws, so the model is not
just a three-temperature lookup table.

### Why the policy is masked

With 64 sites at L=8, evaluating a conventional MLP separately for every one of
64 trajectory steps is expensive during training. A masked autoregressive MLP
instead outputs all 64 binary logits in one forward pass. Its fixed connection
masks guarantee that logit \(k\) can depend only on spins \(0,\ldots,k-1\), never
on the current or future spins. Therefore

\[
\log P_F(x\mid\beta)=\sum_{k=0}^{N-1}
\log P_F(s_k\mid s_0,\ldots,s_{k-1},\beta)
\]

is still a valid normalized autoregressive probability. Tests explicitly change
future spins and verify that earlier logits do not move; full enumeration at
L=2 verifies normalization.

The input includes both each prefix spin and its product with normalized beta.
The \(\beta s_i\) feature and a causal linear skip make it easier to learn the
temperature-scaled local Ising interaction, while the nonlinear hidden layers
learn the effect of unassigned parts of the lattice.

### Enforcing the exact zero-field symmetry

At zero field, \(E(x)=E(-x)\), so the target distribution gives equal weight to
the positive and negative ordered modes. For a raw causal logit function
\(g_k\), the implemented policy uses

\[
\ell_k(s_{<k},\beta)=\frac12\left[
g_k(s_{<k},\beta)-g_k(-s_{<k},\beta)\right].
\]

This makes \(\ell_k(-s_{<k},\beta)=-\ell_k(s_{<k},\beta)\). Consequently, every
trajectory and its global spin flip have exactly equal policy probability:
\(P_F(x\mid\beta)=P_F(-x\mid\beta)\). The first-spin logit is identically zero,
so generation begins in either sign with probability one half. This is a known
physical symmetry, not information from the validation samples. At T=1.8, the
measured L=8 fraction with positive magnetization was 0.498825.

### Coverage and CPU choices

Trajectory balance can train off-policy, because its equality is valid for any
complete trajectory. Batches therefore combine:

- uniform configurations for full support;
- current-policy configurations for accuracy where the model places mass;
- noisy ordered configurations for the low-temperature sector; and
- block-domain configurations for correlated, low-magnetization states.

All are paired with global spin flips. The largest model has two 256-unit hidden
layers, the allowed CPU ceiling. It trained for 16,000 steps with batch size 512.
This longer run was needed because susceptibility is a variance: the mean energy
and |m| could look accurate while a small over-weighting of rare domain walls
still biased \(\chi\).

### What passed

For L=4, the exact KL divergence stayed between 0.00031 and 0.00326 nats at the
three validation temperatures. The maximum \(\log Z(T)\) error over a 16-point
grid was 0.105%, demonstrating interpolation by the learned normalization
network.

For L=8, 200,000 fresh generated configurations at each temperature were
compared with 256,000 independent Metropolis samples. All mean energy and |m|
errors were below 0.45%; all susceptibility errors were below 4.35%. This clears
the respective 3%, 3%, and 10% requirements without relaxing any criterion.

## M4 — Predicting the critical temperature from the model

### Route 1: curvature of the learned partition function

The partition function contains thermodynamics before any terminal samples are
drawn. In inverse-temperature coordinates,

\[
Z(\beta)=\sum_x e^{-\beta E(x)},\qquad
U=-\frac{\partial\ln Z}{\partial\beta}.
\]

Differentiating again gives the energy variance,

\[
\frac{\partial^2\ln Z}{\partial\beta^2}
=\langle E^2\rangle-\langle E\rangle^2,
\]

so heat capacity per site is

\[
c(\beta)=\frac{\beta^2}{N}
\frac{\partial^2\ln Z}{\partial\beta^2}.
\]

M4 evaluates \(\ln Z\) on 256 evenly spaced beta values, fits a degree-10
Chebyshev polynomial, differentiates that smooth fit analytically, and searches
for the maximum over T in [1.8, 2.8]. The exact L=4 oracle is the non-negotiable
calibration: direct exact observables put its finite-size peak at T=2.438950,
while the logZ pipeline returns T=2.439257, only 0.0126% away.

Applying exactly the same procedure to the learned normalization networks gives
T=2.281360 for L=4 and T=2.342234 for L=8. The larger-lattice value is the
declared primary logZ prediction and is 3.22% above Onsager's thermodynamic-limit
Tc=2.269185. Both are finite-size model predictions, so agreement to all digits
is neither expected nor claimed.

Differentiation is less forgiving than value prediction: tiny smooth errors in
\(\ln Z\) are magnified by a second derivative. In particular, the learned
curves yield negative estimated heat capacity near the T=1.5 boundary. Negative
equilibrium heat capacity is not physical for this canonical model; it is a
boundary-curvature/model-fit artifact. The peak validation passes, but the
artifact is an important limitation of using learned \(\ln Z\) derivatives.

### Route 2: finite-size observables from generated states

For a finite lattice, susceptibility has a rounded maximum at a pseudo-critical
temperature \(T_\chi(L)\). The maximum shifts with lattice size. With only three
small sizes, M4 uses the simplest stated extrapolation,

\[
T_\chi(L)=T_c+\frac{a}{L}.
\]

Local quadratic fits give peaks at 2.812028 for generated L=4 states, 2.552567
for generated L=8 states, and 2.447225 for L=12 Metropolis states. Linear
regression against \(1/L\) has \(R^2=0.99825\) and intercept

\[
T_c^{(\chi)}=2.273526,
\]

which is 0.19% above the exact Tc. The very high \(R^2\) describes these three
points only; it is not a substitute for larger lattices or uncertainty analysis.

The Binder cumulant is nearly size-independent at criticality. Linear
interpolation of the generated/sample curves gives

\[
T_\times^{(4,8)}=2.242534,\qquad
T_\times^{(8,12)}=2.249908.
\]

Their mean, 2.246221, is 1.01% below exact Tc. Giving the susceptibility family
and the mean Binder family equal weight produces the declared observable-route
consensus

\[
T_c^{(\mathrm{obs})}=2.259873,
\]

0.41% below exact. The weighting is a transparent summary convention, not a
new estimator optimized after seeing the exact answer; the component estimates
remain the scientifically useful outputs.

### What trajectories reveal

For a binary action with probability \(p_t\), per-step policy entropy is

\[
H_t=-p_t\ln p_t-(1-p_t)\ln(1-p_t).
\]

The first spin always has \(H_0=\ln2\) because of enforced spin-flip symmetry.
At low T, later actions become predictable after an ordered sign is selected,
so mean L=8 entropy is only 0.04587 nats at T=1.5. It rises smoothly through
0.28136 nats at exact Tc to 0.56129 at T=3.2 as disorder makes both actions more
plausible. This monotonic change is a useful learned signature, but no separate
Tc is extracted from it.

Raw \(P(m>0)\) drifts slightly below one half at high T because a growing
fraction of finite-lattice samples has exactly zero magnetization. Conditioning
on nonzero magnetization removes that bookkeeping effect; at exact Tc the
positive fraction is 0.50014. Thus the trajectories show no spontaneous
preference for either ordered sign, consistent with the zero-field symmetry.

## M5 — How to present the result honestly

The report separates three kinds of statements that should not be blurred in a
presentation:

1. **Validation facts:** exact L=4 KL, MCMC errors, and L=8 observable errors say
   whether the sampler is trustworthy at the sizes tested.
2. **Finite-size estimates:** the learned-logZ peak, susceptibility intercept,
   and Binder crossings are different estimators with different biases. The
   observable consensus is a declared equal weighting, not a fitted law.
3. **Thermodynamic truth:** Onsager's Tc is the comparison target. It is never
   fed into training or used to adjust an acceptance threshold.

The clean headline is: “The model predicted Tc=2.3422 from its learned partition
function and Tc=2.2599 from generated observables, compared with exact 2.2692.”
Immediately follow it with the important qualification that the logZ derivative
has a nonphysical low-temperature boundary artifact and that only L=4/L=8
GFlowNets were feasible on the CPU budget.

When comparing samplers, avoid saying the GFlowNet replaces MCMC. Metropolis is
simpler and reached L=12 without training. The GFlowNet's distinct advantages
are amortized independent generation across T, explicit two-mode coverage, and
access to a learned normalization function. That complementary comparison is
both more accurate and more scientifically defensible.
