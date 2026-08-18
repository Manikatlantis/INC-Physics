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
