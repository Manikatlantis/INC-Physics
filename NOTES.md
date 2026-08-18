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
