# GFlowNet Sampling of the 2D Ising Model Near Criticality

This repository validates a temperature-conditioned generative flow network
against exact and Markov-chain references for the periodic square-lattice Ising
model. Work proceeds through the acceptance-gated milestones in `AGENTS.md`.

## Current commands

```bash
python3 -m unittest discover -s tests -v
python3 scripts/m1_validate.py
```

All stochastic code requires an explicit seed. Validators write timestamped JSON
metrics and figures under `results/`, including the Git revision and complete
sampling/training configuration.

