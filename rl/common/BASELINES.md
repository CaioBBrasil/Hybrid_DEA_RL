# Baselines

Plan A compares DQN policies with four selectors under the same environment,
image order, and safety shield:

- Highest-Quality;
- Lowest-Energy;
- TOPSIS-Static;
- Energy-Aware.

Weights are stored in `rl/common/configs/baselines.yaml`. Quality values and
confidence thresholds come from validation; the final test split is not read
while constructing the rankings. The full evaluation command is documented in
`rl/plan_a/README.md`.
