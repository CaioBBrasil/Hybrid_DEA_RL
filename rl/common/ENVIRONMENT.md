# Mission environment

Each episode is a linear route of image-bearing cells. An action selects one
detector. The normalized observation contains remaining battery, route
progress, and estimated return energy.

The energy balance records movement, hover during inference, and embedded
computer energy separately. A common safety shield controls return to home;
the DQN does not learn a return action.

The Plan A scenario contains up to 152 cells spaced 100 m apart. Coverage is
not fixed: the safety shield returns the UAV when processing another cell with
the selected detector would make a safe return infeasible.

```bash
.env/bin/python -m rl.common.validate_env \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml
```
