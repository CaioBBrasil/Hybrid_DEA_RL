# DEA + RL core

The active implementation has two parts:

```text
rl/common/   shared implementation and unit tests
rl/plan_a/   experiment using real predictions and Raspberry Pi 4 measurements
```

The policy selects only the detector. A common safety shield controls return
to home; `RETURN` is not learned by the DQN.

The mission contains up to 152 cells spaced 100 m apart. Coverage is an outcome:
the selected detectors, their inference costs, and the safety boundary determine
how much of the route can be processed before returning home.

```bash
.env/bin/python -m unittest discover -s rl/common/tests -v
.env/bin/python -m rl.common.validate_env \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml
```

See [`plan_a/README.md`](plan_a/README.md) for the complete protocol.
