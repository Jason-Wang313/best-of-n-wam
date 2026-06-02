# RoboCasa Catalog Probe

- status: `verified`
- registered RoboCasa task IDs: `396`
- task IDs covered by verified rollout-pool artifacts: `55`
- rollout-pool coverage fraction: `0.1389`
- task IDs covered by micro-rollout viability probes: `27`
- any-artifact task coverage: `55`
- any-artifact coverage fraction: `0.1389`

## Coverage By Category

- `cleaning`: rollout-pool `8`, micro-rollout `8`, any `8` / registered `35`
- `close`: rollout-pool `7`, micro-rollout `0`, any `7` / registered `13`
- `cooking`: rollout-pool `5`, micro-rollout `5`, any `5` / registered `40`
- `long_horizon_or_compositional`: rollout-pool `8`, micro-rollout `8`, any `8` / registered `249`
- `manipulate`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `5`
- `move`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `4`
- `open`: rollout-pool `7`, micro-rollout `0`, any `7` / registered `13`
- `pick_place`: rollout-pool `13`, micro-rollout `4`, any `13` / registered `25`
- `turn`: rollout-pool `7`, micro-rollout `2`, any `7` / registered `12`

This is a registry coverage audit. It deliberately does not promote uncovered task IDs to benchmark evidence; uncovered IDs still need reset, rollout-pool, learned-WAM, and CI artifacts before the README can claim them.
