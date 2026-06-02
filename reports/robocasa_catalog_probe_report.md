# RoboCasa Catalog Probe

- status: `verified`
- registered RoboCasa task IDs: `396`
- task IDs covered by verified rollout-pool artifacts: `28`
- rollout-pool coverage fraction: `0.0707`
- task IDs covered by micro-rollout viability probes: `4`
- any-artifact task coverage: `28`
- any-artifact coverage fraction: `0.0707`

## Coverage By Category

- `cleaning`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `35`
- `close`: rollout-pool `6`, micro-rollout `0`, any `6` / registered `13`
- `cooking`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `40`
- `long_horizon_or_compositional`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `249`
- `manipulate`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `5`
- `move`: rollout-pool `0`, micro-rollout `0`, any `0` / registered `4`
- `open`: rollout-pool `7`, micro-rollout `0`, any `7` / registered `13`
- `pick_place`: rollout-pool `11`, micro-rollout `4`, any `11` / registered `25`
- `turn`: rollout-pool `4`, micro-rollout `0`, any `4` / registered `12`

This is a registry coverage audit. It deliberately does not promote uncovered task IDs to benchmark evidence; uncovered IDs still need reset, rollout-pool, learned-WAM, and CI artifacts before the README can claim them.
