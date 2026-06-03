# RoboCasa Catalog Probe

- status: `verified`
- registered RoboCasa task IDs: `396`
- task IDs covered by verified rollout-pool artifacts: `97`
- rollout-pool coverage fraction: `0.2449`
- task IDs covered by micro-rollout viability probes: `69`
- any-artifact task coverage: `97`
- any-artifact coverage fraction: `0.2449`

## Coverage By Category

- `cleaning`: rollout-pool `19`, micro-rollout `19`, any `19` / registered `35`
- `close`: rollout-pool `7`, micro-rollout `0`, any `7` / registered `13`
- `cooking`: rollout-pool `13`, micro-rollout `13`, any `13` / registered `40`
- `long_horizon_or_compositional`: rollout-pool `18`, micro-rollout `18`, any `18` / registered `249`
- `manipulate`: rollout-pool `2`, micro-rollout `2`, any `2` / registered `5`
- `move`: rollout-pool `4`, micro-rollout `4`, any `4` / registered `4`
- `open`: rollout-pool `7`, micro-rollout `0`, any `7` / registered `13`
- `pick_place`: rollout-pool `19`, micro-rollout `10`, any `19` / registered `25`
- `turn`: rollout-pool `8`, micro-rollout `3`, any `8` / registered `12`

This is a registry coverage audit. It deliberately does not promote uncovered task IDs to benchmark evidence; uncovered IDs still need reset, rollout-pool, learned-WAM, and CI artifacts before the README can claim them.
