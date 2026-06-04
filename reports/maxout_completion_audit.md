# Max-Out Completion Audit

Audit date: 2026-05-30.

## Execution Tier

Benchmark-visual validated: theorem layer, learned toy, multi-env toy, Gymnasium/MuJoCo Reacher-v5 benchmark, Gymnasium Robotics Fetch benchmark, Meta-World ML1, RoboSuite Panda benchmark, ManiSkill3 state-mode benchmark, RoboCasa single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned WAM-lite, LIBERO Spatial three-task rollout-pool WAM-lite, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned autonomous low-dimensional BC smoke, LIBERO RGB/proprio/language BC smoke, toy visual mode, Reacher RGB WAM-lite, and Fetch RGB WAM-lite.

## Artifact Coverage

- Environments: `block_push, drawer_pull, slippery_grasp, nonstationary_shift, deformable_toy`.
- Learned backbones: `horizon_wam, mlp_dynamics_wam, ensemble_wam`.
- Multi-env seeds: `5`.
- Benchmark attempted: `True`; any benchmark available: `True`.
- Benchmark suite: `Reacher-v5`; rollout pools: `25`; exact-law MAE: `0.0188`.
- Gymnasium Robotics suite: `['FetchReach-v4', 'FetchPush-v4', 'FetchPickAndPlace-v4']`; rollout pools: `60`; exact-law MAE: `0.0126`; learned-random N32 CI lower: `0.4409`.
- Meta-World suite: `['reach-v3', 'push-v3', 'drawer-open-v3']`; rollout pools: `45`; exact-law MAE: `0.0298`; learned-random N32 CI lower: `0.0975`; reward-random N32 CI lower: `0.4666`.
- RoboSuite suite: `['Lift', 'Stack', 'Door']`; rollout pools: `30`; exact-law MAE: `0.0024`; learned-random N32 CI lower: `0.2447`; reward-random N32 CI lower: `0.2617`; closed-loop learned-random N8 CI lower: `0.0798`.
- ManiSkill suite: `['PickCube-v1', 'PushCube-v1', 'PegInsertionSide-v1']`; rollout pools: `30`; exact-law MAE: `0.0034`; control: `pd_joint_delta_pos`.
- RoboCasa learned WAM-lite: verified `True`; train samples `80`; eval samples `80`; utility corr `0.7640`; learned-random N8 CI lower `0.0503`.
- RoboCasa three-task WAM-lite: verified `True`; tasks `['robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave']`; train/eval samples `144`/`240`; utility corr `0.6752`; promoted scorer `learned_energy_regularized`; learned-random N8 CI lower `0.1691`.
- RoboCasa broad task family WAM-lite: verified `True`; tasks `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet']`; train/eval samples `64`/`128`; utility corr `0.8599`; promoted scorer `learned_wam`; learned-random N8 CI lower `0.2112`.
- RoboCasa 12-task family WAM-lite: verified `True`; tasks `12`; train/eval samples `96`/`192`; utility corr `0.8330`; promoted scorer `learned_energy_regularized`; learned-random N8 CI lower `0.1830`.
- RoboCasa 24-task family WAM-lite: verified `True`; tasks `24`; train/eval samples `192`/`384`; utility corr `0.8523`; promoted scorer `learned_energy_regularized`; learned-random N8 CI lower `0.2393`.
- RoboCasa extra four-task WAM-lite: verified `True`; tasks `4`; train/eval samples `64`/`128`; utility corr `0.7993`; promoted scorer `learned_wam`; learned-random N8 CI lower `0.2469`.
- RoboCasa combined 28-task family WAM-lite: verified `True`; tasks `28`; train/eval samples `448`/`448`; utility corr `0.8335`; promoted scorer `learned_wam`; learned-random N8 CI lower `0.2353`.
- RoboCasa combined 32-task family WAM-lite: verified `True`; tasks `32`; train/eval samples `512`/`512`; utility corr `0.8384`; promoted scorer `learned_energy_regularized`; learned-random N8 CI lower `0.2276`.
- RoboCasa stratified 55-task WAM-lite: verified `True`; tasks `55`; train/eval samples `880`/`880`; utility corr `0.8326`; promoted scorer `learned_wam`; learned-random N8 CI lower `0.2743`.
- RoboCasa stratified 97-task WAM-lite: verified `True`; tasks `97`; train/validation/eval samples `1552`/`776`/`1552`; rollout pools `194`; utility corr `0.8380`; promoted scorer `learned_wam`; learned-random N8 CI lower `0.3149`; oracle-learned N8 CI lower `0.0369`.
- RoboCasa residual 35-task clean/cook WAM-lite: verified `True`; tasks `35`; train/validation/eval samples `140`/`140`/`280`; rollout pools `35`; horizon `1`; Nmax `4`; utility corr `0.8345`; promoted scorer `learned_wam`; learned-random Nmax CI lower `0.1974`; oracle-learned Nmax CI lower `0.0137`.
- RoboCasa micro-rollout probe: verified `True`; nondegenerate extra task IDs `4`; rollouts per task `2`; horizon `1`.
- RoboCasa stratified micro-rollout probe: verified `True`; nondegenerate task IDs `23`; rollouts per task `2`; horizon `1`.
- RoboCasa frontier micro-rollout probe: verified `True`; candidates `64`; runnable `52`; nondegenerate task IDs `42`; rollouts per task `2`; horizon `1`.
- RoboCasa residual clean/cook micro-rollout sweep: verified `True`; candidates `43`; completed chunks `41`; timed-out chunks `2`; runnable `39`; nondegenerate task IDs `35`; rollouts per task `2`; horizon `1`.
- RoboCasa registry coverage audit: registered task IDs `396`; rollout-pool task IDs `132`; micro-rollout task IDs `106`; any-artifact task IDs `134`.
- LIBERO WAM-lite: verified `True`; tasks `['libero_spatial/0', 'libero_spatial/1', 'libero_spatial/2']`; train/eval samples `192`/`240`; utility corr `0.3526`; exact-law MAE `0.0001`; learned-random N8 CI lower `0.2653`.
- LIBERO scripted sparse-success smoke: verified `True`; episodes `50`; successes `50`; success-rate CI lower `1.0000`.
- LIBERO learned action-head smoke: verified `True`; train examples `5014`; eval successes `30`/`30`; success-rate CI lower `1.0000`.
- LIBERO time-conditioned autonomous low-dimensional BC smoke: verified `True`; train examples `12535`; eval successes `50`/`50`; success-rate CI lower `1.0000`; uses phase labels `False`; uses target commands `False`; uses step clock `True`.
- LIBERO RGB/proprio/language BC smoke: verified `True`; policy `rgb_proprio_language_knn_behavior_cloning`; train examples `7521`; eval successes `30`/`30`; success-rate CI lower `1.0000`; uses RGB `True`; uses language `True`; uses object state `False`.
- Visual attempted: `True`; visual verified: `True`.
- Benchmark visual verified: `True`.
- Benchmark RGB WAM-lite: `extra_trees_visual_wam`; verified: `True`; utility corr: `0.2199`; utility MAE: `0.5208`; exact-law MAE: `0.0157`.
- Gymnasium Robotics RGB WAM-lite: verified: `True`; mean utility corr: `0.7325`; exact-law MAE: `0.0106`; visual-random N32 CI lower: `0.3475`.
- ManiSkill visual/EE probe: attempted `True`; state baseline ok `True`; any visual success `False`; blocker `vk::Device::allocateDescriptorSetsUnique: ErrorOutOfPoolMemory`.
- ManiSkill dependency probe: Pinocchio import `False`; binary `pin` wheel `False`; source install attempted `True`.
- Inference audit tail/gain correlation: `0.9864`.
- Learned-backend inference audit present: `True`.
- Scorer repair N64 gain over predicted utility: `0.3489`.
- Compute frontier predicted N128-N1 gain: `0.0255`.

## Acceptance Status

- Pytest: run by the final execution sequence.
- Smoke: run by the final execution sequence.
- Learned WAM toy: run by the final execution sequence.
- Multi-env: artifacts cover BlockPush2D, DrawerPull, SlipperyGrasp, and Nonstationary.
- Backbones: MLP, horizon, and ensemble WAM artifacts are present.
- EXP10: anti-scorer and randomized-dynamics falsification artifacts are present when multi-env is regenerated.
- Benchmark: Gymnasium/MuJoCo Reacher-v5 artifacts generated.
- Gymnasium Robotics: FetchReach-v4, FetchPush-v4, and FetchPickAndPlace-v4 artifacts generated.
- Meta-World: reach-v3, push-v3, and drawer-open-v3 ML1 artifacts generated.
- RoboSuite: Lift, Stack, and Door Panda manipulation artifacts generated, including small closed-loop traces.
- ManiSkill: PickCube-v1, PushCube-v1, and PegInsertionSide-v1 state-mode artifacts generated.
- RoboCasa: `PickPlaceCounterToCabinet` kitchen smoke artifact generated in a separate RoboCasa-compatible environment; exact-law utility MAE `0.0003` over `80` rollouts.
- RoboCasa learned WAM-lite: single-task `PickPlaceCounterToCabinet` ridge state/action-sequence WAM trained on `80` rollouts and evaluated on `80` heldout rollouts.
- RoboCasa three-task learned WAM-lite: task conditioned ridge WAM over `['robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave']` trained on `144` rollouts and evaluated on `240` heldout rollouts.
- RoboCasa broad task family learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet']` trained on `64` rollouts and evaluated on `128` heldout rollouts.
- RoboCasa 12-task family learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven']` trained on `96` rollouts and evaluated on `192` heldout rollouts.
- RoboCasa 24-task family learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender']` trained on `192` rollouts and evaluated on `384` heldout rollouts.
- RoboCasa extra four-task learned WAM-lite: task conditioned ridge WAM over `['robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter']` trained on `64` rollouts and evaluated on `128` heldout rollouts.
- RoboCasa combined 28-task learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter']` trained on `448` rollouts and evaluated on `448` heldout rollouts.
- RoboCasa combined 32-task learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter', 'robocasa/CloseFridgeDrawer', 'robocasa/TurnOnBlender', 'robocasa/PickPlaceCabinetToCounter', 'robocasa/PickPlaceSinkToCounter']` trained on `512` rollouts and evaluated on `512` heldout rollouts.
- RoboCasa stratified 55-task learned WAM-lite: task conditioned ridge WAM over `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter', 'robocasa/CloseFridgeDrawer', 'robocasa/TurnOnBlender', 'robocasa/PickPlaceCabinetToCounter', 'robocasa/PickPlaceSinkToCounter', 'robocasa/CleanMicrowave', 'robocasa/ClearSink', 'robocasa/ClearCuttingBoard', 'robocasa/RinseCuttingBoard', 'robocasa/RinseSinkBasin', 'robocasa/WashFruitColander', 'robocasa/WashLettuce', 'robocasa/CollectWashingSupplies', 'robocasa/MicrowavePressButton', 'robocasa/ToastBagel', 'robocasa/AdjustHeat', 'robocasa/LowerHeat', 'robocasa/TransportCookware', 'robocasa/FillKettle', 'robocasa/FillBlenderJug', 'robocasa/LoadDishwasher', 'robocasa/EmptyDishRack', 'robocasa/LoadFridgeByType', 'robocasa/ArrangeDrinkware', 'robocasa/GatherVegetables', 'robocasa/OrganizeCondiments', 'robocasa/TurnOnMicrowave', 'robocasa/TurnOnToaster']` trained on `880` rollouts and evaluated on `880` heldout rollouts.
- RoboCasa stratified 97-task learned WAM-lite: task conditioned ridge WAM over `97` task IDs trained on `1552` rollouts, validated on `776` rollouts, and evaluated on `1552` heldout rollouts from `194` rollout pools.
- RoboCasa residual 35-task clean/cook learned WAM-lite: task conditioned ridge WAM over `35` task IDs trained on `140` rollouts, validated on `140` rollouts, and evaluated on `280` heldout rollouts from `35` rollout pools with horizon `1` and Nmax `4`.
- RoboCasa micro-rollout probe: task IDs `['robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter']` reset and produced short nondegenerate rollouts. This remains lower-tier viability evidence; the separate extra four-task WAM artifact is the stronger learned-WAM validation for the same task IDs.
- RoboCasa stratified micro-rollout probe: `23` task IDs reset and produced short nondegenerate rollouts across wider kitchen categories. This remains lower-tier viability evidence; the separate stratified 55-task and 97-task WAM artifacts are stronger learned-WAM validation for promoted task IDs.
- RoboCasa frontier micro-rollout probe: `42` of `64` candidate task IDs reset and produced short nondegenerate rollouts across manipulation, movement, pick-place, cleaning, washing, cooking, and arrangement families. This remains lower-tier viability evidence; the separate 97-task WAM artifact is the stronger learned-WAM validation for promoted task IDs.
- RoboCasa residual clean/cook micro-rollout sweep: `35` of `43` cleaning/cooking task IDs produced nondegenerate short rollouts; two timeout chunks are documented, and this sweep is not promoted as full learned-WAM evidence beyond the separate residual 35-task artifact.
- RoboCasa catalog coverage audit: local registry contains `396` task IDs; verified rollout-pool artifacts cover `132` of them, micro-rollout probes cover `106`, and any committed artifact covers `134`. This is coverage accounting, not validation for uncovered IDs.
- LIBERO learned WAM-lite: three Spatial tasks `['libero_spatial/0', 'libero_spatial/1', 'libero_spatial/2']` trained on `192` rollout samples and evaluated on `240` heldout rollout samples with dense progress utility.
- LIBERO sparse-success scripted smoke: all 10 Object tasks evaluated over `5` seeds with `50` successes over `50` episodes.
- LIBERO learned action-head smoke: `knn` action head trained on `5014` scripted action examples and evaluated on `30` heldout sparse-success episodes over all 10 Object tasks.
- LIBERO time-conditioned autonomous low-dimensional BC smoke: kNN behavior cloning trained on `12535` scripted action examples and evaluated on `50` heldout sparse-success episodes over all 10 Object tasks without phase labels or target-point commands at evaluation time.
- LIBERO RGB/proprio/language BC smoke: `feature-kNN RGB/proprio/language behavior cloning` trained on `7521` scripted action examples and evaluated on `30` heldout sparse-success episodes over all 10 Object tasks without object state, task IDs, phase labels, or target-point commands at evaluation time; this is `a lightweight feature-kNN smoke, not full LIBERO or modern VLA evidence`.
- Visual: toy visual mode verified with MAE `0.0185`.
- Benchmark visual WAM: Reacher-v5 RGB-frame/action-sequence model verified with visual-random N32 CI lower bound `0.1998`.
- Gymnasium Robotics visual WAM: Fetch RGB-frame/action-sequence models verified with visual-random N32 CI lower bound `0.3475`.
- ManiSkill visual/EE-control probe: generated artifact-backed blocker report when local RGB/RGB-D and EE-control attempts failed.
- Audit framework: inference-value profiles, deployment gates, scorer repair, and compute frontiers generated.
- README overclaims: `0`.

## Key Numerical Results

- EXP1 success MAE: `0.0021`.
- EXP1 utility MAE: `0.0138`.
- EXP2 max AUC identity error: `0.00000000`.
- EXP2 same-p/kappa N64 gap: `0.9988`.
- EXP3 relative MAE reduction: `0.7280`.
- EXP4 oracle-random N64 utility gap: `7.0843`.
- EXP5 severe mismatch gap growth: `16.7257`.
- EXP6 moment-law improvement over uniform: `0.0767`.
- EXP7 learned useful N64-N1 success gain: `0.2167`.
- EXP8 conditional-law MAE: `0.0027`.
- Gymnasium Robotics Fetch exact-law MAE: `0.0126`.
- Meta-World exact-law MAE: `0.0298`.
- RoboSuite exact-law MAE: `0.0024`.
- RoboCasa smoke exact-law MAE: `0.0003`.
- RoboCasa learned WAM utility corr: `0.7640`.
- RoboCasa learned-random N8 CI lower: `0.0503`.
- RoboCasa three-task WAM utility corr: `0.6752`.
- RoboCasa three-task learned-random N8 CI lower: `0.1691`.
- RoboCasa broad WAM utility corr: `0.8599`.
- RoboCasa broad learned-random N8 CI lower: `0.2112`.
- RoboCasa broad exact-law utility MAE: `0.0004`.
- RoboCasa 12-task family WAM utility corr: `0.8330`.
- RoboCasa 12-task family learned-random N8 CI lower: `0.1830`.
- RoboCasa 12-task family exact-law utility MAE: `0.0003`.
- RoboCasa 24-task family WAM utility corr: `0.8523`.
- RoboCasa 24-task family learned-random N8 CI lower: `0.2393`.
- RoboCasa 24-task family exact-law utility MAE: `0.0003`.
- RoboCasa extra four-task WAM utility corr: `0.7993`.
- RoboCasa extra four-task learned-random N8 CI lower: `0.2469`.
- RoboCasa extra four-task exact-law utility MAE: `0.0003`.
- RoboCasa combined 28-task WAM utility corr: `0.8335`.
- RoboCasa combined 28-task learned-random N8 CI lower: `0.2353`.
- RoboCasa combined 28-task exact-law utility MAE: `0.0003`.
- RoboCasa combined 32-task WAM utility corr: `0.8384`.
- RoboCasa combined 32-task learned-random N8 CI lower: `0.2276`.
- RoboCasa combined 32-task exact-law utility MAE: `0.0003`.
- RoboCasa stratified 55-task WAM utility corr: `0.8326`.
- RoboCasa stratified 55-task learned-random N8 CI lower: `0.2743`.
- RoboCasa stratified 55-task exact-law utility MAE: `0.0003`.
- RoboCasa stratified 97-task WAM utility corr: `0.8380`.
- RoboCasa stratified 97-task learned-random N8 CI lower: `0.3149`.
- RoboCasa stratified 97-task oracle-learned N8 CI lower: `0.0369`.
- RoboCasa stratified 97-task exact-law utility MAE: `0.0003`.
- RoboCasa residual 35-task WAM utility corr: `0.8345`.
- RoboCasa residual 35-task learned-random Nmax CI lower: `0.1974`.
- RoboCasa residual 35-task oracle-learned Nmax CI lower: `0.0137`.
- RoboCasa residual 35-task exact-law utility MAE: `0.0002`.
- RoboCasa micro-rollout extra tasks: `4` / `4` nondegenerate.
- RoboCasa stratified micro-rollout tasks: `23` / `28` nondegenerate.
- RoboCasa frontier micro-rollout tasks: `42` / `64` nondegenerate.
- RoboCasa residual clean/cook micro-rollout tasks: `35` / `43` nondegenerate.
- RoboCasa catalog coverage: `132` rollout-pool task IDs and `106` micro-rollout task IDs out of `396` registered task IDs.
- LIBERO WAM utility corr: `0.3526`.
- LIBERO learned-random N8 CI lower: `0.2653`.
- LIBERO exact-law utility MAE: `0.0001`.
- LIBERO Object scripted success rate: `1.0000` with CI [`1.0000`, `1.0000`].
- LIBERO learned action-head heldout success rate: `1.0000` with CI [`1.0000`, `1.0000`].
- LIBERO time-conditioned autonomous low-dimensional BC heldout success rate: `1.0000` with CI [`1.0000`, `1.0000`].
- LIBERO RGB/proprio/language BC heldout success rate: `1.0000` with CI [`1.0000`, `1.0000`].
- Falsification anti-scorer N64 mean utility: `-26.5657`.
