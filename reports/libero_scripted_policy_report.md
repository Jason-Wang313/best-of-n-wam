# LIBERO Scripted Policy Smoke Report

This optional artifact runs a hand scripted OSC pick-place controller against LIBERO's real sparse success predicate. It is separate from the LIBERO rollout-pool WAM-lite artifact and should not be described as a learned policy or full LIBERO benchmark result.

## Summary

- Available: `True`.
- Attempted: `True`.
- Verified: `True`.
- Suite: `libero_object`.
- Episodes: `50` across `10` tasks and `5` seeds.
- Success rate: `0.6` with bootstrap CI [`0.46`, `0.74`].
- Successful episodes: `30`.

## Successful Episodes

- `libero_object/0` seed `100`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5277381614383393`.
- `libero_object/0` seed `101`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5199142473766224`.
- `libero_object/0` seed `102`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5340922928655296`.
- `libero_object/0` seed `103`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5418402527419623`.
- `libero_object/0` seed `104`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5425487442577821`.
- `libero_object/2` seed `100`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.3755318516977355`.
- `libero_object/2` seed `101`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.3678338501052798`.
- `libero_object/2` seed `102`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.36249078419281144`.
- `libero_object/2` seed `103`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.36511256908401857`.
- `libero_object/2` seed `104`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.3758084263942425`.
- `libero_object/3` seed `100`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3732769282038757`.
- `libero_object/3` seed `101`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3625056392299334`.
- `libero_object/3` seed `102`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3748420331513875`.
- `libero_object/3` seed `103`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.38984518133684176`.
- `libero_object/3` seed `104`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3965507469531576`.
- `libero_object/4` seed `100`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5316791493215227`.
- `libero_object/4` seed `101`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5218927848407622`.
- `libero_object/4` seed `102`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5211434996547314`.
- `libero_object/4` seed `103`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.52675696205257`.
- `libero_object/4` seed `104`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5371677078124485`.
- `libero_object/7` seed `100`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.516893886827215`.
- `libero_object/7` seed `101`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5102934863333072`.
- `libero_object/7` seed `102`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.526002348520287`.
- `libero_object/7` seed `103`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5297950220470373`.
- `libero_object/7` seed `104`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5286788297493599`.
- `libero_object/9` seed `100`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.37828152296679796`.
- `libero_object/9` seed `101`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.3705684806365335`.
- `libero_object/9` seed `102`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.36523431695437003`.
- `libero_object/9` seed `103`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.36786897553386805`.
- `libero_object/9` seed `104`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.378574818249496`.

## Failed Episodes

- `libero_object/1` seed `100`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; final distance `0.5766895280352258`, progress `-0.04766038709359521`.
- `libero_object/1` seed `101`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; final distance `0.573030535734826`, progress `-0.045456553946692346`.
- `libero_object/1` seed `102`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; final distance `0.5896533032340623`, progress `-0.04950657051993779`.
- `libero_object/1` seed `103`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; final distance `0.5980974297504985`, progress `-0.05224944513579921`.
- `libero_object/1` seed `104`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; final distance `0.5950620523312524`, progress `-0.05244669775558142`.
- `libero_object/5` seed `100`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; final distance `0.5761441924972395`, progress `-0.056769330753846825`.
- `libero_object/5` seed `101`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; final distance `0.5697980145940698`, progress `-0.054949249163106284`.
- `libero_object/5` seed `102`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; final distance `0.58414519781486`, progress `-0.059106644105885175`.
- `libero_object/5` seed `103`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; final distance `0.5966254448433392`, progress `-0.06279007852673213`.
- `libero_object/5` seed `104`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; final distance `0.5966638430312401`, progress `-0.0627939012616825`.
- `libero_object/6` seed `100`: `pick_up_the_butter_and_place_it_in_the_basket`; final distance `0.7980393211187327`, progress `-0.10310464874970848`.
- `libero_object/6` seed `101`: `pick_up_the_butter_and_place_it_in_the_basket`; final distance `0.7855013159188677`, progress `-0.09984813910320767`.
- `libero_object/6` seed `102`: `pick_up_the_butter_and_place_it_in_the_basket`; final distance `0.8051649445165734`, progress `-0.10646740165781599`.
- `libero_object/6` seed `103`: `pick_up_the_butter_and_place_it_in_the_basket`; final distance `0.820777530525308`, progress `-0.11037058022826374`.
- `libero_object/6` seed `104`: `pick_up_the_butter_and_place_it_in_the_basket`; final distance `0.8234246394901358`, progress `-0.1104872070563575`.
- `libero_object/8` seed `100`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; final distance `0.8154048357759744`, progress `-0.12484375156660543`.
- `libero_object/8` seed `101`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; final distance `0.8008542116091375`, progress `-0.12035226278526345`.
- `libero_object/8` seed `102`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; final distance `0.8000053935311479`, progress `-0.12007769324409223`.
- `libero_object/8` seed `103`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; final distance `0.8084820196650403`, progress `-0.12268581083271335`.
- `libero_object/8` seed `104`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; final distance `0.8239381636995988`, progress `-0.12747058838693237`.

## Limitations

- This is a scripted sparse-success smoke, not a learned WAM policy and not a demonstration that the project solves LIBERO.
- The controller uses object and target positions exposed by the simulator, so it is diagnostic benchmark evidence rather than deployable perception.
- Failed tasks remain reported in the CSV/JSON artifact; the claim is limited to the measured success subset.
