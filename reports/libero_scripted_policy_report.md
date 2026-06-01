# LIBERO Scripted Policy Smoke Report

This optional artifact runs a hand scripted OSC pick-place controller against LIBERO's real sparse success predicate. It is separate from the LIBERO rollout-pool WAM-lite artifact and should not be described as a learned policy or full LIBERO benchmark result.

## Summary

- Available: `True`.
- Attempted: `True`.
- Verified: `True`.
- Suite: `libero_object`.
- Episodes: `50` across `10` tasks and `5` seeds.
- Success rate: `1.0` with bootstrap CI [`1.0`, `1.0`].
- Successful episodes: `50`.

## Successful Episodes

- `libero_object/0` seed `100`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5363086846852569`.
- `libero_object/0` seed `101`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5283750412077174`.
- `libero_object/0` seed `102`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5427259335415153`.
- `libero_object/0` seed `103`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5505254836850222`.
- `libero_object/0` seed `104`: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5512551391364635`.
- `libero_object/1` seed `100`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; reward `1.0`, progress `0.3867370929675663`.
- `libero_object/1` seed `101`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; reward `1.0`, progress `0.3853902692509572`.
- `libero_object/1` seed `102`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; reward `1.0`, progress `0.3986051465036626`.
- `libero_object/1` seed `103`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; reward `1.0`, progress `0.4038370764325351`.
- `libero_object/1` seed `104`: `pick_up_the_cream_cheese_and_place_it_in_the_basket`; reward `1.0`, progress `0.4002290936815351`.
- `libero_object/2` seed `100`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.37420296742663517`.
- `libero_object/2` seed `101`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.3664911630063894`.
- `libero_object/2` seed `102`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.36115667083993286`.
- `libero_object/2` seed `103`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.3637910859333675`.
- `libero_object/2` seed `104`: `pick_up_the_salad_dressing_and_place_it_in_the_basket`; reward `1.0`, progress `0.37449710586375423`.
- `libero_object/3` seed `100`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3831288529546186`.
- `libero_object/3` seed `101`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3723056727359616`.
- `libero_object/3` seed `102`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3772427021444006`.
- `libero_object/3` seed `103`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3922545888495457`.
- `libero_object/3` seed `104`: `pick_up_the_bbq_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3989839726409943`.
- `libero_object/4` seed `100`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5290766012807302`.
- `libero_object/4` seed `101`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5192724719961412`.
- `libero_object/4` seed `102`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5185294162456582`.
- `libero_object/4` seed `103`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5241548195733502`.
- `libero_object/4` seed `104`: `pick_up_the_ketchup_and_place_it_in_the_basket`; reward `1.0`, progress `0.5345844009877517`.
- `libero_object/5` seed `100`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3676078701015679`.
- `libero_object/5` seed `101`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3632615810702296`.
- `libero_object/5` seed `102`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.37319017636738083`.
- `libero_object/5` seed `103`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.3819371134147449`.
- `libero_object/5` seed `104`: `pick_up_the_tomato_sauce_and_place_it_in_the_basket`; reward `1.0`, progress `0.38193398899169745`.
- `libero_object/6` seed `100`: `pick_up_the_butter_and_place_it_in_the_basket`; reward `1.0`, progress `0.551171265542219`.
- `libero_object/6` seed `101`: `pick_up_the_butter_and_place_it_in_the_basket`; reward `1.0`, progress `0.5419805563833828`.
- `libero_object/6` seed `102`: `pick_up_the_butter_and_place_it_in_the_basket`; reward `1.0`, progress `0.5547903271230878`.
- `libero_object/6` seed `103`: `pick_up_the_butter_and_place_it_in_the_basket`; reward `1.0`, progress `0.566452546151617`.
- `libero_object/6` seed `104`: `pick_up_the_butter_and_place_it_in_the_basket`; reward `1.0`, progress `0.5689957936501743`.
- `libero_object/7` seed `100`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5074388791824416`.
- `libero_object/7` seed `101`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5009118776552713`.
- `libero_object/7` seed `102`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5163874935919872`.
- `libero_object/7` seed `103`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5198902394262268`.
- `libero_object/7` seed `104`: `pick_up_the_milk_and_place_it_in_the_basket`; reward `1.0`, progress `0.5186472470698037`.
- `libero_object/8` seed `100`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; reward `1.0`, progress `0.5426390476685378`.
- `libero_object/8` seed `101`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; reward `1.0`, progress `0.532783883939478`.
- `libero_object/8` seed `102`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; reward `1.0`, progress `0.5320524092821508`.
- `libero_object/8` seed `103`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; reward `1.0`, progress `0.5377215010756988`.
- `libero_object/8` seed `104`: `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`; reward `1.0`, progress `0.5482036883145378`.
- `libero_object/9` seed `100`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.38289099341876853`.
- `libero_object/9` seed `101`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.3751442452242535`.
- `libero_object/9` seed `102`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.3698353817312267`.
- `libero_object/9` seed `103`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.3725035916689718`.
- `libero_object/9` seed `104`: `pick_up_the_orange_juice_and_place_it_in_the_basket`; reward `1.0`, progress `0.3832333525482142`.

## Failed Episodes

- None.

## Limitations

- This is a scripted sparse-success smoke, not a learned WAM policy and not a demonstration that the project solves LIBERO.
- The controller uses object and target positions exposed by the simulator, so it is diagnostic benchmark evidence rather than deployable perception.
- The default smoke uses hand-coded object-conditioned grasp heights for LIBERO Object; this is benchmark engineering, not learned policy evidence.
- Failed tasks, if any, remain reported in the CSV/JSON artifact; the claim is limited to the measured success subset.
