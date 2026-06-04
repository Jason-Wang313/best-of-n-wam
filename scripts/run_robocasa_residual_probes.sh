#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROBOCASA_PYTHON:-}" ]]; then
  echo "Set ROBOCASA_PYTHON to a RoboCasa-compatible Python interpreter." >&2
  exit 2
fi

RESIDUAL_ATOMIC_IDS=(
  robocasa/CloseBlenderLid
  robocasa/CloseDoor
  robocasa/CloseDropDownDoor
  robocasa/CloseElectricKettleLid
  robocasa/CloseStandMixerHead
  robocasa/CloseToasterOvenDoor
  robocasa/OpenBlenderLid
  robocasa/OpenDoor
  robocasa/OpenDropDownDoor
  robocasa/OpenElectricKettleLid
  robocasa/OpenStandMixerHead
  robocasa/OpenToasterOvenDoor
  robocasa/TurnOffSimmeredSauceHeat
  robocasa/TurnOnElectricKettle
  robocasa/TurnOnToasterOven
  robocasa/TurnSinkSpout
  robocasa/ManipulateDoor
  robocasa/ManipulateDrawer
  robocasa/ManipulateLowerDoor
  robocasa/PickPlace
  robocasa/PickPlaceBread
  robocasa/PickPlaceCan
  robocasa/PickPlaceCereal
  robocasa/PickPlaceMilk
  robocasa/PickPlaceSingle
)

RESIDUAL_CLEAN_COOK_IDS=(
  robocasa/AdjustToasterOvenTemperature
  robocasa/AssembleCookingArray
  robocasa/BeginSlowCooking
  robocasa/BoilCorn
  robocasa/BoilEggs
  robocasa/BoilPot
  robocasa/CookieDoughPrep
  robocasa/CoolBakedCake
  robocasa/CoolBakedCookies
  robocasa/FreezeCookedFood
  robocasa/HeatKebabSandwich
  robocasa/HeatMultipleWater
  robocasa/MicrowaveDefrostMeat
  robocasa/MicrowaveThawing
  robocasa/MicrowaveThawingFridge
  robocasa/OvenBroilFish
  robocasa/PlaceLidToBoil
  robocasa/PreheatOven
  robocasa/SlideOvenRack
  robocasa/SlideToasterOvenRack
  robocasa/SteamInMicrowave
  robocasa/StopSlowCooking
  robocasa/SweetSavoryToastSetup
  robocasa/ToastHeatableIngredients
  robocasa/ToastOnCorrectRack
  robocasa/ToastOneSlotPair
  robocasa/ToasterOvenBroilFish
  robocasa/CandleCleanup
  robocasa/CleanBoard
  robocasa/ClearFreezer
  robocasa/ClearReceptaclesForCleaning
  robocasa/ClearSinkArea
  robocasa/ClearSinkSpace
  robocasa/ClusterItemsForClearing
  robocasa/CupcakeCleanup
  robocasa/DryDishes
  robocasa/DryDrinkware
  robocasa/OrganizeCleaningSupplies
  robocasa/PreRinseStation
  robocasa/PrepFridgeForCleaning
  robocasa/PrepSinkForCleaning
  robocasa/ReturnWashingSupplies
  robocasa/SortingCleanup
)

"$ROBOCASA_PYTHON" experiments/benchmark_robocasa_micro_rollout_probe.py \
  --output-tag residual_atomic_probe \
  --env-ids "${RESIDUAL_ATOMIC_IDS[@]}" \
  --rollouts 2 \
  --horizon 1 \
  --min-tasks 1

"$ROBOCASA_PYTHON" experiments/benchmark_robocasa_residual_frontier_sweep.py \
  --env-ids "${RESIDUAL_CLEAN_COOK_IDS[@]}" \
  --chunk-size 1 \
  --timeout-seconds "${ROBOCASA_RESIDUAL_TIMEOUT_SECONDS:-420}" \
  --rollouts 2 \
  --horizon 1 \
  --output-tag-prefix residual_clean_cook_sweep \
  --skip-existing

mapfile -t RESIDUAL35_IDS < <("$ROBOCASA_PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("results") / "benchmark_robocasa_residual_frontier_sweep.json"
payload = json.loads(path.read_text(encoding="utf-8"))
for env_id in payload.get("nondegenerate_env_ids") or []:
    print(env_id)
PY
)

"$ROBOCASA_PYTHON" experiments/benchmark_robocasa_multitask_wam.py \
  --output-tag residual35_h1_n4 \
  --train-states 1 \
  --train-rollouts 4 \
  --val-states 1 \
  --val-rollouts 4 \
  --eval-states 1 \
  --eval-rollouts 8 \
  --horizon 1 \
  --mc-trials 500 \
  --n-values 1 2 4 \
  --min-tasks 35 \
  --min-eval-pools 35 \
  --max-exact-mae 0.04 \
  --env-ids "${RESIDUAL35_IDS[@]}"

"$ROBOCASA_PYTHON" experiments/benchmark_robocasa_catalog_probe.py
