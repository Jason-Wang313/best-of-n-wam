# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/AssembleCookingArray', 'robocasa/BeginSlowCooking', 'robocasa/BoilCorn', 'robocasa/BoilEggs', 'robocasa/BoilPot', 'robocasa/CleanBoard', 'robocasa/ClearFreezer', 'robocasa/ClearReceptaclesForCleaning', 'robocasa/ClearSinkArea', 'robocasa/ClearSinkSpace', 'robocasa/ClusterItemsForClearing', 'robocasa/CoolBakedCake', 'robocasa/CoolBakedCookies', 'robocasa/CupcakeCleanup', 'robocasa/DryDishes', 'robocasa/DryDrinkware', 'robocasa/FreezeCookedFood', 'robocasa/HeatKebabSandwich', 'robocasa/HeatMultipleWater', 'robocasa/MicrowaveDefrostMeat', 'robocasa/MicrowaveThawing', 'robocasa/MicrowaveThawingFridge', 'robocasa/OrganizeCleaningSupplies', 'robocasa/OvenBroilFish', 'robocasa/PlaceLidToBoil', 'robocasa/PreRinseStation', 'robocasa/PrepFridgeForCleaning', 'robocasa/PrepSinkForCleaning', 'robocasa/SortingCleanup', 'robocasa/SteamInMicrowave', 'robocasa/StopSlowCooking', 'robocasa/SweetSavoryToastSetup', 'robocasa/ToastOnCorrectRack', 'robocasa/ToastOneSlotPair', 'robocasa/ToasterOvenBroilFish']`
- train samples: `140`
- validation samples: `140`
- eval samples: `280`
- eval rollout pools: `35`
- exact-law utility MAE: `0.00024620294740340757`
- validation utility correlation: `0.8345242481462408`
- validation learned-physics correlation: `0.8345242481462406`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N4 CI: `{'n': 35, 'mean': 0.2408146016473912, 'std': 0.13106289940391155, 'stderr': 0.02215367341365428, 'ci95': 0.04342119989076239, 'lo': 0.19739340175662884, 'hi': 0.2842358015381536}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
