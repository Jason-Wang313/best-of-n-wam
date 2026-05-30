# Claims Report

## Counts

- verified: `42`
- partial: `0`
- unsupported: `0`
- failed: `0`
- README overclaims: `0`

## Verified

- 1. Exact finite binary law verified. Evidence: success MAE=0.006959852139023545
- 2. Utility-valued finite law verified. Evidence: utility MAE=0.0451116620102325
- 3. N=2 AUC identity verified. Evidence: max identity error=0.0
- 4. High-N moment hierarchy verified. Evidence: same-p/kappa gap=0.9988209815422198
- 5. Pilot-to-heldout improves with K. Evidence: relative MAE reduction=0.43698115249631736
- 6. Pilot uncertainty is reported. Evidence: pilot improvement CI={'n': 16, 'mean': 0.44172358371190434, 'std': 0.6445133007518349, 'stderr': 0.16112832518795872, 'ci95': 0.3158115173683991, 'lo': 0.12591206634350527, 'hi': 0.7575351010803034}
- 7. Score function controls inference value. Evidence: oracle-random N64=6.976857029097962
- 8. Best non-oracle beats random with CI. Evidence: learned CI={'n': 5, 'mean': 5.97838149808149, 'std': 1.2959235541324863, 'stderr': 0.5795546321366736, 'ci95': 1.1359270789878801, 'lo': 4.8424544190936105, 'hi': 7.11430857706937}
- 9. Oracle remains above learned/non-oracle. Evidence: oracle-learned CI={'n': 5, 'mean': 1.9951266518211337, 'std': 0.18826614525085428, 'stderr': 0.08419517972855187, 'ci95': 0.16502255226796164, 'lo': 1.830104099553172, 'hi': 2.1601492040890955}
- 10. Real-vs-imagined utility gap verified. Evidence: severe-none=16.435472824968038
- 11. Mismatch gap grows with N. Evidence: learned severe gap CI={'n': 5, 'mean': 13.8807751738171, 'std': 0.7005630946072382, 'stderr': 0.3133013404138802, 'ci95': 0.6140706272112052, 'lo': 13.266704546605894, 'hi': 14.494845801028305}
- 12. Bad scorer falsification verified. Evidence: anti N64=-26.565706083129044, N1=-14.101561337022474
- 13. Randomized dynamics falsification verified. Evidence: randomized-oracle N64 gap=13.393867700179667
- 14. Moment/adaptive allocation beats uniform with CI. Evidence: moment-uniform CI={'n': 5, 'mean': 0.30066408929367955, 'std': 0.028847241897066837, 'stderr': 0.012900878769044288, 'ci95': 0.025285722387326803, 'lo': 0.27537836690635276, 'hi': 0.32594981168100634}
- 15. Adaptive allocation reduces oracle regret. Evidence: oracle-uniform=0.3055113080821638
- 16. Closed-loop high-N gain verified. Evidence: analytic useful N64-N1=0.5; learned CI={'n': 5, 'mean': 0.2166666666666667, 'std': 0.11180339887498948, 'stderr': 0.049999999999999996, 'ci95': 0.09799999999999999, 'lo': 0.11866666666666671, 'hi': 0.3146666666666667}
- 17. Useful scorer beats random in closed loop. Evidence: learned useful-random CI={'n': 5, 'mean': 0.3166666666666667, 'std': 0.06972166887783962, 'stderr': 0.031180478223116172, 'ci95': 0.061113737317307695, 'lo': 0.255552929349359, 'hi': 0.3777804039839744}
- 18. Oracle first-action remains upper bound. Evidence: oracle-useful CI={'n': 5, 'mean': 0.16666666666666666, 'std': 0.11785113019775793, 'stderr': 0.052704627669472995, 'ci95': 0.10330107023216707, 'lo': 0.06336559643449959, 'hi': 0.26996773689883374}
- 19. Conditional law verified under distribution shift. Evidence: MAE=0.006278872119413491
- 20. Stale estimates fail/degrade under shift. Evidence: stale post-pre CI={'n': 48, 'mean': 0.06053148468706062, 'std': 0.1237430819653073, 'stderr': 0.017860775420756023, 'ci95': 0.035007119824681805, 'lo': 0.025524364862378815, 'hi': 0.09553860451174243}
- 21. Adaptive re-estimation helps under shift. Evidence: stale-adaptive post CI={'n': 48, 'mean': 0.10238434930665018, 'std': 0.14521811815578564, 'stderr': 0.020960429902113432, 'ci95': 0.041082442608142325, 'lo': 0.06130190669850785, 'hi': 0.1434667919147925}
- 22. Learned WAM trained. Evidence: model=results\models\learned_wam_lite_toy.npz
- 23. Learned WAM ID error reported. Evidence: validation={'split': 'validation', 'mismatch': 'mild', 'n_samples': 768, 'final_delta_mae': 0.0736601208533958, 'final_position_l2_mae': 0.11172250197070284, 'utility_mae': 0.8623689603284173, 'utility_rmse': 1.1687153364750877, 'utility_corr': 0.8944539743322287}
- 24. Learned WAM OOD error reported. Evidence: ood count=3
- 25. Learned WAM reproduces key inference-value claims. Evidence: learned-analytic CI={'n': 5, 'mean': 1.169836264042242, 'std': 0.24964325385176853, 'stderr': 0.11164385714735812, 'ci95': 0.21882196000882193, 'lo': 0.95101430403342, 'hi': 1.3886582240510639}
- 26. BlockPush verified. Evidence: multi-env or canonical artifacts
- 27. DrawerPull verified. Evidence: multi-env artifact
- 28. SlipperyGrasp verified. Evidence: multi-env artifact
- 29. Nonstationary verified. Evidence: multi-env/canonical artifact
- 30. Deformable optional. Evidence: multi-env deformable artifact
- 31. Benchmark adapter available. Evidence: attempted=True, any_available=True
- 32. Benchmark rollout pools collected. Evidence: pools=25
- 33. Benchmark exact law verified. Evidence: utility MAE=0.018753143169510465
- 34. Benchmark score comparison verified. Evidence: oracle-random CI={'n': 5, 'mean': 1.7107893171571196, 'std': 0.3902710091578448, 'stderr': 0.17453450122487676, 'ci95': 0.34208762240075846, 'lo': 1.368701694756361, 'hi': 2.052876939557878}
- 35. Benchmark real-vs-imagined gap verified. Evidence: gap growth=0.14372151651340698
- 36. Benchmark closed-loop verified. Evidence: learned-random closed-loop CI={'n': 5, 'mean': 1.7643643582266495, 'std': 1.5449273707487574, 'stderr': 0.6909125242588483, 'ci95': 1.3541885475473427, 'lo': 0.41017581067930675, 'hi': 3.118552905773992}
- 37. Benchmark learned WAM trained. Evidence: model=C:\Users\wangz\best-of-n-wam\results\models\benchmark_gym_manip_horizon_wam.npz
- 38. Visual toy WAM attempted. Evidence: visual=True
- 39. Visual toy WAM verified if artifacts exist. Evidence: test MAE=0.018500254396187476
- 40. Benchmark visual optional. Evidence: verified=True
- 41. README has no unsupported claims. Evidence: README overclaims=0
- 42. paper_outline has no unsupported claims. Evidence: paper overclaims=0

## Partial

- none

## Unsupported

- none

## Failed

- none
