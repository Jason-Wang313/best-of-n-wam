# Claims Report

## Counts

- verified: `83`
- partial: `0`
- unsupported: `0`
- failed: `0`
- README overclaims: `0`

## Verified

- 1. Exact finite binary law verified. Evidence: success MAE=0.002104612037108505
- 2. Utility-valued finite law verified. Evidence: utility MAE=0.013762680427723635
- 3. N=2 AUC identity verified. Evidence: max identity error=0.0
- 4. High-N moment hierarchy verified. Evidence: same-p/kappa gap=0.9988209815422198
- 5. Pilot-to-heldout improves with K. Evidence: relative MAE reduction=0.36981841294484663
- 6. Pilot uncertainty is reported. Evidence: pilot improvement CI={'n': 3, 'mean': 0.4562385545786219, 'std': 0.6526343629570743, 'stderr': 0.3767986251356668, 'ci95': 0.738525305265907, 'lo': -0.2822867506872851, 'hi': 1.194763859844529}
- 7. Score function controls inference value. Evidence: oracle-random N64=6.6877116595064265
- 8. Best non-oracle beats random with CI. Evidence: learned CI={'n': 5, 'mean': 5.97838149808149, 'std': 1.2959235541324863, 'stderr': 0.5795546321366736, 'ci95': 1.1359270789878801, 'lo': 4.8424544190936105, 'hi': 7.11430857706937}
- 9. Oracle remains above learned/non-oracle. Evidence: oracle-learned CI={'n': 5, 'mean': 1.9951266518211337, 'std': 0.18826614525085428, 'stderr': 0.08419517972855187, 'ci95': 0.16502255226796164, 'lo': 1.830104099553172, 'hi': 2.1601492040890955}
- 10. Real-vs-imagined utility gap verified. Evidence: severe-none=15.179989463689903
- 11. Mismatch gap grows with N. Evidence: learned severe gap CI={'n': 5, 'mean': 13.880775173817053, 'std': 0.7005630946072381, 'stderr': 0.31330134041388014, 'ci95': 0.6140706272112051, 'lo': 13.266704546605848, 'hi': 14.494845801028259}
- 12. Bad scorer falsification verified. Evidence: anti N64=-26.565706083129044, N1=-14.101561337022474
- 13. Randomized dynamics falsification verified. Evidence: randomized-oracle N64 gap=13.393867700179667
- 14. Moment/adaptive allocation beats uniform with CI. Evidence: moment-uniform CI={'n': 5, 'mean': 0.30066408929367955, 'std': 0.028847241897066837, 'stderr': 0.012900878769044288, 'ci95': 0.025285722387326803, 'lo': 0.27537836690635276, 'hi': 0.32594981168100634}
- 15. Adaptive allocation reduces oracle regret. Evidence: oracle-uniform=0.3055113080821638
- 16. Closed-loop high-N gain verified. Evidence: analytic useful N64-N1=0.0; learned CI={'n': 5, 'mean': 0.2166666666666667, 'std': 0.11180339887498948, 'stderr': 0.049999999999999996, 'ci95': 0.09799999999999999, 'lo': 0.11866666666666671, 'hi': 0.3146666666666667}
- 17. Useful scorer beats random in closed loop. Evidence: learned useful-random CI={'n': 5, 'mean': 0.3166666666666667, 'std': 0.06972166887783962, 'stderr': 0.031180478223116172, 'ci95': 0.061113737317307695, 'lo': 0.255552929349359, 'hi': 0.3777804039839744}
- 18. Oracle first-action remains upper bound. Evidence: oracle-useful CI={'n': 5, 'mean': 0.16666666666666666, 'std': 0.11785113019775793, 'stderr': 0.052704627669472995, 'ci95': 0.10330107023216707, 'lo': 0.06336559643449959, 'hi': 0.26996773689883374}
- 19. Conditional law verified under distribution shift. Evidence: MAE=0.006278872119413491
- 20. Stale estimates fail/degrade under shift. Evidence: stale post-pre CI={'n': 48, 'mean': 0.06053148468706062, 'std': 0.1237430819653073, 'stderr': 0.017860775420756023, 'ci95': 0.035007119824681805, 'lo': 0.025524364862378815, 'hi': 0.09553860451174243}
- 21. Adaptive re-estimation helps under shift. Evidence: stale-adaptive post CI={'n': 48, 'mean': 0.10238434930665018, 'std': 0.14521811815578564, 'stderr': 0.020960429902113432, 'ci95': 0.041082442608142325, 'lo': 0.06130190669850785, 'hi': 0.1434667919147925}
- 22. Learned WAM trained. Evidence: model=results\models\learned_wam_lite_toy.npz
- 23. Learned WAM ID error reported. Evidence: validation={'split': 'validation', 'mismatch': 'mild', 'n_samples': 768, 'final_delta_mae': 0.07366012085339552, 'final_position_l2_mae': 0.11172250197070248, 'utility_mae': 0.8623689603284147, 'utility_rmse': 1.1687153364750862, 'utility_corr': 0.894453974332229}
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
- 35. Benchmark real-vs-imagined gap verified. Evidence: gap growth=0.14372151651340617
- 36. Benchmark closed-loop verified. Evidence: learned-random closed-loop CI={'n': 5, 'mean': 1.7643643582266495, 'std': 1.5449273707487574, 'stderr': 0.6909125242588483, 'ci95': 1.3541885475473427, 'lo': 0.41017581067930675, 'hi': 3.118552905773992}
- 37. Benchmark learned WAM trained. Evidence: model=C:\Users\wangz\best-of-n-wam\results\models\benchmark_gym_manip_horizon_wam.npz
- 38. Visual toy WAM attempted. Evidence: visual=True
- 39. Visual toy WAM verified if artifacts exist. Evidence: test MAE=0.018500254396187476
- 40. Benchmark visual optional. Evidence: verified=True
- 41. Inference-value audit profiles generated. Evidence: profiles=15, decisions=13
- 42. Tail alignment predicts high-N inference value. Evidence: tail-gain corr CI={'n': 5, 'mean': 0.9865000978504973, 'std': 0.001804695185387798, 'stderr': 0.0008070842226387402, 'ci95': 0.0015818850763719308, 'lo': 0.9849182127741254, 'hi': 0.9880819829268692}
- 43. Audit gate blocks harmful high-N bad-scorer deployments. Evidence: anti block CI={'n': 5, 'mean': 1.0, 'std': 0.0, 'stderr': 0.0, 'ci95': 0.0, 'lo': 1.0, 'hi': 1.0}; anti harm CI={'n': 5, 'mean': 0.5458182454465957, 'std': 0.007548888768848233, 'stderr': 0.003375965688345869, 'ci95': 0.006616892749157903, 'lo': 0.5392013526974379, 'hi': 0.5524351381957536}
- 44. Stop-rule compute savings are reported. Evidence: saved rollout fraction CI={'n': 5, 'mean': 0.7341830801560019, 'std': 0.014683879008794358, 'stderr': 0.006566830327409283, 'ci95': 0.012870987441722194, 'lo': 0.7213120927142797, 'hi': 0.7470540675977241}
- 45. Learned-backend inference audit reproduced. Evidence: learned tail-gain CI={'n': 5, 'mean': 0.9823361910668508, 'std': 0.003404324580102718, 'stderr': 0.001522460235716621, 'ci95': 0.002984022062004577, 'lo': 0.9793521690048462, 'hi': 0.9853202131288553}; anti block CI={'n': 5, 'mean': 1.0, 'std': 0.0, 'stderr': 0.0, 'ci95': 0.0, 'lo': 1.0, 'hi': 1.0}
- 46. Pilot-calibrated scorer repair improves heldout high-N utility. Evidence: repair-predicted CI={'n': 5, 'mean': 0.34893208610613924, 'std': 0.007783860739302779, 'stderr': 0.0034810483480945563, 'ci95': 0.0068228547622653304, 'lo': 0.3421092313438739, 'hi': 0.3557549408684046}
- 47. Robot-imagination compute frontier is measured. Evidence: pred gain CI={'n': 5, 'mean': 0.025522691537548704, 'std': 0.013112519318734733, 'stderr': 0.005864096910594019, 'ci95': 0.011493629944764275, 'lo': 0.014029061592784428, 'hi': 0.03701632148231298}; oracle-pred gain CI={'n': 5, 'mean': 0.4091337942554179, 'std': 0.022021280046922203, 'stderr': 0.00984821582729556, 'ci95': 0.019302503021499298, 'lo': 0.38983129123391863, 'hi': 0.4284362972769172}
- 48. ManiSkill state benchmark suite verified. Evidence: envs=['PickCube-v1', 'PushCube-v1', 'PegInsertionSide-v1'], control=pd_joint_delta_pos
- 49. ManiSkill rollout pools collected. Evidence: pools=30, rollouts=32
- 50. ManiSkill exact law verified. Evidence: utility MAE=0.0034490668019852445
- 51. ManiSkill score comparison verified. Evidence: dense-random CI={'n': 5, 'mean': 0.24732126597363005, 'std': 0.023695650387774738, 'stderr': 0.010597017007626712, 'ci95': 0.020770153334948357, 'lo': 0.2265511126386817, 'hi': 0.2680914193085784}; oracle-random CI={'n': 5, 'mean': 0.24785881464044768, 'std': 0.024027854839175085, 'stderr': 0.010745583354778553, 'ci95': 0.021061343375365964, 'lo': 0.2267974712650817, 'hi': 0.26892015801581365}
- 52. ManiSkill WAM-lite trained and evaluated. Evidence: model metric rows=6
- 53. ManiSkill closed-loop learned scorer beats random. Evidence: learned-random closed-loop CI={'n': 5, 'mean': 0.027762312500152514, 'std': 0.020055963136210424, 'stderr': 0.008969299385359276, 'ci95': 0.01757982679530418, 'lo': 0.010182485704848334, 'hi': 0.04534213929545669}
- 54. ManiSkill learned open-loop scorer is honestly reported. Evidence: learned-random open-loop CI={'n': 5, 'mean': -0.005754620067846242, 'std': 0.04861601605708501, 'stderr': 0.021741743339772673, 'ci95': 0.04261381694595444, 'lo': -0.04836843701380068, 'hi': 0.036859196878108194}
- 55. Benchmark RGB visual WAM-lite trained and evaluated. Evidence: model=extra_trees_visual_wam, validation={'utility_mae': 0.5207678201477979, 'utility_corr': 0.2199328908664021, 'success_mae': 0.18271949404761903}
- 56. Benchmark RGB visual WAM exact law verified. Evidence: utility MAE=0.01571480996432407
- 57. Benchmark RGB visual WAM scorer beats random with CI. Evidence: visual-random CI={'n': 5, 'mean': 0.2367775727718704, 'std': 0.04220964852617569, 'stderr': 0.01887672868218053, 'ci95': 0.03699838821707384, 'lo': 0.19977918455479654, 'hi': 0.27377596098894424}
- 58. Benchmark RGB visual WAM oracle gap reported. Evidence: oracle-visual CI={'n': 5, 'mean': 0.3433881226502652, 'std': 0.09104423341118655, 'stderr': 0.04071621897335413, 'ci95': 0.0798037891877741, 'lo': 0.2635843334624911, 'hi': 0.4231919118380393}
- 59. Gymnasium Robotics Fetch benchmark suite verified. Evidence: envs=['FetchReach-v4', 'FetchPush-v4', 'FetchPickAndPlace-v4'], pools=60
- 60. Gymnasium Robotics Fetch exact law verified. Evidence: utility MAE=0.012637772476737578
- 61. Gymnasium Robotics learned WAM scorer beats random with CI. Evidence: learned-random CI={'n': 5, 'mean': 0.5183591980928537, 'std': 0.08832562568455588, 'stderr': 0.03950042063717366, 'ci95': 0.07742082444886038, 'lo': 0.4409383736439933, 'hi': 0.595780022541714}
- 62. Gymnasium Robotics closed-loop learned scorer beats random. Evidence: closed-loop learned-random CI={'n': 5, 'mean': 0.7086378606835364, 'std': 0.5149831233310606, 'stderr': 0.23030745420668186, 'ci95': 0.4514026102450964, 'lo': 0.25723525043844, 'hi': 1.160040470928633}
- 63. Gymnasium Robotics oracle gap reported. Evidence: oracle-learned CI={'n': 5, 'mean': 0.03423794265616582, 'std': 0.021599643753640783, 'stderr': 0.009659654344583902, 'ci95': 0.018932922515384448, 'lo': 0.015305020140781375, 'hi': 0.05317086517155027}
- 64. Gymnasium Robotics RGB visual WAM trained and evaluated. Evidence: envs=['FetchReach-v4', 'FetchPush-v4', 'FetchPickAndPlace-v4'], mean corr=0.7325062733110719
- 65. Gymnasium Robotics RGB visual exact law verified. Evidence: utility MAE=0.010612997381058541
- 66. Gymnasium Robotics RGB visual scorer beats random with CI. Evidence: visual-random CI={'n': 5, 'mean': 0.4888505704773543, 'std': 0.16122603373528005, 'stderr': 0.0721024742349521, 'ci95': 0.1413208495005061, 'lo': 0.3475297209768482, 'hi': 0.6301714199778604}
- 67. Gymnasium Robotics RGB visual oracle gap is reported without requiring significance. Evidence: oracle-visual CI={'n': 5, 'mean': 0.02680333126025487, 'std': 0.03203449057950047, 'stderr': 0.014326259712067935, 'ci95': 0.028079469035653153, 'lo': -0.0012761377753982839, 'hi': 0.05488280029590802}
- 68. ManiSkill RGB/RGB-D and EE-control probe is artifact-documented. Evidence: visual_success=False, blocker=vk::Device::allocateDescriptorSetsUnique: ErrorOutOfPoolMemory; pinocchio=False, pin_binary=False
- 69. Meta-World ML1 benchmark suite verified. Evidence: tasks=['reach-v3', 'push-v3', 'drawer-open-v3'], pools=45
- 70. Meta-World exact law verified. Evidence: utility MAE=0.029769105761603815
- 71. Meta-World learned WAM scorer beats random open-loop with CI. Evidence: learned-random CI={'n': 5, 'mean': 0.11054551953348397, 'std': 0.014837925371412848, 'stderr': 0.0066357219551095884, 'ci95': 0.013006015032014793, 'lo': 0.09753950450146918, 'hi': 0.12355153456549876}
- 72. Meta-World oracle and benchmark-reward scorers beat random with CI. Evidence: reward-random CI={'n': 5, 'mean': 0.5208496009041265, 'std': 0.06185776418102653, 'stderr': 0.027663633128985385, 'ci95': 0.054220720932811356, 'lo': 0.4666288799713151, 'hi': 0.5750703218369378}; oracle-random CI={'n': 5, 'mean': 0.5576417182681597, 'std': 0.04380737850483259, 'stderr': 0.019591255250573755, 'ci95': 0.038398860291124555, 'lo': 0.5192428579770351, 'hi': 0.5960405785592843}
- 73. RoboSuite Panda manipulation benchmark suite verified. Evidence: envs=['Lift', 'Stack', 'Door'], pools=30
- 74. RoboSuite exact law verified. Evidence: utility MAE=0.0023698818100493587
- 75. RoboSuite learned WAM scorer beats random open-loop with CI. Evidence: learned-random CI={'n': 5, 'mean': 0.32020186708958887, 'std': 0.08614749302461966, 'stderr': 0.038526330098847705, 'ci95': 0.0755116069937415, 'lo': 0.24469026009584738, 'hi': 0.39571347408333035}
- 76. RoboSuite reward, progress, and oracle scorers beat random with CI. Evidence: reward-random CI={'n': 5, 'mean': 0.33580473945491357, 'std': 0.0845123155588677, 'stderr': 0.03779505650510826, 'ci95': 0.07407831075001219, 'lo': 0.26172642870490137, 'hi': 0.40988305020492577}; progress-random CI={'n': 5, 'mean': 0.1829135769086768, 'std': 0.11887225635263822, 'stderr': 0.053161289168656055, 'ci95': 0.10419612677056586, 'lo': 0.07871745013811093, 'hi': 0.28710970367924266}; oracle-random CI={'n': 5, 'mean': 0.3554064768118671, 'std': 0.09637875214875093, 'stderr': 0.0431018882782422, 'ci95': 0.08447970102535471, 'lo': 0.27092677578651236, 'hi': 0.43988617783722184}
- 77. RoboSuite closed-loop learned and reward scorers beat random. Evidence: learned-random CI={'n': 5, 'mean': 0.08618701545420379, 'std': 0.00732792326762759, 'stderr': 0.0032771469120635353, 'ci95': 0.006423207947644529, 'lo': 0.07976380750655926, 'hi': 0.09261022340184832}; reward-random CI={'n': 5, 'mean': 0.0742545661373815, 'std': 0.02836672038992971, 'stderr': 0.012685983018122434, 'ci95': 0.02486452671551997, 'lo': 0.04939003942186153, 'hi': 0.09911909285290148}
- 78. RoboCasa kitchen benchmark smoke verified. Evidence: env=robocasa/PickPlaceCounterToCabinet, pools=5, rollouts=80, exact MAE=0.0002724664778796843, oracle-random CI={'n': 5, 'mean': 0.2737591862068588, 'std': 0.13943158358342508, 'stderr': 0.06235569982059644, 'ci95': 0.12221717164836901, 'lo': 0.15154201455848978, 'hi': 0.3959763578552278}
- 79. RoboCasa learned WAM-lite scorer beats random with CI. Evidence: train=80, val=32, eval=80, utility corr=0.7639608394479505, learned-random CI={'n': 5, 'mean': 0.14832993183038654, 'std': 0.11178217873717447, 'stderr': 0.04999051006587074, 'ci95': 0.09798139972910665, 'lo': 0.05034853210127989, 'hi': 0.2463113315594932}
- 80. RoboCasa three-task learned WAM-lite scorer beats random with CI. Evidence: tasks=['robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave'], train=144, val=96, eval=240, utility corr=0.6751791364461345, promoted=learned_energy_regularized, learned-random CI={'n': 15, 'mean': 0.23492695125782703, 'std': 0.13017328673923353, 'stderr': 0.03361059811080895, 'ci95': 0.06587677229718554, 'lo': 0.1690501789606415, 'hi': 0.30080372355501256}, oracle-learned CI={'n': 15, 'mean': 0.05812674657491653, 'std': 0.04447418542048683, 'stderr': 0.01148318529797908, 'ci95': 0.022507043184038997, 'lo': 0.035619703390877534, 'hi': 0.08063378975895552}
- 81. README has no unsupported claims. Evidence: README overclaims=0
- 82. paper_outline has no unsupported claims. Evidence: paper overclaims=0
- 83. LIBERO rollout-pool learned WAM-lite benchmark verified. Evidence: tasks=['libero_spatial/0', 'libero_spatial/1', 'libero_spatial/2'], train=192, val=96, eval=240, exact MAE=0.0001402348651464976, utility corr=0.3526483541014925, learned-random CI={'n': 15, 'mean': 0.3375863022441559, 'std': 0.14290935532341625, 'stderr': 0.03689903687898863, 'ci95': 0.07232211228281771, 'lo': 0.2652641899613382, 'hi': 0.40990841452697363}

## Partial

- none

## Unsupported

- none

## Failed

- none
