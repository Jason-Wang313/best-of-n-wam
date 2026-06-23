# External Benchmark Runtime Probe

- Verified: `True`
- LIBERO import available: `True`
- RoboCasa import available: `True`
- joint LIBERO+SmolVLA runtime available: `True`
- Checks: `4`
- Issues: `0`

This is a runtime/import probe only; it does not promote benchmark-performance claims.

## libero_success

- Candidate: `external_libero310`
- Python: `C:\Users\wangz\external_benchmarks\.venvs\libero310\Scripts\python.exe`
- Source: `C:\Users\wangz\external_benchmarks\LIBERO`
- Config: `C:\Users\wangz\external_benchmarks\.libero`

## robocasa_success

- Candidate: `external_robocasa`
- Python: `C:\Users\wangz\external_benchmarks\.venvs\robocasa\Scripts\python.exe`
- Source: `C:\Users\wangz\external_benchmarks\robocasa`
- Config: `None`

## VLA Runtime Compatibility

- `vla_external_libero310`: ok=`True`, joint_ready=`True`, available={'libero': True, 'lerobot_smolvla': True, 'torch': True, 'transformers': True, 'huggingface_hub': True}
- `vla_external_libero38`: ok=`True`, joint_ready=`False`, available={'libero': True, 'lerobot_smolvla': False, 'torch': True, 'transformers': False, 'huggingface_hub': False}
- `vla_external_robocasa`: ok=`False`, joint_ready=`None`, available={}
- `vla_vla_external_robocasa_with_libero_source`: ok=`True`, joint_ready=`True`, available={'libero': True, 'lerobot_smolvla': True, 'torch': True, 'transformers': True, 'huggingface_hub': True}
