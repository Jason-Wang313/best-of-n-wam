# Modern VLA Availability Probe

- verified audit: `True`
- VLA package importable: `False`
- local VLA-like matches: `0`
- Hugging Face models reachable: `5`
- joint LIBERO+VLA runtime available: `True`
- pretrained VLA loaded: `True`
- pretrained VLA parameters: `450046212`
- LIBERO execution probe present: `True`
- LIBERO execution verified: `False`
- LIBERO execution failure stage: `policy_load`
- ready for policy eval: `False`

## Missing For Ideal Claim

- pretrained VLA can select an action and step LIBERO in a compatible runtime
- LIBERO-compatible sparse-success VLA evaluation artifact

## Importable Packages

- `torch`: importable=`True`
- `transformers`: importable=`True`
- `huggingface_hub`: importable=`True`
- `lerobot`: importable=`False`
- `openvla`: importable=`False`
- `octo`: importable=`False`
- `openpi`: importable=`False`
- `gr00t`: importable=`False`

## Hugging Face Metadata Probe

- `openvla/openvla-7b`: reachable=`True`, private=`False`, files=`18`
- `openvla/openvla-7b-finetuned-libero-spatial`: reachable=`True`, private=`False`, files=`16`
- `openvla/openvla-7b-finetuned-libero-object`: reachable=`True`, private=`False`, files=`16`
- `physical-intelligence/fast`: reachable=`True`, private=`False`, files=`7`
- `nvidia/GR00T-N1.5-3B`: reachable=`True`, private=`False`, files=`13`

This is an availability/blocker artifact only. It does not download checkpoints, expose secrets, or validate a modern VLA policy.
