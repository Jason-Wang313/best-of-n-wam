# Universal Recipe Boundary

- verified: `True`
- result type: `no_free_lunch_boundary`

## Claim

No artifact-limited empirical optimizer can prove a universal WAM train/inference recipe over unrestricted future robot distributions without additional assumptions.

## Construction

Two worlds match all committed evidence but assign opposite utilities to the same recipes in an unobserved deployment context.

- `world_A`: optimal `recipe_A`, utility `{'recipe_A': 1.0, 'recipe_B': 0.0}`
- `world_B`: optimal `recipe_B`, utility `{'recipe_A': 0.0, 'recipe_B': 1.0}`

A deterministic recipe therefore fails in one compatible world. A randomized recipe has positive worst-case regret; with a 50/50 mixture the lower bound is `0.5`.

## Needed For A Positive Universal Claim

- A specified restricted task/environment class.
- A distributional assumption linking observed artifacts to future deployments.
- A learnability/realizability assumption for the WAM family and scorer family.
- A proof or large heldout benchmark suite matching that restricted claim.

This is a boundary result. It keeps the README and paper honest: the repo can claim exact inference laws and evidence-bound optimization, not a universal WAM training recipe.
