# best-of-n-wam v2 Review Audit

## Scope

This audit records the v2 self-review pass for `iclr_submission.tex`. The main
reviewer threat is duplicate-wrapper risk: the old public face read as another
"Best-of-N + exact law" paper. The v2 paper must instead read as the
world-action planning paper: it audits real utility in the high planner-score
tail of imagined rollouts and uses the finite identity as a tool, not as a
generic theorem wrapper.

Source of truth:

- Local folder: `C:\Users\wangz\best-of-n-wam`
- GitHub repo: `Jason-Wang313/best-of-n-wam`
- Desktop source PDF: `C:\Users\wangz\OneDrive\Desktop\best-of-n-wam.pdf`
- Desktop v2 artifact after compile: `C:\Users\wangz\OneDrive\Desktop\best-of-n-wam-v2.pdf`

## Fixes Applied Before Final Review

- Replaced the title with `When Imagination Hurts: Score-Tail Audits for
  World-Action Rollout Planning`.
- Rewrote the abstract around score-tail real-utility auditing, imagined-real
  mismatch, bad-scorer falsification, WAM-lite checks, and simulated rollout
  pools.
- Reframed the introduction and contributions so the finite identity supports
  the WAM audit rather than serving as the generic headline.
- Renamed the main theorem from a finite Best-of-N law to a finite score-tail
  rollout identity.
- Updated README, pyproject description, claim map, and handoff facts to
  match the new v2 identity.

## 50-Round Attack Pass

1. Title looks like the LLM paper with world-action words swapped in. Fixed:
   title no longer says Best-of-N, exact law, or inference law.
2. Abstract leads with theorem novelty. Fixed: abstract leads with high
   planner-score tail risk.
3. The paper could be dismissed as order statistics. Mitigated: novelty is the
   WAM score-tail audit and simulator-backed mismatch evidence.
4. It may still look like an LLM reranking paper. Mitigated: first page names
   imagined rollouts, real utility, WAM-lite, and simulated rollout pools.
5. It may overclaim robotics. Guarded: no real-world robot validation, no VLA
   scale, no full RoboCasa-wide claim.
6. It may overclaim training laws. Guarded: fixed generator/scorer only.
7. It may imply more candidates always help. Guarded: title and abstract
   foreground harm and saturation.
8. It may hide model/scorer mismatch. Fixed: mismatch is central.
9. The theorem name repeats sibling papers. Fixed: finite score-tail rollout
   identity.
10. The setup still needs N-candidate language. Fixed: uses N-candidate
    planner before theorem.
11. AUC claims may sound generic. Mitigated: tied to binary success boundary.
12. AUC may be framed as useless. Guarded: complete for two candidates only.
13. Adaptive allocation may sound optimal. Guarded: diagnostic heuristic.
14. Receding-horizon claims may sound global. Guarded: conditional per visited
    state only.
15. Simulator evidence may be mistaken for hardware. Guarded repeatedly.
16. LIBERO smokes may sound like modern VLA performance. Guarded.
17. RoboCasa breadth may sound complete. Guarded.
18. ManiSkill missing visual/EE paths may be hidden. README still discloses
    blocker-backed limitations.
19. Learned WAM-lite may sound like a training recipe. Guarded.
20. Falsification may look like theorem failure. Fixed: failure is alignment,
    not the identity.
21. Figures may carry old theorem language. Captions now say audit/identity.
22. Related work may overclaim new probability theory. Guarded: contribution
    is deployment audit.
23. The claim map may preserve old headline. Fixed.
24. README may preserve old paper title. Fixed.
25. pyproject may preserve old package description. Fixed.
26. Handoff may send future agents to stale repo facts. Fixed.
27. Pre-existing untracked files may contaminate commit scope. Guarded in
    handoff and staging plan.
28. Lean formalization may be overclaimed. Current main paper does not promote
    Lean verification.
29. Exact-law metric names may remain in artifact text. Acceptable where they
    name historical artifact metrics, not the headline.
30. Reviewers may compare to `best-of-n-llm-v2`. Result: WAM v2 has a distinct
    failure-mode and audit identity.
31. Reviewers may compare to diffusion policy. Result: WAM v2 is about
    planner-score tail real utility, not denoising/action generation.
32. Reviewers may compare to MCTS/CEM papers. Result: WAM v2 audits fixed
    rollout/scorer stacks rather than tree/search algorithm novelty.
33. Reviewers may compare to Dreamer/RSSM papers. Result: WAM v2 does not
    claim latent dynamics training or representation learning.
34. The title may sound too negative. Acceptable: harm is the core WAM-specific
    risk and is supported by bad-scorer artifacts.
35. The abstract may be too broad. Guarded with exact exclusions.
36. The paper may need hardware to be accepted. Residual risk disclosed; not
    hidden.
37. Evidence may be too simulator-heavy. Residual risk disclosed.
38. The finite identity may be mathematically simple. Mitigated by audit
    framing, claim gates, and mismatch experiments.
39. The paper may be viewed as a diagnostic tool, not a method. Acceptable:
    adaptive budgeting is a scoped application.
40. The paper may need stronger empirical scale. Existing artifacts are broad
    but still scoped; v2 does not inflate them.
41. The first-page story must differ from LLM. Fixed.
42. The contribution list must differ from theorem wrapper. Fixed.
43. The conclusion must not return to generic Best-of-N. Fixed.
44. The README must help future agents identify the paper. Fixed.
45. The Desktop v2 artifact must be versioned. Build target will be
    `best-of-n-wam-v2.pdf`.
46. GitHub push must not include unrelated untracked files. Guarded.
47. Claims must remain artifact-backed. Existing claim map and scripts remain
    source of truth.
48. Paper compile must be verified. Pending final build.
49. Desktop source map must include v2 artifact. Pending after build.
50. Final reviewer stance: after v2, remaining risk is empirical scope, not
    duplicate-wrapper identity.
