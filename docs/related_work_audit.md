# Related Work Audit

Date: 2026-06-06

Local source of truth read for this audit: `iclr_submission.tex`,
`iclr_submission.bib`, `docs/theory.md`, `claim_evidence_map.md`,
`paper_story.md`, `iclr_reviewer_risks.md`, and `theorem_map.md`.

This audit is intentionally reviewer-facing. It separates the mathematical
lineage from the paper's defensible WAM-specific contribution and flags where
the manuscript should soften novelty, adaptivity, training, and robotics claims.

## Executive Verdict

The safest novelty claim is not "new order statistics" or "new universal
test-time scaling law." The exact best-of-N law is mathematically close to
standard order-statistic and reranking identities. The finite theorem adds
paper-useful details: empirical score ties, uniform tie-breaking, and a readout
of real rollout utility over score tie groups.

The ICLR contribution should instead be framed as:

- applying the score-order-statistic view to fixed WAM rollout generators and
  fixed planner/scorer stacks;
- making the selected quantity real utility, not the score used for selection;
- showing that best-of-N rollout planning is governed by real utility in the
  upper planner-score tail;
- proving the exact N=2 tie-aware AUC boundary while showing why AUC is
  insufficient for N>2;
- turning the curve N -> V_N into inference-value profiles and mismatch audits
  for rollout allocation, saturation, and high-N harm.

Recommended core phrasing:

> We give an exact inference law for a common rollout-selection primitive under
> fixed generator/scorer assumptions, and use it to audit score-tail alignment
> between imagined/planner scores and real rollout utility.

Avoid:

- "a new order-statistic law";
- "a universal WAM scaling law";
- "optimal controller";
- "solves OOD/model mismatch";
- "real-robot evidence";
- "AUC controls best-of-N";
- "more imagination always helps."

## Reviewer Risk: "Is This Just Order Statistics?"

Yes, partly. Concede this directly.

The theorem conditions on the tie group containing the maximum sampled score.
That is standard order-statistic reasoning. A reviewer familiar with maxima,
rank distributions, or reranking can derive the probability mass term quickly.
Trying to hide this will make the paper look less trustworthy.

The stronger answer is:

- the paper is not claiming a fundamentally new probability identity for
  maxima;
- the useful object for WAM planning is the joint distribution of planner score
  and real utility;
- the order statistic is taken over score, but the reported value is real
  utility;
- this distinction exposes imagined-real mismatch, high-score-tail failures,
  and the N=2 versus high-N AUC boundary.

Suggested main-text sentence near the theorem:

> The proof is an order-statistic/rank argument; the WAM-specific point is that
> selection is by planner score while deployment value is real utility, so the
> controlling object is real utility conditioned on the upper score tail.

## What Is Novel Versus Not Novel

Novel or defensible in this manuscript:

- conditional inference-value curve N -> V_N for a fixed state, generator,
  scorer, and real utility distribution;
- finite, tie-aware empirical law aligned with rollout-pool experiments;
- real utility selected by imagined/planner score rather than by real utility;
- exact tie-aware AUC identity only at N=2;
- high-N dependence on upper-tail rank moments beyond AUC;
- inference-value profiles for rollout-budget allocation;
- audit language for saturation, harm, and imagined-real mismatch;
- explicit scope boundary separating fixed-inference analysis from WAM training.

Not novel and should be cited rather than sold as original:

- sampling multiple candidates and reranking them;
- verifier-based best-of-N;
- CEM, random shooting, MPC, and sampling-based planning;
- world models and latent imagination;
- learned-model planning and model-bias/model-exploitation concerns;
- generic test-time compute scaling and adaptive allocation;
- order statistics, maxima, rank distributions, and AUC/Wilcoxon
  interpretations.

## Must-Cite Prior Art

| Area | Citation | Why it matters | Local bib status | Verified source |
|---|---|---|---|---|
| Order statistics | David and Nagaraja, Order Statistics, 3rd ed., 2003 | Classic order-statistics lineage. Cite to defuse "just order statistics" and distinguish the WAM readout. | Already present as `david2003orderstatistics`. | https://www.wiley-vch.de/en/areas-interest/mathematics-statistics/order-statistics-978-0-471-38926-2 |
| AUC/Wilcoxon | Hanley and McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," Radiology, 1982 | Establishes the probability/ranking interpretation of AUC and its relation to Wilcoxon. Essential for the N=2 AUC identity framing. | Missing. Candidate key: `hanley1982roc`. | DOI page: https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747 ; PubMed metadata: https://pubmed.ncbi.nlm.nih.gov/7063747/ |
| Best-of-N alignment | Beirami et al., "Theoretical guarantees on the best-of-n alignment policy," 2024/ICML 2025 | Direct BoN inference-time alignment neighbor. Shows BoN is already studied as a policy induced by sampling and reward ranking. | Missing. Candidate key: `beirami2024bestofn`. | https://arxiv.org/abs/2401.01879 |
| Verifier reranking | Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021 | Strong precedent for generating many candidates and selecting the top verifier-ranked answer. | Missing. Candidate key: `cobbe2021verifiers`. | https://arxiv.org/abs/2110.14168 and https://openai.com/research/solving-math-word-problems |
| Test-time compute allocation | Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters," 2024 | Prior work on adaptive test-time compute and difficulty-dependent allocation. Use to position adaptive rollout budgeting as analogous but WAM/utility-tail-specific. | Missing. Candidate key: `snell2024testtime`. | https://arxiv.org/abs/2408.03314 |
| Model bias/mismatch | Janner et al., "When to Trust Your Model: Model-Based Policy Optimization," NeurIPS 2019 | Anchors model-bias risk: generated model rollouts can be useful but biased, and model usage must be limited/audited. | Missing. Candidate key: `janner2019mbpo`. | https://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization and https://arxiv.org/abs/1906.08253 |
| World models | Ha and Schmidhuber; PlaNet; Dreamer | Existing anchors for learned latent dynamics and imagination. | Already present as `ha2018worldmodels`, `hafner2019planet`, `hafner2020dreamer`. | Existing `.bib` URLs. |
| Sampling MPC/CEM | PETS, CEM, IT-MPC | Existing anchors for sampled action-sequence planning and model-based control. | Already present as `chua2018pets`, `rubinstein1999cem`, `williams2017itmmpc`. | Existing `.bib` URLs. |

## Nice-To-Cite Prior Art

| Area | Citation | Why it may help | Local bib status | Verified source |
|---|---|---|---|---|
| Variational BoN | Amini et al., "Variational Best-of-N Alignment," ICLR 2025 | Shows contemporary work deriving/approximating the BoN-induced distribution. Helpful if related work discusses BoN policy distributions. | Missing. Candidate key: `amini2024vbon`. | https://arxiv.org/abs/2407.06057 |
| Efficient BoN decoding | Sun et al., "Fast Best-of-N Decoding via Speculative Rejection," NeurIPS 2024 | Shows BoN cost is active enough that acceleration papers exist. Useful for inference-cost motivation. | Missing. Candidate key: `sun2024fastbon`. | https://openreview.net/forum?id=348hfcprUs&noteId=lvi6jxweeC |
| Self-consistency | Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," ICLR 2023 | Another sample-many/test-time-selection pattern, though answer aggregation rather than score-max reranking. Use only if discussing broad inference-time sampling. | Missing. Candidate key: `wang2022selfconsistency`. | https://arxiv.org/abs/2203.11171 |
| Process reward models | Lightman et al., "Let's Verify Step by Step," 2023 | Relevant for verifier/scorer literature and process reward models. | Missing. Candidate key: `lightman2023verify`. | https://arxiv.org/abs/2305.20050 |
| Learned model planning | Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model," Nature 2020 | Strong learned-model planning anchor if the paper wants a broader planning-with-learned-models paragraph. | Missing. Candidate key: `schrittwieser2020muzero`. | https://www.nature.com/articles/s41586-020-03051-4 |
| WAM term collision | Han and Yilmaz, "Enhancing Policy Learning with World-Action Model," 2026 | Contemporary paper using "World-Action Model" for an action-regularized world model. Cite if this manuscript leans hard on the term WAM. | Missing. Candidate key: `han2026wam`. | https://arxiv.org/abs/2603.28955 |

## Related Work Placement

Main text should carry the claims that protect the theorem:

- Setup: fixed generator/scorer/state distribution; real utility is separate from
  planner score.
- Theorem section: finite law is exact under with-replacement sampling and
  uniform tie-breaking; proof uses order-statistic reasoning.
- AUC section: AUC identity is exact for N=2 only; high N depends on upper-tail
  moments.
- Adaptive section: inference-budgeting rule is diagnostic and evidence-backed,
  not a globally optimal controller theorem.
- Limitations: no universal WAM training law, no real-world robot validation, no
  OOD guarantee, no full closed-loop closed-form law.

Related work should absorb the broader lineage:

- Order statistics and AUC/Wilcoxon: cite David and Nagaraja plus Hanley and
  McNeil. Say the proof lineage is classical.
- BoN/reranking/verifiers: cite Cobbe, Beirami, and optionally Amini/Sun.
- Test-time compute: cite Snell for adaptive compute allocation and contrast
  LLM prompt difficulty with WAM score-tail real utility.
- World models and model-based planning: cite existing Ha/Schmidhuber, PlaNet,
  Dreamer, PETS, CEM, IT-MPC.
- Model bias/exploitation: cite MBPO/Janner and connect to imagined-real tail
  mismatch.
- Robot policy/generative action models: keep existing RT-1, RT-2, Octo,
  Diffusion Policy as motivation, but do not imply the experiments validate
  those systems.

## Recommended Related Work Framing

Suggested paragraph skeleton:

1. Best-of-N and verifier reranking.
   Prior LLM alignment and reasoning work samples many candidates and selects
   the highest-ranked answer using a verifier or reward model. This manuscript
   studies an analogous selection primitive in WAM rollout planning, but asks
   for selected real utility under a fixed score/utility distribution.

2. Order statistics and AUC.
   The theorem uses classical order-statistic reasoning. The N=2 binary special
   case connects to the ranking interpretation of AUC/Wilcoxon. The paper's
   point is the boundary: AUC explains two-sample success but not high-N
   selected utility.

3. World-model planning and sampling MPC.
   World models, PETS, CEM, and IT-MPC motivate sampling candidate futures or
   action sequences. This manuscript does not replace those planners; it gives a
   conditional law for the value of increasing the candidate count once the
   generator and scorer are fixed.

4. Test-time compute and allocation.
   Test-time compute papers motivate adaptive inference budgets. This work's
   allocation signal is not generic prompt difficulty; it is the marginal gain
   of the score-tail real-utility curve.

5. Model bias and mismatch.
   Model-based RL already warns that model rollouts can be biased. The theorem
   sharpens the failure mode for best-of-N selection: increasing N concentrates
   selection on whatever lives in the high-score tail, including hallucinated or
   anti-aligned rollouts.

## Claim Boundary For The Manuscript

Use this boundary in abstract, introduction, theorem discussion, and limitations.

| Claim family | Safe wording | Unsafe wording |
|---|---|---|
| Novelty | "exact inference law for a fixed best-of-N rollout-selection primitive" | "new order-statistic scaling law" |
| WAM scope | "WAM rollout planning is the motivating domain and evaluated setting" | "the theorem is specific only to WAMs" |
| Utility | "selected real utility under score-based selection" | "score accuracy alone determines value" |
| AUC | "AUC is exact for N=2 and insufficient alone for N>2" | "AUC is irrelevant" or "AUC controls all N" |
| Adaptivity | "inference-value profiles can guide rollout budgeting in tested settings" | "globally optimal adaptive controller" |
| Model mismatch | "high N can amplify score/real mismatch" | "the method solves OOD or prevents model exploitation" |
| Robotics | "simulated robotics and rollout-pool evidence" | "real-world robot validation" |
| Training | "fixed generator and scorer assumptions" | "universal WAM training recipe" |

## Main-Text Citation Suggestions

The current `iclr_submission.bib` already contains the world-model and
sampling-MPC anchors, plus David and Nagaraja. It is missing the strongest BoN,
AUC, verifier, adaptive compute, and model-bias citations.

Suggested insertion points:

- Introduction first paragraph after "sample N rollouts, score them": cite CEM,
  IT-MPC, PETS, and verifier/BoN work.
- End of contributions: state that the law uses order-statistic reasoning and
  cite David and Nagaraja.
- AUC proposition paragraph: cite Hanley and McNeil.
- Adaptive budgeting section: cite Snell.
- Imagined-real mismatch section: cite Janner/MBPO.
- Related work BoN paragraph: cite Cobbe, Beirami, Amini, Sun, Lightman as
  appropriate.

## Citation Candidate Policy

I did not create `docs/citation_candidates.bib` because this implementation was
scoped to a related-work audit only. If a BibTeX file is later requested, include
only verified candidates not already in `iclr_submission.bib`, likely:

- `hanley1982roc`
- `beirami2024bestofn`
- `cobbe2021verifiers`
- `snell2024testtime`
- `janner2019mbpo`
- optional: `amini2024vbon`, `sun2024fastbon`,
  `wang2022selfconsistency`, `lightman2023verify`,
  `schrittwieser2020muzero`, `han2026wam`

Do not add unverified BibTeX generated from memory. Prefer DOI, arXiv, dblp,
OpenReview, official publication pages, or Crossref metadata.

## Bottom Line

The paper can survive "this is order statistics" if it says so first. The
submission should make the theorem look simple and inevitable, then make the
conceptual move look valuable: in WAM rollout selection, the planner optimizes
score but deployment cares about real utility, so test-time imagination is only
as good as the real utility hiding in the selected score tail.
