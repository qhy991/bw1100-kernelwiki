---
name: bw1100-kernelwiki
description: Retrieve and maintain evidence-backed BW1100/gfx938 kernel optimization knowledge, including DTK/HCU, AITER ports, fusion, mapping, precision and profiling mechanisms. Use for BW1100 kernel ideas or recording bounded local results; this skill does not launch GPU experiments or publish content.
---

# BW1100 kernel knowledge

The `knowledge` link points to the canonical dedicated library. Begin with its
README and `queries/pages.json`, then read the mechanism and cited source pages.

```bash
bash scripts/wiki.sh query "GEMM 融合" --limit 5
bash scripts/wiki.sh query --architecture gfx938 --kernel-type moe --type kernel
bash scripts/wiki.sh get technique-execution-groups
bash scripts/wiki.sh get exp-width-qualification
bash scripts/wiki.sh validate
```

Commands are relative to this skill, or use its absolute launcher path. They are
CPU-only. Report optimization ideas through cost, transformation, conditions and
tradeoffs; do not substitute a parameter leaderboard for a mechanism.

`source-reported/experimental` and the source's `evidence_scope` describe different
axes. Compile-only, resident components, search scores and fresh whole-call
confirmation are not interchangeable. Keep gfx938/Hygon distinct from AMD CDNA.
Read original locators when a new claim would change an action; missing live
observation remains unknown. The ongoing version pilot has no final result here.

For maintenance, read `MAINTENANCE.md`, record the owning evidence in `sources/`,
then revise synthesis. Use existing templates and controlled tags, regenerate
indices and validate. Source-specific reversals stay linked to their successor.
The original rocm-kernelwiki remains a separate upstream reference library.
