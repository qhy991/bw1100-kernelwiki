# BW1100 KernelWiki owner rules

Write objective mechanism explanations: cost, transformation, conditions, tradeoffs and bounded observations. Keep detailed experiment numbers and locators in source pages. A local result is not a general fastest-parameter or hardware guarantee.

Experiment artifacts and pinned code own facts. `sources/` owns their scoped reading; `wiki/` owns synthesis; generated `queries/` is a projection. Preserve superseded/negative evidence. Do not import raw datasets, weights, credentials, private IPs or model transcripts. Do not update live/frozen experiments as part of editing this library.

Before committing: `./bwiki index`, `./bwiki validate`, and the focused tests under `tests/`. Never raise the zero-warning budget. Querying or authoring the wiki does not authorize new GPU jobs, publication or automations.
