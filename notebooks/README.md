# Notebooks

Notebook workflows are grouped by purpose:

- `demo/`: interactive AIM-Flow demonstrations.
- `benchmarks/`: combined SPFC benchmark workflows.
- `geneval/`: one-method-per-notebook GenEval generation runs.
- `t2i_compbench/generation/`: full T2I-CompBench generation runs.
- `t2i_compbench/shards/`: four merge-compatible T2I-CompBench SPFC notebooks: full and target-only-uniform runs for indexes `0:50` and `50:100`.
- `t2i_compbench/evaluation/`: official T2I-CompBench evaluation and comparison notebooks.
- `t2i_compbench/workflows/`: older all-in-one T2I-CompBench workflows retained for reference.
- `utilities/`: small inspection helpers.

All notebook filenames use lowercase snake case. The shard ranges are end-exclusive in the generation CLI.
