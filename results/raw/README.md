# Raw benchmark logs

One directory per Slurm job (`<tag>_<jobid>`), one subdirectory per arm (`<framework>_<setting>[_rN]`, `_rN` = the
N-th run of the same arm in that job). `log.jsonl` is what each harness wrote (window records, val, summary),
`arms.jsonl` the arm start/end times, `smi.csv` nvidia-smi samples at 1 Hz, `gpu_equiv.json` the GPU equivalence
check. Cluster-specific absolute paths were replaced by `$BENCH` / `$SHARED` and the user name by `$USER`;
nothing else was edited, except `node2_4803689/pk_12288` (see `SPLIT_NOTE.txt` there).

| tag | what |
|---|---|
| smoke | 12-step run of every harness + GPU checks |
| tune1 | 24-step baseline tuning sweep (micro-batch size, gradient checkpointing) and the first Unsloth attempt |
| ablate1, ablate2 | perf_kit ablations (`bench/e2e/pk_ablate.py`), 1 GPU; ablate1 also holds a profiled perf_kit run (`pkprof`) |
| main1-3 | 1-GPU comparison, 60 steps (main1 also ran the test suite) |
| warm1 | 1-GPU perf_kit cold + 2 warm-cache runs, Unsloth repeats |
| node1-3 | 4-GPU comparison (node3: ms-swift after the LOCAL_WORLD_SIZE fix, perf_kit cold + warm) |
| q27b | perf_kit on Qwen3.8-27B-FP8, 1 GPU |
