#!/bin/bash
# Synchronous DDP on one 4-GPU node vs the pool fed by four small jobs that start at different times (1, 2, 1 and
# 2 GPUs; the third is hard-killed mid-run), same model, same per-GPU batch, same total samples.
#   export SITE_ENV=... REPO=... DATA=<dir with train.jsonl, val.jsonl> RUNS=<dir> ACCT=<account> PART=<partition>
#   bash bench/pool/compare.sh <tag> [total_samples] [inner_steps]
# Then: python bench/pool/summarize.py --sacct --run $RUNS/<tag>-ddp --run $RUNS/<tag>-pool
set -euo pipefail
TAG=${1:?tag}
TOTAL=${2:-768000}
H=${3:-500}
: "${SITE_ENV:?}" "${REPO:?}" "${DATA:?}" "${RUNS:?}" "${ACCT:?}" "${PART:?}"
MODEL=${MODEL:-"--dim 768 --layers 12 --heads 12"}
DIE_AFTER_S=${DIE_AFTER_S:-60}
LIMIT=${LIMIT:-00:15:00}
STAGGER=${STAGGER:-60}                   # seconds between the pool jobs' earliest start times
COMMON="--train $DATA/train.jsonl --val $DATA/val.jsonl --total_samples $TOTAL --batch 64 $MODEL --val_windows 2048"
export SITE_ENV REPO POOL_SCRIPT=perf_kit/pool/example_lm.py
mkdir -p "$RUNS/logs"
sub() {   # name gpus begin_offset_s extra_args
  local name=$1 g=$2 begin=$3; shift 3
  POOL_ARGS="$*" sbatch --parsable -A "$ACCT" -p "$PART" --nodes 1 --gpus-per-node "$g" --ntasks-per-node "$g" \
    --cpus-per-task 16 --mem $((64 * g))G -t "$LIMIT" --begin "now+$begin" -J "$name" \
    -o "$RUNS/logs/%x-%j.out" "$REPO/bench/pool/pool_job.sbatch"
}
# SCHEDULE defaults to the lr schedule the 2026-11-24 table ran with; "--schedule_by progress_est" is the fixed one
POOL="--mode pool --out $RUNS/$TAG-pool --inner_steps $H ${SCHEDULE:---schedule_by progress} ${OUTER:-} $COMMON"   # OUTER="--outer_momentum 0.5"
if [ -z "${POOL_ONLY:-}" ]; then
  echo "ddp   $(sub "pkp_${TAG}_ddp" 4 0 --mode ddp --out "$RUNS/$TAG-ddp" $COMMON)"
fi
echo "poolA $(sub "pkp_${TAG}_a" 1 0 $POOL)"
echo "poolB $(sub "pkp_${TAG}_b" 2 "$STAGGER" $POOL)"
echo "poolC $(sub "pkp_${TAG}_c" 1 $((2 * STAGGER)) $POOL --die_after_s "$DIE_AFTER_S")"
echo "poolD $(sub "pkp_${TAG}_d" 2 $((3 * STAGGER)) $POOL)"
