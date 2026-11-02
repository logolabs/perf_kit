#!/bin/bash
# One rank of a benchmark arm, launched by `srun --ntasks-per-node <GPUs> --cpu-bind=cores` (one task per GPU, each
# pinned to the cores next to its GPU; on GH200 its own Grace). Every arm -- perf_kit, transformers + PEFT and
# ms-swift -- is launched this way, so host-side placement is identical across frameworks.
#
#   rank.sh <arm name> <python script> [args...]
#
# Maps Slurm's task ids to RANK / WORLD_SIZE / LOCAL_RANK (torch.distributed, transformers.Trainer and perf_kit all
# read them), keeps host pages on the local NUMA node when numactl exists, and runs the script under the energy
# monitor named by $PERUN (optional; record name <arm>_<jobid>_r<rank>) or under $PY.
set -euo pipefail
ARM=$1; shift
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
# torchrun's LOCAL_WORLD_SIZE: ms-swift uses it to detect DDP (without it, it falls back to device_map model parallel)
LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-$SLURM_NTASKS}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE%%(*}          # Slurm may write "4(x2)"
NUMA=""
if command -v numactl >/dev/null 2>&1; then NUMA="numactl --localalloc"; fi
echo "[$ARM] rank $RANK/$WORLD_SIZE on $(hostname) cpus $(taskset -cp $$ 2>/dev/null | sed 's/.*: //')"
if [ -n "${PERUN:-}" ]; then
  exec $NUMA "$PERUN" monitor --app_name "${ARM}_${SLURM_JOB_ID}_r${RANK}" "$@"
else
  exec $NUMA "${PY:-python}" "$@"
fi
