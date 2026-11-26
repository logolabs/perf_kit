#!/bin/bash
# One rank of a pool job (launched by srun, one task per GPU). Local rank 0 of each node runs under the energy
# monitor when $PERUN is set (one record per node: NVML sees all the node's GPUs), the others plain.
#   PERUN      e.g. "perun -c /path/.perun.ini"   (word-split on purpose)
#   PERUN_ARGS e.g. "--format text --data_out /path/energy"
set -uo pipefail
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
cd "$REPO"
NUMA=""
if command -v numactl >/dev/null 2>&1; then NUMA="numactl --localalloc"; fi
set -f
if [ -n "${PERUN:-}" ] && [ "$LOCAL_RANK" = 0 ]; then
  # shellcheck disable=SC2086
  exec $NUMA $PERUN monitor ${PERUN_ARGS:-} --app_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}_n${SLURM_NODEID}" \
    "$POOL_SCRIPT" $POOL_ARGS
else
  # shellcheck disable=SC2086
  exec $NUMA python -u "$POOL_SCRIPT" $POOL_ARGS
fi
