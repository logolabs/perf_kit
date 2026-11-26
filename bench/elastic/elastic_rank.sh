#!/bin/bash
# One rank of an elastic job (srun task, or run directly for a 1-GPU job). Local rank 0 runs under the energy
# monitor when $PERUN is set. With $PERUN_EXEC (a tiny Python wrapper that runs a command and raises on a non-zero
# exit) the script runs as a child of the monitor, so a rank that crashes (os._exit, a kill) still leaves an energy
# record; without it, in-process.  EL_ARGS2 / EL_ARGS3: further runs of $EL_SCRIPT, one after the other (a job that
# left or finished the first run goes on to the next; each is its own elastic run).
#   PERUN       e.g. "perun -c /path/.perun.ini"      PERUN_ARGS e.g. "--format text --data_out /path/energy"
set -uo pipefail
export LOCAL_RANK=$SLURM_LOCALID
cd "$REPO"
set -f
rc=0
n=0
for ARGS in "$EL_ARGS" "${EL_ARGS2:-}" "${EL_ARGS3:-}"; do
  [ -n "$ARGS" ] || continue
  n=$((n + 1))
  [ -f "$EL_RUN/leave/${SLURM_JOB_ID}" ] && { echo "=== leave requested: skipping run $n"; continue; }
  if [ -n "${PERUN:-}" ] && [ "$LOCAL_RANK" = 0 ]; then
    if [ -n "${PERUN_EXEC:-}" ]; then
      # shellcheck disable=SC2086
      $PERUN monitor ${PERUN_ARGS:-} --app_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}_r${n}" "$PERUN_EXEC" \
        python -u "$EL_SCRIPT" $ARGS || rc=$?
    else
      # shellcheck disable=SC2086
      $PERUN monitor ${PERUN_ARGS:-} --app_name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}_r${n}" "$EL_SCRIPT" $ARGS || rc=$?
    fi
  else
    # shellcheck disable=SC2086
    python -u "$EL_SCRIPT" $ARGS || rc=$?
  fi
done
exit $rc
