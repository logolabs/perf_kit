#!/bin/bash
# One job of an ElasticTCP run (perf_kit/elastic): 1 node, G GPUs, one task per GPU. Every task registers itself in
# the run folder and joins the run's current group at the next step boundary; the job may start at any time.
# Site specifics come from $SITE_ENV (modules, venv, PERUN, PERUN_ARGS, PERUN_EXEC, SRUN_FLAGS); account, partition
# and layout go on the sbatch command line (or use python -m perf_kit.elastic.submit):
#
#   export SITE_ENV=... REPO=... EL_RUN=<shared run dir> EL_SCRIPT=perf_kit/elastic/example.py EL_ARGS="--mode elastic ..."
#   sbatch -A <acct> -p <part> --nodes 1 --gpus-per-node 1 --ntasks-per-node 1 --cpus-per-task 16 --mem 60G \
#          -t 00:30:00 bench/elastic/elastic_job.sbatch
#
# Network: between jobs NCCL runs its own socket transport over plain IP (NCCL_NET=Socket on $EL_IFACE, default
# hsn0); inside the job NCCL uses NVLink (NCCL_HOSTID is per job, so two jobs that land on one node also talk over
# sockets instead of trying P2P/SHM across job cgroups). Do NOT load a libfabric/OFI NCCL plugin here: Slingshot
# gives every job its own VNI, so that path cannot reach another job.
# Leaving: Slurm's --signal=B:TERM@120 reaches this batch shell, whose trap touches $EL_RUN/leave/$SLURM_JOB_ID; the
# ranks see it within a step or two, finish the step and leave the group. ELASTIC_LEAVE_AT (end time - EL_MARGIN_S)
# is a second, signal-free trigger. Long argument strings travel in the environment, never in --export=...,X="...".
#SBATCH -J elastic
#SBATCH --signal=B:TERM@120
#SBATCH -o %x-%j.out
set -uo pipefail
EL_RUN=${EL_RUN:-${POOL_RUN:-}}
: "${SITE_ENV:?}" "${REPO:?}" "${EL_RUN:?}" "${EL_SCRIPT:?}" "${EL_ARGS:?}"
# shellcheck disable=SC1090
source "$SITE_ENV"
[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ] || { echo "REFUSING: an elastic job is one node ($SLURM_JOB_NUM_NODES)"; exit 3; }
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_NET=Socket NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=${EL_IFACE:-hsn0} GLOO_SOCKET_IFNAME=${EL_IFACE:-hsn0}
export ELASTIC_IFACE=${EL_IFACE:-hsn0} NCCL_HOSTID="job${SLURM_JOB_ID}" ELASTIC_JOB_SIZE=${SLURM_NTASKS:-1}
unset NCCL_NET_PLUGIN FI_PROVIDER
export TORCH_NCCL_ASYNC_ERROR_HANDLING=2 TORCH_NCCL_BLOCKING_WAIT=1   # abort the communicator, keep the process
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
if [ -n "$END" ] && END_S=$(date -d "$END" +%s 2>/dev/null); then
  export ELASTIC_LEAVE_AT=$((END_S - ${EL_MARGIN_S:-150}))
fi
mkdir -p "$EL_RUN/leave"
export ELASTIC_LEAVE_FILE="$EL_RUN/leave/$SLURM_JOB_ID"           # seen by every run this job takes part in
echo "=== elastic job $SLURM_JOB_ID host=$(hostname) tasks=${SLURM_NTASKS:-1} start $(date +%T) end ${END:-?} leave_at ${ELASTIC_LEAVE_AT:-none}"
echo "=== $EL_SCRIPT $EL_ARGS"
ip -4 -br addr 2>/dev/null | grep -E "^${EL_IFACE:-hsn0}" || echo "WARNING: no ${EL_IFACE:-hsn0} interface"
if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
  # shellcheck disable=SC2086
  srun --kill-on-bad-exit=0 --cpu-bind=none ${SRUN_FLAGS:-} bash "$REPO/bench/elastic/elastic_rank.sh" &
else
  SLURM_LOCALID=0 bash "$REPO/bench/elastic/elastic_rank.sh" &      # one GPU: no job step needed
fi
PID=$!
trap 'touch "$EL_RUN/leave/$SLURM_JOB_ID"; echo "=== $(date +%T) SIGTERM: leave requested"' TERM
rc=0
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done     # a trapped TERM interrupts wait: wait again
echo "=== job exit $rc $(date +%T)"
exit $rc
