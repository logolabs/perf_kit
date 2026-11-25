#!/bin/bash
# One job of a growable pool (or the synchronous-DDP baseline): N nodes x G GPUs, one task per GPU, ordinary DDP
# inside the job. Site specifics come from $SITE_ENV (modules, venv, PERUN, SRUN_FLAGS; see
# bench/slurm/site_env.example.sh); account, partition and layout go on the sbatch command line:
#
#   export SITE_ENV=... REPO=... POOL_SCRIPT=perf_kit/pool/example_lm.py POOL_ARGS="--mode pool --out RUN ..."
#   sbatch -A <acct> -p <part> --nodes 1 --gpus-per-node 2 --ntasks-per-node 2 --cpus-per-task 16 --mem 128G \
#          -t 00:20:00 bench/pool/pool_job.sbatch
#
# Long argument strings travel in the environment (sbatch inherits it), never as --export=...,ARGS="...", which
# splits at commas. Optional: POOL_TOPUP="<perf_kit.pool.launch args>" runs one top-up after the training step,
# so the pool keeps itself alive without a login-node loop.
#SBATCH -J pool
#SBATCH --signal=TERM@120
#SBATCH -o %x-%j.out
set -uo pipefail
: "${SITE_ENV:?}" "${REPO:?}" "${POOL_SCRIPT:?}" "${POOL_ARGS:?}"
# shellcheck disable=SC1090
source "$SITE_ENV"
[ "${SLURM_JOB_NUM_NODES:-1}" -le 4 ] || { echo "REFUSING: $SLURM_JOB_NUM_NODES nodes > cap of 4"; exit 3; }
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
echo "=== pool job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS start $(date +%T)"
echo "=== $POOL_SCRIPT $POOL_ARGS"
python -c "import perf_kit, sys; print('=== perf_kit from', perf_kit.__file__)"
srun --kill-on-bad-exit=1 --cpu-bind=cores ${SRUN_FLAGS:-} bash "$REPO/bench/pool/pool_rank.sh"
rc=$?
echo "=== training step exit $rc $(date +%T)"
if [ -n "${POOL_TOPUP:-}" ]; then
  # shellcheck disable=SC2086
  python -m perf_kit.pool.launch $POOL_TOPUP || true
fi
exit $rc
