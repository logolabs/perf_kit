#!/bin/bash
# End-to-end LoRA benchmark arms on ONE node (1 GPU or all GPUs), one after another in one allocation, each arm a
# separate process tree launched the same way (bench/slurm/rank.sh under srun, one task per GPU). Site specifics
# (modules, virtualenvs, energy monitor, extra srun flags) come from $SITE_ENV (see site_env.example.sh); the account,
# partition and layout go on the sbatch command line, e.g.
#
#   1 GPU:  sbatch -A <acct> -p <part> --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 --cpus-per-task 72 \
#               --export=ALL,SITE_ENV=...,REPO=...,MODEL=...,DATA=...,RUNS=...,ARMS="pk hf hfx swift" bench_arms.sbatch
#   1 node: ... --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 --cpus-per-task 72 ...
#
# ARMS (space-separated, run in this order; each may carry a variant suffix, see the loop below):
#   pk      perf_kit.train (run_train.py), sequential step sets, bf16 base, micro-batch budget $PK_MICRO tokens
#   hf      transformers + PEFT, SDPA attention, gradient checkpointing, padded micro-batches of $HF_MICRO rows
#   hfx     transformers + PEFT + flash-attention 2 + Liger (RMSNorm, SwiGLU, fused linear-CE), checkpointing
#   swift   ms-swift (transformers backend): padding-free + flash-attention 2 + Liger, checkpointing, $SW_MICRO rows
#   unsloth hf_peft_train.py --unsloth (Unsloth's loader / patched kernels, same loop) under $UNSLOTH_PY
#   ablate  bench/e2e/pk_ablate.py on one GPU ($ABLATE configs)
#   tests   pytest tests/ (CPU + GPU tests) and bench/gpu_equiv.py on one GPU
# All arms share the workload: $STEPS optimizer steps of $GB rows (step s = rows [s*GB, (s+1)*GB) of train.jsonl),
# LoRA r16 / alpha 32 on every projection from the same init adapter, lr 2e-4, 6 warm-up steps, cosine to 0.1x.
#SBATCH -J pk_bench
#SBATCH -t 01:30:00
#SBATCH --signal=TERM@120
#SBATCH -o %x-%j.out
set -euo pipefail
: "${SITE_ENV:?}" "${REPO:?}" "${MODEL:?}" "${DATA:?}" "${RUNS:?}"
# shellcheck disable=SC1090
source "$SITE_ENV"
NG=${SLURM_NTASKS:-1}
STEPS=${STEPS:-60}
GB=${GB:-64}
PK_MICRO=${PK_MICRO:-8192}
HF_MICRO=${HF_MICRO:-8}
HFX_MICRO=${HFX_MICRO:-8}
SW_MICRO=${SW_MICRO:-8}
ARMS=${ARMS:-pk hf hfx swift}
JOBDIR=$RUNS/${TAG:-bench}_${SLURM_JOB_ID}
INIT=$RUNS/_init_r16
CACHE=$RUNS/_tokcache
mkdir -p "$JOBDIR"
cd "${PERUN_CWD:-$RUNS}"                    # the energy monitor reads its config from the working directory
export MASTER_ADDR=$(hostname) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
TKW='{"enable_thinking": false}'
if [ -f "$REPO/SHA256SUMS" ]; then
  (cd "$REPO" && sha256sum -c --quiet SHA256SUMS) || { echo "ABORT: $REPO differs from its SHA256SUMS"; exit 2; }
fi
echo "pk_bench $SLURM_JOB_ID on $(hostname): $NG task(s), arms: $ARMS, steps $STEPS, global batch $GB"
bash "$REPO/bench/slurm/smi_sample.sh" "$JOBDIR/smi.csv" 1000 &
SMI=$!
trap 'kill $SMI 2>/dev/null || true; if [ -n "${LOCAL:-}" ]; then rm -rf "$LOCAL"; fi' EXIT

ranks() {   # name, script, args...: one task per GPU
  local name=$1; shift
  srun --ntasks "$NG" --cpu-bind=cores ${SRUN_FLAGS:-} --kill-on-bad-exit=1 bash "$REPO/bench/slurm/rank.sh" "$name" "$@"
}
one() {     # name, script, args...: one task, one GPU
  local name=$1; shift
  srun --ntasks 1 --cpu-bind=cores ${SRUN_FLAGS:-} bash "$REPO/bench/slurm/rank.sh" "$name" "$@"
}
stamp() { echo "{\"arm\": \"$1\", \"event\": \"$2\", \"t\": $(date +%s.%N), \"tasks\": $NG}" >> "$JOBDIR/arms.jsonl"; }

if [ "${STAGE_MODEL:-0}" = 1 ]; then         # one sequential copy to node-local disk instead of per-arm mmap reads
  LOCAL=${TMPDIR:-/tmp}/$USER/model_$SLURM_JOB_ID
  mkdir -p "$LOCAL"
  t0=$(date +%s)
  cp "$MODEL"/*.json "$MODEL"/*.safetensors "$MODEL"/*.txt "$LOCAL"/ 2>/dev/null || cp "$MODEL"/* "$LOCAL"/
  echo "staged $MODEL -> $LOCAL in $(( $(date +%s) - t0 )) s"
  MODEL=$LOCAL
fi
if [ ! -f "$INIT/adapter_model.safetensors" ]; then
  one init "$REPO/bench/e2e/make_init_adapter.py" --model "$MODEL" --out "$INIT" --rank 16 --alpha 32 --seed 0
fi
COMMON=(--model "$MODEL" --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --init_adapter "$INIT"
        --chat_template_kwargs "$TKW" --rank 16 --alpha 32 --lr 2e-4 --warmup 6 --min_lr_ratio 0.1
        --max_steps "$STEPS" --global_batch "$GB" --log_every 10 --val_rows 256 --max_len 4096 --token_cache "$CACHE")

for a in $ARMS; do
  # an arm token is <base>[@<micro>][+nockpt]: e.g. hfx@16, swift@8+nockpt, pk@12288 (micro = rows per micro-batch
  # for the padded / swift arms, tokens per micro-batch for perf_kit; +nockpt turns gradient checkpointing off)
  base=${a%%[@+]*}
  micro=""
  if [[ $a =~ @([0-9]+) ]]; then micro=${BASH_REMATCH[1]}; fi
  CK=--grad_ckpt; SWCK=true
  if [[ $a == *+nockpt* ]]; then CK=""; SWCK=false; fi
  name=$(echo "$a" | tr '@+' '__')
  stamp "$name" start
  case $base in
    pk)
      ranks "$name" "$REPO/run_train.py" "${COMMON[@]}" --out "$JOBDIR/$name" --step_order sequential --storage bf16 \
        --micro_tokens "${micro:-$PK_MICRO}" --no_save --eval_every 0 ${PK_EXTRA:-} || echo "=== arm $name FAILED" ;;
    hf)
      ranks "$name" "$REPO/bench/e2e/hf_peft_train.py" "${COMMON[@]}" --out "$JOBDIR/$name" \
        --micro_bs "${micro:-$HF_MICRO}" --attn sdpa $CK ${HF_EXTRA:-} || echo "=== arm $name FAILED" ;;
    hfx)
      ranks "$name" "$REPO/bench/e2e/hf_peft_train.py" "${COMMON[@]}" --out "$JOBDIR/$name" \
        --micro_bs "${micro:-$HFX_MICRO}" --attn flash_attention_2 --liger $CK ${HFX_EXTRA:-} \
        || echo "=== arm $name FAILED" ;;
    unsloth)
      PYTHONPATH="${UNSLOTH_PYTHONPATH:+$UNSLOTH_PYTHONPATH:}$PYTHONPATH" \
        ranks "$name" "$REPO/bench/e2e/exec_cmd.py" "${UNSLOTH_PY:-$PY}" "$REPO/bench/e2e/hf_peft_train.py" \
        "${COMMON[@]}" --out "$JOBDIR/$name" --micro_bs "${micro:-$HFX_MICRO}" --unsloth $CK ${UL_EXTRA:-} \
        || echo "=== arm $name FAILED" ;;
    swift)
      MB=${micro:-$SW_MICRO}
      ACC=$((GB / NG / MB))
      export BENCH_OUT=$JOBDIR/$name BENCH_MODEL=$MODEL BENCH_VAL=$DATA/val.jsonl BENCH_VAL_ROWS=256
      export BENCH_TEMPLATE_KWARGS=$TKW BENCH_LOG_EVERY=10 BENCH_E2E_DIR=$REPO/bench/e2e
      ranks "$name" "$REPO/bench/e2e/exec_cmd.py" "$SWIFT_PY" -m swift.cli.sft \
        --model "$MODEL" --model_type qwen3_5 --template qwen3_5 --tuner_type lora --adapters "$INIT" \
        --dataset "$DATA/train.jsonl" --split_dataset_ratio 0 --dataset_shuffle false \
        --train_dataloader_shuffle false --torch_dtype bfloat16 --max_length 4096 --truncation_strategy delete \
        --enable_thinking false --add_non_thinking_prefix true --loss_scale ignore_empty_think \
        --per_device_train_batch_size "$MB" --gradient_accumulation_steps "$ACC" --max_steps "$STEPS" \
        --learning_rate 2e-4 --warmup_steps 6 --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --weight_decay 0.0 --adam_beta2 0.99 --max_grad_norm 1.0 \
        --gradient_checkpointing "$SWCK" --use_liger_kernel true --padding_free true --attn_impl flash_attn \
        --freeze_vit true --freeze_aligner true --ddp_find_unused_parameters false \
        --eval_strategy no --save_strategy no --logging_steps 10 --report_to none \
        --dataloader_num_workers 4 --dataset_num_proc 16 --seed 0 --data_seed 0 \
        --external_plugins "$REPO/bench/e2e/swift_bench_plugin.py" --callbacks perf_kit_bench \
        --output_dir "/tmp/$USER/swift_out_${SLURM_JOB_ID}_$name" ${SW_EXTRA:-} || echo "=== arm $name FAILED" ;;
    ablate)
      one ablate "$REPO/bench/e2e/pk_ablate.py" --model "$MODEL" --train "$DATA/train.jsonl" --init_adapter "$INIT" \
        --chat_template_kwargs "$TKW" --out "$JOBDIR/ablate" --global_batch "$GB" --micro_tokens "$PK_MICRO" \
        --configs ${ABLATE:-base} || echo "=== arm ablate FAILED" ;;
    tests)
      PYTHONPATH="${TEST_PYTHONPATH:+$TEST_PYTHONPATH:}$PYTHONPATH" one tests "$REPO/bench/e2e/exec_cmd.py" "$PY" \
        -m pytest -q -rs -p no:cacheprovider "$REPO/tests" || echo "=== arm tests FAILED"
      one equiv "$REPO/bench/gpu_equiv.py" --out "$JOBDIR/gpu_equiv.json" || echo "=== arm equiv FAILED" ;;
    *) echo "unknown arm $a" ;;
  esac
  stamp "$name" end
done
echo "=== $(date '+%F %T %Z') pk_bench done: $JOBDIR"
