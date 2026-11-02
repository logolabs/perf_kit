#!/bin/bash
# Sample every visible GPU once per second (SM clock, power, memory, utilisation, power-cap / slowdown reasons) into a
# CSV until killed. Used to report the clock the GPUs actually ran at (power-capped GH200s run well below the
# 1,980 MHz the 989 TFLOPS bf16 peak assumes) and an allocator-independent peak memory per arm.
#   smi_sample.sh <out.csv> [period_ms]
OUT=$1
P=${2:-1000}
FULL="timestamp,index,clocks.sm,clocks.max.sm,power.draw,enforced.power.limit,memory.used,utilization.gpu"
FULL="$FULL,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown"
MIN="timestamp,index,clocks.sm,power.draw,memory.used,utilization.gpu"
if nvidia-smi --query-gpu="$FULL" --format=csv,noheader,nounits >/dev/null 2>&1; then Q=$FULL; else Q=$MIN; fi
echo "$Q" > "$OUT"
exec nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -lms "$P" >> "$OUT"
