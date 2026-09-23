#!/bin/bash
# 全量转换保真度校验（容器内跑，CPU-only，不占 GPU）
# 48 分片逐个校验，结果追加日志，可随时查阅部分结果
SRC=/mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash
OUT=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4
LOG=/w/quark-int8/logs/verify_all_ct_int4.log
: > "$LOG"
echo "=== 全量保真度校验 $(date +%F' '%T) ===" | tee -a "$LOG"
bad=0
for i in $(seq -w 1 48); do
  S="model-000${i}-of-00048.safetensors"
  echo "--- [$S] $(date +%T) ---" >> "$LOG"
  timeout 2400 python3 /w/quark-int8/verify_ct_int4.py "$SRC" "$OUT" "$S" --sample 64 >> "$LOG" 2>&1
  rc=$?
  echo "    exit=$rc" >> "$LOG"
  [ $rc -ne 0 ] && bad=$((bad+1))
done
echo "=== 完成 $(date +%F' '%T)  非零退出分片数=$bad ===" | tee -a "$LOG"
