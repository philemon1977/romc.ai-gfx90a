#!/usr/bin/env bash
# 停服 → 用 8117 脚本默认值（现在含 gfx90a a8w8 表）重启 → 就绪判据 → 交付态复测。
# 与 int8aiter_arm*.sh 的区别：**跑完不停服**（服务保持运行）。
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8
LAUNCH=/home/qiba/ai/models/ornith-ai/launcher/ornith_1.5_397b_int8w8a8attn_aiter_vllm_rocm72_mtp5_256k_8117_ornith_mi250dx8.sh
PIDF=/home/qiba/ai/logs/ornith397b-8117.pid
exec > >(tee -a "$REPO/logs/restart_8117.log") 2>&1
echo "=== 停服/重启 8117  $(date -u +%FT%TZ) ==="

# ── 1) 停服（只认 PID 文件；负号=进程组）────────────────────────────────
if [ -f "$PIDF" ]; then
  p=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
    echo "停服：kill -TERM -$p"; kill -TERM -"$p" 2>/dev/null
  else
    echo "PID 文件里的 $p 已不在（陈旧文件）"
  fi
else
  echo "⚠️ 没有 PID 文件；8117 若是手工起的，本脚本不会去 pkill（项目禁 pkill -f）"
fi
rm -f "$PIDF"

# 等端口释放 + 显存回收（launcher 的 gate 要求每 die ≥62 GiB）
for i in $(seq 1 90); do
  nc -z 127.0.0.1 8117 2>/dev/null && { sleep 5; continue; }
  free=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${free:-0}" -ge 62 ] && { echo "已停稳：端口空闲，min free ${free} GiB（用了 $((i*5))s）"; break; }
  sleep 5
done

# ── 2) 重启（脚本默认值；不传任何 env 覆盖）─────────────────────────────
echo "启动：env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE -u AITER_CONFIG_GEMM_A8W8 PORT=8117 bash $(basename "$LAUNCH")"
env -u SPEC -u VLLM_ROCM_USE_AITER -u VLLM_DISABLE_COMPILE_CACHE -u AITER_CONFIG_GEMM_A8W8 PORT=8117 bash "$LAUNCH" || { echo "❌ LAUNCH FAILED"; exit 1; }

t0=$(date +%s)
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1 && break; sleep 5; done
if ! curl -sf http://127.0.0.1:8117/health >/dev/null 2>&1; then
  S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
  echo "❌ NOT READY；$S 里的异常："
  grep -aoE "(ValueError|RuntimeError|AssertionError|AttributeError|ImportError|Consumer error): .{0,170}" "$S" | sort -u | head -5
  exit 1
fi
echo "✅ ready，用时 $(( $(date +%s)-t0 ))s"
S=$(ls -t /home/qiba/ai/logs/ornith397b/server-8117-*.log | head -1)
echo "服务日志：$S"

# ── 3) 就绪判据（四行都要对）+ 本次新增：a8w8 表是否吃上 ──────────────────
echo "--- 就绪判据 ---"
grep -acE "Selected AiterInt8ScaledMMLinearKernel" "$S" | sed 's/^/  Selected AiterInt8ScaledMMLinearKernel: /'
grep -acE "Selected TritonInt8ScaledMMLinearKernel" "$S" | sed 's/^/  Selected TritonInt8ScaledMMLinearKernel(反向门，应为0): /'
grep -aoE "Using [A-Za-z_]+ Int8 MoE backend" "$S" | sort -u | sed 's/^/  /'
grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$S" | sort -u | sed 's/^/  /'
grep -aoE "Available KV cache memory: [0-9.]+ GiB" "$S" | sort -u | sed 's/^/  /'
echo "--- a8w8 表（本次新增）---"
echo "  not found tuned config 行数: $(grep -ac 'not found tuned config' "$S")   （未注入时应为 432，期望 0）"
echo "  [aiter] import [module_gemm_a8w8] 行数: $(grep -ac 'import \[module_gemm_a8w8\]' "$S")"
echo "  编译路径: 载入=$(grep -acE 'Directly load AOT' "$S") 保存=$(grep -acE 'saved AOT compiled' "$S")  （期望 0/0，= DISABLE_COMPILE_CACHE=1 生效）"

# ── 4) 交付态复测（与提交进仓的数对照）─────────────────────────────────
echo "--- count n=5 ---"
python3 "$REPO/measure_median.py" 8117 restart-count 5 256 count 2>&1 | sed 's/^/  /'
echo "--- explain n=5 ---"
python3 "$REPO/measure_median.py" 8117 restart-explain 5 256 explain 2>&1 | sed 's/^/  /'
echo "--- 步时分解 count ---"
python3 "$REPO/step_probe.py" 8117 restart 5 count 256 2>&1 | sed 's/^/  /'
echo "--- 步时分解 explain ---"
python3 "$REPO/step_probe.py" 8117 restart 5 explain 256 2>&1 | sed 's/^/  /'

echo "=== 完成，服务保持运行（$(date -u +%FT%TZ)） ==="
echo "停服：kill -TERM -\$(cat $PIDF)"
