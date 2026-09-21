#!/usr/bin/env bash
# 复现 Hyperloom 会话 #3 的 v05 变体 aiter-int8-full-atomic（+2.9%，未过 KEEP 门）并上数值尺子。
#
# 为什么必须单独一个脚本、且不能图省事用 glm53flash_0918_arm.sh：
#   ① 臂脚本把 -e VLLM_ROCM_USE_AITER=0 **硬设**在 ENVS 里，传不进去；
#   ② v05 跑在 hyperloom-local（bridge 网络），宿主尺子打不到它的端口 ⇒ 这里改用 --network host；
#   ③ **环境必须逐条复刻 Magpie runner 的 preamble**（MI250_MOE_GEMV=1 /
#      PYTHONPATH=/patches/moe_gemv / DSV41_IDX_AITER_KERNEL=1 / FASTSAFETENSORS_ODIRECT=1 /
#      MI250_FST_MAX_BATCH_MB=2560 / PYTORCH_HIP_ALLOC_CONF / HF_HUB_OFFLINE=1），
#      否则复现出来的不是同一个臂（GEMV 补丁与自研 indexer 内核都会改变数值与速度）。
#   依据：session/…/runs/explore/9d747f3a…/v01_aiter-int8-full-atomic/…/config.yaml 的 envs 全文。
#
# 判据（顺序即优先级）：
#   硬门 A = NLL 量级（tools/probe_nll.py，坏模型会趋近 ln V=11.95）
#   硬门 B = 事实召回（quark-int8/scripts_local/taskcheck.py，与 8121 glm53_verify32k 同题同判据）
#   不采信「贪心两次逐字一致」——本机从未有 TP8 臂通过它（含健康的 8121），是阳性指标非必要条件。
# 用法（会话结束后）：bash quark-int8/scripts_local/verify_aiter_linear_arm.sh
set -uo pipefail
NAME=verify-aiter-linear
PORT=8128
IMAGE=rocm-ai/vllm:glm53-int4-hl-fl1
MODEL=/mnt/stripe-3mix-3t2/models/ZhipuAI/GLM-5.3-Flash-Quark-Int8
LOGD=/home/qiba/ai/logs/verify-aiter; mkdir -p "$LOGD"
LOG="$LOGD/server-$(date +%Y%m%d-%H%M%S).log"

docker rm -f "$NAME" >/dev/null 2>&1 || true
set -x
docker run -d --name "$NAME" --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 64g \
  -v "$MODEL:/models:ro" \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 \
  -e VLLM_ROCM_USE_AITER_TRITON_GEMM=0 -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 \
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
  -e MI250_MOE_GEMV=1 -e MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs \
  -e DSV41_IDX_AITER_KERNEL=1 -e FASTSAFETENSORS_ODIRECT=1 -e MI250_FST_MAX_BATCH_MB=2560 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -e HF_HUB_OFFLINE=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  -e AITER_LOG_TUNED_CONFIG=1 \
  "$IMAGE" vllm serve /models --port "$PORT" --served-model-name glm53flash-int8 \
    --tensor-parallel-size 8 --dtype bfloat16 --gpu-memory-utilization 0.95 \
    --max-model-len 8192 --max-num-seqs 8 --block-size 128 --kv-cache-dtype bfloat16 \
    --language-model-only --max-num-batched-tokens 2048 --enforce-eager --trust-remote-code
set +x
docker logs -f "$NAME" > "$LOG" 2>&1 &
echo "LOG=$LOG"

# 就绪/死亡双签名（装载 ~12 min）
for i in $(seq 1 90); do
  grep -qaE "Application startup complete" "$LOG" && { echo "READY $(date +%T)"; break; }
  grep -qaE "Segfault encountered|EngineCore failed|died unexpectedly|hipError|No available memory" "$LOG" && {
    echo "DIED $(date +%T)"; grep -anE "Segfault|died unexpectedly|Traceback|RuntimeError" "$LOG" | tail -6 | cut -c1-190
    docker rm -f "$NAME" >/dev/null; exit 2; }
  sleep 20
done
grep -qaE "Application startup complete" "$LOG" || { echo "TIMEOUT 未就绪"; exit 3; }

echo "=== 硬门 A：NLL 量级（对照 eager/lazy 基线 1.811 / 0.48 与坏模型 ln V=11.95） ==="
python3 /home/qiba/ai/tools/probe_nll.py --url "http://127.0.0.1:$PORT/v1" --model glm53flash-int8 2>&1 | tail -12
echo "=== 硬门 B：事实召回 + 短指令跟随 ==="
python3 /home/qiba/ROCm.AI/quark-int8/scripts_local/taskcheck.py --url "http://127.0.0.1:$PORT/v1" --model glm53flash-int8 2>&1 | tail -20
echo "=== 停服（只杀自己的容器） ==="
docker rm -f "$NAME" >/dev/null && echo stopped
