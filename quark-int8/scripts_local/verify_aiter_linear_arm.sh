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
# AITER=1 = 实验臂（复刻会话里那条 KEEP）；AITER=0 = **对照组**。
# 两者只差下面两个 0/1，其余（镜像、挂载、runner preamble 的那串 env、server args）逐字节相同，
# 这样"数值有没有变坏"才是有对照的结论，而不是拿另一个模型的基线当尺子。
# REPEATS：同一台服务器上连打 N 次 taskcheck。单条读数差不能直接归因——本机 TP8 的
# paged-KV/batch 组成会造成 run-to-run 抖动（09-21 实测同一条中文诗 NLL 在 0.573~1.514 之间跳），
# 所以"实验臂裂了一条召回"必须先排除抖动才能定性。
AITER="${AITER-1}"
REPEATS="${REPEATS-1}"
NAME="verify-aiter-$AITER"
LOGD=/home/qiba/ai/logs/verify-aiter; mkdir -p "$LOGD"
# 整臂输出必须落盘（09-22 实踩：A/B 只走管道送到调用方的作业输出里，读一次就没了，
# 事后想查"到底是哪一条召回裂了"无从查起）。
RUNLOG="$LOGD/arm$AITER-$(date +%Y%m%d-%H%M%S).out"
LOG="$LOGD/server-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$RUNLOG") 2>&1
echo "AITER=$AITER REPEATS=$REPEATS RUNLOG=$RUNLOG"
if [ "$AITER" = "1" ]; then
  ENV_A=(-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_LINEAR=1)
else
  ENV_A=(-e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_LINEAR=0)
fi
echo "=== AITER=$AITER  NAME=$NAME ==="

docker rm -f "$NAME" >/dev/null 2>&1 || true
set -x
docker run -d --name "$NAME" --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 64g \
  -v "$MODEL:/models:ro" \
  "${ENV_A[@]}" \
  -e VLLM_ROCM_USE_AITER_MOE=0 -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 \
  -e VLLM_ROCM_USE_AITER_TRITON_GEMM=0 -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 \
  -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
  -e MI250_MOE_GEMV=1 -e MI250_MOE_GEMV_MODULE=mi250_moe_gemv_gs \
  -e DSV41_IDX_AITER_KERNEL=1 -e FASTSAFETENSORS_ODIRECT=1 -e MI250_FST_MAX_BATCH_MB=2560 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True -e HF_HUB_OFFLINE=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
  -e AITER_LOG_TUNED_CONFIG=1 \
  "$IMAGE" /models --port "$PORT" --served-model-name glm53flash-int8 \
    --tensor-parallel-size 8 --dtype bfloat16 --gpu-memory-utilization 0.95 \
    --max-model-len 8192 --max-num-seqs 8 --block-size 128 --kv-cache-dtype bfloat16 \
    --language-model-only --max-num-batched-tokens 2048 --enforce-eager --trust-remote-code
set +x
docker logs -f "$NAME" > "$LOG" 2>&1 &
echo "LOG=$LOG"

# 就绪/死亡三签名（装载 ~12 min）。
# 为什么要单独查容器已退出：09-22 实踩 —— 本脚本初稿在命令里又写了一遍 vllm serve，
# 而 -fl1 镜像的 ENTRYPOINT 已是 ["vllm","serve"]，于是变成 vllm serve vllm serve /models，
# vllm 直接 argparse 报错 exit 2。那种失败**不留** Segfault/EngineCore 痕迹，只有一行
# "vllm: error: unrecognized arguments"，等待循环就会白转 30 分钟。容器状态才是硬信号。
for i in $(seq 1 90); do
  grep -qaE "Application startup complete" "$LOG" && { echo "READY $(date +%T)"; break; }
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "CONTAINER_EXITED $(date +%T) exit=$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)"
    tail -12 "$LOG" | cut -c1-190
    docker rm -f "$NAME" >/dev/null; exit 4
  fi
  grep -qaE "Segfault encountered|EngineCore failed|died unexpectedly|hipError|No available memory|unrecognized arguments|No such option" "$LOG" && {
    echo "DIED $(date +%T)"; grep -anE "Segfault|died unexpectedly|Traceback|RuntimeError" "$LOG" | tail -6 | cut -c1-190
    docker rm -f "$NAME" >/dev/null; exit 2; }
  sleep 20
done
grep -qaE "Application startup complete" "$LOG" || { echo "TIMEOUT 未就绪"; exit 3; }

echo "=== 硬门 A：NLL 量级（对照 eager/lazy 基线 1.811 / 0.48 与坏模型 ln V=11.95） ==="
python3 /home/qiba/ai/tools/probe_nll.py --url "http://127.0.0.1:$PORT/v1" --model glm53flash-int8 2>&1 | tail -12
echo "=== 硬门 B：事实召回 + 短指令跟随 ==="
# 针尖必须收窄：本臂 --max-model-len 8192，而 taskcheck 默认 needle 目标 9000 token
# ⇒ 直接 HTTP 400（09-22 实踩过，那一条腿等于没跑，不能记成 MISS）。给 8192 留出生成余量取 4000。
NEEDLE_TOKENS="${NEEDLE_TOKENS-4000}"
for k in $(seq 1 "$REPEATS"); do
  echo "----- taskcheck 第 $k/$REPEATS 次（同一台服务器，用来把抖动与真回归分开）-----"
  python3 /home/qiba/ROCm.AI/quark-int8/scripts_local/taskcheck.py --url "http://127.0.0.1:$PORT/v1" --model glm53flash-int8 --tag "aiter$AITER-r$k" --needle-tokens "$NEEDLE_TOKENS" 2>&1 | tail -24
done
echo "=== 停服（只杀自己的容器） ==="
docker rm -f "$NAME" >/dev/null && echo stopped
