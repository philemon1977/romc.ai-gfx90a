#!/bin/bash
# 自动链（用户选项 1）：等并发会话释放 GPU -> ① MoE tile 实测调优 -> ② 起服做稀疏针尖+质量验证
# 只用仓库内路径，不依赖 /tmp（本机今天 22:53 重启过一次）
# 规矩：bash 内不写 ${...} 形式；awk 程序一律用**单引号**（双引号会被 bash 先展开，
#       2026-09-20 实测踩过：awk "...$NF..." ⇒ set -u 下报 NF: unbound variable ⇒
#       探针永远返回空 ⇒ u 回退 999 ⇒ 条件永不成立 ⇒ 整条链永远不触发）
set -u
R=/home/qiba/ROCm.AI/quark-int8
SL=$R/scripts_local
LOG=$R/logs/glm_autorun_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
probe() {
  c=$(docker ps --format "{{.Names}}" | grep -cE "dsv41|glm53|vllm|ornith")
  u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{printf "%d", s/1073741824}')
  test -z "$c" && c=1
  test -z "$u" && u=999
}
echo "=== 自检：先证明探针可用（避免"验证者本身坏了"）$(date +%T) ==="
probe
echo "  探针结果: 容器数=$c  显存合计=${u}GiB"
if [ "$u" -eq 999 ]; then echo "❌ 探针拿不到显存（awk/rocm-smi 有问题）⇒ 不进入等待循环"; exit 1; fi
echo "=== 开始等待 GPU 空闲（别的会话在用，我不抢）$(date +%T) ==="
ok=0
for i in $(seq 1 1440); do
  probe
  if [ "$c" -eq 0 ] && [ "$u" -lt 20 ]; then
    ok=$((ok+1))
    echo "  空闲确认 $ok/3 (容器=$c 总用量=${u}GiB) $(date +%T)"
    [ "$ok" -ge 3 ] && break
  else
    [ "$ok" -ne 0 ] && echo "  又被占用，计数归零 $(date +%T)"
    ok=0
  fi
  sleep 20
done
if [ "$ok" -lt 3 ]; then echo "等待超时(8h)"; exit 1; fi
echo "=== GPU 空闲，开始 $(date +%T) ==="
echo "--- ① MoE tile 实测调优（只把赢默认 >=3% 的桶入表；可能不产出文件）---"
docker run --rm --entrypoint bash --device /dev/kfd --device /dev/dri --group-add video \
  -v $R:/work -v /home/qiba/ai/config/moe-tuned:/out \
  vllm/vllm-openai-rocm:nightly-0918 -c "cd /work && python3 -u moe_tune_w4a16.py --out-dir /out" 2>&1 | tail -25
echo "--- ② 起服：32K + 稀疏针尖 + 事实召回/GSM8K（表若产出会被自动拾取）---"
bash "$SL/glm_verify32k.sh"
echo "AUTORUN_DONE $(date +%T)"
