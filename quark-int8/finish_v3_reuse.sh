#!/bin/bash
# 收尾 v3：等转换结束 → **复用旧模型已验证的 engram 分片 47/48** → 校验 → 交给 auto_eval 跑 A/B。
#
# 为什么可以复用（前提已逐张量验证，见转换记录 §4.23）：
#   q_weight/k_weight : 与源 fp8 模型**逐元素相等**（keep=确定性字节拷贝）⇒ 新链路必然相同
#   wkv.weight        : 源 F8_E4M3 → BF16 的反量化，新转换器走同一条 classify 规则 ⇒ 相同
#   embed.{weight,scale}: requant_engram_int4.py 同脚本同默认(--scale-search 3)，
#                       且本会话已用**独立解码路径**校验：SNR 18.18 dB、
#                       mean|Δ|/amax_blk=0.0395 ≤ 对称 int4 上界 1/14=0.0714 ✓
#   ⇒ 跳过"转换 47/48 + requant"（约 50 分钟），且 engram 在两臂间保持完全一致（更受控 ✓）
#
# 索引安全性：safetensors 的 index 只记 name→file，不记 dtype/shape ⇒ 覆盖文件不使索引失效 ✓
set -u
NEW=${NEW:?}
OLD_ENG=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4
CLOG=${CLOG:?}
REPO=/home/qiba/ROCm.AI/quark-int8
TS=$(date +%m%d_%H%M)

echo "=== $(date +%T) 等转换进程退出 ==="
while ps -eo cmd | grep -q "[c]onvert_dsv41_ct_int4.py --model"; do sleep 20; done
sleep 5
tail -4 "$CLOG"
if ! grep -qa "DONE" "$CLOG"; then echo "❌ 日志无 DONE，终止（不做任何复用）"; exit 1; fi
n=$(ls "$NEW"/model-000*-of-00048.safetensors 2>/dev/null | wc -l)
echo "转换产出分片数=$n（期望 48，转换器自己写 47/48 的 fp8 版）"
if [ "$n" -ne 48 ]; then echo "❌ 分片不足 48，终止"; exit 1; fi

echo "=== $(date +%T) 用旧模型的 int4 engram 分片覆盖 47/48 ==="
for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  cp -f "$OLD_ENG/$f" "$NEW/$f" && echo "  已复用 $f"
done

echo "=== $(date +%T) 校验：48 分片 + index 覆盖 layer39 + engram 是 U8/128 ==="
timeout 240 docker run --rm --entrypoint bash -v "$NEW":/m:ro vllm/vllm-openai-rocm:nightly -c "
python3 - <<'PY'
import json, struct, os
d='/m'
nm=json.load(open(f'{d}/model.safetensors.index.json'))['weight_map']
print('index 张量数 =', len(nm))
need=[f'layers.{i}.ffn.experts.0.w1.weight_packed' for i in (0,2,15,39)]
miss=[k for k in need if k not in nm]
print('抽样必需张量缺失:', miss or '无 ✓')
out=[]
for f in ('model-00047-of-00048.safetensors','model-00048-of-00048.safetensors'):
    with open(os.path.join(d,f),'rb') as fh:
        ln=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(ln))
    for k,v in h.items():
        if k.endswith('.engram.embed.weight'): out.append((v['dtype'],v['shape'][1]))
print('engram.embed 编码:', out)
assert out==[('U8',128),('U8',128)], f'engram 非 int4: {out}'
ig=json.load(open(f'{d}/config.json'))['quantization_config']['ignore']
print('ignore 含 *attn.fused_wqa_wkv:', '*attn.fused_wqa_wkv' in ig, '| 条数', len(ig))
print('✅ 校验通过')
PY
" || { echo "❌ 校验失败，不启动评测"; exit 1; }

du -sh "$NEW"
echo "=== $(date +%T) 模型就绪，交给 auto_eval 跑 A/B ==="
cd "$REPO"
CL=logs/finish_v3_run_${TS}.log
# ★ 顺序很关键：auto_eval 是在轮询 CL 里是否出现"完毕"才开始跑。
#   第一版把这行 echo 到了 stdout 而没写进 CL ⇒ auto_eval 会白等满 4 小时 ✗
{ echo "=== $(date +%T) 复用完成，模型可用 ==="; echo "=== $(date +%T) 完毕 ==="; } > "$CL"
setsid nohup env OUT="$NEW" CL="$CL" bash auto_eval.sh > logs/auto_eval_v3_${TS}.log 2>&1 &
echo "auto_eval PID=$!  日志 logs/auto_eval_v3_${TS}.log"
