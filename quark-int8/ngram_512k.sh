#!/usr/bin/env bash
# ngram speculative decoding @ 512K context, no MTP (no draft model).
# Verifies: no prefix-cache-disable warning, prefix-cache hits > 0, decode speed,
# KV pool size, and needle retrieval at ~300K / ~500K.
set -uo pipefail
REPO=/home/qiba/ROCm.AI/quark-int8; IMG=vllm/vllm-openai-rocm:nightly
NAME=ornith-ngram512k; LOG="$REPO/logs/ngram_512k.log"; mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== ngram@512K start $(date -u +%FT%TZ) ==="
OVR='{"max_position_embeddings":524288,"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
for i in $(seq 1 60); do
  f=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -8 | awk -F, '{printf "%.0f\n",($2-$3)/1073741824}' | sort -n | head -1)
  [ "${f:-0}" -ge 62 ] && break; sleep 10
done
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --network host --device /dev/kfd --device /dev/dri --group-add video \
  --shm-size 64G --ulimit memlock=-1:-1 -e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -v /mnt/kioxia-cm6-3t8:/mnt/kioxia-cm6-3t8 $IMG \
  /mnt/kioxia-cm6-3t8/ai/models/ornith-ai/Ornith-1.5-397B-Quark-Int8-Attn \
  --served-model-name Ornith-ngram512k --port 8100 --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.975 --max-model-len 520000 --max-num-batched-tokens 2048 \
  --max-num-seqs 8 --hf-overrides "$OVR" \
  --language-model-only --trust-remote-code --moe-backend triton \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":5,"prompt_lookup_max":5}' >/dev/null
"$REPO/watch_container.sh" $NAME 900 8100 || exit 1
echo "--- 关键行 ---"
docker logs $NAME 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens.*|Available KV cache memory: [0-9.]+ GiB" | tail -2
echo "--- prefix cache 警告是否还出现（应为 0）---"
docker logs $NAME 2>&1 | grep -c "no KV cache group could be identified"
echo "--- 短上下文解码速度 ---"
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG -c "python3 -u - <<'PY'
import json,time,urllib.request
body=json.dumps({'model':'Ornith-ngram512k','prompt':'Count slowly:','max_tokens':128,'temperature':0,'ignore_eos':True}).encode()
req=urllib.request.Request('http://127.0.0.1:8100/v1/completions',data=body,headers={'Content-Type':'application/json'})
t0=time.time(); d=json.load(urllib.request.urlopen(req,timeout=900)); dt=time.time()-t0
print(f'short-ctx decode {d[\"usage\"][\"completion_tokens\"]} tok in {dt:.1f}s -> {d[\"usage\"][\"completion_tokens\"]/dt:.2f} tok/s')
PY"
echo "--- ngram 接受率 ---"
curl -s localhost:8100/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total\{"
echo "--- 前缀缓存复用测试（同一 200K 前缀两次请求）---"
docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
  -c "python3 -u /work/prefix_reuse_probe.py 8100 Ornith-ngram512k 200000" 2>&1 | tail -4
curl -s localhost:8100/metrics | grep -E "prefix_cache_(queries|hits)_total\{"
echo "--- needle 长上下文 ---"
for T in 262144 438000; do
  echo "=== needle @ target $T (~$((T*114/100)) actual) $(date -u +%H:%M:%S) ==="
  docker run --rm --entrypoint bash --network host -v "$REPO:/work" $IMG \
    -c "python3 -u /work/longctx_test.py 8100 Ornith-ngram512k $T 512" 2>&1 | tail -3
done
echo "=== ngram@512K done $(date -u +%FT%TZ) ==="
