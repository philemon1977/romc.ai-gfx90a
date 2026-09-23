#!/bin/bash
# 等 vLLM 就绪 → 发**真实** chat 请求（触发探针的真实请求门）→ 把 dump 从容器 /tmp 取出到 host。
#
# 为什么必须用 chat 而不是 /v1/completions：checkpoint 不带 chat_template，
# 权威用法是 encoding/encode_messages；裸文本补全是**不支持的用法**，
# 用它得到的退化现象已被我们撤回（见转换记录 §4.10）。
#
# 走真实请求是为了过探针的门：positions == arange(T) 且 16 <= T <= 64。
# warmup 的 dummy 批 positions 是 vLLM 合成的（[2,0,1,...]），会被门挡掉。
set -u
PORT=8119
DUMPDIR=/tmp/dsv41_dump
CT=dsv41-ct-int4
mkdir -p "$DUMPDIR"

echo "=== $(date +%T) 等待服务就绪 ==="
for i in $(seq 1 480); do
  if curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then
    echo "✅ $(date +%T) 服务就绪"; break
  fi
  sleep 15
  [ $((i % 8)) -eq 0 ] && echo "  ...等待中 $((i*15))s"
done
curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" | head -c 300; echo

# 真实请求：中文长问句，token 数落在 16..64，think=false（官方 chat 模式）
echo "=== $(date +%T) 发送真实 chat 请求 ==="
RESP=$(curl -s -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models",
    "messages": [{"role": "user", "content": "请解释张量并行是如何切分权重与注意力头的，通信开销来自哪里。"}],
    "chat_template_kwargs": {"thinking": false},
    "max_tokens": 24,
    "temperature": 0
  }')
echo "$RESP" | head -c 700; echo
echo "$RESP" > "$DUMPDIR/response.json"

sleep 3
echo "=== $(date +%T) 从容器取出 dump ==="
docker cp "${CT}:/tmp/." "$DUMPDIR/ctmp" 2>&1 | tail -2
ls -la "$DUMPDIR"/ctmp/dsv41_* 2>/dev/null | sed 's/^/  /'
# 归位到 host /tmp（与探针打印的路径一致，便于直接引用）
cp -f "$DUMPDIR"/ctmp/dsv41_*.pt "$DUMPDIR"/ 2>/dev/null
echo "=== $(date +%T) 完成；host 侧文件 ==="
ls -la "$DUMPDIR"/dsv41_*.pt 2>/dev/null | sed 's/^/  /'
echo "=== 探针日志（门是否通过） ==="
grep -a "DUMP\]\|FFN\]" /home/qiba/ai/logs/dsv41ctint4/server-8119.current | tail -20
