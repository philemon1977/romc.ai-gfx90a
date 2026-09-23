#!/bin/bash
# 1) 先做「变换等价性」验证：字节内 nibble 互换 == 元素相邻对互换
# 2) 再做「文件机制」验证：复制一个分片 -> 就地修复 -> 与原文件逐元素比对
set -e
T=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/.fixtest
R=/mnt/kioxia-cm6-3t8/ai/models/deepseek-ai/DeepSeek-V4.1-Flash-CT-Int4-W4A16-engram4-opt-allbf16
SHARD=model-00039-of-00048.safetensors
mkdir -p $T
cp -f $R/model.safetensors.index.json $T/
[ -f $T/$SHARD ] || cp $R/$SHARD $T/$SHARD
for f in $R/*.safetensors; do b=$(basename $f); [ "$b" = "$SHARD" ] || ln -sfn $f $T/$b; done
for f in $R/*.json $R/*.py $R/*.txt; do [ -f "$f" ] && cp -fn "$f" $T/ 2>/dev/null; done
python3 /work/fix_fp4_nibble_order.py $T --apply --only $SHARD
