set -u
T=/home/qiba/ai/tools/probe_disaster.py
U=http://127.0.0.1:8128/v1
M=glm53flash-int8
OUT=/home/qiba/ai/logs/glm53flash-0918/ladder-$(date +%H%M%S)
mkdir -p $OUT; echo OUT=$OUT
echo '### 因果阶梯：zh1 pos5'
python3 $T ladder --url $U --model $M --text zh1 --pos 5 --k 5 2>&1 | tee $OUT/ladder-zh1-p5.log
echo
echo '### 因果阶梯：zh1 pos9（换一个位置，排除单点偶然）'
python3 $T ladder --url $U --model $M --text zh1 --pos 9 --k 5 2>&1 | tee $OUT/ladder-zh1-p9.log
echo
echo '### 对齐修正后的 rate（每段 16 次）'
python3 $T rate --url $U --model $M --runs 16 --texts zh1,zh2,en1,seq --out $OUT/rate.json 2>&1 | tee $OUT/rate.log
echo DONE