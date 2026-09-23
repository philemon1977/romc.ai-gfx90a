#!/bin/bash
docker run --rm --entrypoint bash --cpus=4 -v /home/qiba/ROCm.AI:/w \
  -v /mnt/stripe-3mix-3t2/models/deepseek-ai/DeepSeek-V4.1-Flash:/src:ro \
  -v /home/qiba/ROCm.AI/quark-int8/measure_opt_step.py:/m.py:ro \
  vllm/vllm-openai-rocm:nightly -c "nice -n 19 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout 900 python3 /m.py"
