V=/usr/local/lib/python3.12/dist-packages/vllm
python3 -c "import vllm; print('VLLM_VERSION:', vllm.__version__)"
python3 -c "import quark; print('QUARK:', quark.__version__)" 2>&1 | tail -1
pip list 2>/dev/null | grep -iE "quark|aiter|amd" | head
echo "=== aiter import & gfx90a libs ==="
python3 -c "import aiter; print('AITER:', aiter.__version__ if hasattr(aiter,'__version__') else 'n/a', aiter.__file__)" 2>&1 | tail -2
find /usr/local/lib/python3.12/dist-packages -maxdepth 3 -iname "*aiter*" -name "*.so" 2>/dev/null | head -20
find /usr/local/lib/python3.12/dist-packages/aiter* -iname "*gfx9*" 2>/dev/null | sed 's/.*\///' | sort -u | head -20
echo "=== int8 moe oracle backends ==="
sed -n '1,120p' $V/model_executor/layers/fused_moe/oracle/int8.py
echo "=== gfx90a / arch gating in int8 moe backend impls ==="
grep -rn "gfx90a\|gfx942\|is_fp8\|ON_GFX9" $V/model_executor/layers/fused_moe/oracle/int8.py | head
