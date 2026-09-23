
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=2, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_copy__mul_silu_slice_0', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9B7B4BF3E513717766BF1B43F7411CBEBB2EE8CF97883A4ED1B6530B3F9CAF4A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 113246208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_copy__mul_silu_slice_0(in_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + ks1*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (x0 + ks1*x1 + (ks1 // 2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 10.0, tl.float32)
    tmp3 = tl.minimum(tmp1, tmp2, tl.PropagateNan.ALL)
    tmp4 = -tmp3
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp3 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], -10.0, tl.float32)
    tmp13 = tl.maximum(tmp11, tmp12, tl.PropagateNan.ALL)
    tmp14 = tl.minimum(tmp13, tmp2, tl.PropagateNan.ALL)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr1 + (x2), tmp16, xmask)
