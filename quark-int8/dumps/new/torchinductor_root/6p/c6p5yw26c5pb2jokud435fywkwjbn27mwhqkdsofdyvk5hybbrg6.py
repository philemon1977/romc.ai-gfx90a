
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=7, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '9B7B4BF3E513717766BF1B43F7411CBEBB2EE8CF97883A4ED1B6530B3F9CAF4A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 10}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0(in_ptr0, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = ks0
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ks2
    tmp7 = tmp0 >= tmp6
    tmp8 = ks3
    tmp9 = tmp0 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 | tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp0.to(tl.int64)
    tmp14 = tmp5.to(tl.int64)
    tmp15 = tmp14 * tmp1
    tmp16 = tmp10.to(tl.int64)
    tmp17 = ks0 + ks2 + ((-1)*ks1) + ((-1)*ks4)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp13 - tmp19
    tmp21 = tmp12 * tmp20
    tl.store(out_ptr0 + (x0), tmp21, xmask)
