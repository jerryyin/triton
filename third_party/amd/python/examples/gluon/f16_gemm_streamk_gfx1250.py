import pytest
import torch
import math
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# WG-cluster (multi-CTA) layout helper
from triton._C.libtriton.gluon_ir import make_cga_layout

# Handle imports for both pytest (module context) and direct execution
try:
    from .gfx1250_utils import static_profile
    from .f16_gemm_common_gfx1250 import (
        create_tensor_descriptors,
        issue_loads,
        issue_l2_prefetches,
        issue_l2_prefetches_prologue,
        issue_wmma,
        lds_subtile_load,
        slicemn_quad_consume,
        slicemn_subtile_load,
        TileScheduler,
        wc_ksplit_tile,
        wc_ksplit_tile_quad,
    )
except ImportError:
    from gfx1250_utils import static_profile
    from f16_gemm_common_gfx1250 import (
        create_tensor_descriptors,
        issue_loads,
        issue_l2_prefetches,
        issue_l2_prefetches_prologue,
        issue_wmma,
        lds_subtile_load,
        slicemn_quad_consume,
        slicemn_subtile_load,
        TileScheduler,
        wc_ksplit_tile,
        wc_ksplit_tile_quad,
    )


@gluon.jit
def process_remainder_wcsplit(
    a_ptr,
    b_ptr,
    c_ptr,
    scr_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,  #
    pid,
    num_cgas,
    num_full_tiles,
    scheduler,  #
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_CTAS: ttgl.constexpr,
    K_WIDTH: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    load3d: ttgl.constexpr,
    wmma3d: ttgl.constexpr,
    c2d: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
):
    """Phase 2 (lock-free): each remainder tile owned by ONE cluster; K split across the
    cluster's CTAs via a rank-3 batched dot (batch=CGA-split), reduced through scratch +
    cluster.barrier. No P-buffer, no atomic locks. Degenerates correctly to NUM_CTAS==1.
    Large tiles are quad-split (2x2) to keep the accumulator/reduce buffers off the spill path;
    for those load3d/wmma3d/c2d are sized for HALF_M x HALF_N (see _build_remainder_layouts)."""
    if STREAMK_TILES == 0:
        return

    QUAD: ttgl.constexpr = BLOCK_M >= 128 and BLOCK_N >= 128
    scr_cga = pid * (NUM_CTAS * BLOCK_M * BLOCK_N)
    for rt in range(pid, STREAMK_TILES, num_cgas):
        tile_id = num_full_tiles + rt
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_id, GROUP_SIZE_M)
        if QUAD:
            wc_ksplit_tile_quad(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, pid_m * BLOCK_M, pid_n * BLOCK_N, M, N, K,
                                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
                                BLOCK_K, NUM_CTAS, K_WIDTH, load3d, wmma3d, c2d)
        else:
            wc_ksplit_tile(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, pid_m * BLOCK_M, pid_n * BLOCK_N, M, N, K, stride_am,
                           stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS,
                           K_WIDTH, load3d, wmma3d, c2d)


@gluon.jit
def streamk_gemm_tdm_pipelined_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    LOAD3D: ttgl.constexpr,
    WMMA3D: ttgl.constexpr,
    C2D: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
    L2_PREFETCH_DISTANCE: ttgl.constexpr = 0,
):
    """
    StreamK GEMM kernel (4 or 8 warps; warp count flows in via NUM_WARPS / WMMA_LAYOUT).
    Phase 1: standard pipelined persistent loop (prefill / steady-state overlap / drain).
    Phase 2: lock-free within-cluster K-split remainder (cluster.barrier reduce).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8, "This kernel supports NUM_WARPS in {4, 8}")

    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # ============================================================================
    # Phase 1: Process full tiles (persistent scheduling) - pipelined loop
    # ============================================================================
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    # Number of CGAs (StreamK workers) = total programs / CTAs-per-cluster.
    # program_id is the CGA rank, but num_programs() counts CTAs, so divide.
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_cgas = num_sms // num_ctas
    num_full_tiles = scheduler.get_num_full_tiles()

    # Enable chiplet transformation (8 XCDs) to improve l2 reuse
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_cgas, num_xcds=8, chunk_size=2)

    # Trip-count hint: run() guarantees cdiv(K,BLOCK_K) >= NUM_BUFFERS, so the steady K-loop runs
    # at least once -- lets the backend drop the loop-guard and pipeline the body.
    ttgl.assume(ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1) > 0)

    # Persistent loop: each CU processes its assigned tiles with stride NUM_SMS
    for tile_idx in range(pid, num_full_tiles, num_cgas):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        # L2 prefetch for iterations [NUM_BUFFERS, NUM_BUFFERS + L2_PREFETCH_DISTANCE) (no-op if 0)
        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                     NUM_BUFFERS, TRANSPOSE_B)

        # Prefill pipeline
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        # Steady state: overlap load and compute
        for k_iter in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
            producer = issue_loads(
                producer,
                a_desc,
                b_desc,
                off_am,
                off_bn,
                a_buffer,
                b_buffer,
                BLOCK_K,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )
            # Prefetch L2_PREFETCH_DISTANCE-1 ahead (producer already +1 from the issue_loads above)
            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                TRANSPOSE_B)
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 1) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        # Drain pipeline
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            consumer, accumulator = issue_wmma(
                consumer,
                a_buffer,
                OPERAND_LAYOUT_A,
                b_buffer,
                OPERAND_LAYOUT_B,
                accumulator,
                (NUM_BUFFERS - 2 - i) * 2,
                NUM_BUFFERS,
                TRANSPOSE_B,
            )

        # Store result
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # ============================================================================
    # Phase 2 (lock-free): within-cluster K-split remainder (cluster.barrier reduce)
    # ============================================================================
    process_remainder_wcsplit(
        a_ptr,
        b_ptr,
        c_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid,
        num_cgas,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ttgl.num_ctas(),
        8,
        STREAMK_TILES,
        LOAD3D,
        WMMA3D,
        C2D,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_prefetch_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    LOAD3D: ttgl.constexpr,
    WMMA3D: ttgl.constexpr,
    C2D: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
    L2_PREFETCH_DISTANCE: ttgl.constexpr = 0,
):
    """
    StreamK GEMM kernel for 4 warps with TDM, software pipelining, and prologue-epilogue overlap.
    This variant prefetches data for the NEXT tile during the epilogue of the current tile.
    Phase 1: Prefetch-optimized persistent loop.
    Phase 2: lock-free within-cluster K-split remainder (cluster.barrier reduce).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4, "This kernel is only valid for NUM_WARPS == 4")

    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    # Initialize scheduler
    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    # ============================================================================
    # Phase 1: Process full tiles with TRUE prologue-epilogue overlap
    # Uses separate k_counter and buffer_slot tracking to enable cross-tile prefetch
    # ============================================================================
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    # Number of CGAs (StreamK workers) = total programs / CTAs-per-cluster.
    # program_id is the CGA rank, but num_programs() counts CTAs, so divide.
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_cgas = num_sms // num_ctas
    num_full_tiles = scheduler.get_num_full_tiles()

    # Enable chiplet transformation (8 XCDs) to improve l2 reuse
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_cgas, num_xcds=8, chunk_size=2)

    num_k_iters = ttgl.cdiv(K, BLOCK_K)
    # Trip-count hint: run() guarantees num_k_iters >= NUM_BUFFERS, so the steady loop runs >=1x.
    ttgl.assume(num_k_iters - (NUM_BUFFERS - 1) > 0)

    # ===== Initial prologue: Prefetch for FIRST tile =====
    # buffer_slot tracks which shared memory buffer to use (mod NUM_BUFFERS)
    # k_counter tracks which k iteration we're prefetching for current tile
    buffer_slot = 0
    first_tile_idx = pid
    if first_tile_idx < num_full_tiles:
        pid_m_first, pid_n_first = scheduler.get_swizzled_tile_coords(first_tile_idx, GROUP_SIZE_M)
        off_am_first = pid_m_first * BLOCK_M
        off_bn_first = pid_n_first * BLOCK_N

        # Prefill first NUM_BUFFERS-1 slots with first tile's data
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            k_offset = i * BLOCK_K
            slot = buffer_slot % NUM_BUFFERS
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am_first, k_offset], a_buffer.index(slot))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn_first], b_buffer.index(slot))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn_first, k_offset], b_buffer.index(slot))
            buffer_slot += 1

    # k_counter for current tile (starts at NUM_BUFFERS-1 because we prefilled)
    k_counter = NUM_BUFFERS - 1
    consumer_slot = 0

    # ===== Main tile loop =====
    for tile_idx in range(pid, num_full_tiles, num_cgas):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        # Steady state: overlap load and compute
        for _ in range(0, num_k_iters - (NUM_BUFFERS - 1)):
            # Issue load for current tile
            k_offset = k_counter * BLOCK_K
            slot = buffer_slot % NUM_BUFFERS
            ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, k_offset], a_buffer.index(slot))
            if not TRANSPOSE_B:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset, off_bn], b_buffer.index(slot))
            else:
                ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, k_offset], b_buffer.index(slot))
            buffer_slot += 1
            k_counter += 1

            # Wait and consume
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
            cons_slot = consumer_slot % NUM_BUFFERS
            a_operand = a_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(cons_slot).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))
            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)
            consumer_slot += 1

        # Check for next tile
        next_tile_idx = tile_idx + num_cgas
        has_next_tile = next_tile_idx < num_full_tiles
        if has_next_tile:
            pid_m_next, pid_n_next = scheduler.get_swizzled_tile_coords(next_tile_idx, GROUP_SIZE_M)
            off_am_next = pid_m_next * BLOCK_M
            off_bn_next = pid_n_next * BLOCK_N
        else:
            off_am_next = off_am
            off_bn_next = off_bn

        # Drain + Prefetch for next tile
        for i in ttgl.static_range(NUM_BUFFERS - 1):
            # Consume remaining data from current tile
            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
            cons_slot = consumer_slot % NUM_BUFFERS
            a_operand = a_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_A)
            if not TRANSPOSE_B:
                b_operand = b_buffer.index(cons_slot).load(layout=OPERAND_LAYOUT_B)
            else:
                b_operand = (b_buffer.index(cons_slot).permute([1, 0]).load(layout=OPERAND_LAYOUT_B))
            accumulator = ttgl.amd.gfx1250.wmma(a_operand, b_operand, accumulator)
            consumer_slot += 1

            # Prefetch for next tile (overlapped with current drain)
            if has_next_tile:
                k_offset_next = i * BLOCK_K
                slot = buffer_slot % NUM_BUFFERS
                ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am_next, k_offset_next], a_buffer.index(slot))
                if not TRANSPOSE_B:
                    ttgl.amd.gfx1250.tdm.async_load(b_desc, [k_offset_next, off_bn_next], b_buffer.index(slot))
                else:
                    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn_next, k_offset_next], b_buffer.index(slot))
                buffer_slot += 1

        # Reset k_counter for next tile (it starts at NUM_BUFFERS-1 because we just prefilled)
        k_counter = NUM_BUFFERS - 1

        # Store result
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # ============================================================================
    # Phase 2 (lock-free): within-cluster K-split remainder (cluster.barrier reduce)
    # ============================================================================
    process_remainder_wcsplit(
        a_ptr,
        b_ptr,
        c_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid,
        num_cgas,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ttgl.num_ctas(),
        8,
        STREAMK_TILES,
        LOAD3D,
        WMMA3D,
        C2D,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_slicemn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    LOAD3D: ttgl.constexpr,
    WMMA3D: ttgl.constexpr,
    C2D: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
    L2_PREFETCH_DISTANCE: ttgl.constexpr = 0,
):
    """
    StreamK GEMM kernel (4 warps) with a sequential 2x2 accumulator-quad split.
    Phase 1: persistent loop; each output tile is computed as four HALF_M x HALF_N quadrants
    processed ONE AT A TIME (each with its own full K-loop). Only one quadrant accumulator is
    live at a time, so the LLVM backend keeps the kernel out of the register-spill regime that a
    single monolithic BLOCK_M x BLOCK_N accumulator falls into -- this is the VGPR-reduction
    purpose of quad splitting (cf. the general-gemm quad at f16_gemm_gfx1250.py:1308 and the
    remainder's wc_ksplit_tile_quad). The TDM global->LDS pipeline is reused unchanged; each
    quadrant ds_loads its A/B sub-slice from the full LDS tile (so the tile is reloaded per
    quadrant -- a traffic/VGPR trade we accept to avoid spills).
    Phase 2: lock-free within-cluster K-split remainder (cluster.barrier reduce).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8, "sliceMN kernel supports NUM_WARPS in {4, 8}")
    # The 2x2 split slices the M and N dims of the LDS tiles; slicing a CTA-distributed
    # dimension is unsupported, so this variant is single-CTA only (the per-XCC perf target).
    # Multi-CTA WG-cluster configs use the pipelined kernel.
    ttgl.static_assert(ttgl.num_ctas() == 1, "sliceMN kernel supports single-CTA only (num_ctas == 1)")

    HALF_M: ttgl.constexpr = BLOCK_M // 2
    HALF_N: ttgl.constexpr = BLOCK_N // 2

    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(
        a_ptr,
        b_ptr,
        0,
        0,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        SHARED_LAYOUT_A,
        SHARED_LAYOUT_B,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TRANSPOSE_B,
    )
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_cgas = num_sms // num_ctas
    num_full_tiles = scheduler.get_num_full_tiles()

    # Enable chiplet transformation (8 XCDs) to improve l2 reuse
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_cgas, num_xcds=8, chunk_size=2)

    # Trip-count hint: run() guarantees cdiv(K,BLOCK_K) >= NUM_BUFFERS, so each quadrant's steady
    # K-loop runs at least once.
    ttgl.assume(ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1) > 0)

    for tile_idx in range(pid, num_full_tiles, num_cgas):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        # Sequential 2x2 quad: one HALF_M x HALF_N accumulator live at a time. Each quadrant runs
        # its own prefill/steady/drain K-loop and ds_loads only its (QM_OFF, QN_OFF) sub-slice.
        for qm in ttgl.static_range(2):
            for qn in ttgl.static_range(2):
                producer = 0
                consumer = 0
                acc = ttgl.zeros((HALF_M, HALF_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

                # Prefill pipeline (global -> LDS TDM, full tiles)
                for i in ttgl.static_range(NUM_BUFFERS - 1):
                    producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K,
                                           NUM_BUFFERS, TRANSPOSE_B)

                # Steady state: TDM next full tile + consume this quadrant's sub-slice
                for k_iter in range(0, ttgl.cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)):
                    producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K,
                                           NUM_BUFFERS, TRANSPOSE_B)
                    consumer, acc = slicemn_quad_consume(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer,
                                                         OPERAND_LAYOUT_B, acc, (NUM_BUFFERS - 1) * 2, NUM_BUFFERS,
                                                         TRANSPOSE_B, qm * HALF_M, qn * HALF_N, HALF_M, HALF_N)

                # Drain pipeline
                for i in ttgl.static_range(NUM_BUFFERS - 1):
                    consumer, acc = slicemn_quad_consume(consumer, a_buffer, OPERAND_LAYOUT_A, b_buffer,
                                                         OPERAND_LAYOUT_B, acc, (NUM_BUFFERS - 2 - i) * 2, NUM_BUFFERS,
                                                         TRANSPOSE_B, qm * HALF_M, qn * HALF_N, HALF_M, HALF_N)

                # Store this quadrant
                offs_m = pid_m * BLOCK_M + qm * HALF_M + ttgl.arange(0, HALF_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
                offs_n = pid_n * BLOCK_N + qn * HALF_N + ttgl.arange(0, HALF_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
                offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                ttgl.store(c_ptr + offs_c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # ============================================================================
    # Phase 2 (lock-free): within-cluster K-split remainder (cluster.barrier reduce)
    # ============================================================================
    process_remainder_wcsplit(
        a_ptr,
        b_ptr,
        c_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid,
        num_cgas,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ttgl.num_ctas(),
        8,
        STREAMK_TILES,
        LOAD3D,
        WMMA3D,
        C2D,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_subtile_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    LOAD3D: ttgl.constexpr,
    WMMA3D: ttgl.constexpr,
    C2D: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
    L2_PREFETCH_DISTANCE: ttgl.constexpr = 0,
):
    """
    StreamK GEMM kernel with K-direction subtile manual LDS/WMMA interleave.
    Phase 1: persistent loop; each BLOCK_K tile is consumed as NUM_SUBTILES=BLOCK_K//32
    sub-iterations -- the LDS->VGPR ds_load for sub-iter s+1 is issued BEFORE the WMMA of sub-iter
    s, so each WMMA's operands were loaded a sub-iteration earlier (hides LDS latency behind the
    matrix unit). Subtiling is along K (not CTA-sharded), so this is multi-CTA compatible; cluster
    sync is woven around the per-k-tile async_wait like issue_wmma.
    Phase 2: lock-free within-cluster K-split remainder (cluster.barrier reduce).
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4 or NUM_WARPS == 8, "subtile kernel supports NUM_WARPS in {4, 8}")
    SUBTILE_LEN: ttgl.constexpr = 32
    ttgl.static_assert(BLOCK_K % SUBTILE_LEN == 0, "BLOCK_K must be a multiple of 32 (wmma k-dim)")
    NUM_SUBTILES: ttgl.constexpr = BLOCK_K // SUBTILE_LEN

    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)

    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_cgas = num_sms // num_ctas
    num_full_tiles = scheduler.get_num_full_tiles()
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_cgas, num_xcds=8, chunk_size=2)

    num_k_iters = ttgl.cdiv(K, BLOCK_K)
    ttgl.assume(num_k_iters > 0)
    epilogue_lb = num_k_iters - (NUM_BUFFERS - 1)

    for tile_idx in range(pid, num_full_tiles, num_cgas):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        producer = 0
        consumer = 0
        accumulator = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

        issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                     NUM_BUFFERS, TRANSPOSE_B)
        for _ in ttgl.static_range(NUM_BUFFERS - 1):
            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                                   TRANSPOSE_B)

        ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        a_cur, b_cur = lds_subtile_load(consumer, 0, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                        NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
        # Front-load one more tile (matches the general-gemm subtile bookkeeping; predicated off if
        # the tile is already fully prefilled).
        pred0 = ((0 - epilogue_lb) >> 31) & 1
        producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K, NUM_BUFFERS,
                               TRANSPOSE_B, pred=pred0)

        for i in range(0, num_k_iters):
            for s in ttgl.static_range(NUM_SUBTILES):
                if s != NUM_SUBTILES - 1:
                    # Load next sub-iteration's operands, then compute the current one.
                    a_nxt, b_nxt = lds_subtile_load(consumer, (s + 1) * SUBTILE_LEN, a_buffer, OPERAND_LAYOUT_A,
                                                    b_buffer, OPERAND_LAYOUT_B, NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
                    accumulator = ttgl.amd.gfx1250.wmma(a_cur, b_cur, accumulator)
                    a_cur, b_cur = a_nxt, b_nxt
                else:
                    # Last sub-iteration of this k-tile: advance the buffer pipeline, issue the next
                    # tile's TDM load, and prefetch the next k-tile's sub-iter 0.
                    consumer += 1
                    ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
                    if num_ctas > 1:
                        ttgl.amd.gfx1250.cluster.arrive()
                        ttgl.amd.gfx1250.cluster.wait()
                    predi = (((i + 1) - epilogue_lb) >> 31) & 1
                    producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K,
                                           NUM_BUFFERS, TRANSPOSE_B, pred=predi)
                    issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                        TRANSPOSE_B)
                    a_nxt, b_nxt = lds_subtile_load(consumer, 0, a_buffer, OPERAND_LAYOUT_A, b_buffer, OPERAND_LAYOUT_B,
                                                    NUM_BUFFERS, TRANSPOSE_B, SUBTILE_LEN)
                    accumulator = ttgl.amd.gfx1250.wmma(a_cur, b_cur, accumulator)
                    a_cur, b_cur = a_nxt, b_nxt

        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, accumulator, mask=mask_c)

    # ============================================================================
    # Phase 2 (lock-free): within-cluster K-split remainder (cluster.barrier reduce)
    # ============================================================================
    process_remainder_wcsplit(
        a_ptr,
        b_ptr,
        c_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid,
        num_cgas,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ttgl.num_ctas(),
        8,
        STREAMK_TILES,
        LOAD3D,
        WMMA3D,
        C2D,
        GROUP_SIZE_M,
    )


@gluon.jit
def streamk_gemm_tdm_slicemn_subtile_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    NUM_BUFFERS: ttgl.constexpr,
    TRANSPOSE_B: ttgl.constexpr,
    NUM_WARPS: ttgl.constexpr,
    SHARED_LAYOUT_A: ttgl.constexpr,
    SHARED_LAYOUT_B: ttgl.constexpr,
    WMMA_LAYOUT: ttgl.constexpr,
    LOAD3D: ttgl.constexpr,
    WMMA3D: ttgl.constexpr,
    C2D: ttgl.constexpr,
    STREAMK_TILES: ttgl.constexpr,
    GROUP_SIZE_M: ttgl.constexpr = 8,
    L2_PREFETCH_DISTANCE: ttgl.constexpr = 0,
):
    """
    StreamK GEMM: sequential 2x2 acc-quad split + K-direction subtile interleave (combined).
    Each output tile is computed as four HALF_M x HALF_N quadrants one at a time (quad caps the
    live accumulator at 128x128 -> VGPR headroom), and inside each quadrant's K-loop the BLOCK_K
    tile is consumed as NUM_SUBTILES=BLOCK_K//32 sub-iterations with the ds_load for sub-iter s+1
    issued before the WMMA of sub-iter s (latency hiding, now without spilling thanks to the quad
    headroom). Single-CTA only (slices M/N). Phase 2: K-split remainder.
    """
    a_dtype: ttgl.constexpr = a_ptr.type.element_ty
    b_dtype: ttgl.constexpr = b_ptr.type.element_ty
    ttgl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    ttgl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    ttgl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    ttgl.static_assert(NUM_WARPS == 4, "slicemn+subtile kernel is only valid for NUM_WARPS == 4")
    ttgl.static_assert(ttgl.num_ctas() == 1, "slicemn+subtile kernel supports single-CTA only")
    SUBTILE_LEN: ttgl.constexpr = 32
    ttgl.static_assert(BLOCK_K % SUBTILE_LEN == 0, "BLOCK_K must be a multiple of 32 (wmma k-dim)")
    NUM_SUBTILES: ttgl.constexpr = BLOCK_K // SUBTILE_LEN
    HALF_M: ttgl.constexpr = BLOCK_M // 2
    HALF_N: ttgl.constexpr = BLOCK_N // 2

    OPERAND_LAYOUT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_LAYOUT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    a_desc, b_desc = create_tensor_descriptors(a_ptr, b_ptr, 0, 0, stride_am, stride_ak, stride_bn, stride_bk,
                                               SHARED_LAYOUT_A, SHARED_LAYOUT_B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                               TRANSPOSE_B)
    a_buffer = ttgl.allocate_shared_memory(a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=a_desc.layout)
    b_buffer = ttgl.allocate_shared_memory(b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=b_desc.layout)

    scheduler = TileScheduler.initialize(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, STREAMK_TILES)
    pid = scheduler.get_pid()
    num_sms = scheduler.get_num_sms()
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    num_cgas = num_sms // num_ctas
    num_full_tiles = scheduler.get_num_full_tiles()
    pid = scheduler.apply_chiplet_transform_chunked(pid, num_cgas, num_xcds=8, chunk_size=2)

    num_k_iters = ttgl.cdiv(K, BLOCK_K)
    ttgl.assume(num_k_iters > 0)
    epilogue_lb = num_k_iters - (NUM_BUFFERS - 1)

    for tile_idx in range(pid, num_full_tiles, num_cgas):
        pid_m, pid_n = scheduler.get_swizzled_tile_coords(tile_idx, GROUP_SIZE_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N

        for qm in ttgl.static_range(2):
            for qn in ttgl.static_range(2):
                producer = 0
                consumer = 0
                acc = ttgl.zeros((HALF_M, HALF_N), dtype=c_ptr.type.element_ty, layout=WMMA_LAYOUT)

                issue_l2_prefetches_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                                             NUM_BUFFERS, TRANSPOSE_B)
                for _ in ttgl.static_range(NUM_BUFFERS - 1):
                    producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K,
                                           NUM_BUFFERS, TRANSPOSE_B)

                ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
                a_cur, b_cur = slicemn_subtile_load(a_buffer.index(consumer % NUM_BUFFERS),
                                                    b_buffer.index(consumer % NUM_BUFFERS), OPERAND_LAYOUT_A,
                                                    OPERAND_LAYOUT_B, qm * HALF_M, qn * HALF_N, 0, HALF_M, HALF_N,
                                                    SUBTILE_LEN, TRANSPOSE_B)
                pred0 = ((0 - epilogue_lb) >> 31) & 1
                producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K,
                                       NUM_BUFFERS, TRANSPOSE_B, pred=pred0)

                for i in range(0, num_k_iters):
                    for s in ttgl.static_range(NUM_SUBTILES):
                        if s != NUM_SUBTILES - 1:
                            a_nxt, b_nxt = slicemn_subtile_load(a_buffer.index(consumer % NUM_BUFFERS),
                                                                b_buffer.index(consumer % NUM_BUFFERS),
                                                                OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, qm * HALF_M,
                                                                qn * HALF_N, (s + 1) * SUBTILE_LEN, HALF_M, HALF_N,
                                                                SUBTILE_LEN, TRANSPOSE_B)
                            acc = ttgl.amd.gfx1250.wmma(a_cur, b_cur, acc)
                            a_cur, b_cur = a_nxt, b_nxt
                        else:
                            consumer += 1
                            ttgl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
                            predi = (((i + 1) - epilogue_lb) >> 31) & 1
                            producer = issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer,
                                                   BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, pred=predi)
                            issue_l2_prefetches(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc, off_am, off_bn,
                                                BLOCK_K, TRANSPOSE_B)
                            a_nxt, b_nxt = slicemn_subtile_load(a_buffer.index(consumer % NUM_BUFFERS),
                                                                b_buffer.index(consumer % NUM_BUFFERS),
                                                                OPERAND_LAYOUT_A, OPERAND_LAYOUT_B, qm * HALF_M,
                                                                qn * HALF_N, 0, HALF_M, HALF_N, SUBTILE_LEN,
                                                                TRANSPOSE_B)
                            acc = ttgl.amd.gfx1250.wmma(a_cur, b_cur, acc)
                            a_cur, b_cur = a_nxt, b_nxt

                offs_m = pid_m * BLOCK_M + qm * HALF_M + ttgl.arange(0, HALF_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
                offs_n = pid_n * BLOCK_N + qn * HALF_N + ttgl.arange(0, HALF_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
                offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
                ttgl.store(c_ptr + offs_c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # ============================================================================
    # Phase 2 (lock-free): within-cluster K-split remainder (cluster.barrier reduce)
    # ============================================================================
    process_remainder_wcsplit(
        a_ptr,
        b_ptr,
        c_ptr,
        p_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        pid,
        num_cgas,
        num_full_tiles,
        scheduler,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        ttgl.num_ctas(),
        8,
        STREAMK_TILES,
        LOAD3D,
        WMMA3D,
        C2D,
        GROUP_SIZE_M,
    )


def _build_remainder_layouts(BLOCK_M, BLOCK_N, num_ctas, num_warps):
    """3D batched + 2D broadcast layouts for the within-cluster K-split remainder, sized for
    num_warps (warps distributed N=2, M=num_warps//2, matching the cooperative warp pattern)."""
    cga3d = make_cga_layout([num_ctas, 1, 1], [num_ctas, 1, 1], [2, 1, 0])
    cga2d = make_cga_layout([num_ctas, 1], [1, 1], [0, 1])
    n_warps = 2 if num_warps >= 2 else 1
    m_warps = num_warps // n_warps
    tM, tN = 8, 4  # threads per warp (product 32)
    sM = BLOCK_M // (m_warps * tM)
    sN = BLOCK_N // (n_warps * tN)
    load3d = ttgl.BlockedLayout([1, sM, sN], [1, tM, tN], [1, m_warps, n_warps], [2, 1, 0], cga3d)
    c2d = ttgl.BlockedLayout([sM, sN], [tM, tN], [m_warps, n_warps], [1, 0], cga2d)
    wb = [[0, 0, 1]]
    for i in range(int(math.log2(num_warps // 2))):
        wb.append([0, 1 << i, 0])
    wmma3d = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=wb, reg_bases=[], instr_shape=[16, 16, 32],
                                    cga_layout=cga3d, rank=3)
    return load3d, wmma3d, c2d


def run_streamk_gemm_tdm_pipelined(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
    num_warps,
    use_prefetch=False,
    use_slicemn=False,
    use_subtile=False,
    disable_streamk=False,
    ctas_per_cga=[1, 1],
    num_sms=None,
    waves_per_eu=None,
    l2_prefetch_distance=0,
):
    """Helper function for StreamK GEMM kernel testing.

    Args:
        use_prefetch: If True, use the prefetch kernel variant with prologue-epilogue overlap.
        disable_streamk: If True, set STREAMK_TILES=0 (pure persistent mode, no K-splitting).
        num_sms: number of StreamK workers (CGAs). None => auto = device CUs // num_ctas, so
            total CTAs (num_sms * num_ctas) fill the device. Override for sim runs whose modeled
            CU count differs from what the driver reports.
    """
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        print(f"Skipping: K/BLOCK_K ({triton.cdiv(K, BLOCK_K)}) < NUM_BUFFERS ({NUM_BUFFERS})")
        return

    # WG cluster (multi-CTA): scale the per-CTA block up to the CGA-collective block
    # and build cga-aware layouts on the host (mirrors the gemm reference path). The
    # per-CTA tile stays the same; the cluster tiles num_ctas of them together.
    num_ctas = ctas_per_cga[0] * ctas_per_cga[1]
    BLOCK_M = BLOCK_M * ctas_per_cga[0]
    BLOCK_N = BLOCK_N * ctas_per_cga[1]

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    cga_layout_c = make_cga_layout(ctas_per_cga, [ctas_per_cga[0], ctas_per_cga[1]], [0, 1])
    # Derive operand cga layouts by projecting the C cga layout onto each operand, so the
    # shared-memory layouts carry the SAME CTAs-per-CGA as the WMMA layout (mirrors the
    # gemm reference _build_gemm_layouts). Passing the cga args straight through would leave
    # the shared layout at 1 CTA/CGA while WMMA expects num_ctas -> layout mismatch.
    WMMA_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32], cga_layout_c)
    cga_a = ttgl.DotOperandLayout(0, WMMA_LAYOUT, 8).cga_layout
    cga_b = ttgl.DotOperandLayout(1, WMMA_LAYOUT, 8).cga_layout
    if TRANSPOSE_B:
        cga_b_t = tuple([tuple([row[1], row[0]]) for row in cga_b])
    else:
        cga_b_t = cga_b
    SHARED_LAYOUT_A = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0], cga_a)
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0],
                                                                    cga_b_t)
    else:
        SHARED_LAYOUT_B = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0], cga_b_t)

    # 3D layouts for the lock-free within-cluster K-split remainder (batch dim = CGA-split).
    # Large tiles are quad-split into 2x2 sub-blocks (caps spills), so size the layouts for the
    # sub-block; must match the QUAD condition in process_remainder_wcsplit.
    quad = BLOCK_M >= 128 and BLOCK_N >= 128
    rem_m, rem_n = (BLOCK_M // 2, BLOCK_N // 2) if quad else (BLOCK_M, BLOCK_N)
    load3d, wmma3d, c2d = _build_remainder_layouts(rem_m, rem_n, num_ctas, num_warps)

    # StreamK workers (CGAs). Auto: fill the device so total CTAs (num_sms * num_ctas) == #CUs.
    if num_sms is None:
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        num_sms = num_cus // num_ctas

    # Calculate STREAMK_TILES automatically (remainder tiles for load balancing)
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    STREAMK_TILES = 0 if disable_streamk else (total_tiles % num_sms)

    torch.manual_seed(42)

    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = ((b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0)))
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Use persistent grid (StreamK uses persistent kernel infrastructure)
    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)

    # Allocate scratch: per-CGA region of num_ctas slabs for the within-cluster reduce.
    p = torch.zeros(num_sms * num_ctas * BLOCK_M * BLOCK_N, dtype=torch.float32)

    a_device = a.cuda()
    b_device = b.cuda()
    c_device = c.cuda()
    p_device = p.cuda()

    if use_subtile and use_slicemn:
        kernel_name = "slicemn+subtile-4warps"
    elif use_subtile:
        kernel_name = f"subtile-{num_warps}warps"
    elif num_warps == 8:
        kernel_name = "pipelined-8warps"
    elif use_slicemn:
        kernel_name = "slicemn-4warps"
    elif use_prefetch:
        kernel_name = "prefetch-4warps"
    else:
        kernel_name = "pipelined-4warps"
    print(f"\nTesting StreamK {kernel_name} kernel with STREAMK_TILES={STREAMK_TILES}")
    print(f"Grid: {grid}, Total tiles: {total_tiles}")

    # Select kernel. All paths use the lock-free within-cluster K-split remainder
    # (cluster.barrier reduce) + host-built cga-aware layouts.
    args = (a_device, b_device, c_device, p_device, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn)
    common = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_BUFFERS=NUM_BUFFERS, TRANSPOSE_B=TRANSPOSE_B,
                  NUM_WARPS=num_warps, SHARED_LAYOUT_A=SHARED_LAYOUT_A, SHARED_LAYOUT_B=SHARED_LAYOUT_B,
                  WMMA_LAYOUT=WMMA_LAYOUT, LOAD3D=load3d, WMMA3D=wmma3d, C2D=c2d, STREAMK_TILES=STREAMK_TILES,
                  L2_PREFETCH_DISTANCE=l2_prefetch_distance, num_warps=num_warps, num_ctas=num_ctas,
                  waves_per_eu=(waves_per_eu if waves_per_eu is not None else num_warps // 4))
    if use_subtile and use_slicemn:
        kernel_fn = streamk_gemm_tdm_slicemn_subtile_kernel
    elif use_subtile:
        kernel_fn = streamk_gemm_tdm_subtile_kernel
    elif use_slicemn:
        kernel_fn = streamk_gemm_tdm_slicemn_kernel
    elif use_prefetch:
        kernel_fn = streamk_gemm_tdm_prefetch_kernel
    else:
        kernel_fn = streamk_gemm_tdm_pipelined_kernel
    kernel = kernel_fn[grid](*args, **common)
    static_profile(kernel)

    c_triton = c_device.cpu()
    c_torch = a.to(torch.float32) @ (b.to(torch.float32) if not TRANSPOSE_B else b.T.to(torch.float32))
    # fp16 inputs accumulated in fp32 chunk-by-chunk: differs from a single fp32 matmul only by
    # summation-order rounding, which grows with K. At large K (e.g. 4096) the worst near-zero
    # (cancellation) elements reach ~1.5-2.4e-4 even vs an fp64 reference -- the fp32-accumulation
    # floor, identical for the general gemm kernel. 1e-4 only holds at small K; use 1e-3.
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-3, atol=1e-3)
    print(f"✓ StreamK {kernel_name} kernel test passed!")


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [2, 4])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (258, 258, 510)])
@pytest.mark.parametrize("variant", ["pipelined", "prefetch", "slicemn", "subtile"])
@pytest.mark.parametrize("ctas_per_cga", [[1, 1], [2, 1], [2, 2]])
def test_streamk_gemm_tdm_4warps(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
    variant,
    ctas_per_cga,
):
    """Test 4-warp StreamK GEMM kernel (pipelined, prefetch, and sliceMN variants), single- and
    multi-CTA (WG cluster) configurations."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")
    if variant == "slicemn" and ctas_per_cga != [1, 1]:
        pytest.skip("sliceMN variant is single-CTA only (slicing CTA-distributed dims is unsupported)")

    run_streamk_gemm_tdm_pipelined(
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, num_warps=4, use_prefetch=(variant == "prefetch"),
        use_slicemn=(variant == "slicemn"), use_subtile=(variant == "subtile"), ctas_per_cga=ctas_per_cga,
        num_sms=8,  # fixed small grid: keep the functional-sim suite fast (correctness, not perf)
    )


@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("NUM_BUFFERS", [3])
@pytest.mark.parametrize("TRANSPOSE_B", [False, True])
@pytest.mark.parametrize("M,N,K", [(256, 256, 512), (258, 258, 510)])
@pytest.mark.parametrize("ctas_per_cga", [[1, 1], [2, 1], [2, 2]])
def test_streamk_gemm_tdm_8warps(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_BUFFERS,
    TRANSPOSE_B,
    M,
    N,
    K,
    ctas_per_cga,
):
    """Test 8-warp StreamK GEMM kernel, single- and multi-CTA (WG cluster) configurations."""
    if triton.cdiv(K, BLOCK_K) < NUM_BUFFERS:
        pytest.skip("Skip tests where K/BLOCK_K < NUM_BUFFERS")

    run_streamk_gemm_tdm_pipelined(
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, TRANSPOSE_B, M, N, K, num_warps=8, ctas_per_cga=ctas_per_cga,
        num_sms=8,  # fixed small grid: keep the functional-sim suite fast (correctness, not perf)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StreamK GEMM kernel test - automatically calculates StreamK tiles for load balancing",
        epilog="Example: python3 f16_sk_gemm_gfx1250.py -M 258 -N 258 -K 510",
    )
    parser.add_argument("-M", type=int, default=1028, help="problem M size (default: 258)")
    parser.add_argument("-N", type=int, default=1028, help="problem N size (default: 258)")
    parser.add_argument("-K", type=int, default=1024, help="problem K size (default: 510)")
    parser.add_argument("--block-m", type=int, default=256, help="BLOCK_M tile size (default: 32)")
    parser.add_argument("--block-n", type=int, default=256, help="BLOCK_N tile size (default: 32)")
    parser.add_argument("--block-k", type=int, default=128, help="BLOCK_K tile size (default: 128)")
    parser.add_argument(
        "--num-warps",
        type=int,
        choices=[4, 8],
        default=4,
        help="num warps (default: 4)",
    )
    parser.add_argument(
        "--num-buffers",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="num shared memory buffers (default: 2, use 3 for 8-warp warp-pipelining)",
    )
    parser.add_argument("--num-sms", type=int, default=None,
                        help="number of StreamK workers (CGAs); default: auto = device CUs // num_ctas")
    parser.add_argument("--waves-per-eu", type=int, default=None,
                        help="target waves/SIMD (occupancy hint); default: num_warps // 4")
    parser.add_argument("--l2-prefetch-distance", type=int, default=0,
                        help="L2 prefetch distance (tdm.prefetch); 0 disables (default: 0)")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Use prefetch kernel variant with prologue-epilogue overlap",
    )
    parser.add_argument(
        "--slicemn",
        action="store_true",
        help="Use sliceMN kernel variant (2x2 operand-sliced region schedule, 4-warp only)",
    )
    parser.add_argument(
        "--subtile",
        action="store_true",
        help="Use K-direction subtile manual-interleave kernel variant",
    )
    parser.add_argument(
        "--disable-streamk",
        action="store_true",
        help="Disable StreamK (STREAMK_TILES=0), use pure persistent mode",
    )
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    NUM_BUFFERS = args.num_buffers
    NUM_WARPS = args.num_warps
    TRANSPOSE_B = True
    # __main__ uses single-CTA (num_ctas=1), so auto num_sms == device CU count.
    NUM_SMS = args.num_sms if args.num_sms is not None else \
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    # Calculate tile dimensions for display
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    streamk_tiles = 0 if args.disable_streamk else (total_tiles % NUM_SMS)
    mode = ("Persistent (StreamK disabled)"
            if args.disable_streamk else f"StreamK with {streamk_tiles} remainder tiles")

    STREAMK = not args.disable_streamk
    print(f"Mode: {mode} (out of {total_tiles} total tiles)")
    print(
        f"({M=}, {N=}, {K=}), ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}), {TRANSPOSE_B=}, {NUM_WARPS=}, {NUM_BUFFERS=}, PERSISTENT=True, {STREAMK=}, PREFETCH={args.prefetch}"
    )

    # Run StreamK kernel test
    run_streamk_gemm_tdm_pipelined(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BUFFERS,
        TRANSPOSE_B,
        M,
        N,
        K,
        NUM_WARPS,
        use_prefetch=args.prefetch,
        use_slicemn=args.slicemn,
        use_subtile=args.subtile,
        disable_streamk=args.disable_streamk,
        num_sms=args.num_sms,
        waves_per_eu=args.waves_per_eu,
        l2_prefetch_distance=args.l2_prefetch_distance,
    )
