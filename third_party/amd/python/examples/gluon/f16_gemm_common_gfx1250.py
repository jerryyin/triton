# ruff: noqa: E402
"""
Common utilities for GFX1250 GEMM kernels.

This module contains shared functions, classes, and utilities used by both
persistent and StreamK GEMM implementations.
"""

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


@gluon.jit
def chiplet_transform(pid, num_workgroups, num_xcds: ttgl.constexpr):
    """
    Basic chiplet transformation for multi-XCD AMD GPUs.

    Transforms program ID to distribute work evenly across chiplets (XCDs).
    Each XCD gets a contiguous range of work items.
    """
    xcd = pid % num_xcds
    pos_in_xcd = pid // num_xcds
    min_per_xcd = num_workgroups // num_xcds
    extra_sms = num_workgroups % num_xcds
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd


@gluon.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
    """
    Chunked chiplet transformation for improved memory locality.

    Groups work items into chunks of size `chunk_size` per XCD, ensuring
    adjacent work items within a chunk are on the same chiplet for better
    cache utilization and memory bandwidth.
    """
    # Simplified to reduce SGPR temporaries
    if pid >= (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return (local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size)


@gluon.jit
def remap_xcd_chunked(pid, grid_mn, num_xcds: ttgl.constexpr = 8, chunk_size: ttgl.constexpr = 2):
    """
    XCD remapping with chunked distribution (alternative implementation).
    Similar to chiplet_transform_chunked but with default parameters
    """
    # Compute current XCD and local PID
    xcd = pid % num_xcds
    # Distribute the modulo pids in round robin
    if pid > (grid_mn // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid
    local_pid = pid // num_xcds
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    # Calculate new PID
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@gluon.constexpr_function
def create_shared_layouts(BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                          TRANSPOSE_B: ttgl.constexpr, cga_layout_a=[], cga_layout_b=[]):

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0], cga_layout_a)
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0], cga_layout_b)
    else:
        cga_layout_b = tuple([tuple([row[1], row[0]]) for row in cga_layout_b])
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0], cga_layout_b)

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


def build_gemm_layouts(BLOCK_M, BLOCK_N, BLOCK_K, cga_layout_a, cga_layout_b, cga_layout_c, WARP_BASES, TRANSPOSE_B):
    """
    Build all layouts for the GEMM kernel.
    """
    # If TRANSPOSE_B we need to transpose each basis vector of the CGALayout for the
    # shared allocation because the permute will transpose the basis vectors before we
    # load them for wmmas.
    if TRANSPOSE_B:
        # Transpose each basis vector: [a, b] -> [b, a]
        cga_layout_b_transposed = tuple([tuple([row[1], row[0]]) for row in cga_layout_b])
    else:
        cga_layout_b_transposed = cga_layout_b

    SHARED_LAYOUT_A: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                [1, 0], cga_layout_a)
    if not TRANSPOSE_B:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 16]], [BLOCK_K, BLOCK_N],
                                                                                    [1, 0], cga_layout_b_transposed)
    else:
        SHARED_LAYOUT_B: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                    [1, 0], cga_layout_b_transposed)

    WMMA_LAYOUT_A = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_a)
    WMMA_LAYOUT_B = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_b)
    ACCUMULATOR_LAYOUT = ttgl.amd.AMDWMMALayout(3, True, WARP_BASES, [], [16, 16, 32], cga_layout_c)
    OPERAND_LAYOUT_A = ttgl.DotOperandLayout(0, WMMA_LAYOUT_A, 8)
    OPERAND_LAYOUT_B = ttgl.DotOperandLayout(1, WMMA_LAYOUT_B, 8)

    return SHARED_LAYOUT_A, SHARED_LAYOUT_B, ACCUMULATOR_LAYOUT, OPERAND_LAYOUT_A, OPERAND_LAYOUT_B


@gluon.jit
def create_tensor_descriptors(a_ptr, b_ptr, off_am, off_bn, stride_am, stride_ak, stride_bn, stride_bk,
                              shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr, M: ttgl.constexpr,
                              N: ttgl.constexpr, K: ttgl.constexpr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                              BLOCK_K: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr + off_am, shape=(M, K),
                                                         strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K),
                                                         layout=shared_layout_a)
    if not TRANSPOSE_B:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(K, N),
                                                             strides=(stride_bk, stride_bn),
                                                             block_shape=(BLOCK_K, BLOCK_N), layout=shared_layout_b)
    else:
        b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr + off_bn, shape=(N, K),
                                                             strides=(stride_bn, stride_bk),
                                                             block_shape=(BLOCK_N, BLOCK_K), layout=shared_layout_b)

    return a_desc, b_desc


@gluon.jit
def issue_l2_prefetches(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: ttgl.constexpr,
                        TRANSPOSE_B: ttgl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `producer + distance`.
    """
    if distance <= 0:
        return

    prefetch_iteration = producer + distance
    ttgl.amd.gfx1250.tdm.prefetch(a_desc, [off_am, prefetch_iteration * BLOCK_K], pred=pred)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_K, off_bn], pred=pred)
    else:
        ttgl.amd.gfx1250.tdm.prefetch(b_desc, [off_bn, prefetch_iteration * BLOCK_K], pred=pred)


@gluon.jit
def issue_l2_prefetches_prologue(distance, producer, a_desc, b_desc, off_am, off_bn, BLOCK_K: ttgl.constexpr,
                                 NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=True):
    """
    Creates prefetches for iterations [NUM_BUFFERS, NUM_BUFFERS + distance).
    This skips iterations which are preloaded in the prologue because prefetching them does not make sense for GEMMs.
    """
    for i in ttgl.static_range(NUM_BUFFERS, NUM_BUFFERS + distance):
        issue_l2_prefetches(i, producer, a_desc, b_desc, 0, 0, BLOCK_K, TRANSPOSE_B, pred)


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_am, off_bn, a_buffer, b_buffer, BLOCK_K: ttgl.constexpr,
                NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, pred=1):
    # pred is a hardware predicate passed to async_load for conditional execution without branch divergence
    # Convert boolean pred to i32 for hardware predicate (i1 -> i32)
    pred_i32 = pred.to(ttgl.int32) if hasattr(pred, 'to') else pred
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_am, producer * BLOCK_K], a_buffer.index(producer % NUM_BUFFERS),
                                    pred=pred_i32)
    if not TRANSPOSE_B:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [producer * BLOCK_K, off_bn], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_bn, producer * BLOCK_K], b_buffer.index(producer % NUM_BUFFERS),
                                        pred=pred_i32)
    producer += 1
    return producer


@gluon.jit
def issue_wmma(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr, accumulator,
               wait_producers_cnt, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """
    For multi-CTA configurations, we want warps within the CGA (cluster) to stay temporally aligned so we can
    multicast data to multiple CTAs.
    We do this by draining this CTA's async loads (`async_wait`) first so the data is ready, then signaling the
    cluster barrier (`arrive`) and waiting for the cluster (`wait`). This keeps warps of a CGA within one
    iteration of each other. The `async_wait` must not sit between `arrive` and `wait`: a CTA that signals
    arrival and then stalls on its load counter inside the barrier window can hang on real HW.
    """
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    ttgl.amd.gfx1250.tdm.async_wait(wait_producers_cnt)

    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.arrive()
        ttgl.amd.gfx1250.cluster.wait()

    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    consumer += 1
    return consumer, accumulator


@gluon.jit
def slicemn_wait(wait_cnt):
    """Cluster-gated TDM wait for the sliceMN region schedule.

    Mirrors the sync in issue_wmma: drain this CTA's async loads first, then arrive at
    and wait on the cluster barrier, so warps of a CGA stay within one iteration of each
    other (multicast alignment). The async_wait must not sit between arrive and wait or a
    CTA can hang on real HW. Single-CTA runs emit no cluster ops."""
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    ttgl.amd.gfx1250.tdm.async_wait(wait_cnt)
    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.arrive()
        ttgl.amd.gfx1250.cluster.wait()


@gluon.jit
def slicemn_load_a(slot_buffer, start, a_layout: ttgl.constexpr, HALF_M: ttgl.constexpr):
    """Load one A sub-tile (HALF_M rows) from a resident LDS slot via .slice() along the M dim."""
    return slot_buffer.slice(start, HALF_M, 0).load(layout=a_layout)


@gluon.jit
def slicemn_load_b(slot_buffer, start, b_layout: ttgl.constexpr, HALF_N: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """Load one B sub-tile (HALF_N cols) from a resident LDS slot. Non-transposed B slices the
    N dim (1); transposed B slices the stored-N dim (0) then permutes, mirroring lds_subtile_load."""
    if not TRANSPOSE_B:
        return slot_buffer.slice(start, HALF_N, 1).load(layout=b_layout)
    else:
        return slot_buffer.slice(start, HALF_N, 0).permute([1, 0]).load(layout=b_layout)


@gluon.jit
def slicemn_subtile_load(sa, sb, a_layout: ttgl.constexpr, b_layout: ttgl.constexpr, QM_OFF: ttgl.constexpr,
                         QN_OFF: ttgl.constexpr, K_OFF: ttgl.constexpr, HALF_M: ttgl.constexpr, HALF_N: ttgl.constexpr,
                         SUBTILE_LEN: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """Load one quadrant's K-sub-slice from a resident LDS slot: A rows [QM_OFF:+HALF_M] x K
    [K_OFF:+SUBTILE_LEN], B K [K_OFF:+SUBTILE_LEN] x N cols [QN_OFF:+HALF_N]. Double slice (M/N
    quadrant + K subtile). Single-CTA only (slices M/N). sa/sb are a_buffer.index(slot) etc."""
    a = sa.slice(QM_OFF, HALF_M, 0).slice(K_OFF, SUBTILE_LEN, 1).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = sb.slice(K_OFF, SUBTILE_LEN, 0).slice(QN_OFF, HALF_N, 1).load(layout=b_layout)
    else:
        b = sb.slice(QN_OFF, HALF_N, 0).slice(K_OFF, SUBTILE_LEN, 1).permute([1, 0]).load(layout=b_layout)
    return a, b


@gluon.jit
def slicemn_quad_consume(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr, acc,
                         wait_cnt, NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, QM_OFF: ttgl.constexpr,
                         QN_OFF: ttgl.constexpr, HALF_M: ttgl.constexpr, HALF_N: ttgl.constexpr):
    """One k-iteration of ONE accumulator quadrant: wait, load this quadrant's A/B sub-slices
    (rows [QM_OFF:QM_OFF+HALF_M], cols [QN_OFF:QN_OFF+HALF_N]) from the ready buffer slot, and
    accumulate into the single HALF_M x HALF_N accumulator.

    Sequential-quad processing keeps only ONE quadrant accumulator live at a time. The LLVM
    backend allocates that far better than one monolithic BLOCK_M x BLOCK_N accumulator, whose
    huge live range pushes it into the register-spill regime -- this is the VGPR-reduction
    purpose of quad splitting (mirrors the general-gemm quad at f16_gemm_gfx1250.py:1308 and the
    remainder's wc_ksplit_tile_quad)."""
    slicemn_wait(wait_cnt)
    sa = a_buffer.index(consumer % NUM_BUFFERS)
    sb = b_buffer.index(consumer % NUM_BUFFERS)
    a = slicemn_load_a(sa, QM_OFF, a_layout, HALF_M)
    b = slicemn_load_b(sb, QN_OFF, b_layout, HALF_N, TRANSPOSE_B)
    acc = ttgl.amd.gfx1250.wmma(a, b, acc)
    return consumer + 1, acc


@gluon.jit
def lds_subtile_load(consumer, start, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr,
                     NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr, SUBTILE_LEN: ttgl.constexpr):

    index = consumer % NUM_BUFFERS
    a = a_buffer.index(index).slice(start, SUBTILE_LEN, 1).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 0).load(layout=b_layout)
    else:
        b = b_buffer.index(index).slice(start, SUBTILE_LEN, 1).permute([1, 0]).load(layout=b_layout)

    return a, b


@gluon.jit
def lds_load(consumer, a_buffer, a_layout: ttgl.constexpr, b_buffer, b_layout: ttgl.constexpr,
             NUM_BUFFERS: ttgl.constexpr, TRANSPOSE_B: ttgl.constexpr):
    """Load A and B tiles from shared memory (LDS) into registers."""
    a = a_buffer.index(consumer % NUM_BUFFERS).load(layout=a_layout)
    if not TRANSPOSE_B:
        b = b_buffer.index(consumer % NUM_BUFFERS).load(layout=b_layout)
    else:
        b = b_buffer.index(consumer % NUM_BUFFERS).permute([1, 0]).load(layout=b_layout)

    consumer += 1
    return consumer, a, b


@gluon.jit
def issue_wmma_compute(a, b, accumulator):
    """Perform WMMA computation on pre-loaded operands."""
    accumulator = ttgl.amd.gfx1250.wmma(a, b, accumulator)
    return accumulator


@gluon.jit
def swiglu_epilogue(acc):
    """Apply SwiGLU: reshape (M, 2N) -> split into gate/up -> swish(gate) * up."""
    BLOCK_M: ttgl.constexpr = acc.shape[0]
    BLOCK_N: ttgl.constexpr = acc.shape[1] // 2
    acc_3d = ttgl.reshape(acc, (BLOCK_M, BLOCK_N, 2))
    gate, up = ttgl.split(acc_3d)
    # swish(x) = x * sigmoid(x); sigmoid(x) = 1 / (1 + exp(-x))
    return gate * (1.0 / (1.0 + ttgl.exp(-gate))) * up


@gluon.jit
def apply_activation_epilogue(acc, ACTIVATION: ttgl.constexpr, ACC_LAYOUT: ttgl.constexpr):
    if ACTIVATION == "swiglu":
        result = ttgl.convert_layout(swiglu_epilogue(acc), ACC_LAYOUT)
    else:
        result = acc
    return result


@gluon.aggregate
class TileScheduler:
    """
    Tile Scheduler

    Stores essential tile scheduling state. Values like iters_per_tile
    are recomputed when needed to reduce live register pressure.

    Stored fields (4 SGPRs total):
    - num_pid_m: Number of tiles in M dimension
    - num_pid_n: Number of tiles in N dimension
    - total_full_tiles: Number of tiles processed in persistent mode
    - num_streamk_tiles: Number of tiles for StreamK processing
    """
    num_pid_m: ttgl.tensor
    num_pid_n: ttgl.tensor
    total_full_tiles: ttgl.tensor
    num_streamk_tiles: ttgl.tensor

    @gluon.constexpr_function
    def __init__(self, num_pid_m, num_pid_n, total_full_tiles, num_streamk_tiles):
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.total_full_tiles = total_full_tiles
        self.num_streamk_tiles = num_streamk_tiles

    @gluon.jit
    def initialize(M, N, K, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                   STREAMK_TILES: ttgl.constexpr):
        """Initialize scheduler - stores essential state for tile scheduling."""
        num_pid_m = ttgl.cdiv(M, BLOCK_M)
        num_pid_n = ttgl.cdiv(N, BLOCK_N)
        total_tiles = num_pid_m * num_pid_n
        total_full_tiles = total_tiles - STREAMK_TILES
        # Convert constexpr to tensor for storage
        num_streamk_tiles = total_tiles - total_full_tiles
        return TileScheduler(num_pid_m, num_pid_n, total_full_tiles, num_streamk_tiles)

    @gluon.jit
    def get_num_tiles(self):
        """Return total number of full tiles for persistent loop."""
        return self.total_full_tiles

    @gluon.jit
    def get_num_full_tiles(self):
        return self.total_full_tiles

    @gluon.jit
    def get_num_streamk_tiles(self):
        return self.num_streamk_tiles

    @gluon.jit
    def get_pid(self):
        """Return current program ID."""
        return ttgl.program_id(axis=0)

    @gluon.jit
    def get_num_sms(self):
        """Return total number of SMs/CUs available."""
        return ttgl.num_programs(axis=0)

    @gluon.jit
    def get_swizzled_tile_coords(self, tile_id, GROUP_SIZE_M: ttgl.constexpr):
        """Get swizzled tile coordinates using stored num_pid_m and num_pid_n."""
        num_pid_in_group = GROUP_SIZE_M * self.num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = ttgl.minimum(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @gluon.jit
    def apply_chiplet_transform(self, pid, num_sms, num_xcds: ttgl.constexpr):
        """Apply basic chiplet transformation to a program ID."""
        return chiplet_transform(pid, num_sms, num_xcds)

    @gluon.jit
    def apply_chiplet_transform_chunked(self, pid, num_sms, num_xcds: ttgl.constexpr, chunk_size: ttgl.constexpr):
        """Apply chunked chiplet transformation for improved cache locality."""
        return chiplet_transform_chunked(pid, num_sms, num_xcds, chunk_size)


@gluon.jit
def wc_ksplit_tile(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, off_m, off_n,  #
                   M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                   BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr, NUM_CTAS: ttgl.constexpr,
                   K_WIDTH: ttgl.constexpr, load3d: ttgl.constexpr, wmma3d: ttgl.constexpr, c2d: ttgl.constexpr):
    """Compute one output tile [off_m:off_m+BLOCK_M, off_n:off_n+BLOCK_N] via within-cluster
    K-split + lock-free reduce. The cluster's NUM_CTAS CTAs each take a K-slice and compute a
    partial through a rank-3 batched dot (batch dim = CGA-split, one slab per CTA), store the
    partials to this CGA's scratch region, sync once, then every CTA sums all slabs (multicast
    loads off the broadcast c2d layout) and one CTA writes C. Degenerates to NUM_CTAS==1."""
    # 3D layouts -> per-axis 1D layouts (batch / dim1 / dim2 for loads and wmma; M/N for c2d).
    lb: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.SliceLayout(2, load3d))
    l1: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(2, load3d))
    l2: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(1, load3d))
    wb: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.SliceLayout(2, wmma3d))
    w1: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(2, wmma3d))
    w2: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(1, wmma3d))
    cm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, c2d))
    cn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, c2d))

    # Even K-iteration split across the cluster's CTAs; ragged tail handled by masking.
    total_iters = ttgl.cdiv(K, BLOCK_K)
    base = total_iters // NUM_CTAS
    rem = total_iters % NUM_CTAS
    cta = ttgl.arange(0, NUM_CTAS, layout=lb)
    k_start = cta * base + ttgl.minimum(cta, rem)
    k_end = k_start + base + (cta < rem).to(ttgl.int32)
    om = ttgl.arange(0, BLOCK_M, layout=l1)
    ak = ttgl.arange(0, BLOCK_K, layout=l2)
    bk = ttgl.arange(0, BLOCK_K, layout=l1)
    on = ttgl.arange(0, BLOCK_N, layout=l2)

    # Stage 1: each CTA computes its K-slice partial. Loop the max slice length across CTAs
    # CTAs with a shorter slice mask off their tail via `kit < k_end`.
    max_local_iters = ttgl.cdiv(total_iters, NUM_CTAS)
    acc = ttgl.zeros((NUM_CTAS, BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=wmma3d)
    for ki in range(max_local_iters):
        kit = k_start[:, None, None] + ki
        ka = kit * BLOCK_K + ak[None, None, :]
        offs_a = stride_am * (off_m + om[None, :, None]) + stride_ak * ka
        mask_a = (kit < k_end[:, None, None]) & (ka < K) & ((off_m + om[None, :, None]) < M)
        at = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
        kb = kit * BLOCK_K + bk[None, :, None]
        offs_b = stride_bk * kb + stride_bn * (off_n + on[None, None, :])
        mask_b = (kit < k_end[:, None, None]) & (kb < K) & ((off_n + on[None, None, :]) < N)
        bt = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
        at = ttgl.convert_layout(at, ttgl.DotOperandLayout(0, wmma3d, K_WIDTH))
        bt = ttgl.convert_layout(bt, ttgl.DotOperandLayout(1, wmma3d, K_WIDTH))
        acc = ttgl.amd.gfx1250.wmma(at, bt, acc)

    sb = ttgl.arange(0, NUM_CTAS, layout=wb)
    sm = ttgl.arange(0, BLOCK_M, layout=w1)
    sn = ttgl.arange(0, BLOCK_N, layout=w2)
    soff = scr_cga + sb[:, None, None] * (BLOCK_M * BLOCK_N) + sm[None, :, None] * BLOCK_N + sn[None, None, :]
    ttgl.store(scr_ptr + soff, acc)
    ttgl.barrier()  # CTA-level: scratch round-trip needs cross-warp visibility even at NUM_CTAS==1
    if NUM_CTAS > 1:
        ttgl.amd.gfx1250.cluster.arrive()
        ttgl.amd.gfx1250.cluster.wait()

    # Stage 2: every CTA sums all slabs (multicast loads), one CTA stores C.
    red = ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=c2d)
    for cc in ttgl.static_range(NUM_CTAS):
        roff = scr_cga + cc * (BLOCK_M * BLOCK_N) + cm[:, None] * BLOCK_N + cn[None, :]
        red += ttgl.load(scr_ptr + roff)
    offs_c = stride_cm * (off_m + cm[:, None]) + stride_cn * (off_n + cn[None, :])
    mask_c = ((off_m + cm[:, None]) < M) & ((off_n + cn[None, :]) < N)
    ttgl.store(c_ptr + offs_c, red, mask=mask_c)
    ttgl.barrier()  # CTA-level: scratch round-trip needs cross-warp visibility even at NUM_CTAS==1
    if NUM_CTAS > 1:
        ttgl.amd.gfx1250.cluster.arrive()
        ttgl.amd.gfx1250.cluster.wait()


@gluon.jit
def wc_ksplit_tile_quad(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, off_m, off_n,  #
                        M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                        BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                        NUM_CTAS: ttgl.constexpr, K_WIDTH: ttgl.constexpr, load3d: ttgl.constexpr,
                        wmma3d: ttgl.constexpr, c2d: ttgl.constexpr):
    """Quad-split wrapper for wc_ksplit_tile: process the output tile as 2x2 HALF_M x HALF_N
    sub-blocks to cap register pressure (the full-tile acc/reduce buffers are the dominant VGPR
    cost at large tiles). load3d/wmma3d/c2d MUST be sized for HALF_M x HALF_N. The scr_cga
    region is reused across sub-blocks; the trailing cluster barrier in each wc_ksplit_tile
    makes that reuse safe."""
    HALF_M: ttgl.constexpr = BLOCK_M // 2
    HALF_N: ttgl.constexpr = BLOCK_N // 2
    for qm in ttgl.static_range(2):
        for qn in ttgl.static_range(2):
            wc_ksplit_tile(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, off_m + qm * HALF_M, off_n + qn * HALF_N, M, N, K,
                           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, HALF_M, HALF_N, BLOCK_K,
                           NUM_CTAS, K_WIDTH, load3d, wmma3d, c2d)
