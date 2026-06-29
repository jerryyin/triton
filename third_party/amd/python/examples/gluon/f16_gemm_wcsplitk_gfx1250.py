"""Within-cluster split-K GEMM for gfx1250 (WG cluster).
NOTE: K-split mode (no operand multicast — CTAs load different K). The cross-CTA
  reduction passes partials through global scratch, synchronized only by cluster.barrier
  — which orders execution but NOT memory visibility. Real HW needs release/acquire
  fences around the barrier (producer store -> release -> barrier -> acquire -> consumer
  load) so every partial is globally visible before it's summed.
"""
import pytest
import torch
import triton.experimental.gluon.language as ttgl
from triton.experimental import gluon
from triton._C.libtriton.gluon_ir import make_cga_layout

try:
    from .gfx1250_utils import static_profile  # noqa: F401
    from .f16_gemm_common_gfx1250 import wc_ksplit_tile
except Exception:  # noqa: BLE001
    from f16_gemm_common_gfx1250 import wc_ksplit_tile


@gluon.jit
def wcsplitk_gemm_kernel(a_ptr, b_ptr, c_ptr, scr_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                         BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                         NUM_CTAS: ttgl.constexpr, K_WIDTH: ttgl.constexpr, load_layout: ttgl.constexpr,
                         wmma_layout: ttgl.constexpr, c2d_layout: ttgl.constexpr):
    pid = ttgl.program_id(0)
    num_cgas = ttgl.num_programs(0) // NUM_CTAS
    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    scr_cga = pid * (NUM_CTAS * BLOCK_M * BLOCK_N)

    for tile in range(pid, total_tiles, num_cgas):
        off_m = (tile % num_pid_m) * BLOCK_M
        off_n = (tile // num_pid_m) * BLOCK_N
        wc_ksplit_tile(a_ptr, b_ptr, c_ptr, scr_ptr, scr_cga, off_m, off_n, M, N, K, stride_am, stride_ak, stride_bk,
                       stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, K_WIDTH, load_layout,
                       wmma_layout, c2d_layout)


def run_wcsplitk_gemm(BLOCK_M, BLOCK_N, BLOCK_K, M, N, K, num_warps, num_ctas, num_cgas, TRANSPOSE_B=False):
    """Launch the within-cluster split-K GEMM and verify against torch."""
    cga_split = make_cga_layout([num_ctas, 1, 1], [num_ctas, 1, 1], [2, 1, 0])
    cga_bcast = make_cga_layout([num_ctas, 1], [1, 1], [0, 1])
    load_layout = ttgl.BlockedLayout([1, 1, 8], [1, 4, 8], [1, 4, 1], [2, 1, 0], cga_split)
    wmma_layout = ttgl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[[0, 0, 1], [0, 1, 0]], reg_bases=[],
                                         instr_shape=[16, 16, 32], cga_layout=cga_split, rank=3)
    c2d_layout = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0], cga_bcast)

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    if TRANSPOSE_B:
        b = b.T.contiguous()
    c = torch.zeros((M, N), dtype=torch.float32)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    scr = torch.zeros((num_cgas * num_ctas * BLOCK_M * BLOCK_N, ), dtype=torch.float32)
    ad, bd, cd, sd = a.cuda(), b.cuda(), c.cuda(), scr.cuda()

    wcsplitk_gemm_kernel[(num_cgas, )](ad, bd, cd, sd, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                                       stride_cn, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, NUM_CTAS=num_ctas,
                                       K_WIDTH=8, load_layout=load_layout, wmma_layout=wmma_layout,
                                       c2d_layout=c2d_layout, num_warps=num_warps, num_ctas=num_ctas)
    ref = a.float() @ (b.float() if not TRANSPOSE_B else b.T.float())
    torch.testing.assert_close(cd.cpu(), ref, rtol=1e-3, atol=1e-3)


# (M, N, K, transpose_b) — covers divisible, M/N edges, partial-K-block, ragged-iters
_WCSPLITK_SHAPES = [(64, 64, 256, False),  # clean / divisible
                    (70, 66, 256, False),  # M,N not multiples of 32 -> edge masks
                    (64, 64, 250, False),  # K not multiple of 32 -> partial K block
                    (64, 64, 288, False),  # 9 K-iters -> ragged distribution (rem>0 at num_ctas=4)
                    (70, 66, 250, True),  # edges + partial-K + TRANSPOSE_B
                    (256, 256, 512, False),  # multi-tile (8x8 grid) persistent distribution
                    (258, 258, 510, True),  # multi-tile + edges + TRANSPOSE_B
                    ]


@pytest.mark.parametrize("M,N,K,TRANSPOSE_B", _WCSPLITK_SHAPES)
@pytest.mark.parametrize("num_ctas,num_cgas", [(2, 1), (4, 1), (2, 2), (4, 2)])
def test_wcsplitk_gemm(M, N, K, TRANSPOSE_B, num_ctas, num_cgas):
    run_wcsplitk_gemm(32, 32, 32, M, N, K, num_warps=4, num_ctas=num_ctas, num_cgas=num_cgas, TRANSPOSE_B=TRANSPOSE_B)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
