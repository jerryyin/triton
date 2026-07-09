// Detection half of the wave-uniform gather-index scalarization: the side table
// ConvertToLLVM builds over tt.load ops feeding a TDM gather/scatter row-index.
// Unlike uniform_index_sload.mlir (which forces the *emission* via env), this
// runs with NO override, so scalarization only happens if collection fires:
// the load must be wave-uniform AND read-only. readfirstlane on the index
// address is the signal that it did.

// RUN: triton-opt %s -split-input-file --allocate-shared-memory \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

// Wave-uniform (sliced blocked layout projects the lane dim out), read-only
// index feeding a gather -> collected -> scalarized.
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gather_index_detected
  // CHECK: rocdl.readfirstlane
  tt.func public @gather_index_detected(
    %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %tensorDesc: !tt.tensordesc<16x64xf16, #shared>,
    %memDesc: !ttg.memdesc<16x64xf16, #shared, #smem, mutable>) {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice>
    %sp = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<16x!tt.ptr<i32>, #slice>, tensor<16xi32, #slice>
    %row_indices = tt.load %ptrs : tensor<16x!tt.ptr<i32>, #slice>
    amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice>, !ttg.memdesc<16x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<16x64xf16, #shared>
    tt.return
  }
}

// -----

// Same load, but a store to the index buffer forfeits read-only -> not
// collected -> left as a vector load (no address readfirstlane).
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gather_index_written
  // CHECK-NOT: rocdl.readfirstlane
  tt.func public @gather_index_written(
    %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %tensorDesc: !tt.tensordesc<16x64xf16, #shared>,
    %memDesc: !ttg.memdesc<16x64xf16, #shared, #smem, mutable>) {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice>
    %sp = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<16x!tt.ptr<i32>, #slice>, tensor<16xi32, #slice>
    %row_indices = tt.load %ptrs : tensor<16x!tt.ptr<i32>, #slice>
    tt.store %ptrs, %row_indices : tensor<16x!tt.ptr<i32>, #slice>
    amdg.async_tdm_gather %tensorDesc[%row_indices] to %memDesc : tensor<16xi32, #slice>, !ttg.memdesc<16x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<16x64xf16, #shared>
    tt.return
  }
}
