// Wave-uniform load scalarization for TDM gather indices (issue #1885).
// With TRITON_AMD_UNIFORM_SLOAD=1, a load whose result is uniform across lanes
// (the `lane` dim is free in its LinearLayout -- e.g. a sliced/broadcast index
// tensor) has each loaded element lifted into an SGPR via v_readfirstlane at the
// load site. This is gated by the env var and only fires for uniform loads.
//
// RUN: TRITON_AMD_UNIFORM_SLOAD=1 triton-opt %s -split-input-file \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ON
// RUN: triton-opt %s -split-input-file \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,OFF

// (1) Wave-uniform index load: sliced blocked layout projects the lane dim out,
//     so all 32 lanes hold the same values. Flag ON -> readfirstlane; OFF -> none.
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @uniform_index_load
  tt.func @uniform_index_load(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<16x!tt.ptr<i16>, #slice>, tensor<16xi32, #slice>
    // ON: rocdl.readfirstlane
    // OFF-NOT: rocdl.readfirstlane
    %v = tt.load %ptrs : tensor<16x!tt.ptr<i16>, #slice>
    tt.return
  }
}

// -----

// (2) Divergent load: plain blocked layout, lane varies. Even with the flag ON,
//     the load must be left untouched (no readfirstlane).
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @divergent_load
  tt.func @divergent_load(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>, #blocked1>
    %ptrs = tt.addptr %sp, %r : tensor<32x!tt.ptr<i16>, #blocked1>, tensor<32xi32, #blocked1>
    // ON-NOT: rocdl.readfirstlane
    // OFF-NOT: rocdl.readfirstlane
    %v = tt.load %ptrs : tensor<32x!tt.ptr<i16>, #blocked1>
    tt.return
  }
}
