// Wave-uniform gather-index scalarization to scalar load (issue #1885).
// Production trigger is the `amdgpu.uniform_scalar_index` attribute that the
// ConvertToLLVM pass sets on a wave-uniform, read-only tt.load feeding a TDM
// gather/scatter index (see the a8w4 end-to-end test). This unit test exercises
// the *emission* via the manual override env var TRITON_AMD_UNIFORM_SLOAD: a
// wave-uniform load is lowered to per-element `invariant` scalar loads (ISel ->
// s_load) with the addresses lifted to SGPRs via readfirstlane; a divergent load
// is left as a vector load.
//
// RUN: TRITON_AMD_UNIFORM_SLOAD=1 triton-opt %s -split-input-file \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ON
// RUN: triton-opt %s -split-input-file \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,OFF

// (1) Wave-uniform index load (sliced blocked layout projects the lane dim out).
//     Override ON -> invariant scalar loads with readfirstlane'd addresses.
//     Override OFF -> no scalar/invariant load (production attribute is only set
//     for loads feeding a TDM gather, not present in this isolated test).
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @uniform_index_load
  tt.func @uniform_index_load(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<128x!tt.ptr<i16>, #slice>, tensor<128xi32, #slice>
    // ON: rocdl.readfirstlane
    // ON: llvm.load {{.*}}invariant
    // OFF-NOT: llvm.load {{.*}}invariant
    %v = tt.load %ptrs : tensor<128x!tt.ptr<i16>, #slice>
    tt.return
  }
}

// -----

// (2) Divergent load: plain blocked layout, lane varies. Never scalarized.
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @divergent_load
  tt.func @divergent_load(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>, #blocked1>
    %ptrs = tt.addptr %sp, %r : tensor<32x!tt.ptr<i16>, #blocked1>, tensor<32xi32, #blocked1>
    // ON-NOT: llvm.load {{.*}}invariant
    // OFF-NOT: llvm.load {{.*}}invariant
    %v = tt.load %ptrs : tensor<32x!tt.ptr<i16>, #blocked1>
    tt.return
  }
}
