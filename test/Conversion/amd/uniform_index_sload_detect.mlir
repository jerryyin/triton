// Wave-uniform read-only load scalarization (issue: MoE gather-index s_load,
// generalized). A load is scalarized to s_load when its result is lane-uniform
// AND its base argument carries the tt.readonly + tt.noalias contract. This is a
// purely local load-lowering decision -- no gather/scatter consumer required --
// so it applies to any such load, not just TDM indices. readfirstlane on the
// address is the signal it scalarized.

// RUN: triton-opt %s -split-input-file \
// RUN:   --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

// Wave-uniform load (sliced blocked layout projects the lane dim out) + full
// contract -> scalarized.
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @uniform_contract
  // CHECK: rocdl.readfirstlane
  tt.func public @uniform_contract(%arg0: !tt.ptr<i16> {tt.readonly = 1 : i32, tt.noalias = 1 : i32}) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<128x!tt.ptr<i16>, #slice>, tensor<128xi32, #slice>
    %v = tt.load %ptrs : tensor<128x!tt.ptr<i16>, #slice>
    tt.return
  }
}

// -----

// Wave-uniform load but only tt.readonly (no tt.noalias) -> not scalarized.
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @readonly_only
  // CHECK-NOT: rocdl.readfirstlane
  tt.func public @readonly_only(%arg0: !tt.ptr<i16> {tt.readonly = 1 : i32}) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<128x!tt.ptr<i16>, #slice>, tensor<128xi32, #slice>
    %v = tt.load %ptrs : tensor<128x!tt.ptr<i16>, #slice>
    tt.return
  }
}

// -----

// Wave-uniform load but only tt.noalias (no tt.readonly) -> not scalarized.
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @noalias_only
  // CHECK-NOT: rocdl.readfirstlane
  tt.func public @noalias_only(%arg0: !tt.ptr<i16> {tt.noalias = 1 : i32}) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>, #slice>
    %ptrs = tt.addptr %sp, %r : tensor<128x!tt.ptr<i16>, #slice>, tensor<128xi32, #slice>
    %v = tt.load %ptrs : tensor<128x!tt.ptr<i16>, #slice>
    tt.return
  }
}

// -----

// Divergent load (lane varies) with full contract -> not scalarized (the
// uniformity gate, independent of the contract).
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @divergent_contract
  // CHECK-NOT: rocdl.readfirstlane
  tt.func public @divergent_contract(%arg0: !tt.ptr<i16> {tt.readonly = 1 : i32, tt.noalias = 1 : i32}) {
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %sp = tt.splat %arg0 : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>, #blocked1>
    %ptrs = tt.addptr %sp, %r : tensor<32x!tt.ptr<i16>, #blocked1>, tensor<32xi32, #blocked1>
    %v = tt.load %ptrs : tensor<32x!tt.ptr<i16>, #blocked1>
    tt.return
  }
}
