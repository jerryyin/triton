// RUN: triton-opt %s -split-input-file -tritonamdgpu-pipeline | FileCheck %s

// CHECK-LABEL: cluster_sync_loop1
// CHECK: scf.for
// CHECK: amdg.cluster_barrier_arrive
// CHECK: tt.load
// CHECK: tt.store
// CHECK: amdg.cluster_barrier_wait

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 0], [1, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cluster_sync_loop1(%a_ptr: tensor<16x16x!tt.ptr<f16>, #blocked>,
                                     %b_ptr: tensor<16x16x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c100 step %c1 {
      %val = tt.load %a_ptr : tensor<16x16x!tt.ptr<f16>, #blocked>
      tt.store %b_ptr, %val : tensor<16x16x!tt.ptr<f16>, #blocked>
      scf.yield
    }
    tt.return
  }
}

// -----

// Note: The only difference between this testing and the previous one
//   is the CGALayout encoding. While it still use multiple CTAs, the CGALayout
//   encoding indicates there is no multi-casting taking place, obviating the
//   need of inserting cluster-{arrive|wait} operations.
//
// CHECK-LABEL: cluster_sync_loop2
// CHECK-NOT: amdg.cluster_barrier_arrive
// CHECK-NOT: amdg.cluster_barrier_wait

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cluster_sync_loop2(%a_ptr: tensor<16x16x!tt.ptr<f16>, #blocked>,
                                     %b_ptr: tensor<16x16x!tt.ptr<f16>, #blocked>) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c100 step %c1 {
      %val = tt.load %a_ptr : tensor<16x16x!tt.ptr<f16>, #blocked>
      tt.store %b_ptr, %val : tensor<16x16x!tt.ptr<f16>, #blocked>
      scf.yield
    }
    tt.return
  }
}

// -----
