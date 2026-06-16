// RUN: triton-opt %s -split-input-file -tritonamdgpu-plan-cga | FileCheck %s

// NOTE that in this file we need to capture some attributes whose names not
//  start with $. These variables are considered as local variable by FileCheck.
//  Since FILECHECK_OPTS contains --enable-var-scope and local variable cannot
//  live across CHECK_LABEL, we should not use CHECK_LABEL to check function
//  name.

// NOTE on @matmul1()
// - The input tensors have shape M/N/K=512/512/32. The
//  default CTAs are arranged into 4x1 shape. Since M and N is close, plan-cta
//  is expected to change the CTA shape into 2x2.
// - It's a matmul with tt.load
//
// CHECK-DAG: [[OPND_A:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = {{\[\[0, 0\], \[1, 0\]\]}}
// CHECK-DAG: [[OLD_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = {{\[\[1, 0\], \[2, 0\]\]}}
// CHECK-DAG: [[OPND_CD:#.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}
// CHECK-DAG: [[STORE_LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}
// CHECK: tt.func public @matmul1(
// CHECK: scf.for {{.*}} {
// CHECK:   %[[A_PTR:.*]] = ttg.convert_layout %arg5 : {{.*}} -> tensor<512x32x!tt.ptr<f16>, [[OPND_A]]
// CHECK:   %[[A1:.*]] = tt.load %[[A_PTR]] : tensor<512x32x!tt.ptr<f16>, [[OPND_A]]>
// CHECK:   %[[A2:.*]] = ttg.convert_layout %[[A1]] : tensor<512x32xf16, [[OPND_A]]> -> tensor<512x32xf16, [[OLD_LAYOUT]]>
// CHECK:   %[[CVT_A:.*]] = ttg.convert_layout %[[A2]] {{.*}}, #ttg.dot_op<{opIdx = 0, parent = [[OPND_CD]]}
// CHECK:   tt.dot %[[CVT_A]], {{.*}} : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = [[OPND_CD]]}>> * tensor<32x512xf16, #ttg.dot_op<{opIdx = 1, parent = [[OPND_CD]]}>> -> tensor<512x512xf32, [[OPND_CD]]>
// CHECK: } {tt.num_stages = 2 : i32}
// CHECK: tt.store {{.*}} tensor<512x512x!tt.ptr<f32>, [[STORE_LAYOUT]]>

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512x512xf32, #blocked>
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %cst_0 = arith.constant dense<1024> : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_1 = arith.constant dense<512> : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_2 = arith.constant dense<256> : tensor<512x1xi32, #blocked1>
    %cst_3 = arith.constant dense<512> : tensor<32x1xi32, #blocked2>
    %cst_4 = arith.constant dense<512> : tensor<512x1xi32, #blocked3>
    %cst_5 = arith.constant dense<32> : tensor<512x32xi32, #blocked1>
    %cst_6 = arith.constant dense<16384> : tensor<32x512xi32, #blocked2>
    %0 = tt.get_program_id x : i32
    %1 = arith.remsi %0, %c2_i32 : i32
    %2 = arith.divsi %0, %c2_i32 : i32
    %3 = arith.muli %1, %c512_i32 : i32
    %4 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %6 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %7 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %8 = tt.splat %3 : i32 -> tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %9 = tt.splat %3 : i32 -> tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %10 = arith.addi %8, %4 : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %11 = arith.addi %9, %5 : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %12 = arith.remsi %10, %cst_0 : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %13 = arith.muli %2, %c512_i32 : i32
    %14 = tt.splat %13 : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %13 : i32 -> tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %16 = arith.addi %14, %6 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %17 = arith.addi %15, %7 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %18 = arith.remsi %16, %cst_1 : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = tt.expand_dims %12 {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<512x1xi32, #blocked1>
    %20 = arith.muli %19, %cst_2 : tensor<512x1xi32, #blocked1>
    %21 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.expand_dims %21 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %23 = tt.broadcast %20 : tensor<512x1xi32, #blocked1> -> tensor<512x32xi32, #blocked1>
    %24 = tt.broadcast %22 : tensor<1x32xi32, #blocked1> -> tensor<512x32xi32, #blocked1>
    %25 = arith.addi %23, %24 : tensor<512x32xi32, #blocked1>
    %26 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<512x32x!tt.ptr<f16>, #blocked1>
    %27 = tt.addptr %26, %25 : tensor<512x32x!tt.ptr<f16>, #blocked1>, tensor<512x32xi32, #blocked1>
    %28 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
    %30 = arith.muli %29, %cst_3 : tensor<32x1xi32, #blocked2>
    %31 = tt.expand_dims %18 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x512xi32, #blocked2>
    %32 = tt.broadcast %30 : tensor<32x1xi32, #blocked2> -> tensor<32x512xi32, #blocked2>
    %33 = tt.broadcast %31 : tensor<1x512xi32, #blocked2> -> tensor<32x512xi32, #blocked2>
    %34 = arith.addi %32, %33 : tensor<32x512xi32, #blocked2>
    %35 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x512x!tt.ptr<f16>, #blocked2>
    %36 = tt.addptr %35, %34 : tensor<32x512x!tt.ptr<f16>, #blocked2>, tensor<32x512xi32, #blocked2>
    %37:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %27, %arg6 = %36) -> (tensor<512x512xf32, #blocked>, tensor<512x32x!tt.ptr<f16>, #blocked1>, tensor<32x512x!tt.ptr<f16>, #blocked2>)  : i32 {
      %47 = tt.load %arg5 : tensor<512x32x!tt.ptr<f16>, #blocked1>
      %48 = tt.load %arg6 : tensor<32x512x!tt.ptr<f16>, #blocked2>
      %49 = ttg.convert_layout %47 : tensor<512x32xf16, #blocked1> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %50 = ttg.convert_layout %48 : tensor<32x512xf16, #blocked2> -> tensor<32x512xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %51 = tt.dot %49, %50, %arg4 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x512xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<512x512xf32, #blocked>
      %52 = tt.addptr %arg5, %cst_5 : tensor<512x32x!tt.ptr<f16>, #blocked1>, tensor<512x32xi32, #blocked1>
      %53 = tt.addptr %arg6, %cst_6 : tensor<32x512x!tt.ptr<f16>, #blocked2>, tensor<32x512xi32, #blocked2>
      scf.yield %51, %52, %53 : tensor<512x512xf32, #blocked>, tensor<512x32x!tt.ptr<f16>, #blocked1>, tensor<32x512x!tt.ptr<f16>, #blocked2>
    } {tt.num_stages = 2 : i32}
    %38 = tt.expand_dims %11 {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<512x1xi32, #blocked3>
    %39 = arith.muli %38, %cst_4 : tensor<512x1xi32, #blocked3>
    %40 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x1x!tt.ptr<f32>, #blocked3>
    %41 = tt.addptr %40, %39 : tensor<512x1x!tt.ptr<f32>, #blocked3>, tensor<512x1xi32, #blocked3>
    %42 = tt.expand_dims %17 {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x512xi32, #blocked3>
    %43 = tt.broadcast %41 : tensor<512x1x!tt.ptr<f32>, #blocked3> -> tensor<512x512x!tt.ptr<f32>, #blocked3>
    %44 = tt.broadcast %42 : tensor<1x512xi32, #blocked3> -> tensor<512x512xi32, #blocked3>
    %45 = tt.addptr %43, %44 : tensor<512x512x!tt.ptr<f32>, #blocked3>, tensor<512x512xi32, #blocked3>
    %46 = ttg.convert_layout %37#0 : tensor<512x512xf32, #blocked> -> tensor<512x512xf32, #blocked3>
    tt.store %45, %46 : tensor<512x512x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----
// CHECK-DAG: [[OPND_D:#.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}
// CHECK-DAG: [[OPND_A:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = {{\[\[0, 0\], \[1, 0\]\]}}
// CHECK-DAG: [[OPND_B:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = {{\[\[0, 1\], \[0, 0\]\]}}
// CHECK: tt.func public @matmul_tdm(
// CHECK: scf.for {{.*}} {
// CHECK:   tt.descriptor_load {{.*}} : !tt.tensordesc<128x16xf16> -> tensor<128x16xf16, [[OPND_A]]>
// CHECK:   tt.descriptor_load {{.*}} : !tt.tensordesc<16x128xf16> -> tensor<16x128xf16, [[OPND_B]]>
// CHECK:   tt.dot {{.*}} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[OPND_D]]}>> * tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = [[OPND_D]]}>> -> tensor<128x128xf32, [[OPND_D]]>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_tdm(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"), %M: i32 {tt.divisibility = 16 : i32} loc("M"), %N: i32 {tt.divisibility = 16 : i32} loc("N"), %K: i32 {tt.divisibility = 16 : i32} loc("K")) attributes {noinline = false} {
    %c15_i32 = arith.constant 15 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = arith.muli %1, %c128_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f16>, <128x16xf16>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f16>, <16x128xf16>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <128x128xf16>
    %9 = arith.addi %K, %c15_i32 : i32
    %10 = arith.divsi %9, %c16_i32 : i32
    %accumulator:2 = scf.for %k = %c0_i32 to %10 step %c1_i32 iter_args(%offs_k = %c0_i32, %accumulator_0 = %cst) -> (i32, tensor<128x128xf32, #blocked>)  : i32 {
      %13 = tt.descriptor_load %5[%2, %offs_k] : !tt.tensordesc<128x16xf16> -> tensor<128x16xf16, #blocked1>
      %14 = ttg.convert_layout %13 : tensor<128x16xf16, #blocked1> -> tensor<128x16xf16, #blocked2>
      %15 = tt.descriptor_load %7[%offs_k, %3] : !tt.tensordesc<16x128xf16> -> tensor<16x128xf16, #blocked3>
      %16 = ttg.convert_layout %15 : tensor<16x128xf16, #blocked3> -> tensor<16x128xf16, #blocked>
      %17 = ttg.convert_layout %14 : tensor<128x16xf16, #blocked2> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked4}>>
      %18 = ttg.convert_layout %16 : tensor<16x128xf16, #blocked> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked4}>>
      %accumulator_1 = ttg.convert_layout %accumulator_0 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked4>
      %19 = tt.dot %17, %18, %accumulator_1 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked4}>> * tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked4}>> -> tensor<128x128xf32, #blocked4>
      %20 = ttg.convert_layout %19 : tensor<128x128xf32, #blocked4> -> tensor<128x128xf32, #blocked>
      %21 = arith.addi %offs_k, %c16_i32 : i32
      scf.yield %21, %20 : i32, tensor<128x128xf32, #blocked>
    }
    %11 = arith.truncf %accumulator#1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %12 = ttg.convert_layout %11 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked5>
    tt.descriptor_store %8[%2, %3], %12 : !tt.tensordesc<128x128xf16>, tensor<128x128xf16, #blocked5>
    tt.return
  }
}


// -----
// CHECK-DAG: [[OPND_D:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}
// CHECK-DAG: [[OPND_A:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = {{\[\[0, 0\], \[1, 0\]\]}}
// CHECK-DAG: [[OPND_A_SCALE:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 0\], \[1, 0\]\]}}
// CHECK-DAG: [[OPND_B:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = {{\[\[0, 1\], \[0, 0\]\]}}
// CHECK-DAG: [[OPND_B_SCALE:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\], \[0, 0\]\]}}
// CHECK: tt.func public @mxfp_matmul1(
// CHECK:   tt.dot_scaled {{.*}} : tensor<128x64xf8E5M2, [[OPND_A]]>, tensor<128x2xi8, [[OPND_A_SCALE]]> * tensor<64x128xf8E5M2, [[OPND_B]]>, tensor<128x2xi8, [[OPND_B_SCALE]]> -> tensor<128x128xf32, [[OPND_D]]>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mxfp_matmul1(
    %a: tensor<128x64xf8E5M2, #blocked>,
    %b: tensor<64x128xf8E5M2, #blocked1>,
    %d: tensor<128x128x!tt.ptr<f32>, #blocked2>,
    %a_scale: tensor<128x2xi8, #blocked3>,
    %b_scale: tensor<128x2xi8, #blocked3>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %cst lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E5M2, #blocked>, tensor<128x2xi8, #blocked3> * tensor<64x128xf8E5M2, #blocked1>, tensor<128x2xi8, #blocked3> -> tensor<128x128xf32, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked2>
    tt.store %d, %1 : tensor<128x128x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// -----
// Note: Currently the PlanCGA has no problem dealing with multipl dot in the kernel,
//  but as of this moment, it has problem in handling reduction ops, which normally
//  present more stringent constraints.
//
// CHECK-DAG: [[OPND_D:#.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}
// CHECK: tt.func public @mxfp_matmul2(
// CHECK:   tt.dot_scaled {{.*}} -> tensor<128x128xf32, [[OPND_D]]>
// CHECK:   tt.dot_scaled {{.*}} -> tensor<128x128xf32, [[OPND_D]]>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0], [2, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mxfp_matmul2(
    %a: tensor<128x64xf8E5M2, #blocked>,
    %b: tensor<64x128xf8E5M2, #blocked1>,
    %d: tensor<128x128x!tt.ptr<f32>, #blocked2>,
    %a_scale: tensor<128x2xi8, #blocked3>,
    %b_scale: tensor<128x2xi8, #blocked3>,
    %a2: tensor<128x64xf8E5M2, #blocked>,
    %d2: tensor<128x128x!tt.ptr<f32>, #blocked2>) {

    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %cst lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E5M2, #blocked>, tensor<128x2xi8, #blocked3> * tensor<64x128xf8E5M2, #blocked1>, tensor<128x2xi8, #blocked3> -> tensor<128x128xf32, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked2>
    tt.store %d, %1 : tensor<128x128x!tt.ptr<f32>, #blocked2>

    %2 = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %cst lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E5M2, #blocked>, tensor<128x2xi8, #blocked3> * tensor<64x128xf8E5M2, #blocked1>, tensor<128x2xi8, #blocked3> -> tensor<128x128xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked2>
    tt.store %d2, %3 : tensor<128x128x!tt.ptr<f32>, #blocked2>

    tt.return
  }
}
