#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "TritonAMDGPUToLLVM/TypeConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONAMDGPUTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// --- Wave-uniform gather/scatter index scalarization ---
// Identify a wave-uniform, read-only tt.load that feeds a TDM gather/scatter
// row-index operand, so LoadOpConversion loads it straight into SGPRs (s_load)
// instead of a per-lane vector load + in-loop v_readfirstlane. Computed here
// (at the start of ConvertToLLVM, where the tt.load and the gather op still
// coexist and pattern-application order is irrelevant) and threaded into the
// load lowering as a side table.

// Trace back to the producing tt.LoadOp through *layout-preserving* ops only
// (bitcast + integer-elementwise arith). We deliberately do NOT cross
// ConvertLayoutOp: staying layout-preserving means the load shares the gather
// index's encoding, so the gather/scatter verifier's lane-uniform guarantee on
// the index carries to the load — no separate uniformity check needed.
static triton::LoadOp traceToProducingLoad(Value v, int depth = 8) {
  if (depth < 0 || !v)
    return nullptr;
  Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;
  if (auto ld = dyn_cast<triton::LoadOp>(def))
    return ld;
  if (isa<triton::BitcastOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
          arith::RemUIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
          arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp, arith::AndIOp,
          arith::OrIOp>(def)) {
    for (Value operand : def->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (auto ld = traceToProducingLoad(operand, depth - 1))
        return ld;
    }
  }
  return nullptr;
}

// Trace a pointer to its base value (ideally a kernel-arg BlockArgument).
static Value traceToBasePtr(Value ptr) {
  while (ptr) {
    Operation *def = ptr.getDefiningOp();
    if (!def)
      break;
    if (auto ap = dyn_cast<triton::AddPtrOp>(def)) {
      ptr = ap.getPtr();
      continue;
    }
    if (auto sp = dyn_cast<triton::SplatOp>(def)) {
      ptr = sp.getSrc();
      continue;
    }
    if (auto bc = dyn_cast<triton::BitcastOp>(def)) {
      ptr = bc.getSrc();
      continue;
    }
    break;
  }
  return ptr;
}

static bool isPtrOrDescType(Type t) {
  return isa<triton::TensorDescType>(t) ||
         isa<triton::PointerType>(getElementTypeOrSelf(t));
}

// Resolve the base object a global write targets. The target is either a
// (tensor of) pointer or a tensor descriptor; reduce it to its base pointer.
// Descriptor chains (update_tensor_descriptor -> ... -> make_tensor_descriptor)
// are traced to the make's base.
static Value resolveWriteBase(Value written) {
  while (written && isa<triton::TensorDescType>(written.getType())) {
    Operation *def = written.getDefiningOp();
    if (auto mk = dyn_cast_or_null<triton::MakeTensorDescOp>(def))
      return traceToBasePtr(mk.getBase());
    if (auto up =
            dyn_cast_or_null<triton::amdgpu::UpdateTensorDescriptorOp>(def)) {
      written = up.getDesc();
      continue;
    }
    return {}; // unknown descriptor source
  }
  if (!written)
    return {};
  return traceToBasePtr(written);
}

// Read-only iff `base` is a kernel-arg BlockArgument and no op in the function
// writes global memory to a location that may alias it. Global writes are
// discovered generically via the MemoryEffectOpInterface (so all writing ops --
// tt.store, atomics, descriptor_store, async_tdm_copy_local_to_global,
// async_tdm_scatter, and any future one -- are covered), filtered to the
// GlobalMemory resource (shared-memory writes are irrelevant). Some ops declare
// the write effect without pinning a value (e.g. the TDM copy-to-global reports
// a bare GlobalMemory write); for those we fall back to the op's own
// pointer/descriptor operands as the write targets. Aliasing is conservative: a
// write to a *distinct* kernel argument is assumed disjoint (the standard
// no-alias-across-args convention), but a write to the same base, or to a base
// we cannot resolve to a distinct argument, forfeits read-only.
static bool baseIsReadOnly(Value base, Operation *funcScope) {
  if (!isa<BlockArgument>(base))
    return false;
  bool readOnly = true;
  funcScope->walk([&](Operation *op) {
    auto effOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effOp)
      return;
    SmallVector<MemoryEffects::EffectInstance> effects;
    effOp.getEffects(effects);
    for (const auto &eff : effects) {
      if (!isa<MemoryEffects::Write>(eff.getEffect()))
        continue;
      if (eff.getResource() != mlir::triton::GlobalMemory::get())
        continue;
      SmallVector<Value, 2> targets;
      if (Value v = eff.getValue())
        targets.push_back(v);
      else
        for (Value o : op->getOperands())
          if (isPtrOrDescType(o.getType()))
            targets.push_back(o);
      // A global write exposing neither a pinned value nor a ptr/desc operand
      // is unanalyzable -> conservatively writable.
      if (targets.empty())
        readOnly = false;
      for (Value t : targets) {
        Value wbase = resolveWriteBase(t);
        // Provably-distinct kernel argument => assumed disjoint. Otherwise
        // (same base, unresolved, or not an argument) => conservatively
        // writable.
        if (!wbase || !isa<BlockArgument>(wbase) || wbase == base)
          readOnly = false;
      }
    }
  });
  return readOnly;
}

// Collect the read-only tt.load ops that feed a TDM gather/scatter row-index
// operand. The index is already lane-uniform by construction: AsyncTDMGatherOp
// / AsyncTDMScatterOp::verify() reject any index layout that distributes values
// across lanes, and traceToProducingLoad only walks layout-preserving ops, so
// the load shares that lane-uniform layout. Uniformity therefore needs no
// re-check here; read-only is the remaining condition. Returned as a side table
// (threaded into LoadOpConversion) rather than an IR attribute -- no IR
// mutation, and it cannot be dropped by a later rewrite. The op pointers are
// stable: dialect conversion hands the original op to matchAndRewrite and
// replaces it via the rewriter.
static llvm::DenseSet<Operation *>
collectUniformGatherIndexLoads(ModuleOp mod) {
  llvm::DenseSet<Operation *> result;
  if (::triton::tools::getBoolEnv("TRITON_AMD_DISABLE_UNIFORM_SLOAD"))
    return result;
  mod.walk([&](Operation *op) {
    Value idx;
    if (auto g = dyn_cast<triton::amdgpu::AsyncTDMGatherOp>(op))
      idx = g.getSrcRowIndices();
    else if (auto s = dyn_cast<triton::amdgpu::AsyncTDMScatterOp>(op))
      idx = s.getDstRowIndices();
    else
      return;
    triton::LoadOp ld = traceToProducingLoad(idx);
    if (!ld)
      return;
    Operation *func = ld->getParentOfType<FunctionOpInterface>();
    if (!func)
      func = mod;
    if (!baseIsReadOnly(traceToBasePtr(ld.getPtr()), func))
      return;
    result.insert(ld.getOperation());
  });
  return result;
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<triton::instrument::TritonInstrumentDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonAMDGPUToLLVM
    : public triton::impl::ConvertTritonAMDGPUToLLVMBase<
          ConvertTritonAMDGPUToLLVM> {
  explicit ConvertTritonAMDGPUToLLVM(StringRef gfxArch, bool ftz) {
    this->gfxArch = gfxArch.str();
    this->ftz = ftz;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::amdgpu::TritonAMDGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Collect wave-uniform, read-only gather/scatter index loads, threaded into
    // the load lowering below as a side table.
    llvm::DenseSet<Operation *> uniformIndexLoads =
        collectUniformGatherIndexLoads(mod);

    AMD::TargetInfo targetInfo(this->gfxArch.getValue());
    if (targetInfo.getISAFamily() == triton::amdgpu::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->gfxArch.getValue() << "'";
      return signalPassFailure();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonAMDGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    // Allocate shared memory and set barrier
    auto allocationFn = [&targetInfo](Operation *op) {
      return AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
    };
    ModuleAllocation allocation(mod, allocationFn,
                                targetInfo.getSharedMemoryPartitionSize());

    if (targetInfo.requiresAliasInfoForAsyncOps())
      AMD::annotateLocalLoadsSyncedViaAsyncWait(mod);

    ModuleMembarAnalysis membarPass(&allocation,
                                    mlir::triton::AMD::membarFilter);
    membarPass.run();

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::AMD::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for AMD specific patterns higher so they apply before common
    // patterns
    int AMDBenefit = commonBenefit + 1;
    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

    AMD::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, AMDBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    AMD::populateDotOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                     AMDBenefit);
    AMD::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                             axisInfoAnalysis, allocation,
                                             targetInfo, AMDBenefit);
    AMD::populateFpCastOpToLLVMPatterns(typeConverter, patterns, ftz,
                                        axisInfoAnalysis, allocation,
                                        targetInfo, AMDBenefit);
    AMD::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, uniformIndexLoads,
                                           AMDBenefit);
    AMD::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    AMD::populateBarrierOpToLLVMPatterns(typeConverter, patterns, AMDBenefit);
    AMD::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                            AMDBenefit);

    populatePatterns7(mlir::triton::populateReduceOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    AMD::populateHistogramOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                           AMDBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateGatherOpToLLVMPatterns,
                      commonBenefit);

    AMD::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        AMDBenefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    AMD::populateSPMDOpToLLVMPattern(typeConverter, patterns, AMDBenefit);

    mlir::triton::AMD::populateTritonAMDGPUToLLVMPatterns(
        typeConverter, patterns, targetInfo, AMDBenefit);
    mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, AMDBenefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::AMD::populateWarpIdOpToLLVMPattern(typeConverter, targetInfo,
                                                     patterns, commonBenefit);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->gfxArch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->gfxArch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::populateInstrumentationToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo);
    mlir::triton::populateFpSanToLLVMPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    AMD::adjustModeRegister(mod, targetInfo);
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem",
        /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonAMDGPUToLLVMPass(StringRef gfxArch, bool ftz) {
  return std::make_unique<ConvertTritonAMDGPUToLLVM>(gfxArch, ftz);
}

} // namespace mlir::triton
