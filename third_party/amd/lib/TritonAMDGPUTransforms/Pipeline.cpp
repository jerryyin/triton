#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/Passes.h" // IWYU pragma: keep
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define DEBUG_TYPE "tritonamdgpu-pipeline-expand-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                             Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, dotOp->getResult(0).getType(),
                                  pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = scf::YieldOp::create(thenB, loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    auto ifOpBuilder = ifOp.getElseBodyBuilder();
    scf::YieldOp::create(ifOpBuilder, loc, dotOp->getOperand(2));
    return ifOp;
  }
  if (isa<tt::DescriptorLoadLikeOpInterface>(op)) {
    auto loc = op->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, op->getResultTypes(), pred,
                                  /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = scf::YieldOp::create(thenB, loc, op->getResults());
    op->moveBefore(yield);

    auto elseB = ifOp.getElseBodyBuilder();
    SmallVector<Value> zeroValues;
    zeroValues.reserve(op->getNumResults());
    for (Type resultType : op->getResultTypes()) {
      zeroValues.push_back(
          arith::ConstantOp::create(elseB, loc, elseB.getZeroAttr(resultType)));
    }
    scf::YieldOp::create(elseB, loc, zeroValues);
    return ifOp;
  }
  // Gate the copy by chaining a pred-only update_tensor_descriptor onto its
  // descriptor: the chained update inherits the positioning and narrows pred to
  // the loop predicate.
  if (auto copyOp = dyn_cast<triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(rewriter, op->getLoc(),
                                          rewriter.getI32Type(), pred);
    auto updated = triton::amdgpu::UpdateTensorDescriptorOp::create(
        rewriter, op->getLoc(), copyOp.getDesc().getType(), copyOp.getDesc(),
        /*add_offsets=*/ValueRange{}, /*set_bounds=*/ValueRange{},
        /*pred=*/predI32);
    copyOp.getDescMutable().assign(updated.getResult());
    return op;
  }
  if (auto gatherOp = dyn_cast<triton::amdgpu::AsyncTDMGatherOp>(op)) {
    auto predicatedOp = cast<tt::PredicatedOpInterface>(op);
    rewriter.setInsertionPoint(op);
    auto predI32 = arith::ExtUIOp::create(
        rewriter, op->getLoc(), predicatedOp.getPredicateOperand().getType(),
        pred);
    Value mask = arith::AndIOp::create(
        rewriter, op->getLoc(), predicatedOp.getPredicateOperand(), predI32);
    predicatedOp.setPredicateOperand(mask);
    return op;
  }
  if (isa<triton::amdgpu::AsyncTDMWait>(op))
    return op;
  if (isa<tt::DescriptorStoreLikeOpInterface>(op)) {
    auto loc = op->getLoc();
    auto ifOp = scf::IfOp::create(rewriter, loc, pred,
                                  /*withElseRegion=*/false);
    op->moveBefore(ifOp.thenYield());
    return ifOp;
  }
  return tt::wrapInMaskOp(rewriter, op, pred);
}

void expandLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      continue;

    // Create the final schedule for the kernel loop. This will dictate the
    // stages and order of operations to the pipeline expander.
    auto coarseSchedule = schedule.createFinalSchedule(forOp);

    tt::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
    // Annotate loadOp in prologue for further moving up
    options.annotateFn = [](Operation *op,
                            tt::PipeliningOption::PipelinerPart part,
                            unsigned stage) {
      if (part != tt::PipeliningOption::PipelinerPart::Prologue)
        return;

      auto annotateLoad = [](Operation *loadOp) {
        loadOp->setAttr("amd.pipeliner_part",
                        StringAttr::get(loadOp->getContext(), "prologue"));
      };

      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        annotateLoad(loadOp);
        return;
      }
      // loadOp may be wrapped by a MaskOp as predicateFn execution
      // precedes annotation
      if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
        for (auto &innerOp : maskOp.getBody()->without_terminator()) {
          if (auto loadOp = dyn_cast<tt::LoadOp>(&innerOp)) {
            annotateLoad(loadOp);
            return;
          }
        }
      }
    };
    // Set the final schedule as our scheduling function
    options.getScheduleFn =
        [coarseSchedule](scf::ForOp,
                         std::vector<std::pair<Operation *, unsigned>> &s) {
          s = std::move(coarseSchedule);
        };

    LDBG("Loop before sending to expander:\n" << *forOp);

    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        tt::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp))
      continue;
  }

  tt::resolveMaskOp(moduleOp);
}

// Fold consecutive waits of the same kind into a single wait.
void combineWaitOps(ModuleOp moduleOp, bool useAsyncCopy) {
  llvm::SmallSetVector<Operation *, 8> asyncWaitOps;
  llvm::SmallSetVector<Operation *, 8> tdmWaitOps;
  moduleOp.walk([&](Operation *op) {
    if (useAsyncCopy && isa<ttg::AsyncWaitOp>(op))
      asyncWaitOps.insert(op);
    else if (isa<triton::amdgpu::AsyncTDMWait>(op))
      tdmWaitOps.insert(op);
  });

  if (useAsyncCopy) {
    tt::combineRedundantWaitOps(
        asyncWaitOps,
        [](Operation *op) { return isa<ttg::AsyncCommitGroupOp>(op); },
        [](OpBuilder &b, Location loc, ValueRange operands,
           unsigned num) -> Operation * {
          return ttg::AsyncWaitOp::create(b, loc, operands, num);
        });
  }

  tt::combineRedundantWaitOps(
      tdmWaitOps,
      [](Operation *op) { return isa<triton::amdgpu::TDMOpInterface>(op); },
      [](OpBuilder &b, Location loc, ValueRange operands,
         unsigned num) -> Operation * {
        return triton::amdgpu::AsyncTDMWait::create(b, loc, operands, num);
      });
}

// The InsertClusterSync() and its helper function is to
//  - insert clusterArriveOp at very beginning of the loop body
//  - insert clusterWaitOp at the very end of loop body.
//
// The purpose of these two operations is to make all CTAs in the cluster run
// about the same pace as all CTAs sync per iteration. By doing so, all CTAs
// make load requests about the same time and render it possible for hardware
// to multicast duplicated loads.
//
// These two operations need to be sufficiently separated as clusterArriveOp
// may take a while to send its signal to its peer CTAs.
//
// TODO: We may need multiple pairs of arrive/wait if the loop body is big.
//
void InsertClusterSyncIntoLoop(scf::ForOp loop) {

  mlir::OpBuilder builder(loop.getContext());
  mlir::Block *loopBody = loop.getBody();
  mlir::Operation &firstOp = loopBody->front();
  builder.setInsertionPoint(&firstOp);
  triton::amdgpu::ClusterBarrierArriveOp::create(builder, firstOp.getLoc());

  auto terminator = loopBody->getTerminator();
  builder.setInsertionPoint(terminator);
  triton::amdgpu::ClusterBarrierWaitOp::create(builder, terminator->getLoc());
}

void InsertClusterSync(ModuleOp mod) {
  if (auto arch = getAMDArch(mod)) {
    triton::AMD::TargetInfo targetInfo(*arch);
    auto isaFamily = targetInfo.getISAFamily();
    if (isaFamily != triton::amdgpu::ISAFamily::GFX1250)
      // Bail out if we know for sure the underlying architecture is not
      // gfx1250 because it's the only architecture support multicasting as of
      // this moment.
      return;
  }

  auto hasMulticast = [](ValueRange valueRange) {
    for (auto value : valueRange) {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        auto cgaLayout =
            ttg::getCGALayout(type.getEncoding()).getLinearLayout();
        assert(cgaLayout.isSurjective() &&
               "The tensor block is not completely covered by CTAs");
        bool multicast = !cgaLayout.isInjective();
        if (multicast)
          return true;
      }
    }
    return false;
  };

  SmallVector<scf::ForOp> loops;
  mod->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    auto walkResult = forOp.walk([&](Operation *op) {
      if (isa<tt::LoadOp, tt::DescriptorLoadOp, ttg::AsyncCopyGlobalToLocalOp,
              triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(op)) {
        if (hasMulticast(op->getResults()) || hasMulticast(op->getOperands()))
          return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    // No multi-casting occurrence
    if (!walkResult.wasInterrupted())
      continue;

    // Check if cluster{Arrive|Wait} are already inserted in the loop.
    walkResult = forOp.walk([&](Operation *op) {
      if (isa<triton::amdgpu::ClusterBarrierArriveOp,
              triton::amdgpu::ClusterBarrierWaitOp>(op))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      LDBG("See clusterArriveOp/clusterWaitOp in the loop body!");
      continue;
    }

    InsertClusterSyncIntoLoop(forOp);
    LDBG("After insert clusterArriveOp and clusterWaitOp" << forOp);
  }
}

} // namespace

struct PipelinePass : impl::TritonAMDGPUPipelineBase<PipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    lowerLoops(moduleOp, useAsyncCopy, usePingpong);
    expandLoops(moduleOp);

    if (useAsyncCopy) {
      auto targetFeatures = tt::amdgpu::TargetFeatures::fromModuleOp(moduleOp);
      // Only asyncmark targets (CDNA3/CDNA4) need updateWaits here: their
      // lowering reads ttg.async_wait's `num` directly into wait.asyncmark(N),
      // and PR #9883 made UpdateAsyncWaitCount a no-op on those archs, so
      // without this call the pipeliner-authored num=0 would serialize the
      // SWP. Every other family keeps the prior combineRedundantWaitOps-only
      // path: their num is re-derived downstream by UpdateAsyncWaitCount.
      if (targetFeatures.isCDNA3() || targetFeatures.isCDNA4()) {
        mlir::triton::updateWaits(moduleOp);
      }
    }
    combineWaitOps(moduleOp, useAsyncCopy);

    // Pipeline TDM stores / scatters that survive in loop bodies: lift the
    // LDS allocation out of the loop and hoist the wait so the outgoing
    // async store overlaps the next iteration's compute.  The transformation
    // is correct regardless of the loop's pipeline-stage count, so we apply
    // it to every loop unconditionally; it is a no-op when no descriptor
    // stores or scatters are present.
    SmallVector<scf::ForOp> loops;
    moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      pipelineTDMStores(forOp);

    InsertClusterSync(moduleOp);
    tt::removePipeliningAttributes(moduleOp);
  }
};
} // namespace mlir
