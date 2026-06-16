////////////////////////////////////////////////////////////////////////////
//
// ### Overview
//
//   This file is to lay out multiple CTAs. Currently it proactively perform
// CTA layout for tt.dot and tt.to_scaled operations; for those operations
// that are related to dot-operations (results feed to dot-op or depends on
// dot-op, directly or indirectly), their CTA layout will be changed
// accordingly. ConvertLayouts are inserted both before and after the dot-op
// to bridge the transformed dot-op with its old uses and defs.
//
// The preferred layout are propagated forward and backward starting from the
// inserted ConvertLayouts. This pass relies on subsequent
// RemoveLayoutConversions pass to remove and optimize ConvertLayout ops.
//
// Since RemoveLayoutConversions pass will not change layout for some
// "expensive" ops (aka layout "anchor", e.g., tensor load/store), this pass
// needs to proactively change these expensive ops' layout if their existing
// layouts are different from desired layout propagated from dot op.
//
// This file has following major building block
//   * utility: In particular, LayOutCGAData. We keep a single instance of
//      of LayOutCGAData and shared it between other building blocks.
//   * MulticastDot: to decide CTA split and physically transform dot
//   * PropagateLayout: to propagate preferred layout starting from dot
//   * LayOutCGA: sort of driver. It is also responsible for changing layout
//       for some 2nd class citizens (i.e. expensive ops like tensor ld/st).
//
// ### Limitations
//
//   1. OPs on control-flow's join node may need different layouts on different
// path. As of today, we don't have mechanism in place to track all possible
// layouts from different paths. We only track one layout for an Op. So if
// aforementioned situation take place, this Op does not need ConvertLayout in
// some paths and need a ConvertLayout on other passes.
//
//   2. As mentioned above, currently we only passively change dot's CGA layout.
// and as a result some "expensive" ops's layout will be changed accordingly.
//
//   3. We don't take into account of chained dots either
//
// ### TODO
//   In the future we need to evaluate the best and legal CTA split between
// related ops before physically transforming them. The basic idea is that
//  - Infer the CTA split constraint from those OPs which has a very
//    restrictive splitting constraints.
//  - Represent some constraints properly and propagate along the UD and UD
//    chains.
//  - pick the best and legal CGA splitting.
//
//  Constraints should be represented in a way such that it can depict the
// constraints imposed between related Ops (e.g. chained tt.dot).
//
////////////////////////////////////////////////////////////////////////////
//

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-plan-cta"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define DEBUG_DOT ">DOTOP "
#define DEBUG_FWD ">FWD "
#define DEBUG_BKWD ">BKWD "
#define DEBUG_REWRITE ">REWRITE "
#define DEBUG_INDENT "  "

namespace mlir {
namespace {

////////////////////////////////////////////////////////////////////////
//
//         Common utility functions shared by helper classes
//
////////////////////////////////////////////////////////////////////////
//
ttg::DistributedEncodingTrait
replaceCGALayout(ttg::DistributedEncodingTrait layout,
                 const ttg::CGAEncodingAttr &newCGALayout) {
  // FIXME: Do we need to change sizePerThread as well?
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    return BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), newCGALayout);
  }

  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    return SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCGALayout(sliceLayout.getParent(), newCGALayout));
  }

  llvm::report_fatal_error("unexpected layout");
  return layout;
}

// The LayOutCGAData is to keep some information and shared between helper
// classes in this file. These information include
//  - Ops that should not changed again.
//  - inferred preferred encoding
//  - preferred encoding that needs to be propagate forward or backward.
//  - among other misc
class LayOutCGAData {
public:
  bool setValueLayout(Value v, ttg::DistributedEncodingTrait encoding) {
    if (valToEncoding.contains(v))
      return false;
    valToEncoding[v] = encoding;
    return true;
  }

  std::optional<ttg::DistributedEncodingTrait> getLayout(Value v) {
    if (!valToEncoding.contains(v))
      return std::nullopt;
    return valToEncoding[v];
  }

  void addBackwardPropagation(OpOperand *opnd,
                              ttg::DistributedEncodingTrait encoding) {
    LDBG(DEBUG_INDENT << "add backward propagation root: "
                      << opnd->getOperandNumber()
                      << "-th operand of value: " << opnd->get());
    propagateBkwd.push_back(opnd);
    (void)setValueLayout(opnd->get(), encoding);
  }

  void addForwardPropagation(Value result,
                             ttg::DistributedEncodingTrait encoding) {
    propagateFwd.push_back(result);
    (void)setValueLayout(result, encoding);
  }

  FuncOp funcOp;
  std::vector<Value> propagateFwd;
  std::vector<OpOperand *> propagateBkwd;
  llvm::DenseSet<const Operation *> noRewrite;
  llvm::DenseMap<Value, ttg::DistributedEncodingTrait> valToEncoding;
};

std::string printCGALayout(Attribute layout) {
  auto cgaLayout = ttg::getCGALayout(layout).getLinearLayout();
  auto str = cgaLayout.toString();
  std::replace(str.begin(), str.end(), (char)'\n', (char)',');
  return "{" + str + "}";
}

//////////////////////////////////////////////////////////////////////////////
//
//  MulticastDotOp and MulticastDotScaledOp is to decide CTA split and
// change source and result's operands' encodings, accordingly.
//
/////////////////////////////////////////////////////////////////////////////
//
SmallVector<unsigned, 2> decideCTASplit(RankedTensorType ty);

class MulticastDotBase {
public:
  MulticastDotBase(LayOutCGAData &_data) : data(_data) {}

  void changeOperandType(mlir::OpOperand &opnd, RankedTensorType newTy);
  void changeResultType(Operation *dot, RankedTensorType newTy);

protected:
  LayOutCGAData &data;
};

class MulticastDotOp : public MulticastDotBase {
public:
  MulticastDotOp(LayOutCGAData &layoutData, tt::DotOp _dotOp)
      : MulticastDotBase(layoutData), dotOp(_dotOp) {}
  bool updateInPlace(bool &changed);

private:
  static SmallVector<unsigned, 2> decideCTASplit(tt::DotOp dotOp) {
    return ::decideCTASplit(cast<RankedTensorType>(dotOp.getType()));
  }

  void changeC(RankedTensorType newTy) {
    auto &opndC = dotOp.getCMutable();
    changeOperandType(opndC, newTy);
  }
  void changeAOrB(mlir::OpOperand &opnd, ttg::DotOperandEncodingAttr newLayout);

  tt::DotOp dotOp;
};

class MulticastDotScaledOp : public MulticastDotBase {
public:
  MulticastDotScaledOp(LayOutCGAData &layoutData, tt::DotScaledOp _dotScaledOp)
      : MulticastDotBase(layoutData), dotScaledOp(_dotScaledOp) {}
  bool updateInPlace(bool &changed);

  enum DotScaleOpndKind {
    DSOK_A = 0,
    DSOK_B = 1,
    DSOK_A_SCALE = 2,
    DSOK_B_SCALE = 3,
  };

private:
  static SmallVector<unsigned, 2> decideCTASplit(tt::DotScaledOp dotScaledOp) {
    return ::decideCTASplit(cast<RankedTensorType>(dotScaledOp.getType()));
  }

  ttg::CGAEncodingAttr inferOperandCGALayout(ttg::CGAEncodingAttr parent,
                                             DotScaleOpndKind opndKind);

  void changeNonScaleOperand(mlir::OpOperand &opnd,
                             ttg::DotOperandEncodingAttr newLayout);
  void changeScaleOperand(mlir::OpOperand &opnd,
                          ttg::DotOperandEncodingAttr newLayout);

  tt::DotScaledOp dotScaledOp;
};

void MulticastDotBase::changeOperandType(mlir::OpOperand &opnd,
                                         RankedTensorType newTy) {
  auto op = opnd.getOwner();

  OpBuilder builder(op);
  builder.setInsertionPoint(op);

  auto oldDef = opnd.get().getDefiningOp();
  auto cvt =
      ttg::ConvertLayoutOp::create(builder, op->getLoc(), newTy, opnd.get());
  opnd.assign(cvt);

  if (oldDef && mlir::isOpTriviallyDead(oldDef))
    oldDef->erase();

  data.addBackwardPropagation(
      &cvt->getOpOperands()[0],
      cast<ttg::DistributedEncodingTrait>(newTy.getEncoding()));
  data.noRewrite.insert(cvt);
}

void MulticastDotBase::changeResultType(Operation *dot,
                                        RankedTensorType newTy) {
  OpBuilder builder(dot);
  builder.setInsertionPointAfter(dot);

  Value result = dot->getResult(0);
  Type oldType = result.getType();

  // Change result type to the given new type, and insert a cvt-op to convert
  // the new type back to the old type.
  result.setType(newTy);
  auto cvt =
      ttg::ConvertLayoutOp::create(builder, dot->getLoc(), oldType, result);
  result.replaceAllUsesExcept(cvt, cvt);

  // Forward propagate desired layout `newDLayout` from cvt's LHS to RHS (The
  // result of cvt mapped to newDLayout) and then propagate onward.
  //
  data.addForwardPropagation(
      cvt.getResult(),
      cast<ttg::DistributedEncodingTrait>(newTy.getEncoding()));
  data.noRewrite.insert(dot);
}

// Current heuristic is to split in way such that the resulting sub-block is
// relatively balanced in M and N dimension.
SmallVector<unsigned, 2> decideCTASplit(RankedTensorType ty) {
  unsigned M = ty.getShape()[0];
  unsigned N = ty.getShape()[1];

  auto numCTAs = ttg::getNumCTAs(ty.getEncoding());
  assert(numCTAs <= 16 && (((numCTAs - 1) & numCTAs) == 0) &&
         "numCTAs must be power-of-2 and not greater than 16");

  llvm::SmallVector<unsigned, 2> split(2);
  llvm::SmallVector<unsigned, 2> mnSize(2);
  int minSize = 16;
  split[0] = split[1] = 0;

  // We can only multicast to 5 WGs (4 due to pow2 constraint). TODO: make
  // this info configurable.
  int maxMultCastWG = 4;
  for (int sm = 1, smMax = std::min(maxMultCastWG, int(numCTAs)); sm <= smMax;
       sm *= 2) {
    int sn = numCTAs / sm;
    if ((M % sm) || (N % sn) || (sn > maxMultCastWG))
      continue;

    int mSize = M / sm;
    int nSize = N / sn;

    if (mSize < minSize || nSize < minSize)
      continue;

    bool better = false;
    if (split[0] == 0)
      better = true;
    else {
      float ratio1 = std::max(mnSize[0], mnSize[1]) /
                     float(std::min(mnSize[0], mnSize[1]));
      float ratio2 = std::max(mSize, nSize) / float(std::min(mSize, nSize));
      better = ratio2 < ratio1;
    }

    if (better) {
      split[0] = sm;
      split[1] = sn;
      mnSize[0] = mSize;
      mnSize[1] = nSize;
    }
  }

  assert(split[0]);
  LDBG(DEBUG_DOT << "m-split: " << split[0] << ", n-split: " << split[1]);
  return split;
}

// TODO: I don't know for sure if we can directly add ConvertLayoutOp to bridge
// the old A and B and the tt.dot. A/B has special encoding
// DotOperandEncodingAttr. Not sure if it can incur problem if an op, whose
// resulting type has DotOperandEncodingAttr, can feed to a non-tt.dot Op.
//
// To avoid potential bug, we let ConvertLayoutOp to convert from A/B's source
// operand instead of directly from A/B.
//
void MulticastDotOp::changeAOrB(mlir::OpOperand &opnd,
                                ttg::DotOperandEncodingAttr newLayout) {
  auto oldType = cast<RankedTensorType>(opnd.get().getType());
  auto newTy = oldType.cloneWithEncoding(newLayout);

  auto opndDef = opnd.get().getDefiningOp();
  if (auto cvt = dyn_cast_or_null<ttg::ConvertLayoutOp>(opndDef)) {
    // Assume the original op sequence is following:
    //    x = cvt y # x have DotOperandEncodingAttr
    //    tt.dot x (opnd A), ...
    // we change it into following for the timing being
    //    x = cvt y
    //    tt.dot y (opnd A), ...
    // and then call changeOperandType() to insert a new cvt, as following
    //    x = cvt y # likely become dead
    //    z = cvt y # z have DotOperandEncodingAttr
    //    tt.dot z (opnd A), ...
    auto cvtSrc = cvt.getSrc();
    opnd.assign(cvtSrc);
  } else {
    // Never come across this situation, not sure if such situation is illegal.
    LDBG(DEBUG_INDENT
         << " dotOp's A or B operand is not ConvertLayoutOp. Strange!");
  }

  changeOperandType(opnd, newTy);
}

bool MulticastDotOp::updateInPlace(bool &changed) {
  LDBG(DEBUG_DOT << " before transformation: " << dotOp);

  changed = false;
  if (data.noRewrite.contains(dotOp))
    return true;

  auto ctx = dotOp->getContext();
  auto dTy = cast<RankedTensorType>(dotOp.getD().getType());
  auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

  // NOTE that do not return prematurely if newSplit equals to existing split.
  // That is because previous pass set the same split for A/B/C/D, which is
  // wrong. At least one of A and B should have different CGA-layout from
  // C/D's CGA layout.
  auto newSplit = decideCTASplit(dotOp);

  auto aTy = cast<RankedTensorType>(dotOp.getA().getType());
  auto bTy = cast<RankedTensorType>(dotOp.getB().getType());
  auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());

  int numWarps = ttg::lookupNumWarps(dotOp);
  int numThreads = product(dLayout.getThreadsPerWarp());

  auto newCGALayout =
      ttg::CGAEncodingAttr::fromSplitParams(ctx, newSplit, newSplit, {1, 0});
  auto newDLayout = ttg::BlockedEncodingAttr::get(
      ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
      numWarps, numThreads, newCGALayout);

  auto newALayout =
      ttg::DotOperandEncodingAttr::get(ctx, aLayout.getOpIdx(), newDLayout, 0);
  auto newBLayout =
      ttg::DotOperandEncodingAttr::get(ctx, bLayout.getOpIdx(), newDLayout, 0);

  // Change A/B's CGA encoding
  changeAOrB(dotOp.getAMutable(), newALayout);
  changeAOrB(dotOp.getBMutable(), newBLayout);

  // Change C's CGA
  auto newDTy = dTy.cloneWithEncoding(newDLayout);
  changeC(newDTy);

  // Update result's CGA encoding.
  changeResultType(dotOp, newDTy);

  LDBG(DEBUG_DOT << " after transformation: " << dotOp);
  changed = true;
  return true;
}

ttg::CGAEncodingAttr MulticastDotScaledOp::inferOperandCGALayout(
    ttg::CGAEncodingAttr parentCGALayout, DotScaleOpndKind opndKind) {
  const auto &layout = parentCGALayout.getLinearLayout();
  auto ctx = dotScaledOp.getContext();
  auto bases = layout.getBases();
  auto kBlock = StringAttr::get(ctx, "block");
  auto &blockBases = bases[kBlock];
  auto rank = layout.getNumOutDims();
  assert(rank == 2);

  if (opndKind == DSOK_A || opndKind == DSOK_B) {
    // Steal the logic from DotOperandEncodingAttr::getCGALayout()
    auto kDim = opndKind == DSOK_A ? rank - 1 : rank - 2;
    for (auto &basis : blockBases) {
      basis[kDim] = 0;
    }
    auto dims = layout.getOutDims();
    dims[kDim].second = 1;
    return ttg::CGAEncodingAttr::get(
        ctx, LinearLayout(std::move(bases), dims, true));
  }

  if (opndKind == DSOK_A_SCALE) {
    // The cga-layout of A-scale equals to cga-layout of A.
    return inferOperandCGALayout(parentCGALayout, DSOK_A);
  }

  assert(opndKind == DSOK_B_SCALE && "unknown opeerand kind");
  auto resultSplit = parentCGALayout.getCTASplitNum();
  unsigned nSplit = resultSplit[1];
  unsigned numCtas = mlir::product(parentCGALayout.getCTAsPerCGA());

  return ttg::CGAEncodingAttr::fromSplitParams(
      ctx,
      /*CTAsPerCGA=*/{nSplit, numCtas / nSplit},
      /*CTASplitNum=*/{nSplit, 1}, /*CTAOrder*/ {0, 1});
}

bool MulticastDotScaledOp::updateInPlace(bool &changed) {
  LDBG(DEBUG_DOT << " before transformation: " << dotScaledOp);

  changed = false;
  if (data.noRewrite.contains(dotScaledOp))
    return true;

  auto ctx = dotScaledOp->getContext();
  auto dTy = cast<RankedTensorType>(dotScaledOp.getD().getType());

  // step 1: Decide the CGA split and infer the CGA layout accordingly.
  auto newSplit = decideCTASplit(dotScaledOp);
  auto dNewCGALayout =
      ttg::CGAEncodingAttr::fromSplitParams(ctx, newSplit, newSplit, {1, 0});

  // step 2: Change CGA layout for A/B/A-scale/B-scale
  {
    SmallVector<std::pair<OpOperand *, DotScaleOpndKind>, 2> opndInfo;
    opndInfo.push_back(std::make_pair(&dotScaledOp.getAMutable(), DSOK_A));
    opndInfo.push_back(std::make_pair(&dotScaledOp.getBMutable(), DSOK_B));

    auto maybeScale = dotScaledOp.getAScaleMutable();
    if (!maybeScale.empty())
      opndInfo.push_back(std::make_pair(&*maybeScale.begin(), DSOK_A_SCALE));

    maybeScale = dotScaledOp.getBScaleMutable();
    if (!maybeScale.empty())
      opndInfo.push_back(std::make_pair(&*maybeScale.begin(), DSOK_B_SCALE));
    for (auto iter : opndInfo) {
      OpOperand &opnd = *iter.first;
      auto opndKind = iter.second;

      auto ty = cast<RankedTensorType>(opnd.get().getType());
      auto oldEnc = cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
      auto cgaLayout = inferOperandCGALayout(dNewCGALayout, opndKind);
      auto newEnc = replaceCGALayout(oldEnc, cgaLayout);
      changeOperandType(opnd, ty.cloneWithEncoding(newEnc));
    }
  }

  // step 3: Change C's CGA layout
  {
    auto &opnd = dotScaledOp.getCMutable();
    auto ty = cast<RankedTensorType>(opnd.get().getType());
    auto oldEnc = cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
    auto newEnc = replaceCGALayout(oldEnc, dNewCGALayout);
    changeOperandType(opnd, ty.cloneWithEncoding(newEnc));
  }

  // step 4: Change D's CGA layout
  {
    auto dEnc = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());
    auto dNewEnc = replaceCGALayout(dEnc, dNewCGALayout);
    changeResultType(dotScaledOp, dTy.cloneWithEncoding(dNewEnc));
  }

  changed = true;
  LDBG(DEBUG_DOT << " after transformation: " << dotScaledOp);
  return true;
}

////////////////////////////////////////////////////////////////////////
//
//            Layout Propagation
//
////////////////////////////////////////////////////////////////////////
//
class PropagateLayout {
public:
  PropagateLayout(LayOutCGAData &_data) : data(_data) {}
  void propagateForward();
  void propagateBackward();
  void propagateLayout();

private:
  void updateLayout(Value value, ttg::DistributedEncodingTrait layout,
                    SmallVector<Value> &changed);
  void inferLhsLayout(Operation *op, ValueRange opResults,
                      ttg::DistributedEncodingTrait rhsLayout);
  void propagateToUsers(Value from, ttg::DistributedEncodingTrait layout);

private:
  LayOutCGAData &data;
};

// This function is to set layout for some of op results. The "opResults" is a
// set of results need to be check; the "rhsLayout" is a source operand's
// layout. This function needs to infer LHS's layout from given RHS layout.
void PropagateLayout::inferLhsLayout(Operation *op, ValueRange opResults,
                                     ttg::DistributedEncodingTrait rhsLayout) {

  for (Value result : opResults) {
    if (!isa<RankedTensorType>(result.getType()))
      continue;

    auto layout = data.getLayout(result);
    if (layout.has_value()) {
      // The layout was determined before (likely due to backward propagation),
      // respect the layout.
      continue;
    }

    Attribute dstLayout;
    if (isa<ttg::ConvertLayoutOp>(op))
      dstLayout = rhsLayout;
    else
      dstLayout = inferDstEncoding(op, rhsLayout);

    if (!dstLayout || !isa<ttg::DistributedEncodingTrait>(dstLayout))
      continue;

    data.setValueLayout(result, cast<ttg::DistributedEncodingTrait>(dstLayout));
    data.propagateFwd.push_back(result);
  }
}

void PropagateLayout::propagateToUsers(Value from,
                                       ttg::DistributedEncodingTrait layout) {
  for (OpOperand &use : from.getUses()) {
    Operation *op = use.getOwner();

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      inferLhsLayout(op, result, layout);
      continue;
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      inferLhsLayout(op, arg, layout);
      continue;
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto parent = yieldOp->getParentOp();

      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        inferLhsLayout(parent, parent->getResult(use.getOperandNumber()),
                       layout);

      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        auto value = forOp.getRegionIterArg(use.getOperandNumber());
        inferLhsLayout(parent, value, layout);

        // From SSA's perspective, IterArg is both a definition and use, and
        // hence need to propagate backward.
        OpOperand *initOperand = forOp.getTiedLoopInit(value);
        data.addBackwardPropagation(initOperand, layout);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        inferLhsLayout(parent,
                       whileOp.getBeforeArguments()[use.getOperandNumber()],
                       layout);
      continue;
    }

    if (auto conditionOp = dyn_cast<scf::ConditionOp>(op)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      inferLhsLayout(op, {afterArg, result}, layout);
      continue;
    }

    if (auto gatherOp = dyn_cast<GatherOp>(op)) {
      inferLhsLayout(op, gatherOp.getResult(), layout);
      continue;
    }

    if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        op->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ttg::ConvertLayoutOp>(op)) {
      inferLhsLayout(op, op->getResults(), layout);
      continue;
    }
  }
}

void PropagateLayout::propagateForward() {
  LDBG(DEBUG_FWD << "Start forward propagation");
  while (!data.propagateFwd.empty()) {
    Value value = data.propagateFwd.back();
    data.propagateFwd.pop_back();

    auto layout = data.getLayout(value);
    assert(layout.has_value());

    propagateToUsers(value, *layout);
  }
}

void PropagateLayout::propagateBackward() {
  LDBG(DEBUG_BKWD << "Start backward propagation");

  auto addBkwdPropagationRoot = [&](OpOperand *opnd,
                                    ttg::DistributedEncodingTrait encoding) {
    auto value = opnd->get();
    if (data.getLayout(value).has_value())
      return;

    auto defOp = value.getDefiningOp();
    if (defOp && data.noRewrite.contains(defOp))
      return;

    data.addBackwardPropagation(opnd, encoding);
  };

  while (!data.propagateBkwd.empty()) {
    OpOperand *root = data.propagateBkwd.back();
    auto rootValue = root->get();
    data.propagateBkwd.pop_back();
    LDBG(DEBUG_BKWD << " from " << root->getOperandNumber()
                    << "-th operation of :" << *root->getOwner());

    auto encoding = *data.getLayout(rootValue);
    if (auto *defOp = rootValue.getDefiningOp()) {
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(defOp)) {
        addBkwdPropagationRoot(&cvt->getOpOperands()[0], encoding);
        continue;
      }
      auto srcEncoding = inferSrcEncoding(defOp, encoding);
      if (srcEncoding) {
        for (auto [i, opnd] : llvm::enumerate(defOp->getOpOperands())) {
          if (isa<RankedTensorType>(opnd.get().getType()))
            addBkwdPropagationRoot(&defOp->getOpOperands()[0], encoding);
        }
      }
    } else {
      auto blockArg = cast<BlockArgument>(rootValue);
      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
        OpOperand &yieldOperand =
            forOp.getBody()->getTerminator()->getOpOperand(
                blockArg.getArgNumber() - forOp.getNumInductionVars());
        addBkwdPropagationRoot(&*initOperand, encoding);
        addBkwdPropagationRoot(&yieldOperand, encoding);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
//    LayOutCGA is to plan CTA in funcOp scope.
//
////////////////////////////////////////////////////////////////////////
//
class LayOutCGA {
public:
  LayOutCGA(FuncOp funcOp) { data.funcOp = funcOp; }
  bool performDotOpMulticast();
  void propagateLayout() {
    PropagateLayout prop(data);
    while (!data.propagateFwd.empty() && !data.propagateBkwd.empty()) {
      prop.propagateForward();
      prop.propagateBackward();
    }
  }
  void rewrite();

private:
  bool needRewrite(Operation *op) const;
  void generic_rewrite(Operation *op);
  void getOriginalAndPreferedCGALayout(
      Operation *op, SmallVectorImpl<ttg::CGAEncodingAttr> &original,
      SmallVectorImpl<ttg::CGAEncodingAttr> &preferred);
  void insertIfNotDuplicated(SmallVectorImpl<ttg::CGAEncodingAttr> &attrs,
                             ttg::CGAEncodingAttr attr);

private:
  LayOutCGAData data;
};

bool LayOutCGA::performDotOpMulticast() {

  SmallVector<Operation *, 8> dotOps;
  data.funcOp.walk([&](Operation *op) {
    if (isa<tt::DotOp, tt::DotScaledOp>(op))
      dotOps.push_back(op);
  });

  unsigned changedDotOpNum = 0;
  bool succ = true;

  for (auto dot : dotOps) {
    bool localSucc = true;
    bool changed = false;
    if (isa<tt::DotOp>(dot)) {
      MulticastDotOp dotXform(data, cast<tt::DotOp>(dot));
      localSucc = dotXform.updateInPlace(changed);
    } else {
      assert(isa<tt::DotScaledOp>(dot));
      MulticastDotScaledOp dotXform(data, cast<tt::DotScaledOp>(dot));
      localSucc = dotXform.updateInPlace(changed);
    }
    succ = succ && localSucc;
    changedDotOpNum += (changed ? 0 : 1);
  }

  LLVM_DEBUG(DBGS() << changedDotOpNum << " out of " << dotOps.size()
                    << " dot operation were trnasformed.\n";
             if (changedDotOpNum) data.funcOp.dump(););

  return succ;
}

bool LayOutCGA::needRewrite(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (data.valToEncoding.contains(operand))
      return true;
  }

  for (mlir::Value result : op->getResults()) {
    if (data.valToEncoding.contains(result))
      return true;
  }
  return false;
}

void LayOutCGA::rewrite() {
  data.funcOp.walk([&](Operation *op) {
    if (!needRewrite(op))
      return;

    if (isa<StoreOp, tt::DescriptorStoreOp, tt::LoadOp, tt::DescriptorLoadOp>(
            op))
      generic_rewrite(op);
    else if (isa<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      assert(false && "impossible to come across these ops in this pass");
    }
  });
}

void LayOutCGA::insertIfNotDuplicated(
    SmallVectorImpl<ttg::CGAEncodingAttr> &attrs, ttg::CGAEncodingAttr attr) {
  bool duplicated =
      std::any_of(attrs.begin(), attrs.end(),
                  [&](ttg::CGAEncodingAttr iter) { return attr == iter; });

  if (!duplicated)
    attrs.push_back(attr);
}

void LayOutCGA::getOriginalAndPreferedCGALayout(
    Operation *op, SmallVectorImpl<ttg::CGAEncodingAttr> &original,
    SmallVectorImpl<ttg::CGAEncodingAttr> &preferred) {

  original.clear();
  preferred.clear();

  for (OpOperand &operand : op->getOpOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!type)
      continue;

    auto currLayout = cast<ttg::BlockedEncodingAttr>(type.getEncoding());
    auto currCGALayout = ttg::getCGALayout(currLayout);
    insertIfNotDuplicated(original, currCGALayout);

    auto preferLayout = data.getLayout(operand.get());
    if (preferLayout.has_value())
      insertIfNotDuplicated(preferred, ttg::getCGALayout(*preferLayout));
  }

  for (auto result : op->getResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type)
      continue;

    auto currLayout = cast<ttg::BlockedEncodingAttr>(type.getEncoding());
    auto currCGALayout = ttg::getCGALayout(currLayout);
    insertIfNotDuplicated(original, currCGALayout);

    auto preferLayout = data.getLayout(result);
    if (preferLayout.has_value())
      insertIfNotDuplicated(preferred, ttg::getCGALayout(*preferLayout));
  }
}

// This function is applicable to those ops whose RankedTensorType-ed source
// operands or results share the same CGA layout.
//
void LayOutCGA::generic_rewrite(Operation *op) {
  LDBG(DEBUG_REWRITE << *op);
  if (data.noRewrite.contains(op)) {
    LDBG(DEBUG_REWRITE << DEBUG_INDENT << "skip, should not modify\n");
    return;
  }

  llvm::SmallVector<ttg::CGAEncodingAttr, 4> originalCGAs;
  llvm::SmallVector<ttg::CGAEncodingAttr, 4> preferCGAs;
  getOriginalAndPreferedCGALayout(op, originalCGAs, preferCGAs);

  assert(originalCGAs.size() == 1 && "Operands have different CGA layout");

  if (preferCGAs.empty()) {
    LDBG(DEBUG_REWRITE << DEBUG_INDENT << "skip, no preferred CGA\n");
    return;
  }

  if (preferCGAs.size() != 1) {
    LDBG(DEBUG_REWRITE << DEBUG_INDENT
                       << "Op has multiple preferred layout:" << op);
  }

  auto preferCGALayout = *preferCGAs.begin();
  for (OpOperand &operand : op->getOpOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!type)
      continue;

    auto currLayout = cast<ttg::BlockedEncodingAttr>(type.getEncoding());
    auto newLayout = replaceCGALayout(currLayout, preferCGALayout);
    OpBuilder builder(op);
    builder.setInsertionPoint(op);
    Value newOperand = ttg::ConvertLayoutOp::create(
        builder, operand.get().getLoc(), type.cloneWithEncoding(newLayout),
        operand.get());
    op->setOperand(operand.getOperandNumber(), newOperand);
  }

  for (OpResult result : op->getOpResults()) {
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type)
      continue;

    auto currLayout = cast<ttg::BlockedEncodingAttr>(type.getEncoding());
    auto newLayout = replaceCGALayout(currLayout, preferCGALayout);

    auto newType = type.cloneWithEncoding(newLayout);
    result.setType(newType);

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto cvt =
        ttg::ConvertLayoutOp::create(builder, op->getLoc(), type, result);
    result.replaceAllUsesExcept(cvt, cvt);
  }

  data.noRewrite.insert(op);
  LDBG(DEBUG_REWRITE << "after rewrite:" << *op);
}

} // end of anonymous namespace

#define GEN_PASS_DEF_TRITONAMDGPUPLANCGA
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct PlanCGAPass : impl::TritonAMDGPUPlanCGABase<PlanCGAPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet p(ctx);
    ModuleOp mod = getOperation();
    if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
      return;

    LLVM_DEBUG({
      DBGS() << "Module before PlanCTA:\n";
      mod.dump();
    });

    mod.walk([&](FuncOp funcOp) {
      LayOutCGA layoutCTA(funcOp);
      layoutCTA.performDotOpMulticast();
      layoutCTA.propagateLayout();
      layoutCTA.rewrite();
    });

    LLVM_DEBUG({
      DBGS() << "Module after PlanCTA:\n";
      mod.dump();
    });
  }
};

} // namespace mlir
