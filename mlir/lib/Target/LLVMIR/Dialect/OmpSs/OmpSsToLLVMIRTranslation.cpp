//===- OmpSsToLLVMIRTranslation.cpp - Translate OmpSs dialect to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OmpSs dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/OmpSs/OmpSsToLLVMIRTranslation.h"
#include "mlir/Dialect/OmpSs/OmpSsDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OmpSs/OSSIRBuilder.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`.
static void convertOSSOpRegions(Region &region, StringRef blockName,
                                llvm::BasicBlock &sourceBlock,
                                llvm::BasicBlock &continuationBlock,
                                llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation,
                                LogicalResult &bodyGenStatus) {
  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock.getTerminator();

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  SetVector<Block *> blocks = getTopologicallySortedBlocks(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == &continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      bodyGenStatus = failure();
      return;
    }

    // Special handling for `oss.terminator` (we may have more
    // than one): they return the control to the parent OpenMP dialect operation
    // so replace them with the branch to the continuation block. We handle this
    // here to avoid relying inter-function communication through the
    // ModuleTranslation class to set up the correct insertion point. This is
    // also consistent with MLIR's idiom of handling special region terminators
    // in the same code that handles the region-owning operation.
    if (isa<oss::TerminatorOp>(bb->getTerminator()))
      builder.CreateBr(&continuationBlock);
  }
  // Finally, after all blocks have been traversed and values mapped,
  // connect the PHI nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);
}

template<typename Op>
static void gatherLoopClauses(
    Operation &opInst,
    llvm::OmpSsIRBuilder::DirectiveClausesInfo &DirClauses,
    LLVM::ModuleTranslation &moduleTranslation) {
  if (auto lBExprVar = cast<Op>(opInst).getLowerBound())
    DirClauses.LowerBound = moduleTranslation.lookupValue(lBExprVar);
  if (auto uBExprVar = cast<Op>(opInst).getUpperBound())
    DirClauses.UpperBound = moduleTranslation.lookupValue(uBExprVar);
  if (auto stepExprVar = cast<Op>(opInst).getStep())
    DirClauses.Step = moduleTranslation.lookupValue(stepExprVar);
  if (auto loopTypeExprVar = cast<Op>(opInst).getLoopType())
    DirClauses.LoopType = moduleTranslation.lookupValue(loopTypeExprVar);
  if (auto indVarExprVar = cast<Op>(opInst).getIndVar())
    DirClauses.IndVar = moduleTranslation.lookupValue(indVarExprVar);
}

static Type getDSAInnerType(Value val) {
  mlir::LLVM::UndefOp undef = val.template getDefiningOp<mlir::LLVM::UndefOp>();
  return undef.getRes().getType();
}

template<typename Op>
static void gatherDirectiveClauses(
    Operation &opInst,
    llvm::OmpSsIRBuilder::DirectiveClausesInfo &DirClauses,
    LLVM::ModuleTranslation &moduleTranslation) {
  if constexpr (Op::allowsIf())
    if (auto ifExprVar = cast<Op>(opInst).getIfExprVar())
      DirClauses.If = moduleTranslation.lookupValue(ifExprVar);
  if constexpr (Op::allowsFinal())
    if (auto finalExprVar = cast<Op>(opInst).getFinalExprVar())
      DirClauses.Final = moduleTranslation.lookupValue(finalExprVar);
  if constexpr (Op::allowsCost())
    if (auto costExprVar = cast<Op>(opInst).getCostExprVar())
      DirClauses.Cost = moduleTranslation.lookupValue(costExprVar);
  if constexpr (Op::allowsPriority())
    if (auto priorityExprVar = cast<Op>(opInst).getPriorityExprVar())
      DirClauses.Priority = moduleTranslation.lookupValue(priorityExprVar);
  if constexpr (Op::allowsShared()) {
    assert(cast<Op>(opInst).getSharedVars().size() == cast<Op>(opInst).getSharedTypeVars().size());
    int i = 0;
    for (auto sharedExprVar : cast<Op>(opInst).getSharedVars())
      DirClauses.Shareds.insert(std::make_pair(
        moduleTranslation.lookupValue(sharedExprVar),
        moduleTranslation.convertType(getDSAInnerType(cast<Op>(opInst).getSharedTypeVars()[i++]))));
  }
  if constexpr (Op::allowsPrivate()) {
    assert(cast<Op>(opInst).getPrivateVars().size() == cast<Op>(opInst).getPrivateTypeVars().size());
    int i = 0;
    for (auto privateExprVar : cast<Op>(opInst).getPrivateVars())
      DirClauses.Privates.insert(std::make_pair(
        moduleTranslation.lookupValue(privateExprVar),
        moduleTranslation.convertType(getDSAInnerType(cast<Op>(opInst).getPrivateTypeVars()[i++]))));
    for (auto initExprVar : cast<Op>(opInst).getInitVars()) {
      Operation *op = initExprVar.getDefiningOp();
      auto copyOp = cast<oss::CopyOp>(op);
      DirClauses.Inits.emplace_back(
        moduleTranslation.lookupValue(copyOp.getBase()),
        moduleTranslation.lookupFunction(copyOp.getFunction()));
    }
  }
  if constexpr (Op::allowsFirstprivate()) {
    assert(cast<Op>(opInst).getFirstprivateVars().size() == cast<Op>(opInst).getFirstprivateTypeVars().size());
    int i = 0;
    for (auto firstprivateExprVar : cast<Op>(opInst).getFirstprivateVars())
      DirClauses.Firstprivates.insert(std::make_pair(
        moduleTranslation.lookupValue(firstprivateExprVar),
        moduleTranslation.convertType(getDSAInnerType(cast<Op>(opInst).getFirstprivateTypeVars()[i++]))));
    for (auto copyExprVar : cast<Op>(opInst).getCopyVars()) {
      Operation *op = copyExprVar.getDefiningOp();
      auto copyOp = cast<oss::CopyOp>(op);
      DirClauses.Copies.emplace_back(
        moduleTranslation.lookupValue(copyOp.getBase()),
        moduleTranslation.lookupFunction(copyOp.getFunction()));
    }
  }
  if constexpr (Op::allowsPrivate() || Op::allowsFirstprivate())
    for (auto deinitExprVar : cast<Op>(opInst).getDeinitVars()) {
      Operation *op = deinitExprVar.getDefiningOp();
      auto deinitOp = cast<oss::CopyOp>(op);
      DirClauses.Deinits.emplace_back(
        moduleTranslation.lookupValue(deinitOp.getBase()),
        moduleTranslation.lookupFunction(deinitOp.getFunction()));
    }
  if constexpr (Op::allowsVlaDims())
    for (auto vlaDimExprVar : cast<Op>(opInst).getVlaDimsVars()){
      Operation *op = vlaDimExprVar.getDefiningOp();
      auto vlaDimOp = cast<oss::VlaDimOp>(op);

      SmallVector<llvm::Value *, 4> TmpList;

      TmpList.push_back(moduleTranslation.lookupValue(vlaDimOp.getPointer()));
      for (auto arg : vlaDimOp.getSizes()){
        TmpList.push_back(moduleTranslation.lookupValue(arg));
      }

      DirClauses.VlaDims.push_back(TmpList);
    }
  if constexpr (Op::allowsCaptures())
    for (auto captureExprVar : cast<Op>(opInst).getCapturesVars())
      DirClauses.Captures.push_back(moduleTranslation.lookupValue(captureExprVar));
  auto processDep =
    [&moduleTranslation](
      mlir::Operation::operand_range range, SmallVector< SmallVector<llvm::Value *, 4>, 4> &List) {
    for (auto exprVar : range) {
      // The operation is just used to contain the real values.
      Operation *op = exprVar.getDefiningOp();
      auto depOp = cast<oss::DepOp>(op);

      SmallVector<llvm::Value *, 4> TmpList;

      TmpList.push_back(moduleTranslation.lookupValue(depOp.getBase()));

      TmpList.push_back(
        llvm::ConstantDataArray::getString(moduleTranslation.getLLVMContext(), "dep string"));

      TmpList.push_back(moduleTranslation.lookupFunction(depOp.getFunction()));
      for (auto arg : depOp.getArguments())
        TmpList.push_back(moduleTranslation.lookupValue(arg));
      List.push_back(TmpList);
    }
  };
  if constexpr (Op::allowsIn())
    processDep(cast<Op>(opInst).getInVars(), DirClauses.Ins);
  if constexpr (Op::allowsOut())
    processDep(cast<Op>(opInst).getOutVars(), DirClauses.Outs);
  if constexpr (Op::allowsInout())
    processDep(cast<Op>(opInst).getInoutVars(), DirClauses.Inouts);
  if constexpr (Op::allowsConcurrent())
    processDep(cast<Op>(opInst).getConcurrentVars(), DirClauses.Concurrents);
  if constexpr (Op::allowsCommutative())
    processDep(cast<Op>(opInst).getCommutativeVars(), DirClauses.Commutatives);
  if constexpr (Op::allowsWeakIn())
    processDep(cast<Op>(opInst).getWeakinVars(), DirClauses.WeakIns);
  if constexpr (Op::allowsWeakOut())
    processDep(cast<Op>(opInst).getWeakoutVars(), DirClauses.WeakOuts);
  if constexpr (Op::allowsWeakInout())
    processDep(cast<Op>(opInst).getWeakinoutVars(), DirClauses.WeakInouts);
  if constexpr (Op::allowsWeakConcurrent())
    processDep(cast<Op>(opInst).getWeakconcurrentVars(), DirClauses.WeakConcurrents);
  if constexpr (Op::allowsWeakCommutative())
    processDep(cast<Op>(opInst).getWeakcommutativeVars(), DirClauses.WeakCommutatives);
}

static LogicalResult
convertOSSTask(
    Operation &opInst, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {

  using InsertPointTy = llvm::OmpSsIRBuilder::InsertPointTy;
  using DirectiveClausesInfo = llvm::OmpSsIRBuilder::DirectiveClausesInfo;
  // TODO: support error propagation in OmpSsIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy beforeRegionIP, llvm::BasicBlock &continuationBB) {
    // TaskOp has only `1` region associated with it.
    auto &region = cast<oss::TaskOp>(opInst).getRegion();
    convertOSSOpRegions(
      region, "oss.task.region", *beforeRegionIP.getBlock(),
      continuationBB, builder, moduleTranslation, bodyGenStatus);
  };

  DirectiveClausesInfo DirClauses;
  gatherDirectiveClauses<oss::TaskOp>(opInst, DirClauses, moduleTranslation);

  llvm::OmpSsIRBuilder::LocationDescription ossLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(
      moduleTranslation.getOmpSsBuilder()->createTask(ossLoc, bodyGenCB, DirClauses));

  if (failed(bodyGenStatus))
    return failure();
  return success();
}

static LogicalResult
convertOSSTaskFor(
    Operation &opInst, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {

  using InsertPointTy = llvm::OmpSsIRBuilder::InsertPointTy;
  using DirectiveClausesInfo = llvm::OmpSsIRBuilder::DirectiveClausesInfo;
  using OmpSsDirectiveKind = llvm::OmpSsIRBuilder::OmpSsDirectiveKind;
  // TODO: support error propagation in OmpSsIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy beforeRegionIP, llvm::BasicBlock &continuationBB) {
    // TaskForOp has only `1` region associated with it.
    auto &region = cast<oss::TaskForOp>(opInst).getRegion();
    convertOSSOpRegions(
      region, "oss.taskfor.region", *beforeRegionIP.getBlock(),
      continuationBB, builder, moduleTranslation, bodyGenStatus);
  };

  DirectiveClausesInfo DirClauses;
  if (auto chunksizeExprVar = cast<oss::TaskForOp>(opInst).getChunksizeExprVar())
    DirClauses.Chunksize = moduleTranslation.lookupValue(chunksizeExprVar);
  gatherDirectiveClauses<oss::TaskForOp>(opInst, DirClauses, moduleTranslation);
  gatherLoopClauses<oss::TaskForOp>(opInst, DirClauses, moduleTranslation);

  llvm::OmpSsIRBuilder::LocationDescription ossLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(
      moduleTranslation.getOmpSsBuilder()->createLoop(
        OmpSsDirectiveKind::OSSD_task_for, ossLoc, bodyGenCB, DirClauses));

  if (failed(bodyGenStatus))
    return failure();
  return success();
}

static LogicalResult
convertOSSTaskloop(
    Operation &opInst, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {

  using InsertPointTy = llvm::OmpSsIRBuilder::InsertPointTy;
  using DirectiveClausesInfo = llvm::OmpSsIRBuilder::DirectiveClausesInfo;
  using OmpSsDirectiveKind = llvm::OmpSsIRBuilder::OmpSsDirectiveKind;
  // TODO: support error propagation in OmpSsIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy beforeRegionIP, llvm::BasicBlock &continuationBB) {
    // TaskloopOp has only `1` region associated with it.
    auto &region = cast<oss::TaskloopOp>(opInst).getRegion();
    convertOSSOpRegions(
      region, "oss.taskloop.region", *beforeRegionIP.getBlock(),
      continuationBB, builder, moduleTranslation, bodyGenStatus);
  };

  DirectiveClausesInfo DirClauses;
  if (auto grainsizeExprVar = cast<oss::TaskloopOp>(opInst).getGrainsizeExprVar())
    DirClauses.Grainsize = moduleTranslation.lookupValue(grainsizeExprVar);
  gatherDirectiveClauses<oss::TaskloopOp>(opInst, DirClauses, moduleTranslation);
  gatherLoopClauses<oss::TaskloopOp>(opInst, DirClauses, moduleTranslation);

  llvm::OmpSsIRBuilder::LocationDescription ossLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(
      moduleTranslation.getOmpSsBuilder()->createLoop(
        OmpSsDirectiveKind::OSSD_taskloop, ossLoc, bodyGenCB, DirClauses));

  if (failed(bodyGenStatus))
    return failure();
  return success();
}

static LogicalResult
convertOSSTaskloopFor(
    Operation &opInst, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {

  using InsertPointTy = llvm::OmpSsIRBuilder::InsertPointTy;
  using DirectiveClausesInfo = llvm::OmpSsIRBuilder::DirectiveClausesInfo;
  using OmpSsDirectiveKind = llvm::OmpSsIRBuilder::OmpSsDirectiveKind;
  // TODO: support error propagation in OmpSsIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy beforeRegionIP, llvm::BasicBlock &continuationBB) {
    // TaskloopForOp has only `1` region associated with it.
    auto &region = cast<oss::TaskloopForOp>(opInst).getRegion();
    convertOSSOpRegions(
      region, "oss.taskloopfor.region", *beforeRegionIP.getBlock(),
      continuationBB, builder, moduleTranslation, bodyGenStatus);
  };

  DirectiveClausesInfo DirClauses;
  if (auto chunksizeExprVar = cast<oss::TaskloopForOp>(opInst).getChunksizeExprVar())
    DirClauses.Chunksize = moduleTranslation.lookupValue(chunksizeExprVar);
  if (auto grainsizeExprVar = cast<oss::TaskloopForOp>(opInst).getGrainsizeExprVar())
    DirClauses.Grainsize = moduleTranslation.lookupValue(grainsizeExprVar);
  gatherDirectiveClauses<oss::TaskloopForOp>(opInst, DirClauses, moduleTranslation);
  gatherLoopClauses<oss::TaskloopForOp>(opInst, DirClauses, moduleTranslation);

  llvm::OmpSsIRBuilder::LocationDescription ossLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  builder.restoreIP(
      moduleTranslation.getOmpSsBuilder()->createLoop(
        OmpSsDirectiveKind::OSSD_taskloop_for, ossLoc, bodyGenCB, DirClauses));

  if (failed(bodyGenStatus))
    return failure();
  return success();
}

static LogicalResult
convertOSSRelease(
    Operation &opInst, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) {

  using DirectiveClausesInfo = llvm::OmpSsIRBuilder::DirectiveClausesInfo;

  DirectiveClausesInfo DirClauses;
  gatherDirectiveClauses<oss::ReleaseOp>(opInst, DirClauses, moduleTranslation);

  llvm::OmpSsIRBuilder::LocationDescription ossLoc(
      builder.saveIP(), builder.getCurrentDebugLocation());
  moduleTranslation.getOmpSsBuilder()->createRelease(ossLoc, DirClauses);

  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OmpSs-2 dialect to LLVM IR.
class OmpSsDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // end namespace

/// Given an OmpSs-2 MLIR operation, create the corresponding LLVM IR
/// (including OmpSs-2 runtime calls).
LogicalResult
OmpSsDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  llvm::OmpSsIRBuilder *ossBuilder = moduleTranslation.getOmpSsBuilder();

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](oss::TaskOp) { return convertOSSTask(*op, builder, moduleTranslation); })
      .Case([&](oss::TaskForOp) { return convertOSSTaskFor(*op, builder, moduleTranslation); })
      .Case([&](oss::TaskloopOp) { return convertOSSTaskloop(*op, builder, moduleTranslation); })
      .Case([&](oss::TaskloopForOp) { return convertOSSTaskloopFor(*op, builder, moduleTranslation); })
      .Case([&](oss::TaskwaitOp) {
        llvm::OmpSsIRBuilder::LocationDescription ossLoc(
            builder.saveIP(), builder.getCurrentDebugLocation());
        ossBuilder->createTaskwait(ossLoc);
        return success();
      })
      .Case([&](oss::ReleaseOp) { return convertOSSRelease(*op, builder, moduleTranslation); })
      .Case([&](oss::TerminatorOp) {
        // `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OmpSs-2 terminator with operands");
        return success();
      })
      .Case([&](oss::DepOp) {
        return success();
      })
      .Case([&](oss::CopyOp) {
        return success();
      })
      .Case([&](oss::VlaDimOp) {
        return success();
      })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OmpSs-2 operation: ")
               << inst->getName();
      });
}

void mlir::registerOmpSsDialectTranslation(DialectRegistry &registry) {
  registry.insert<oss::OmpSsDialect>();
  registry.addExtension(+[](MLIRContext *ctx, oss::OmpSsDialect *dialect) {
    dialect->addInterfaces<OmpSsDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerOmpSsDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOmpSsDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
