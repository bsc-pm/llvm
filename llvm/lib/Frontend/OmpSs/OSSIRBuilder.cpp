//===- OmpSsIRBuilder.cpp - Builder for LLVM-IR for OmpSs directives ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the OmpSsIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for OmpSs directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OmpSs/OSSIRBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include <sstream>

#define DEBUG_TYPE "ompss-2-ir-builder"

using namespace llvm;

void OmpSsIRBuilder::initialize() {
}

void OmpSsIRBuilder::finalize() {
}

static llvm::Type *getDsaOrVlaElemType(
    llvm::Value *Val,
    llvm::Type *Ty,
    const llvm::OmpSsIRBuilder::DirectiveClausesInfo &DirClauses) {

  for (auto &List : DirClauses.VlaDims) {
    if (find(List, Val) != List.end()) {
      while (llvm::ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
        Ty = ArrTy->getElementType();
      }
      return Ty;
    }
  }
  return Ty;
}


void OmpSsIRBuilder::emitDirectiveData(
    SmallVector<llvm::OperandBundleDef, 8> &TaskInfo,
    const DirectiveClausesInfo &DirClauses) {

  // TODO: remove constants from captures because
  // they are globals in llvm
  SmallVector<llvm::Value*, 4> CapturedList;
  CapturedList.append(DirClauses.Captures.begin(), DirClauses.Captures.end());

  if (DirClauses.LowerBound) {
    TaskInfo.emplace_back("QUAL.OSS.LOOP.LOWER.BOUND", DirClauses.LowerBound);
    CapturedList.push_back(DirClauses.LowerBound);
  }
  if (DirClauses.UpperBound) {
    TaskInfo.emplace_back("QUAL.OSS.LOOP.UPPER.BOUND", DirClauses.UpperBound);
    CapturedList.push_back(DirClauses.UpperBound);
  }
  if (DirClauses.Step) {
    TaskInfo.emplace_back("QUAL.OSS.LOOP.STEP", DirClauses.Step);
    CapturedList.push_back(DirClauses.Step);
  }
  if (DirClauses.LoopType) {
    // [Fortran]
    // DO constructs are signed
    SmallVector<llvm::Value*, 4> LoopTypeList;
    LoopTypeList.push_back(DirClauses.LoopType);
    LoopTypeList.push_back(Builder.getInt64(1));
    LoopTypeList.push_back(Builder.getInt64(1));
    LoopTypeList.push_back(Builder.getInt64(1));
    LoopTypeList.push_back(Builder.getInt64(1));
    TaskInfo.emplace_back("QUAL.OSS.LOOP.TYPE", LoopTypeList);
  }
  if (DirClauses.IndVar)
    TaskInfo.emplace_back("QUAL.OSS.LOOP.IND.VAR", DirClauses.IndVar);
  if (DirClauses.If)
    TaskInfo.emplace_back("QUAL.OSS.IF", DirClauses.If);
  if (DirClauses.Final)
    TaskInfo.emplace_back("QUAL.OSS.FINAL", DirClauses.Final);
  if (DirClauses.Cost) {
    TaskInfo.emplace_back("QUAL.OSS.COST", DirClauses.Cost);
    CapturedList.push_back(DirClauses.Cost);
  }
  if (DirClauses.Priority) {
    TaskInfo.emplace_back("QUAL.OSS.PRIORITY", DirClauses.Priority);
    CapturedList.push_back(DirClauses.Priority);
  }
  for (auto &p : DirClauses.Shareds)
    TaskInfo.emplace_back(
      "QUAL.OSS.SHARED", ArrayRef<Value*>{p.first, llvm::UndefValue::get(getDsaOrVlaElemType(p.first, p.second, DirClauses))});
  for (auto &p : DirClauses.Privates)
    TaskInfo.emplace_back(
      "QUAL.OSS.PRIVATE", ArrayRef<Value*>{p.first, llvm::UndefValue::get(getDsaOrVlaElemType(p.first, p.second, DirClauses))});
  for (auto &p : DirClauses.Firstprivates)
    TaskInfo.emplace_back(
      "QUAL.OSS.FIRSTPRIVATE", ArrayRef<Value*>{p.first, llvm::UndefValue::get(getDsaOrVlaElemType(p.first, p.second, DirClauses))});
  for (auto &p : DirClauses.Copies)
    TaskInfo.emplace_back(
      "QUAL.OSS.COPY", ArrayRef<Value*>{p.first, p.second});
  for (auto &p : DirClauses.Inits)
    TaskInfo.emplace_back(
      "QUAL.OSS.INIT", ArrayRef<Value*>{p.first, p.second});
  for (auto &p : DirClauses.Deinits)
    TaskInfo.emplace_back(
      "QUAL.OSS.DEINIT", ArrayRef<Value*>{p.first, p.second});

  for (auto &List : DirClauses.VlaDims)
    TaskInfo.emplace_back("QUAL.OSS.VLA.DIMS", List);

  for (auto &List : DirClauses.Ins)
    TaskInfo.emplace_back("QUAL.OSS.DEP.IN", List);
  for (auto &List : DirClauses.Outs)
    TaskInfo.emplace_back("QUAL.OSS.DEP.OUT", List);
  for (auto &List : DirClauses.Inouts)
    TaskInfo.emplace_back("QUAL.OSS.DEP.INOUT", List);
  for (auto &List : DirClauses.Concurrents)
    TaskInfo.emplace_back("QUAL.OSS.DEP.CONCURRENT", List);
  for (auto &List : DirClauses.Commutatives)
    TaskInfo.emplace_back("QUAL.OSS.DEP.COMMUTATIVE", List);
  for (auto &List : DirClauses.WeakIns)
    TaskInfo.emplace_back("QUAL.OSS.DEP.WEAKIN", List);
  for (auto &List : DirClauses.WeakOuts)
    TaskInfo.emplace_back("QUAL.OSS.DEP.WEAKOUT", List);
  for (auto &List : DirClauses.WeakInouts)
    TaskInfo.emplace_back("QUAL.OSS.DEP.WEAKINOUT", List);
  for (auto &List : DirClauses.WeakConcurrents)
    TaskInfo.emplace_back("QUAL.OSS.DEP.WEAKCONCURRENT", List);
  for (auto &List : DirClauses.WeakCommutatives)
    TaskInfo.emplace_back("QUAL.OSS.DEP.WEAKCOMMUTATIVE", List);

  if (DirClauses.Chunksize) {
    TaskInfo.emplace_back("QUAL.OSS.LOOP.CHUNKSIZE", DirClauses.Chunksize);
    CapturedList.push_back(DirClauses.Chunksize);
  }
  if (DirClauses.Grainsize) {
    TaskInfo.emplace_back("QUAL.OSS.LOOP.GRAINSIZE", DirClauses.Grainsize);
    CapturedList.push_back(DirClauses.Grainsize);
  }
  if (!CapturedList.empty())
    TaskInfo.emplace_back("QUAL.OSS.CAPTURED", CapturedList);
}

llvm::OmpSsIRBuilder::InsertPointTy OmpSsIRBuilder::createLoop(
    OmpSsDirectiveKind DKind,
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    const DirectiveClausesInfo &DirClauses) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  std::string DKindStr = "";
  if (DKind == OmpSsDirectiveKind::OSSD_task_for) {
    DKindStr = "TASK.FOR";
  } else if (DKind == OmpSsDirectiveKind::OSSD_taskloop) {
    DKindStr = "TASKLOOP";
  } else if (DKind == OmpSsDirectiveKind::OSSD_taskloop_for) {
    DKindStr = "TASKLOOP.FOR";
  } else {
    llvm_unreachable("unexpected loop directive");
  }

  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      "DIR.OSS",
      llvm::ConstantDataArray::getString(
        M.getContext(), DKindStr));

  emitDirectiveData(TaskInfo, DirClauses);

  // Create entry/exit intrinsics
  Function *EntryFn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_region_entry);
  llvm::Instruction *EntryCall = Builder.CreateCall(EntryFn, {}, TaskInfo);

  Function *ExitFn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_region_exit);
  llvm::Instruction *ExitCall = Builder.CreateCall(ExitFn, EntryCall);

  // Create an artificial insertion point that will also ensure the blocks we
  // are about to split are not degenerated.
  auto *UI = Builder.CreateUnreachable();
  BasicBlock *ExitCallBB = ExitCall->getParent();
  ExitCallBB = ExitCallBB->splitBasicBlock(ExitCall, "oss.end");
  UI->removeFromParent();

  // Let the caller create the body.
  assert(BodyGenCB && "Expected body generation callback!");
  InsertPointTy BeforeRegionIP(EntryCall->getParent(), EntryCall->getParent()->end());
  BodyGenCB(BeforeRegionIP, *ExitCallBB);

  InsertPointTy AfterRegionIP(ExitCallBB, ExitCallBB->end());

  return AfterRegionIP;

}

llvm::OmpSsIRBuilder::InsertPointTy OmpSsIRBuilder::createTask(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    const DirectiveClausesInfo &DirClauses) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  SmallVector<llvm::Value*, 4> CapturedList;
  TaskInfo.emplace_back(
      "DIR.OSS",
      llvm::ConstantDataArray::getString(
        M.getContext(), "TASK"));

  emitDirectiveData(TaskInfo, DirClauses);

  // Create entry/exit intrinsics
  Function *EntryFn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_region_entry);
  llvm::Instruction *EntryCall = Builder.CreateCall(EntryFn, {}, TaskInfo);

  Function *ExitFn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_region_exit);
  llvm::Instruction *ExitCall = Builder.CreateCall(ExitFn, EntryCall);

  // Create an artificial insertion point that will also ensure the blocks we
  // are about to split are not degenerated.
  auto *UI = Builder.CreateUnreachable();
  BasicBlock *ExitCallBB = ExitCall->getParent();
  ExitCallBB = ExitCallBB->splitBasicBlock(ExitCall, "oss.end");
  UI->removeFromParent();

  // Let the caller create the body.
  assert(BodyGenCB && "Expected body generation callback!");
  InsertPointTy BeforeRegionIP(EntryCall->getParent(), EntryCall->getParent()->end());
  BodyGenCB(BeforeRegionIP, *ExitCallBB);

  InsertPointTy AfterRegionIP(ExitCallBB, ExitCallBB->end());

  return AfterRegionIP;
}

void OmpSsIRBuilder::emitTaskwaitImpl(const LocationDescription &Loc) {
  Function *Fn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_marker);

  Builder.CreateCall(
    Fn, {},
    llvm::OperandBundleDef(
      "DIR.OSS",
      llvm::ConstantDataArray::getString(
        M.getContext(), "TASKWAIT")));
}

void OmpSsIRBuilder::createTaskwait(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitTaskwaitImpl(Loc);
}

void OmpSsIRBuilder::createRelease(
    const LocationDescription &Loc,
    const DirectiveClausesInfo &DirClauses) {
  if (!updateToLocation(Loc))
    return;

  Function *Fn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::directive_marker);

  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      "DIR.OSS",
      llvm::ConstantDataArray::getString(
        M.getContext(), "RELEASE"));

  emitDirectiveData(TaskInfo, DirClauses);
  Builder.CreateCall(Fn, {}, TaskInfo);
}

