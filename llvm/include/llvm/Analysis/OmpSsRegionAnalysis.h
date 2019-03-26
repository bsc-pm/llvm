//===- llvm/Analysis/OmpSsRegionAnalysis.h - OmpSs Region Analysis -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_OMPSSREGIONANALYSIS_H
#define LLVM_ANALYSIS_OMPSSREGIONANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Dominators.h"

namespace llvm {

struct TaskDSAInfo {
  SetVector<Value *> Shared;
  SetVector<Value *> Private;
  SetVector<Value *> Firstprivate;
};

struct TaskInfo {
  TaskDSAInfo DSAInfo;
  Instruction *Entry;
  Instruction *Exit;
};

struct TaskFunctionInfo {
  SmallVector<TaskInfo, 4> PostOrder;
};

class OmpSsRegionAnalysisPass : public FunctionPass {
private:

  struct TaskPrintInfo {
    // Task nesting level
    size_t Depth;
    size_t Idx;
  };

  struct TaskAnalysisInfo {
    SetVector<Value *> UsesBeforeEntry;
    SetVector<Value *> UsesAfterExit;
  };

  struct TaskFunctionAnalysisInfo {
    SmallVector<TaskAnalysisInfo, 4> PostOrder;
  };

  // AnalysisInfo only used in this pass
  TaskFunctionAnalysisInfo FuncAnalysisInfo;
  // Info used by the transform pass
  TaskFunctionInfo FuncInfo;
  // Used to keep the task layout to print it properly, since
  // we save the tasks in post order
  MapVector<Instruction *, TaskPrintInfo> TaskProgramOrder;

  static const int PrintSpaceMultiplier = 2;
  // Walk over each task in RPO identifying uses before entry
  // and after exit. Uses before entry are then matched with DSA info
  // in OperandBundles
  static void getTaskFunctionUsesInfo(
      Function &F, DominatorTree &DT, TaskFunctionInfo &FuncInfo,
      TaskFunctionAnalysisInfo &FuncAnalysisInfo,
      MapVector<Instruction *, TaskPrintInfo> &TaskProgramOrder);

public:
  static char ID;

  OmpSsRegionAnalysisPass();

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "OmpSs-2 Region Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void print(raw_ostream &OS, const Module *M) const override;

  void releaseMemory() override;

  TaskFunctionInfo& getFuncInfo();

};

} // end namespace llvm

#endif // LLVM_ANALYSIS_OMPSSREGIONANALYSIS_H

