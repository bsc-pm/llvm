//===- OmpSsRegionAnalysis.cpp - OmpSs Region Analysis -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OmpSsRegionAnalysis.h"
#include "llvm/Analysis/OrderedInstructions.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

static cl::opt<bool>
DisableChecks("disable-checks",
                  cl::desc("Avoid checking OmpSs-2 task uses after task body and DSA matching"),
                  cl::Hidden,
                  cl::init(false));

enum PrintVerbosity {
  PV_Task,
  PV_Uses,
  PV_DsaMissing
};

static cl::opt<PrintVerbosity>
PrintVerboseLevel("print-verbosity",
  cl::desc("Choose verbosity level"),
  cl::Hidden,
  cl::values(
  clEnumValN(PV_Task, "task", "Print task layout only"),
  clEnumValN(PV_Uses, "uses", "Print task layout with uses"),
  clEnumValN(PV_DsaMissing, "dsa_missing", "Print task layout with uses without DSA")));

char OmpSsRegionAnalysisPass::ID = 0;

OmpSsRegionAnalysisPass::OmpSsRegionAnalysisPass() : FunctionPass(ID) {
  initializeOmpSsRegionAnalysisPassPass(*PassRegistry::getPassRegistry());
}

static bool valueInDSABundles(const TaskDSAInfo& DSAInfo,
                              const Value *V) {
  auto SharedIt = find(DSAInfo.Shared, V);
  auto PrivateIt = find(DSAInfo.Private, V);
  auto FirstprivateIt = find(DSAInfo.Firstprivate, V);
  if (SharedIt == DSAInfo.Shared.end()
      && PrivateIt == DSAInfo.Private.end()
      && FirstprivateIt == DSAInfo.Firstprivate.end())
    return false;

  return true;
}

void OmpSsRegionAnalysisPass::print(raw_ostream &OS, const Module *M) const {
  for (auto it = TaskProgramOrder.begin(); it != TaskProgramOrder.end(); ++it) {
    Instruction *I = it->first;
    int Depth = it->second.Depth;
    int Idx = it->second.Idx;
    const TaskAnalysisInfo &AnalysisInfo = FuncAnalysisInfo.PostOrder[Idx];
    const TaskInfo &Info = FuncInfo.PostOrder[Idx];

    dbgs() << std::string(Depth*PrintSpaceMultiplier, ' ') << "[" << Depth << "] ";
    I->printAsOperand(dbgs(), false);

    if (PrintVerboseLevel == PV_Uses) {
      for (size_t j = 0; j < AnalysisInfo.UsesBeforeEntry.size(); ++j) {
        dbgs() << "\n";
        dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
               << "[Before] ";
        AnalysisInfo.UsesBeforeEntry[j]->printAsOperand(dbgs(), false);
      }
      for (size_t j = 0; j < AnalysisInfo.UsesAfterExit.size(); ++j) {
        dbgs() << "\n";
        dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
               << "[After] ";
        AnalysisInfo.UsesAfterExit[j]->printAsOperand(dbgs(), false);
      }
    }
    if (PrintVerboseLevel == PV_DsaMissing) {
      for (size_t j = 0; j < AnalysisInfo.UsesBeforeEntry.size(); ++j) {
        if (!valueInDSABundles(Info.DSAInfo, AnalysisInfo.UsesBeforeEntry[j])) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
          AnalysisInfo.UsesBeforeEntry[j]->printAsOperand(dbgs(), false);
        }
      }
    }
    dbgs() << "\n";
  }
}

TaskFunctionInfo& OmpSsRegionAnalysisPass::getFuncInfo() { return FuncInfo; }

static void getOperandBundlesAsDefsWithID(const IntrinsicInst *I,
                                          SmallVectorImpl<OperandBundleDef> &OpBundles,
                                          uint32_t Id) {

  for (unsigned i = 0, e = I->getNumOperandBundles(); i != e; ++i) {
    OperandBundleUse U = I->getOperandBundleAt(i);
    if (U.getTagID() == Id)
      OpBundles.emplace_back(U);
  }
}

// Gather Value from each OperandBundle Id.
// Error if there is more than one Value in OperandBundle
static void getValueFromOperandBundlesWithID(const IntrinsicInst *I,
                                              SetVector<Value *> &Values,
                                              uint32_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  for (OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 1 && "Only allowed one Value per OperandBundle");
    Values.insert(OBDef.inputs()[0]);
  }
}

void OmpSsRegionAnalysisPass::getTaskFunctionUsesInfo(
    Function &F, DominatorTree &DT, TaskFunctionInfo &FuncInfo,
    TaskFunctionAnalysisInfo &FuncAnalysisInfo,
    MapVector<Instruction *, TaskPrintInfo> &TaskProgramOrder) {

  OrderedInstructions OI(&DT);

  struct Task {
    TaskAnalysisInfo AnalysisInfo;
    TaskInfo Info;
  };

  SmallVector<Task, 2> Stack;

  ReversePostOrderTraversal<BasicBlock *> RPOT(&F.getEntryBlock());
  for (BasicBlock *BB : RPOT) {
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II) {
      Instruction *I = &*II;
      if (auto *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
          assert(II->hasOneUse() && "Task entry has more than one user.");

          TaskPrintInfo &TPI = TaskProgramOrder[II];
          TPI.Depth = Stack.size();

          Instruction *Exit = dyn_cast<Instruction>(II->user_back());
          assert(Exit && "Task exit is not a Instruction.");
          assert(OI.dominates(II, Exit) && "Task entry does not dominate exit.");

          Task T;
          T.Info.Entry = II;
          T.Info.Exit = Exit;

          getValueFromOperandBundlesWithID(II, T.Info.DSAInfo.Shared,
                                            LLVMContext::OB_oss_shared);
          getValueFromOperandBundlesWithID(II, T.Info.DSAInfo.Private,
                                            LLVMContext::OB_oss_private);
          getValueFromOperandBundlesWithID(II, T.Info.DSAInfo.Firstprivate,
                                            LLVMContext::OB_oss_firstprivate);

          Stack.push_back(T);
        } else if (II->getIntrinsicID() == Intrinsic::directive_region_exit) {
          if (Stack.empty())
            llvm_unreachable("Task exit hit without and entry.");

          Task &T = Stack.back();
          Instruction *Entry = T.Info.Entry;

          TaskPrintInfo &TPI = TaskProgramOrder[&*Entry];
          TPI.Idx = FuncInfo.PostOrder.size();

          FuncAnalysisInfo.PostOrder.push_back(T.AnalysisInfo);
          FuncInfo.PostOrder.push_back(T.Info);

          Stack.pop_back();
        }
      } else if (!Stack.empty()) {
        Task &T = Stack.back();
        Instruction *Entry = T.Info.Entry;
        Instruction *Exit = T.Info.Exit;
        for (Use &U : I->operands()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U.get())) {
            if (OI.dominates(I2, Entry)) {
              T.AnalysisInfo.UsesBeforeEntry.insert(I2);
              if (!DisableChecks && !valueInDSABundles(T.Info.DSAInfo, I2)) {
                llvm_unreachable("Value supposed to be inside task entry "
                                 "OperandBundle not found.");
              }
            }
          } else if (Argument *A = dyn_cast<Argument>(U.get())) {
            T.AnalysisInfo.UsesBeforeEntry.insert(A);
            if (!DisableChecks && !valueInDSABundles(T.Info.DSAInfo, A)) {
              llvm_unreachable("Value supposed to be inside task entry "
                               "OperandBundle not found.");
            }
          }
        }
        for (User *U : I->users()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U)) {
            if (OI.dominates(Exit, I2)) {
              T.AnalysisInfo.UsesAfterExit.insert(I);
              if (!DisableChecks) {
                llvm_unreachable("Value inside the task body used after it.");
              }
            }
          }
        }
      }
    }
  }
}

bool OmpSsRegionAnalysisPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  getTaskFunctionUsesInfo(F, DT, FuncInfo, FuncAnalysisInfo, TaskProgramOrder);

  return false;
}

void OmpSsRegionAnalysisPass::releaseMemory() {
  FuncInfo = TaskFunctionInfo();
  FuncAnalysisInfo = TaskFunctionAnalysisInfo();
  TaskProgramOrder.clear();
}

void OmpSsRegionAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
}

INITIALIZE_PASS_BEGIN(OmpSsRegionAnalysisPass, "ompss-2-regions",
                      "Classify OmpSs-2 inside region uses", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(OmpSsRegionAnalysisPass, "ompss-2-regions",
                    "Classify OmpSs-2 inside region uses", false, true)

