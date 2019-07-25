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
  PV_Unpack,
  PV_DsaMissing
};

static cl::opt<PrintVerbosity>
PrintVerboseLevel("print-verbosity",
  cl::desc("Choose verbosity level"),
  cl::Hidden,
  cl::values(
  clEnumValN(PV_Task, "task", "Print task layout only"),
  clEnumValN(PV_Uses, "uses", "Print task layout with uses"),
  clEnumValN(PV_Unpack, "unpack-ins", "Print task layout with unpack instructions needed in dependencies"),
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

static void dump_dependency(int Depth, int PrintSpaceMultiplier, std::string DepType,
                            const SmallVectorImpl<DependInfo> &DepList) {
  for (const DependInfo &DI : DepList) {
    dbgs() << "\n";
    dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
                          << "[" << DepType << "] ";
    dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
    DI.Base->printAsOperand(dbgs(), false);
    for (Instruction * const &I : DI.UnpackInstructions) {
      dbgs() << "\n";
      dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
      I->printAsOperand(dbgs(), false);
    }
  }
}

void OmpSsRegionAnalysisPass::print(raw_ostream &OS, const Module *M) const {
  for (auto it = TaskProgramOrder.begin(); it != TaskProgramOrder.end(); ++it) {
    Instruction *I = it->first;
    int Depth = it->second.Depth;
    int Idx = it->second.Idx;
    const TaskAnalysisInfo &AnalysisInfo = TaskFuncAnalysisInfo.PostOrder[Idx];
    const TaskInfo &Info = FuncInfo.TaskFuncInfo.PostOrder[Idx];

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
    else if (PrintVerboseLevel == PV_DsaMissing) {
      for (size_t j = 0; j < AnalysisInfo.UsesBeforeEntry.size(); ++j) {
        if (!valueInDSABundles(Info.DSAInfo, AnalysisInfo.UsesBeforeEntry[j])) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
          AnalysisInfo.UsesBeforeEntry[j]->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_Unpack) {
      const TaskDependsInfo &TDI = Info.DependsInfo;
      dump_dependency(Depth, PrintSpaceMultiplier, "In", TDI.Ins);
      dump_dependency(Depth, PrintSpaceMultiplier, "Out", TDI.Outs);
      dump_dependency(Depth, PrintSpaceMultiplier, "Inout", TDI.Inouts);

      dump_dependency(Depth, PrintSpaceMultiplier, "WeakIn", TDI.WeakIns);
      dump_dependency(Depth, PrintSpaceMultiplier, "WeakOut", TDI.WeakOuts);
      dump_dependency(Depth, PrintSpaceMultiplier, "WeakInout", TDI.WeakInouts);
    }
    dbgs() << "\n";
  }
}

FunctionInfo& OmpSsRegionAnalysisPass::getFuncInfo() { return FuncInfo; }

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

// Gather Value list from each OperandBundle Id.
static void getValueListFromOperandBundlesWithID(const IntrinsicInst *I,
                                              SetVector<Value *> &Values,
                                              uint32_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  for (OperandBundleDef &OBDef : OpBundles) {
    Values.insert(OBDef.input_begin(), OBDef.input_end());
  }
}

static void gatherDSAInfo(const IntrinsicInst *I, TaskInfo &TI) {
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Shared,
                                    LLVMContext::OB_oss_shared);
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Private,
                                    LLVMContext::OB_oss_private);
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Firstprivate,
                                    LLVMContext::OB_oss_firstprivate);

  getValueListFromOperandBundlesWithID(I, TI.DSAInfo.Shared,
                                    LLVMContext::OB_oss_shared_vla);
  getValueListFromOperandBundlesWithID(I, TI.DSAInfo.Private,
                                    LLVMContext::OB_oss_private_vla);
  getValueListFromOperandBundlesWithID(I, TI.DSAInfo.Firstprivate,
                                    LLVMContext::OB_oss_firstprivate_vla);
}

// Inserts I into InstList ensuring program order if I it's not already in the list
static bool insertInstructionInProgramOrder(SmallVectorImpl<Instruction *> &InstList,
                                            Instruction *I,
                                            const OrderedInstructions &OI) {

  auto It = InstList.begin();
  while (It != InstList.end() && OI.dominates(*It, I))
    ++It;

  if (*It == I)
    return false;
  InstList.insert(It, I);
  return true;
}

static void gatherUnpackInstructions(DependInfo &DI,
                                     const TaskDSAInfo &DSAInfo,
                                     TaskAnalysisInfo &TAI,
                                     const OrderedInstructions &OI) {
  SmallVectorImpl<Instruction *> &UnpackIns = DI.UnpackInstructions;
  SmallPtrSet<ConstantExpr *, 4> &UnpackConsts = DI.UnpackConstants;
  SmallPtrSet<Value *, 4> DSAMerge;

  DSAMerge.insert(DSAInfo.Shared.begin(), DSAInfo.Shared.end());
  DSAMerge.insert(DSAInfo.Private.begin(), DSAInfo.Private.end());
  DSAMerge.insert(DSAInfo.Firstprivate.begin(), DSAInfo.Firstprivate.end());

  Value *Base = DI.Base;

  // First element is the current instruction, second is
  // the Instruction where we come from (the dependency)
  SmallVector<std::pair<Value *, Value *>, 4> WorkList;
  WorkList.emplace_back(Base, Base);
  for (Value *V : DI.Dims)
    WorkList.emplace_back(V, V);

  while (!WorkList.empty()) {
    auto It = WorkList.begin();
    Value *Cur = It->first;
    Value *Dep = It->second;
    WorkList.erase(It);

    if (DSAMerge.find(Cur) == DSAMerge.end()) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Cur)) {
        for (Use &U : CE->operands()) {
          WorkList.emplace_back(U.get(), Dep);
        }
        UnpackConsts.insert(CE);
      }

      if (Instruction *I = dyn_cast<Instruction>(Cur)) {
        for (Use &U : I->operands()) {
          WorkList.emplace_back(U.get(), Dep);
        }
        insertInstructionInProgramOrder(UnpackIns, I, OI);
      }
    } else if (Dep == Base) {
      // Found DSA associated with Dependency, assign SymbolIndex
      if (!TAI.DepSymToIdx.count(Base)) {
        TAI.DepSymToIdx[Base] = TAI.DepSymToIdx.size();
      }
      DI.SymbolIndex = TAI.DepSymToIdx[Base];
    }
  }
}

// Process each OpBundle gathering dependency information
static void gatherDependsInfoFromBundles(const SmallVectorImpl<OperandBundleDef> &OpBundles,
                                    const OrderedInstructions &OI,
                                    const TaskDSAInfo &DSAInfo,
                                    TaskAnalysisInfo &TAI,
                                    SmallVectorImpl<DependInfo> &DependsList) {
  for (const OperandBundleDef &OBDef : OpBundles) {
    DependInfo DI;
    ArrayRef<Value *> OBArgs = OBDef.inputs();

    DI.SymbolIndex = -1;
    // TODO: Support RegionText stringifying clause content
    DI.RegionText = "";
    DI.Base = OBArgs[0];
    for (size_t i = 1; i < OBArgs.size(); ++i) {
      DI.Dims.push_back(OBArgs[i]);
    }

    gatherUnpackInstructions(DI, DSAInfo, TAI, OI);

    DependsList.push_back(DI);
  }
}

// Gathers dependencies needed information of type Id
static void gatherDependsInfoWithID(const IntrinsicInst *I,
                                    const OrderedInstructions &OI,
                                    const TaskDSAInfo &DSAInfo,
                                    TaskAnalysisInfo &TAI,
                                    SmallVectorImpl<DependInfo> &DependsList,
                                    uint64_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  gatherDependsInfoFromBundles(OpBundles, OI, DSAInfo, TAI, DependsList);
}

// Gathers all dependencies needed information
static void gatherDependsInfo(const IntrinsicInst *I, TaskInfo &TI,
                              TaskAnalysisInfo &TAI,
                              const OrderedInstructions &OI) {
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.Ins, LLVMContext::OB_oss_dep_in);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.Outs, LLVMContext::OB_oss_dep_out);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.Inouts, LLVMContext::OB_oss_dep_inout);

  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.WeakIns, LLVMContext::OB_oss_dep_weakin);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.WeakOuts, LLVMContext::OB_oss_dep_weakout);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TAI, TI.DependsInfo.WeakInouts, LLVMContext::OB_oss_dep_weakinout);
  TI.DependsInfo.NumSymbols = TAI.DepSymToIdx.size();
}

void OmpSsRegionAnalysisPass::getOmpSsFunctionInfo(
    Function &F, DominatorTree &DT, FunctionInfo &FI,
    TaskFunctionAnalysisInfo &TFAI,
    MapVector<Instruction *, TaskPrintInfo> &TPO) {

  OrderedInstructions OI(&DT);

  struct Task {
    TaskAnalysisInfo AnalysisInfo;
    TaskInfo Info;
  };

  SmallVector<Task, 2> Stack;

  ReversePostOrderTraversal<BasicBlock *> RPOT(&F.getEntryBlock());
  for (BasicBlock *BB : RPOT) {
    for (Instruction &I : *BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
          assert(II->hasOneUse() && "Task entry has more than one user.");

          TaskPrintInfo &TPI = TPO[II];
          TPI.Depth = Stack.size();

          Instruction *Exit = dyn_cast<Instruction>(II->user_back());
          assert(Exit && "Task exit is not a Instruction.");
          assert(OI.dominates(II, Exit) && "Task entry does not dominate exit.");

          Task T;
          T.Info.Entry = II;
          T.Info.Exit = Exit;

          gatherDSAInfo(II, T.Info);
          gatherDependsInfo(II, T.Info, T.AnalysisInfo, OI);

          Stack.push_back(T);
        } else if (II->getIntrinsicID() == Intrinsic::directive_region_exit) {
          if (Stack.empty())
            llvm_unreachable("Task exit hit without and entry.");

          Task &T = Stack.back();
          Instruction *Entry = T.Info.Entry;

          TaskPrintInfo &TPI = TPO[&*Entry];
          TPI.Idx = FI.TaskFuncInfo.PostOrder.size();

          TFAI.PostOrder.push_back(T.AnalysisInfo);
          FI.TaskFuncInfo.PostOrder.push_back(T.Info);

          Stack.pop_back();
        } else if (II->getIntrinsicID() == Intrinsic::directive_marker) {
          FI.TaskwaitFuncInfo.PostOrder.push_back({II});
        }
      } else if (!Stack.empty()) {
        Task &T = Stack.back();
        Instruction *Entry = T.Info.Entry;
        Instruction *Exit = T.Info.Exit;
        for (Use &U : I.operands()) {
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
        for (User *U : I.users()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U)) {
            if (OI.dominates(Exit, I2)) {
              T.AnalysisInfo.UsesAfterExit.insert(&I);
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
  getOmpSsFunctionInfo(F, DT, FuncInfo, TaskFuncAnalysisInfo, TaskProgramOrder);

  return false;
}

void OmpSsRegionAnalysisPass::releaseMemory() {
  FuncInfo = FunctionInfo();
  TaskFuncAnalysisInfo = TaskFunctionAnalysisInfo();
  TaskProgramOrder.clear();
}

void OmpSsRegionAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
}

INITIALIZE_PASS_BEGIN(OmpSsRegionAnalysisPass, "ompss-2-regions",
                      "Classify OmpSs-2 inside region uses", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(OmpSsRegionAnalysisPass, "ompss-2-regions",
                    "Classify OmpSs-2 inside region uses", false, true)

