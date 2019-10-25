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
  PV_UnpackAndConst,
  PV_DsaMissing,
  PV_DsaVLADimsMissing
};

static cl::opt<PrintVerbosity>
PrintVerboseLevel("print-verbosity",
  cl::desc("Choose verbosity level"),
  cl::Hidden,
  cl::values(
  clEnumValN(PV_Task, "task", "Print task layout only"),
  clEnumValN(PV_Uses, "uses", "Print task layout with uses"),
  clEnumValN(PV_UnpackAndConst, "unpack", "Print task layout with unpack instructions/constexprs needed in dependencies"),
  clEnumValN(PV_DsaMissing, "dsa_missing", "Print task layout with uses without DSA"),
  clEnumValN(PV_DsaVLADimsMissing, "dsa_vla_dims_missing", "Print task layout with DSAs without VLA info or VLA info without DSAs")));

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

static bool valueInVLADimsBundles(const TaskVLADimsInfo& VLADimsInfo,
                                  const Value *V) {
  for (auto &VLAWithDimsMap : VLADimsInfo) {
    auto Res = find(VLAWithDimsMap.second, V);
    if (Res != VLAWithDimsMap.second.end())
      return true;
  }
  return false;
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
    else if (PrintVerboseLevel == PV_UnpackAndConst) {
      const SmallVector<Instruction *, 4> &UnpackInsts = Info.DependsInfo.UnpackInstructions;
      const SetVector<ConstantExpr *> &UnpackConsts = Info.DependsInfo.UnpackConstants;

      for (Instruction * const I : UnpackInsts) {
        dbgs() << "\n";
        dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
               << "[Inst] ";
        I->printAsOperand(dbgs(), false);
      }
      for (ConstantExpr * const CE : UnpackConsts) {
        dbgs() << "\n";
        dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
               << "[Const] ";
        CE->printAsOperand(dbgs(), false);
      }
    }
    else if (PrintVerboseLevel == PV_DsaVLADimsMissing) {
      // Count VLAs and DSAs, Well-formed VLA must have a DSA and dimensions.
      // Thai is, it must have a frequency of 2
      std::map<const Value *, size_t> DSAVLADimsFreqMap;
      for (Value *V : Info.DSAInfo.Shared) DSAVLADimsFreqMap[V]++;
      for (Value *V : Info.DSAInfo.Private) DSAVLADimsFreqMap[V]++;
      for (Value *V : Info.DSAInfo.Firstprivate) DSAVLADimsFreqMap[V]++;

      for (const auto &VLAWithDimsMap : Info.VLADimsInfo) {
        DSAVLADimsFreqMap[VLAWithDimsMap.first]++;
      }
      for (const auto &Pair : DSAVLADimsFreqMap) {
        // It's expected to have only two VLA bundles, the DSA and de dimensions
        if (Pair.second != 2) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
          Pair.first->printAsOperand(dbgs(), false);
        }
      }
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

// Gather Value unique OperandBundle Id.
// Error if there is more than one Value in OperandBundle
// or more than one OperandBundle
static void getValueFromOperandBundleWithID(const IntrinsicInst *I,
                                            Value *&V,
                                            uint32_t Id) {
  SmallVector<OperandBundleDef, 1> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  assert(OpBundles.size() <= 1 && "Only allowed one OperandBundle with this Id");
  if (OpBundles.size() == 1) {
    assert(OpBundles[0].input_size() == 1 && "Only allowed one Value per OperandBundle");
    V = OpBundles[0].inputs()[0];
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
}

// After gathering DSAInfo we can assert if we find a VLA.DIMS bundle
// without its corresponding DSA
static void gatherVLADimsInfo(const IntrinsicInst *I, TaskInfo &TI) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_vla_dims);
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() > 1 && "VLA dims OperandBundle must have at least a value for the VLA and one dimension");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks && !valueInDSABundles(TI.DSAInfo, OBArgs[0]))
      llvm_unreachable("VLA dims OperandBundle must have an associated DSA");
    assert(TI.VLADimsInfo[OBArgs[0]].empty() && "There're VLA dims duplicated OperandBundles");
    TI.VLADimsInfo[OBArgs[0]].insert(&OBArgs[1], OBArgs.end()); 
  }
}

// Inserts I into InstList ensuring program order if I it's not already in the list
static bool insertUniqInstInProgramOrder(SmallVectorImpl<Instruction *> &InstList,
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

static void gatherUnpackInstructions(const TaskDSAInfo &DSAInfo,
                                     const TaskVLADimsInfo &VLADimsInfo,
                                     const OrderedInstructions &OI,
                                     DependInfo &DI,
                                     TaskAnalysisInfo &TAI,
                                     SmallVectorImpl<Instruction *> &UnpackInsts,
                                     SetVector<ConstantExpr *> &UnpackConsts) {

  // First element is the current instruction, second is
  // the Instruction where we come from (origin of the dependency)
  SmallVector<std::pair<Value *, Value *>, 4> WorkList;
  WorkList.emplace_back(DI.Base, DI.Base);
  for (Value *V : DI.Dims)
    WorkList.emplace_back(V, V);

  while (!WorkList.empty()) {
    auto It = WorkList.begin();
    Value *Cur = It->first;
    Value *Dep = It->second;
    WorkList.erase(It);
    bool IsDSA = valueInDSABundles(DSAInfo, Cur);
    // TODO: this will be get from captured info
    bool IsVLADim = valueInVLADimsBundles(VLADimsInfo, Cur);
    // Go over all uses until:
    //  1. We get a DSA so assign a symbol index
    //  2. We get a VLA dimension, so we're done. We don't want to move
    //  instructions that generate the vla dimension
    if (!IsDSA && !IsVLADim) {
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
        insertUniqInstInProgramOrder(UnpackInsts, I, OI);
      }
    } else if (IsDSA && Dep == DI.Base) {
      // Found DSA associated with Dependency, assign SymbolIndex
      // Cur is the DSA Value
      if (!TAI.DepSymToIdx.count(Cur)) {
        TAI.DepSymToIdx[Cur] = TAI.DepSymToIdx.size();
      }
      DI.SymbolIndex = TAI.DepSymToIdx[Cur];
    }
  }
}

// Process each OpBundle gathering dependency information
static void gatherDependsInfoFromBundles(const SmallVectorImpl<OperandBundleDef> &OpBundles,
                                    const OrderedInstructions &OI,
                                    const TaskDSAInfo &DSAInfo,
                                    const TaskVLADimsInfo &VLADimsInfo,
                                    TaskAnalysisInfo &TAI,
                                    SmallVectorImpl<DependInfo> &DependsList,
                                    SmallVectorImpl<Instruction *> &UnpackInsts,
                                    SetVector<ConstantExpr *> &UnpackConsts) {
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

    gatherUnpackInstructions(DSAInfo, VLADimsInfo, OI, DI, TAI, UnpackInsts, UnpackConsts);

    DependsList.push_back(DI);
  }
}

// Gathers dependencies needed information of type Id
static void gatherDependsInfoWithID(const IntrinsicInst *I,
                                    const OrderedInstructions &OI,
                                    const TaskDSAInfo &DSAInfo,
                                    const TaskVLADimsInfo &VLADimsInfo,
                                    TaskAnalysisInfo &TAI,
                                    SmallVectorImpl<DependInfo> &DependsList,
                                    SmallVectorImpl<Instruction *> &UnpackInsts,
                                    SetVector<ConstantExpr *> &UnpackConsts,
                                    uint64_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  gatherDependsInfoFromBundles(OpBundles, OI, DSAInfo, VLADimsInfo, TAI, DependsList, UnpackInsts, UnpackConsts);
}

// Gathers all dependencies needed information
static void gatherDependsInfo(const IntrinsicInst *I, TaskInfo &TI,
                              TaskAnalysisInfo &TAI,
                              const OrderedInstructions &OI) {
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.Ins,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_in);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.Outs,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_out);
  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.Inouts,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_inout);

  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.WeakIns,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_weakin);

  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.WeakOuts,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_weakout);

  gatherDependsInfoWithID(I, OI, TI.DSAInfo, TI.VLADimsInfo, TAI,
                          TI.DependsInfo.WeakInouts,
                          TI.DependsInfo.UnpackInstructions,
                          TI.DependsInfo.UnpackConstants,
                          LLVMContext::OB_oss_dep_weakinout);
  TI.DependsInfo.NumSymbols = TAI.DepSymToIdx.size();
}

static void gatherIfFinalInfo(const IntrinsicInst *I, TaskInfo &TI) {
  getValueFromOperandBundleWithID(I, TI.Final, LLVMContext::OB_oss_final);
  getValueFromOperandBundleWithID(I, TI.If, LLVMContext::OB_oss_if);
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
          gatherVLADimsInfo(II, T.Info);
          gatherDependsInfo(II, T.Info, T.AnalysisInfo, OI);
          gatherIfFinalInfo(II, T.Info);

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
            if (!DisableChecks
                && !valueInDSABundles(T.Info.DSAInfo, I2)
                && !valueInVLADimsBundles(T.Info.VLADimsInfo, I2)) {
                llvm_unreachable("Value supposed to be inside task entry "
                                 "OperandBundle not found.");
              }
            }
          } else if (Argument *A = dyn_cast<Argument>(U.get())) {
            T.AnalysisInfo.UsesBeforeEntry.insert(A);
            if (!DisableChecks
                && !valueInDSABundles(T.Info.DSAInfo, A)
                && !valueInVLADimsBundles(T.Info.VLADimsInfo, A)) {
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

