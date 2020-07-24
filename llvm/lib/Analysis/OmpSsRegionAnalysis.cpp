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
#include "llvm/InitializePasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
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
  PV_DsaVLADimsMissing,
  PV_VLADimsCaptureMissing,
  PV_NonPODDSAMissing,
  PV_ReductionInitsCombiners
};

static cl::opt<PrintVerbosity>
PrintVerboseLevel("print-verbosity",
  cl::desc("Choose verbosity level"),
  cl::Hidden,
  cl::values(
  clEnumValN(PV_Task, "task", "Print task layout only"),
  clEnumValN(PV_Uses, "uses", "Print task layout with uses"),
  clEnumValN(PV_DsaMissing, "dsa_missing", "Print task layout with uses without DSA"),
  clEnumValN(PV_DsaVLADimsMissing, "dsa_vla_dims_missing", "Print task layout with DSAs without VLA info or VLA info without DSAs"),
  clEnumValN(PV_VLADimsCaptureMissing, "vla_dims_capture_missing", "Print task layout with VLA dimensions without capture"),
  clEnumValN(PV_NonPODDSAMissing, "non_pod_dsa_missing", "Print task layout with non-pod info without according DSA"),
  clEnumValN(PV_ReductionInitsCombiners, "reduction_inits_combiners", "Print task layout with reduction init and combiner functions"))
  );

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

static bool valueInCapturedBundle(const TaskCapturedInfo& CapturedInfo,
                                  Value *const V) {
  return CapturedInfo.count(V);
}

static void print_verbose(
  const MapVector<Instruction *, TaskWithAnalysisInfo> &TEntryToTaskWithAnalysisInfo,
  const MapVector<Instruction *, SmallVector<Instruction *, 4>> &TasksTree,
  Instruction *Cur, int Depth, int PrintSpaceMultiplier) {
  if (Cur) {
    dbgs() << std::string(Depth*PrintSpaceMultiplier, ' ') << "[" << Depth << "] ";
    Cur->printAsOperand(dbgs(), false);
    const TaskAnalysisInfo &AnalysisInfo = TEntryToTaskWithAnalysisInfo.lookup(Cur).AnalysisInfo;
    const TaskInfo &Info = TEntryToTaskWithAnalysisInfo.lookup(Cur).Info;
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
        // It's expected to have only two VLA bundles, the DSA and dimensions
        if (Pair.second != 2) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
          Pair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_VLADimsCaptureMissing) {
      for (auto &VLAWithDimsMap : Info.VLADimsInfo) {
        for (Value *const &V : VLAWithDimsMap.second) {
          if (!valueInCapturedBundle(Info.CapturedInfo, V)) {
            dbgs() << "\n";
            dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
            V->printAsOperand(dbgs(), false);
          }
        }
      }
    }
    else if (PrintVerboseLevel == PV_NonPODDSAMissing) {
      for (auto &InitsPair : Info.NonPODsInfo.Inits) {
        auto It = find(Info.DSAInfo.Private, InitsPair.first);
        if (It == Info.DSAInfo.Private.end()) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
                 << "[Init] ";
          InitsPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (auto &CopiesPair : Info.NonPODsInfo.Copies) {
        auto It = find(Info.DSAInfo.Firstprivate, CopiesPair.first);
        if (It == Info.DSAInfo.Firstprivate.end()) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
                 << "[Copy] ";
          CopiesPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (auto &DeinitsPair : Info.NonPODsInfo.Deinits) {
        auto PrivateIt = find(Info.DSAInfo.Private, DeinitsPair.first);
        auto FirstprivateIt = find(Info.DSAInfo.Firstprivate, DeinitsPair.first);
        if (FirstprivateIt == Info.DSAInfo.Firstprivate.end()
            && PrivateIt == Info.DSAInfo.Private.end()) {
          dbgs() << "\n";
          dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ')
                 << "[Deinit] ";
          DeinitsPair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_ReductionInitsCombiners) {
      for (auto &RedInfo : Info.ReductionsInitCombInfo) {
        dbgs() << "\n";
        dbgs() << std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
        RedInfo.first->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Init->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Comb->printAsOperand(dbgs(), false);
      }
    }
    dbgs() << "\n";
  }
  for (auto II : TasksTree.lookup(Cur)) {
    print_verbose(TEntryToTaskWithAnalysisInfo,
                  TasksTree, II, Depth + 1, PrintSpaceMultiplier);
  }
}

void OmpSsRegionAnalysisPass::print(raw_ostream &OS, const Module *M) const {
  print_verbose(TEntryToTaskWithAnalysisInfo, TasksTree, nullptr, -1, PrintSpaceMultiplier);
}

FunctionInfo& OmpSsRegionAnalysisPass::getFuncInfo() { return FuncInfo; }

static bool isOmpSsLoopDirective(TaskInfo::OmpSsTaskKind TaskKind) {
  return TaskKind == TaskInfo::OSSD_task_for ||
         TaskKind == TaskInfo::OSSD_taskloop ||
         TaskKind == TaskInfo::OSSD_taskloop_for;
}

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

static void gatherTaskKindInfo(const IntrinsicInst *I, TaskInfo &TI) {
  Value *TaskKindValue = nullptr;
  getValueFromOperandBundleWithID(I, TaskKindValue, LLVMContext::OB_oss_dir);
  assert(TaskKindValue && "Expected task kind value in bundles");
  ConstantDataArray *TaskKindDataArray = cast<ConstantDataArray>(TaskKindValue);
  assert(TaskKindDataArray->isCString() && "Task kind must be a C string");
  StringRef TaskKindStringRef = TaskKindDataArray->getAsCString();

  if (TaskKindStringRef == "TASK")
    TI.TaskKind = TaskInfo::OSSD_task;
  else if (TaskKindStringRef == "TASK.FOR")
    TI.TaskKind = TaskInfo::OSSD_task_for;
  else if (TaskKindStringRef == "TASKLOOP")
    TI.TaskKind = TaskInfo::OSSD_taskloop;
  else if (TaskKindStringRef == "TASKLOOP.FOR")
    TI.TaskKind = TaskInfo::OSSD_taskloop_for;
  else
    llvm_unreachable("Unhandled TaskKind string");
}

static void gatherDSAInfo(const IntrinsicInst *I, TaskInfo &TI) {
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Shared,
                                    LLVMContext::OB_oss_shared);
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Private,
                                    LLVMContext::OB_oss_private);
  getValueFromOperandBundlesWithID(I, TI.DSAInfo.Firstprivate,
                                    LLVMContext::OB_oss_firstprivate);
}

static void gatherNonPODInfo(const IntrinsicInst *I, TaskInfo &TI) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  // INIT
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_init);
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 2 && "Non-POD info must have a Value matching a DSA and a function pointer Value");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks) {
      // INIT may only be in private clauses
      auto It = find(TI.DSAInfo.Private, OBArgs[0]);
      if (It == TI.DSAInfo.Private.end())
        llvm_unreachable("Non-POD INIT OperandBundle must have a PRIVATE DSA");
    }
    TI.NonPODsInfo.Inits[OBArgs[0]] = OBArgs[1];
  }

  OpBundles.clear();

  // DEINIT
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_deinit);
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 2 && "Non-POD info must have a Value matching a DSA and a function pointer Value");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks) {
      // DEINIT may only be in firstprivate clauses
      auto PrivateIt = find(TI.DSAInfo.Private, OBArgs[0]);
      auto FirstprivateIt = find(TI.DSAInfo.Firstprivate, OBArgs[0]);
      if (FirstprivateIt == TI.DSAInfo.Firstprivate.end()
          && PrivateIt == TI.DSAInfo.Private.end())
        llvm_unreachable("Non-POD DEINIT OperandBundle must have a PRIVATE or FIRSTPRIVATE DSA");
    }
    TI.NonPODsInfo.Deinits[OBArgs[0]] = OBArgs[1];
  }

  OpBundles.clear();

  // COPY
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_copy);
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 2 && "Non-POD info must have a Value matching a DSA and a function pointer Value");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks) {
      // COPY may only be in firstprivate clauses
      auto It = find(TI.DSAInfo.Firstprivate, OBArgs[0]);
      if (It == TI.DSAInfo.Firstprivate.end())
        llvm_unreachable("Non-POD COPY OperandBundle must have a FIRSTPRIVATE DSA");
    }
    TI.NonPODsInfo.Copies[OBArgs[0]] = OBArgs[1];
  }
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

// After gathering DSAInfo we can assert if we find a DEP.REDUCTION.INIT/COMBINE bundle
// without its corresponding DSA
static void gatherReductionsInitCombInfo(const IntrinsicInst *I, TaskInfo &TI) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_reduction_init);
  // Different reductions may have same init/comb, assign the same ReductionIndex
  DenseMap<Value *, int> SeenInits;
  int ReductionIndex = 0;
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 2 && "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks && !valueInDSABundles(TI.DSAInfo, OBArgs[0]))
      llvm_unreachable("Reduction init/combiner OperandBundle must have an associated DSA");

    // This assert should not trigger since clang allows an unique reduction per DSA
    assert(!TI.ReductionsInitCombInfo.count(OBArgs[0])
           && "Two or more reductions of the same DSA in the same task are not allowed");
    TI.ReductionsInitCombInfo[OBArgs[0]].Init = OBArgs[1];

    if (SeenInits.count(OBArgs[1])) {
      TI.ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = SeenInits[OBArgs[1]];
    } else {
      SeenInits[OBArgs[1]] = ReductionIndex;
      TI.ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = ReductionIndex;
      ReductionIndex++;
    }
  }

  OpBundles.clear();
  getOperandBundlesAsDefsWithID(I, OpBundles, LLVMContext::OB_oss_reduction_comb);
  for (const OperandBundleDef &OBDef : OpBundles) {
    assert(OBDef.input_size() == 2 && "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
    ArrayRef<Value *> OBArgs = OBDef.inputs();
    if (!DisableChecks && !valueInDSABundles(TI.DSAInfo, OBArgs[0]))
      llvm_unreachable("Reduction init/combiner OperandBundle must have an associated DSA");
    TI.ReductionsInitCombInfo[OBArgs[0]].Comb = OBArgs[1];
  }
}

// Process OpBundle gathering dependency information
static void gatherDependInfoFromBundle(ArrayRef<Value *> OBArgs,
                                       TaskDSAInfo &DSAInfo,
                                       TaskCapturedInfo &CapturedInfo,
                                       DependInfo &DI) {
  // First operand has to be the DSA over the dependency is made
  Value *DepBaseDSA = OBArgs[0];
  assert(valueInDSABundles(DSAInfo, DepBaseDSA) && "Dependency has no associated DSA");
  DI.Base = DepBaseDSA;

  Function *ComputeDepFun = cast<Function>(OBArgs[1]);
  DI.ComputeDepFun = ComputeDepFun;

  // Gather compute_dep function params
  for (size_t i = 2; i < OBArgs.size(); ++i) {
    assert((valueInDSABundles(DSAInfo, OBArgs[i])
            || valueInCapturedBundle(CapturedInfo, OBArgs[i]))
           && "Dependency has no associated DSA or capture");
    DI.Args.push_back(OBArgs[i]);
  }

  if (!DSAInfo.DepSymToIdx.count(DepBaseDSA)) {
    DSAInfo.DepSymToIdx[DepBaseDSA] = DSAInfo.DepSymToIdx.size();
  }
  DI.SymbolIndex = DSAInfo.DepSymToIdx[DepBaseDSA];

  // TODO: Support RegionText stringifying clause content
  DI.RegionText = "";
}

// Gathers dependencies needed information of type Id
static void gatherDependsInfoWithID(const IntrinsicInst *I,
                                    SmallVectorImpl<DependInfo> &DependsList,
                                    TaskDSAInfo &DSAInfo,
                                    TaskCapturedInfo &CapturedInfo,
                                    uint64_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  // TODO: maybe do a bundle gather with asserts?
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  for (const OperandBundleDef &OBDef : OpBundles) {
    DependInfo DI;

    gatherDependInfoFromBundle(OBDef.inputs(), DSAInfo, CapturedInfo, DI);

    DependsList.push_back(DI);
  }
}

// Gathers dependencies needed information of type Id
static void gatherReductionsInfoWithID(const IntrinsicInst *I,
                                       SmallVectorImpl<ReductionInfo> &ReductionsList,
                                       TaskDSAInfo &DSAInfo,
                                       TaskCapturedInfo &CapturedInfo,
                                       uint64_t Id) {
  SmallVector<OperandBundleDef, 4> OpBundles;
  getOperandBundlesAsDefsWithID(I, OpBundles, Id);
  for (const OperandBundleDef &OBDef : OpBundles) {
    ReductionInfo RI;

    ArrayRef<Value *> OBArgs = OBDef.inputs();
    RI.RedKind = OBArgs[0];

    // Skip the reduction kind
    gatherDependInfoFromBundle(OBArgs.drop_front(1), DSAInfo, CapturedInfo, RI.DepInfo);

    ReductionsList.push_back(RI);
  }
}

// Gathers all dependencies needed information
static void gatherDependsInfo(const IntrinsicInst *I, TaskInfo &TI,
                              TaskAnalysisInfo &TAI,
                              const OrderedInstructions &OI) {

  gatherDependsInfoWithID(I,
                          TI.DependsInfo.Ins,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_in);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.Outs,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_out);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.Inouts,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_inout);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.Concurrents,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_concurrent);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.Commutatives,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_commutative);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.WeakIns,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_weakin);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.WeakOuts,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_weakout);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.WeakInouts,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_weakinout);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.WeakConcurrents,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_weakconcurrent);
  gatherDependsInfoWithID(I,
                          TI.DependsInfo.WeakCommutatives,
                          TI.DSAInfo,
                          TI.CapturedInfo,
                          LLVMContext::OB_oss_dep_weakcommutative);

  gatherReductionsInfoWithID(I,
                             TI.DependsInfo.Reductions,
                             TI.DSAInfo,
                             TI.CapturedInfo,
                             LLVMContext::OB_oss_dep_reduction);

  gatherReductionsInfoWithID(I,
                             TI.DependsInfo.WeakReductions,
                             TI.DSAInfo,
                             TI.CapturedInfo,
                             LLVMContext::OB_oss_dep_weakreduction);

  TI.DependsInfo.NumSymbols = TI.DSAInfo.DepSymToIdx.size();
}

// TODO: change function name for this
static void gatherIfFinalCostPrioWaitInfo(const IntrinsicInst *I, TaskInfo &TI) {
  getValueFromOperandBundleWithID(I, TI.Final, LLVMContext::OB_oss_final);
  getValueFromOperandBundleWithID(I, TI.If, LLVMContext::OB_oss_if);
  getValueFromOperandBundleWithID(I, TI.Cost, LLVMContext::OB_oss_cost);
  getValueFromOperandBundleWithID(I, TI.Priority, LLVMContext::OB_oss_priority);
  getValueFromOperandBundleWithID(I, TI.Label, LLVMContext::OB_oss_label);
  getValueFromOperandBundleWithID(I, TI.Wait, LLVMContext::OB_oss_wait);
}

// It's expected to have VLA dims info before calling this
static void gatherCapturedInfo(const IntrinsicInst *I, TaskInfo &TI) {
  getValueListFromOperandBundlesWithID(I, TI.CapturedInfo, LLVMContext::OB_oss_captured);
  if (!DisableChecks) {
    // VLA Dims that are not Captured is an error
    for (auto &VLAWithDimsMap : TI.VLADimsInfo) {
      for (Value *const &V : VLAWithDimsMap.second) {
        if (!valueInCapturedBundle(TI.CapturedInfo, V))
          llvm_unreachable("VLA dimension has not been captured");
      }
    }
  }
}

static void gatherLoopInfo(const IntrinsicInst *I, TaskInfo &TI) {
  assert(isOmpSsLoopDirective(TI.TaskKind) && "gatherLoopInfo expects a loop directive");

  // TODO: add stepincr
  SmallVector<OperandBundleDef, 4> LoopTypeBundles;
  getOperandBundlesAsDefsWithID(I, LoopTypeBundles, LLVMContext::OB_oss_loop_type);
  assert(LoopTypeBundles.size() == 1 && LoopTypeBundles[0].input_size() == 5);

  TI.LoopInfo.LoopType = cast<ConstantInt>(LoopTypeBundles[0].inputs()[0])->getSExtValue();
  TI.LoopInfo.IndVarSigned = cast<ConstantInt>(LoopTypeBundles[0].inputs()[1])->getSExtValue();
  TI.LoopInfo.LBoundSigned = cast<ConstantInt>(LoopTypeBundles[0].inputs()[2])->getSExtValue();
  TI.LoopInfo.UBoundSigned = cast<ConstantInt>(LoopTypeBundles[0].inputs()[3])->getSExtValue();
  TI.LoopInfo.StepSigned = cast<ConstantInt>(LoopTypeBundles[0].inputs()[4])->getSExtValue();

  getValueFromOperandBundleWithID(I, TI.LoopInfo.IndVar, LLVMContext::OB_oss_loop_ind_var);
  getValueFromOperandBundleWithID(I, TI.LoopInfo.LBound, LLVMContext::OB_oss_loop_lower_bound);
  getValueFromOperandBundleWithID(I, TI.LoopInfo.UBound, LLVMContext::OB_oss_loop_upper_bound);
  getValueFromOperandBundleWithID(I, TI.LoopInfo.Step, LLVMContext::OB_oss_loop_step);
  getValueFromOperandBundleWithID(I, TI.LoopInfo.Chunksize, LLVMContext::OB_oss_loop_chunksize);
  getValueFromOperandBundleWithID(I, TI.LoopInfo.Grainsize, LLVMContext::OB_oss_loop_grainsize);

  if (TI.LoopInfo.empty())
    llvm_unreachable("LoopInfo is missing some information");
}

// in: TasksTree, Cur
// out: tasks are ordered in post order, which means that
// child tasks will be placed before its parent tasks
static void convertTasksTreeToVectorImpl(
  MapVector<Instruction *, TaskWithAnalysisInfo> &TEntryToTaskWithAnalysisInfo,
  MapVector<Instruction *, SmallVector<Instruction *, 4>> &TasksTree,
  Instruction *Cur,
  SmallVectorImpl<TaskInfo *> &PostOrder,
  SmallVectorImpl<Instruction *> &Stack) {

  if (Cur)
    Stack.push_back(Cur);

  // TODO: Why using operator[] does weird things?
  // for (auto II : TasksTree[Cur]) {
  for (auto II : TasksTree.lookup(Cur)) {
    convertTasksTreeToVectorImpl(TEntryToTaskWithAnalysisInfo,
                                 TasksTree, II, PostOrder, Stack);
  }
  if (Cur) {
    Stack.pop_back();

    TaskWithAnalysisInfo &T = TEntryToTaskWithAnalysisInfo[Cur];
    for (Instruction *I : Stack) {
      // Annotate the current task as inner of all tasks in stack
      TaskWithAnalysisInfo &TStack = TEntryToTaskWithAnalysisInfo[I];
      TStack.Info.InnerTaskInfos.push_back(&T.Info);
    }
    PostOrder.push_back(&T.Info);
  }
}

// in: TasksTree, Cur
// out: tasks are ordered in post order, which means that
// child tasks will be placed before its parent tasks
static void convertTasksTreeToVector(
  MapVector<Instruction *, TaskWithAnalysisInfo> &TEntryToTaskWithAnalysisInfo,
  MapVector<Instruction *, SmallVector<Instruction *, 4>> &TasksTree,
  SmallVectorImpl<TaskInfo*> &PostOrder) {

  SmallVector<Instruction *, 4> Stack;

  convertTasksTreeToVectorImpl(TEntryToTaskWithAnalysisInfo,
                               TasksTree, nullptr, PostOrder, Stack);
}

void OmpSsRegionAnalysisPass::getOmpSsFunctionInfo(
  Function &F, DominatorTree &DT, FunctionInfo &FI,
  MapVector<Instruction *, TaskWithAnalysisInfo> &TEntryToTaskWithAnalysisInfo,
  MapVector<Instruction *, SmallVector<Instruction *, 4>> &TasksTree) {

  OrderedInstructions OI(&DT);

  MapVector<BasicBlock *, SmallVector<Instruction *, 4>> BBTaskStacks;
  SmallVector<BasicBlock*, 8> Worklist;
  SmallPtrSet<BasicBlock*, 8> Visited;

  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());
  while (!Worklist.empty()) {
    auto WIt = Worklist.begin();
    BasicBlock *BB = *WIt;
    Worklist.erase(WIt);

    SmallVectorImpl<Instruction *> &Stack = BBTaskStacks[BB];

    for (Instruction &I : *BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
          assert(II->hasOneUse() && "Task entry has more than one user.");

          Instruction *Exit = cast<Instruction>(II->user_back());
          // This should not happen because it will crash before this pass
          assert(OI.dominates(II, Exit) && "Task entry does not dominate exit.");

          if (Stack.empty()) {
            // outer task, insert into nullptr
            TasksTree[nullptr].push_back(II);
          } else {
            TasksTree[Stack.back()].push_back(II);
          }
          Stack.push_back(II);

          TaskWithAnalysisInfo &T = TEntryToTaskWithAnalysisInfo[II];
          T.Info.Entry = II;
          T.Info.Exit = Exit;

          gatherTaskKindInfo(II, T.Info);
          gatherDSAInfo(II, T.Info);
          gatherNonPODInfo(II, T.Info);
          gatherVLADimsInfo(II, T.Info);
          gatherCapturedInfo(II, T.Info);
          gatherDependsInfo(II, T.Info, T.AnalysisInfo, OI);
          gatherReductionsInitCombInfo(II, T.Info);
          gatherIfFinalCostPrioWaitInfo(II, T.Info);
          if (isOmpSsLoopDirective(T.Info.TaskKind))
            gatherLoopInfo(II, T.Info);
          // TODO: missing grainsize, chunksize
        } else if (II->getIntrinsicID() == Intrinsic::directive_region_exit) {
          if (Stack.empty())
            llvm_unreachable("Task exit hit without and entry.");

          Instruction *StackEntry = Stack.back();
          Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
          assert(StackExit == II && "unexpected task exit instr.");

          Stack.pop_back();
        } else if (II->getIntrinsicID() == Intrinsic::directive_marker) {
          FI.TaskwaitFuncInfo.PostOrder.push_back({II});
        }
      } else if (!Stack.empty()) {
        Instruction *StackEntry = Stack.back();
        Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
        TaskWithAnalysisInfo &T = TEntryToTaskWithAnalysisInfo[StackEntry];
        for (Use &U : I.operands()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U.get())) {
            if (OI.dominates(I2, StackEntry)) {
              T.AnalysisInfo.UsesBeforeEntry.insert(I2);
            if (!DisableChecks
                && !valueInDSABundles(T.Info.DSAInfo, I2)
                && !valueInCapturedBundle(T.Info.CapturedInfo, I2)) {
                llvm_unreachable("Value supposed to be inside task entry "
                                 "OperandBundle not found.");
              }
            }
          } else if (Argument *A = dyn_cast<Argument>(U.get())) {
            T.AnalysisInfo.UsesBeforeEntry.insert(A);
            if (!DisableChecks
                && !valueInDSABundles(T.Info.DSAInfo, A)
                && !valueInCapturedBundle(T.Info.CapturedInfo, A)) {
              llvm_unreachable("Value supposed to be inside task entry "
                               "OperandBundle not found.");
            }
          }
        }
        for (User *U : I.users()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U)) {
            if (OI.dominates(StackExit, I2)) {
              T.AnalysisInfo.UsesAfterExit.insert(&I);
              if (!DisableChecks) {
                llvm_unreachable("Value inside the task body used after it.");
              }
            }
          }
        }
      }
    }

    std::unique_ptr<std::vector<Instruction *>> StackCopy;

    for (auto It = succ_begin(BB); It != succ_end(BB); ++It) {
      if (!Visited.count(*It)) {
        Worklist.push_back(*It);
        Visited.insert(*It);
        // Forward Stack, since we are setting visited here
        // we do this only once per BB
        if (!StackCopy) {
          // We need to copy Stacki, otherwise &Stack as an iterator would be
          // invalidated after BBTaskStacks[*It].
          StackCopy.reset(
              new std::vector<Instruction *>(Stack.begin(), Stack.end()));
        }

        BBTaskStacks[*It].append(StackCopy->begin(), StackCopy->end());
      }
    }
  }

  convertTasksTreeToVector(TEntryToTaskWithAnalysisInfo,
                           TasksTree, FI.TaskFuncInfo.PostOrder);

}

bool OmpSsRegionAnalysisPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  getOmpSsFunctionInfo(F, DT, FuncInfo, TEntryToTaskWithAnalysisInfo, TasksTree);

  return false;
}

void OmpSsRegionAnalysisPass::releaseMemory() {
  FuncInfo = FunctionInfo();
  TEntryToTaskWithAnalysisInfo.clear();
  TasksTree.clear();
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

