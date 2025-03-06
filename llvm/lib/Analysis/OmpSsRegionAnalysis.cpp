//===- OmpSsRegionAnalysis.cpp - OmpSs Region Analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OmpSsRegionAnalysis.h"

#include "llvm/InitializePasses.h"
#include "llvm/IR/IntrinsicsOmpSs.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool>
DisableChecks("disable-checks",
                  cl::desc("Avoid checking OmpSs-2 directive bundle correctness"),
                  cl::Hidden,
                  cl::init(false));

enum PrintVerbosity {
  PV_Directive,
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
  clEnumValN(PV_Directive, "directive", "Print directive layout only"),
  clEnumValN(PV_Uses, "uses", "Print directive layout with uses"),
  clEnumValN(PV_DsaMissing, "dsa_missing", "Print directive layout with uses without DSA"),
  clEnumValN(PV_DsaVLADimsMissing, "dsa_vla_dims_missing", "Print directive layout with DSAs without VLA info or VLA info without DSAs"),
  clEnumValN(PV_VLADimsCaptureMissing, "vla_dims_capture_missing", "Print directive layout with VLA dimensions without capture"),
  clEnumValN(PV_NonPODDSAMissing, "non_pod_dsa_missing", "Print directive layout with non-pod info without according DSA"),
  clEnumValN(PV_ReductionInitsCombiners, "reduction_inits_combiners", "Print directive layout with reduction init and combiner functions"))
  );

/// NOTE: from old OrderedInstructions
static bool localDominates(
    const Instruction *InstA, const Instruction *InstB) {
  assert(InstA->getParent() == InstB->getParent() &&
         "Instructions must be in the same basic block");

  return InstA->comesBefore(InstB);
}

/// Given 2 instructions, check for dominance relation if the instructions are
/// in the same basic block. Otherwise, use dominator tree.
/// NOTE: from old OrderedInstructions
static bool orderedInstructions(
    DominatorTree &DT, const Instruction *InstA, const Instruction *InstB) {
  // Use ordered basic block to do dominance check in case the 2 instructions
  // are in the same basic block.
  if (InstA->getParent() == InstB->getParent())
    return localDominates(InstA, InstB);
  return DT.dominates(InstA->getParent(), InstB->getParent());
}

static DependInfo::DependType getDependTypeFromId(uint64_t Id) {
  switch (Id) {
  case LLVMContext::OB_oss_dep_in:
  case LLVMContext::OB_oss_multidep_range_in:
    return DependInfo::DT_in;
  case LLVMContext::OB_oss_dep_out:
  case LLVMContext::OB_oss_multidep_range_out:
    return DependInfo::DT_out;
  case LLVMContext::OB_oss_dep_inout:
  case LLVMContext::OB_oss_multidep_range_inout:
    return DependInfo::DT_inout;
  case LLVMContext::OB_oss_dep_concurrent:
  case LLVMContext::OB_oss_multidep_range_concurrent:
    return DependInfo::DT_concurrent;
  case LLVMContext::OB_oss_dep_commutative:
  case LLVMContext::OB_oss_multidep_range_commutative:
    return DependInfo::DT_commutative;
  case LLVMContext::OB_oss_dep_reduction:
    return DependInfo::DT_reduction;
  case LLVMContext::OB_oss_dep_weakin:
  case LLVMContext::OB_oss_multidep_range_weakin:
    return DependInfo::DT_weakin;
  case LLVMContext::OB_oss_dep_weakout:
  case LLVMContext::OB_oss_multidep_range_weakout:
    return DependInfo::DT_weakout;
  case LLVMContext::OB_oss_dep_weakinout:
  case LLVMContext::OB_oss_multidep_range_weakinout:
    return DependInfo::DT_weakinout;
  case LLVMContext::OB_oss_dep_weakconcurrent:
  case LLVMContext::OB_oss_multidep_range_weakconcurrent:
    return DependInfo::DT_weakconcurrent;
  case LLVMContext::OB_oss_dep_weakcommutative:
  case LLVMContext::OB_oss_multidep_range_weakcommutative:
    return DependInfo::DT_weakcommutative;
  case LLVMContext::OB_oss_dep_weakreduction:
    return DependInfo::DT_weakreduction;
  }
  llvm_unreachable("unknown depend type id");
}

void DirectiveEnvironment::gatherDirInfo(OperandBundleDef &OB) {
  assert(DirectiveKind == OSSD_unknown && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() >= 1 && "Needed at least one Value per OperandBundle");
  ConstantDataArray *DirectiveKindDataArray = cast<ConstantDataArray>(OB.inputs()[0]);
  assert(DirectiveKindDataArray->isCString() && "Directive kind must be a C string");
  DirectiveKindStringRef = DirectiveKindDataArray->getAsCString();

  if (DirectiveKindStringRef == "TASK")
    DirectiveKind = OSSD_task;
  else if (DirectiveKindStringRef == "CRITICAL.START")
    DirectiveKind = OSSD_critical_start;
  else if (DirectiveKindStringRef == "CRITICAL.END")
    DirectiveKind = OSSD_critical_end;
  else if (DirectiveKindStringRef == "TASK.FOR")
    DirectiveKind = OSSD_task_for;
  else if (DirectiveKindStringRef == "TASKITER.FOR")
    DirectiveKind = OSSD_taskiter_for;
  else if (DirectiveKindStringRef == "TASKITER.WHILE")
    DirectiveKind = OSSD_taskiter_while;
  else if (DirectiveKindStringRef == "TASKLOOP")
    DirectiveKind = OSSD_taskloop;
  else if (DirectiveKindStringRef == "TASKLOOP.FOR")
    DirectiveKind = OSSD_taskloop_for;
  else if (DirectiveKindStringRef == "TASKWAIT")
    DirectiveKind = OSSD_taskwait;
  else if (DirectiveKindStringRef == "RELEASE")
    DirectiveKind = OSSD_release;
  else
    llvm_unreachable("Unhandled DirectiveKind string");

  if (isOmpSsCriticalDirective()) {
    ConstantDataArray *CriticalNameDataArray = cast<ConstantDataArray>(OB.inputs()[1]);
    assert(CriticalNameDataArray->isCString() && "Critical name must be a C string");
    CriticalNameStringRef = CriticalNameDataArray->getAsCString();
  }
}

void DirectiveEnvironment::gatherSharedInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 && "Only allowed two Values per OperandBundle");
  DSAInfo.Shared.insert(
    std::make_pair(OB.inputs()[0], OB.inputs()[1]->getType()));
}

void DirectiveEnvironment::gatherPrivateInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 && "Only allowed two Values per OperandBundle");
  DSAInfo.Private.insert(
    std::make_pair(OB.inputs()[0], OB.inputs()[1]->getType()));
}

void DirectiveEnvironment::gatherFirstprivateInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 && "Only allowed two Values per OperandBundle");
  DSAInfo.Firstprivate.insert(
    std::make_pair(OB.inputs()[0], OB.inputs()[1]->getType()));
}

void DirectiveEnvironment::gatherVLADimsInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 1 &&
    "VLA dims OperandBundle must have at least a value for the VLA and one dimension");
  ArrayRef<Value *> OBArgs = OB.inputs();
  assert(VLADimsInfo[OBArgs[0]].empty() && "There're VLA dims duplicated OperandBundles");
  VLADimsInfo[OBArgs[0]].insert(&OBArgs[1], OBArgs.end());
}

static void gatherDependInfo(
    ArrayRef<Value *> OBArgs, std::map<Value *, int> &DepSymToIdx,
    DirectiveDependsInfo &DependsInfo, DependInfo &DI, uint64_t Id) {
  DI.DepType = getDependTypeFromId(Id);

  if (DI.isReduction()) {
    DI.RedKind = OBArgs[0];
    // Skip the reduction kind
    OBArgs = OBArgs.drop_front(1);
  }

  // First operand has to be the DSA over the dependency is made
  DI.Base = OBArgs[0];

  ConstantDataArray *DirectiveKindDataArray = cast<ConstantDataArray>(OBArgs[1]);
  assert(DirectiveKindDataArray->isCString() && "Region text must be a C string");
  DI.RegionText = DirectiveKindDataArray->getAsCString();

  DI.ComputeDepFun = cast<Function>(OBArgs[2]);

  // Gather compute_dep function params
  for (size_t i = 3; i < OBArgs.size(); ++i) {
    DI.Args.push_back(OBArgs[i]);
  }

  if (!DepSymToIdx.count(DI.Base)) {
    DepSymToIdx[DI.Base] = DepSymToIdx.size();
    DependsInfo.NumSymbols++;
  }
  DI.SymbolIndex = DepSymToIdx[DI.Base];

}

void DirectiveEnvironment::gatherDependInfo(
    OperandBundleDef &OB, uint64_t Id) {

  assert(OB.input_size() > 2 &&
    "Depend OperandBundle must have at least depend base, function and one argument");
  DependInfo *DI = new DependInfo();

  ::gatherDependInfo(OB.inputs(), DepSymToIdx, DependsInfo, *DI, Id);

  DependsInfo.List.emplace_back(DI);
}

void DirectiveEnvironment::gatherReductionInitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();

  // This assert should not trigger since clang allows an unique reduction per DSA
  assert(!ReductionsInitCombInfo.count(OBArgs[0])
         && "Two or more reductions of the same DSA in the same directive are not allowed");
  ReductionsInitCombInfo[OBArgs[0]].Init = OBArgs[1];

  if (SeenInits.count(OBArgs[1])) {
    ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = SeenInits[OBArgs[1]];
  } else {
    SeenInits[OBArgs[1]] = ReductionIndex;
    ReductionsInitCombInfo[OBArgs[0]].ReductionIndex = ReductionIndex;
    ReductionIndex++;
  }
}

void DirectiveEnvironment::gatherReductionCombInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();

  ReductionsInitCombInfo[OBArgs[0]].Comb = OBArgs[1];
}

void DirectiveEnvironment::gatherFinalInfo(OperandBundleDef &OB) {
  assert(!Final && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Final = OB.inputs()[0];
}

void DirectiveEnvironment::gatherIfInfo(OperandBundleDef &OB) {
  assert(!If && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  If = OB.inputs()[0];
}

void DirectiveEnvironment::gatherCostInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Cost OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  CostInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    CostInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherPriorityInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Priority OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  PriorityInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    PriorityInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherLabelInfo(OperandBundleDef &OB) {
  assert((!Label || !InstanceLabel) && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() <= 2 && "Only allowed one Value per OperandBundle");
  Label = OB.inputs()[0];
  if (OB.input_size() == 2)
    InstanceLabel = OB.inputs()[1];
}

void DirectiveEnvironment::gatherOnreadyInfo(OperandBundleDef &OB) {
  assert(OB.input_size() > 0 &&
    "Onready OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  OnreadyInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    OnreadyInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherWaitInfo(OperandBundleDef &OB) {
  assert(!Wait && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Wait = OB.inputs()[0];
}

void DirectiveEnvironment::gatherDeviceInfo(OperandBundleDef &OB) {
  assert(!DeviceInfo.Kind && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  DeviceInfo.Kind = OB.inputs()[0];
}

void DirectiveEnvironment::gatherDeviceNdrangeInfo(OperandBundleDef &OB) {
  assert(DeviceInfo.Ndrange.empty() && "Only allowed one OperandBundle with this Id");
  DeviceInfo.NumDims = cast<ConstantInt>(OB.inputs()[0])->getZExtValue();
  for (size_t i = 1; i < OB.input_size(); i++)
    DeviceInfo.Ndrange.push_back(OB.inputs()[i]);
  DeviceInfo.IsGrid = false;
}

void DirectiveEnvironment::gatherDeviceGridInfo(OperandBundleDef &OB) {
  assert(DeviceInfo.Ndrange.empty() && "Only allowed one OperandBundle with this Id");
  DeviceInfo.NumDims = 3;
  for (size_t i = 0; i < OB.input_size(); i++)
    DeviceInfo.Ndrange.push_back(OB.inputs()[i]);
  DeviceInfo.IsGrid = true;
}

void DirectiveEnvironment::gatherDeviceDevFuncInfo(OperandBundleDef &OB) {
  assert(DeviceInfo.DevFuncStringRef.empty() && "Only allowed one OperandBundle with this Id");
  ConstantDataArray *DevFuncDataArray = cast<ConstantDataArray>(OB.inputs()[0]);
  assert(DevFuncDataArray->isCString() && "Region text must be a C string");
  DeviceInfo.DevFuncStringRef = DevFuncDataArray->getAsCString();
}

void DirectiveEnvironment::gatherDeviceCallOrderInfo(OperandBundleDef &OB) {
  assert(DeviceInfo.CallOrder.empty() && "Only allowed one OperandBundle with this Id");
  DeviceInfo.CallOrder.append(OB.input_begin(), OB.input_end());
}

void DirectiveEnvironment::gatherDeviceShmemInfo(OperandBundleDef &OB) {
  assert(!DeviceInfo.Shmem && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  DeviceInfo.Shmem = OB.inputs()[0];
}

void DirectiveEnvironment::gatherCapturedInfo(OperandBundleDef &OB) {
  assert(CapturedInfo.empty() && "Only allowed one OperandBundle with this Id");
  CapturedInfo.insert(OB.input_begin(), OB.input_end());
}

void DirectiveEnvironment::gatherNonPODInitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Inits[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherNonPODDeinitInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 &&
    "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Deinits[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherNonPODCopyInfo(OperandBundleDef &OB) {
  assert(OB.input_size() == 2 && "Non-POD info must have a Value matching a DSA and a function pointer Value");
  ArrayRef<Value *> OBArgs = OB.inputs();
  NonPODsInfo.Copies[OBArgs[0]] = OBArgs[1];
}

void DirectiveEnvironment::gatherLoopTypeInfo(OperandBundleDef &OB) {
  assert(LoopInfo.LoopType.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.IndVarSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.LBoundSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.UBoundSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(LoopInfo.StepSigned.empty() && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size()%5 == 0 && "Expected loop type and indvar, lb, ub, step signedness");

  for (size_t i = 0; i < OB.input_size()/5; i++) {
    LoopInfo.LoopType.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 0])->getSExtValue());
    LoopInfo.IndVarSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 1])->getSExtValue());
    LoopInfo.LBoundSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 2])->getSExtValue());
    LoopInfo.UBoundSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 3])->getSExtValue());
    LoopInfo.StepSigned.push_back(cast<ConstantInt>(OB.inputs()[i*5 + 4])->getSExtValue());
  }
}

void DirectiveEnvironment::gatherLoopIndVarInfo(OperandBundleDef &OB) {
  assert(LoopInfo.IndVar.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++)
    LoopInfo.IndVar.push_back(OB.inputs()[i]);
}

void DirectiveEnvironment::gatherLoopLowerBoundInfo(OperandBundleDef &OB) {
  assert(LoopInfo.LBound.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.LBound.emplace_back();
      LoopInfo.LBound.back().Fun = F;
    } else
      LoopInfo.LBound.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopUpperBoundInfo(OperandBundleDef &OB) {
  assert(LoopInfo.UBound.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.UBound.emplace_back();
      LoopInfo.UBound.back().Fun = F;
    } else
      LoopInfo.UBound.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopStepInfo(OperandBundleDef &OB) {
  assert(LoopInfo.Step.empty() && "Only allowed one OperandBundle with this Id");
  for (size_t i = 0; i < OB.input_size(); i++) {
    if (auto *F = dyn_cast<Function>(OB.inputs()[i])) {
      LoopInfo.Step.emplace_back();
      LoopInfo.Step.back().Fun = F;
    } else
      LoopInfo.Step.back().Args.push_back(OB.inputs()[i]);
  }
}

void DirectiveEnvironment::gatherLoopChunksizeInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Chunksize && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Chunksize = OB.inputs()[0];
}

void DirectiveEnvironment::gatherLoopGrainsizeInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Grainsize && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Grainsize = OB.inputs()[0];
}

void DirectiveEnvironment::gatherLoopUnrollInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Unroll && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Unroll = OB.inputs()[0];
}

void DirectiveEnvironment::gatherLoopUpdateInfo(OperandBundleDef &OB) {
  assert(!LoopInfo.Update && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  LoopInfo.Update = OB.inputs()[0];
}

void DirectiveEnvironment::gatherWhileCondInfo(OperandBundleDef &OB) {
  assert(WhileInfo.empty() && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() > 0 &&
    "WhileCond OperandBundle must have at least function");
  ArrayRef<Value *> OBArgs = OB.inputs();
  WhileInfo.Fun = cast<Function>(OBArgs[0]);
  for (size_t i = 1; i < OBArgs.size(); ++i)
    WhileInfo.Args.push_back(OBArgs[i]);
}

void DirectiveEnvironment::gatherMultiDependInfo(
    OperandBundleDef &OB, uint64_t Id) {
  // TODO: add asserts

  MultiDependInfo *MDI = new MultiDependInfo();
  MDI->DepType = getDependTypeFromId(Id);

  ArrayRef<Value *> OBArgs = OB.inputs();

  size_t i;
  size_t ComputeFnCnt = 0;
  // 1. Gather iterators from begin to compute multidep function
  // 2. Gather compute multidep function args from 1. to compute dep function previous element
  //    which is the dep base
  for (i = 0; i < OBArgs.size(); ++i) {
    if (auto *ComputeFn = dyn_cast<Function>(OBArgs[i])) {
      if (ComputeFnCnt == 0) // ComputeMultiDepFun
        MDI->ComputeMultiDepFun = ComputeFn;
      else // Seen ComputeDepFun
        break;
      ++ComputeFnCnt;
      continue;
    }
    if (ComputeFnCnt == 0)
      MDI->Iters.push_back(OBArgs[i]);
    else if (ComputeFnCnt == 1)
      MDI->Args.push_back(OBArgs[i]);
  }
  // TODO: this is used because we add dep base and region text too
  // which is wrong...
  MDI->Args.pop_back();
  MDI->Args.pop_back();

  ::gatherDependInfo(OBArgs.drop_front(i - 2), DepSymToIdx, DependsInfo, *MDI, Id);

  DependsInfo.List.emplace_back(MDI);
}

void DirectiveEnvironment::gatherDeclSource(OperandBundleDef &OB) {
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  ConstantDataArray *DeclSourceDataArray = cast<ConstantDataArray>(OB.inputs()[0]);
  assert(DeclSourceDataArray->isCString() && "Region text must be a C string");
  DeclSourceStringRef = DeclSourceDataArray->getAsCString();
}

void DirectiveEnvironment::gatherCoroHandle(OperandBundleDef &OB) {
  assert(!CoroInfo.Handle && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  CoroInfo.Handle = OB.inputs()[0];
}

void DirectiveEnvironment::gatherCoroSizeStore(OperandBundleDef &OB) {
  assert(!CoroInfo.SizeStore && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  CoroInfo.SizeStore = OB.inputs()[0];
}

void DirectiveEnvironment::gatherImmediateInfo(OperandBundleDef &OB) {
  assert(!Immediate && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Immediate = OB.inputs()[0];
}

void DirectiveEnvironment::gatherMicrotaskInfo(OperandBundleDef &OB) {
  assert(!Microtask && "Only allowed one OperandBundle with this Id");
  assert(OB.input_size() == 1 && "Only allowed one Value per OperandBundle");
  Microtask = OB.inputs()[0];
}

void DirectiveEnvironment::verifyVLADimsInfo() {
  for (const auto &VLAWithDimsMap : VLADimsInfo) {
    if (!valueInDSABundles(VLAWithDimsMap.first))
      llvm_unreachable("VLA dims OperandBundle must have an associated DSA");
    if (!(getDSAType(VLAWithDimsMap.first)->isSingleValueType()
          || getDSAType(VLAWithDimsMap.first)->isStructTy()))
      llvm_unreachable("VLA type is not scalar");

    // VLA Dims that are not Captured is an error
    for (auto *V : VLAWithDimsMap.second) {
      if (!valueInCapturedBundle(V))
        llvm_unreachable("VLA dimension has not been captured");
    }
  }
}

void DirectiveEnvironment::verifyDependInfo() {
  for (auto &DI : DependsInfo.List) {
    if (!valueInDSABundles(DI->Base))
      llvm_unreachable("Dependency has no associated DSA");
    for (auto *V : DI->Args) {
      if (!valueInDSABundles(V)
          && !valueInCapturedBundle(V))
        llvm_unreachable("Dependency has no associated DSA or capture");
    }
  }
}

void DirectiveEnvironment::verifyReductionInitCombInfo() {
  for (const auto &RedInitCombMap : ReductionsInitCombInfo) {
    if (!valueInDSABundles(RedInitCombMap.first))
      llvm_unreachable(
        "Reduction init/combiner must have a Value matching a DSA and a function pointer Value");
    if (!RedInitCombMap.second.Init)
      llvm_unreachable("Missing reduction initializer");
    if (!RedInitCombMap.second.Comb)
      llvm_unreachable("Missing reduction combiner");
  }
}

void DirectiveEnvironment::verifyCostInfo() {
  for (auto *V : CostInfo.Args) {
    if (!valueInDSABundles(V)
        && !valueInCapturedBundle(V))
      llvm_unreachable("Cost function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyPriorityInfo() {
  for (auto *V : PriorityInfo.Args) {
    if (!valueInDSABundles(V)
        && !valueInCapturedBundle(V))
      llvm_unreachable("Priority function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyOnreadyInfo() {
  for (auto *V : OnreadyInfo.Args) {
    if (!valueInDSABundles(V)
        && !valueInCapturedBundle(V))
      llvm_unreachable("Onready function argument has no associated DSA or capture");
  }
}

void DirectiveEnvironment::verifyDeviceInfo() {
  // TODO: add a check for DeviceInfo.Kind != (cuda | opencl)
  if (!DeviceInfo.Kind && !DeviceInfo.Ndrange.empty())
    llvm_unreachable("It is expected to have a device kind when used ndrange");
  if (!DeviceInfo.Kind && !DeviceInfo.CallOrder.empty())
    llvm_unreachable("It is expected to have a device kind when call order is set");
  if (!DeviceInfo.Kind && DeviceInfo.Shmem)
    llvm_unreachable("It is expected to have a device kind when used shmem");
  for (auto *V : DeviceInfo.CallOrder)
    if (!valueInDSABundles(V)
        && !valueInCapturedBundle(V))
    llvm_unreachable("Call order value has no associated DSA or capture");
  if (DeviceInfo.NumDims != 0) {
    if (DeviceInfo.NumDims < 1 || DeviceInfo.NumDims > 3)
      llvm_unreachable("Num dimensions is expected to be 1, 2 or 3");
    if (DeviceInfo.NumDims != DeviceInfo.Ndrange.size() &&
        2*DeviceInfo.NumDims != DeviceInfo.Ndrange.size())
      llvm_unreachable("Num dimensions does not match with ndrange list length");

    DeviceInfo.HasLocalSize = (2*DeviceInfo.NumDims) == DeviceInfo.Ndrange.size();
  }
}

void DirectiveEnvironment::verifyNonPODInfo() {
  for (const auto &InitMap : NonPODsInfo.Inits) {
    // INIT may only be in private clauses
    if (!DSAInfo.Private.count(InitMap.first))
      llvm_unreachable("Non-POD INIT OperandBundle must have a PRIVATE DSA");
  }
  for (const auto &DeinitMap : NonPODsInfo.Deinits) {
      // DEINIT may only be in private and firstprivate clauses
      if (!DSAInfo.Private.count(DeinitMap.first)
          && !DSAInfo.Firstprivate.count(DeinitMap.first))
        llvm_unreachable("Non-POD DEINIT OperandBundle must have a PRIVATE or FIRSTPRIVATE DSA");
  }
  for (const auto &CopyMap : NonPODsInfo.Copies) {
    // COPY may only be in firstprivate clauses
    if (!DSAInfo.Firstprivate.count(CopyMap.first))
      llvm_unreachable("Non-POD COPY OperandBundle must have a FIRSTPRIVATE DSA");
  }
}

void DirectiveEnvironment::verifyLoopInfo() {
  if (isOmpSsLoopDirective() || isOmpSsTaskIterForDirective()) {
    if (LoopInfo.empty())
      llvm_unreachable("LoopInfo is missing some information");
    for (size_t i = 0; i < LoopInfo.IndVar.size(); ++i) {
      if (!valueInDSABundles(LoopInfo.IndVar[i]))
        llvm_unreachable("Loop induction variable has no associated DSA");
      for (size_t j = 0; j < LoopInfo.LBound[i].Args.size(); ++j) {
        if (!valueInDSABundles(LoopInfo.LBound[i].Args[j])
            && !valueInCapturedBundle(LoopInfo.LBound[i].Args[j]))
          llvm_unreachable("Loop lbound argument value has no associated DSA or capture");
      }
      for (size_t j = 0; j < LoopInfo.UBound[i].Args.size(); ++j) {
        if (!valueInDSABundles(LoopInfo.UBound[i].Args[j])
            && !valueInCapturedBundle(LoopInfo.UBound[i].Args[j]))
          llvm_unreachable("Loop ubound argument value has no associated DSA or capture");
      }
      for (size_t j = 0; j < LoopInfo.Step[i].Args.size(); ++j) {
        if (!valueInDSABundles(LoopInfo.Step[i].Args[j])
            && !valueInCapturedBundle(LoopInfo.Step[i].Args[j]))
          llvm_unreachable("Loop step argument value has no associated DSA or capture");
      }
    }
  }
}

void DirectiveEnvironment::verifyWhileInfo() {
  if (isOmpSsTaskIterWhileDirective()) {
    if (WhileInfo.empty())
      llvm_unreachable("WhileInfo is missing some information");
    for (size_t j = 0; j < WhileInfo.Args.size(); ++j) {
      if (!valueInDSABundles(WhileInfo.Args[j])
          && !valueInCapturedBundle(WhileInfo.Args[j]))
        llvm_unreachable("WhileCond argument value has no associated DSA or capture");
    }
  }
}

void DirectiveEnvironment::verifyMultiDependInfo() {
  for (auto &DI : DependsInfo.List)
    if (auto *MDI = dyn_cast<MultiDependInfo>(DI.get())) {
      for (auto *V : MDI->Iters)
        if (!valueInDSABundles(V)
            && !valueInCapturedBundle(V))
          llvm_unreachable("Multidependency value has no associated DSA or capture");
      for (auto *V : MDI->Args)
        if (!valueInDSABundles(V)
            && !valueInCapturedBundle(V))
          llvm_unreachable("Multidependency value has no associated DSA or capture");
    }
}

void DirectiveEnvironment::verifyLabelInfo() {
  if (Label && !isa<Constant>(Label))
    llvm_unreachable("Expected a constant as a label");
}

void DirectiveEnvironment::verifyCoroInfo() {
  if (!CoroInfo.empty()) {
    if (!CoroInfo.Handle)
      llvm_unreachable("Missing coroutine handle");
    if (!CoroInfo.SizeStore)
      llvm_unreachable("Missing coroutine handle");
    // TODO: check coro handle is firstprivate
  }
}

void DirectiveEnvironment::verify() {
  verifyVLADimsInfo();

  // release directive does not need data-sharing checks
  if (DirectiveKind != OSSD_release)
    verifyDependInfo();

  verifyReductionInitCombInfo();
  verifyCostInfo();
  verifyPriorityInfo();
  verifyOnreadyInfo();
  verifyDeviceInfo();
  verifyNonPODInfo();
  verifyLoopInfo();
  verifyWhileInfo();
  verifyMultiDependInfo();
  verifyLabelInfo();
  verifyCoroInfo();
}

DirectiveEnvironment::DirectiveEnvironment(const Instruction *I) {
  const IntrinsicInst *II = cast<IntrinsicInst>(I);
  for (unsigned i = 0, e = II->getNumOperandBundles(); i != e; ++i) {
    OperandBundleUse OBUse = II->getOperandBundleAt(i);
    OperandBundleDef OBDef(OBUse);
    uint64_t Id = OBUse.getTagID();
    switch (Id) {
    case LLVMContext::OB_oss_dir:
      gatherDirInfo(OBDef);
      break;
    case LLVMContext::OB_oss_shared:
      gatherSharedInfo(OBDef);
      break;
    case LLVMContext::OB_oss_private:
      gatherPrivateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_firstprivate:
      gatherFirstprivateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_vla_dims:
      gatherVLADimsInfo(OBDef);
      break;
    case LLVMContext::OB_oss_dep_in:
    case LLVMContext::OB_oss_dep_out:
    case LLVMContext::OB_oss_dep_inout:
    case LLVMContext::OB_oss_dep_concurrent:
    case LLVMContext::OB_oss_dep_commutative:
    case LLVMContext::OB_oss_dep_weakin:
    case LLVMContext::OB_oss_dep_weakout:
    case LLVMContext::OB_oss_dep_weakinout:
    case LLVMContext::OB_oss_dep_weakconcurrent:
    case LLVMContext::OB_oss_dep_weakcommutative:
    case LLVMContext::OB_oss_dep_reduction:
    case LLVMContext::OB_oss_dep_weakreduction:
      gatherDependInfo(OBDef, Id);
      break;
    case LLVMContext::OB_oss_reduction_init:
      gatherReductionInitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_reduction_comb:
      gatherReductionCombInfo(OBDef);
      break;
    case LLVMContext::OB_oss_final:
      gatherFinalInfo(OBDef);
      break;
    case LLVMContext::OB_oss_if:
      gatherIfInfo(OBDef);
      break;
    case LLVMContext::OB_oss_cost:
      gatherCostInfo(OBDef);
      break;
    case LLVMContext::OB_oss_priority:
      gatherPriorityInfo(OBDef);
      break;
    case LLVMContext::OB_oss_label:
      gatherLabelInfo(OBDef);
      break;
    case LLVMContext::OB_oss_onready:
      gatherOnreadyInfo(OBDef);
      break;
    case LLVMContext::OB_oss_wait:
      gatherWaitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device:
      gatherDeviceInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device_ndrange:
      gatherDeviceNdrangeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device_grid:
      gatherDeviceGridInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device_dev_func:
      gatherDeviceDevFuncInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device_call_order:
      gatherDeviceCallOrderInfo(OBDef);
      break;
    case LLVMContext::OB_oss_device_shmem:
      gatherDeviceShmemInfo(OBDef);
      break;
    case LLVMContext::OB_oss_captured:
      gatherCapturedInfo(OBDef);
      break;
    case LLVMContext::OB_oss_init:
      gatherNonPODInitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_deinit:
      gatherNonPODDeinitInfo(OBDef);
      break;
    case LLVMContext::OB_oss_copy:
      gatherNonPODCopyInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_type:
      gatherLoopTypeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_ind_var:
      gatherLoopIndVarInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_lower_bound:
      gatherLoopLowerBoundInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_upper_bound:
      gatherLoopUpperBoundInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_step:
      gatherLoopStepInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_chunksize:
      gatherLoopChunksizeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_grainsize:
      gatherLoopGrainsizeInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_unroll:
      gatherLoopUnrollInfo(OBDef);
      break;
    case LLVMContext::OB_oss_loop_update:
      gatherLoopUpdateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_while_cond:
      gatherWhileCondInfo(OBDef);
      break;
    case LLVMContext::OB_oss_multidep_range_in:
    case LLVMContext::OB_oss_multidep_range_out:
    case LLVMContext::OB_oss_multidep_range_inout:
    case LLVMContext::OB_oss_multidep_range_concurrent:
    case LLVMContext::OB_oss_multidep_range_commutative:
    case LLVMContext::OB_oss_multidep_range_weakin:
    case LLVMContext::OB_oss_multidep_range_weakout:
    case LLVMContext::OB_oss_multidep_range_weakinout:
    case LLVMContext::OB_oss_multidep_range_weakconcurrent:
    case LLVMContext::OB_oss_multidep_range_weakcommutative:
      gatherMultiDependInfo(OBDef, Id);
      break;
    case LLVMContext::OB_oss_decl_source:
      gatherDeclSource(OBDef);
      break;
    case LLVMContext::OB_oss_coro_handle:
      gatherCoroHandle(OBDef);
      break;
    case LLVMContext::OB_oss_coro_size_store:
      gatherCoroSizeStore(OBDef);
      break;
    case LLVMContext::OB_oss_immediate:
      gatherImmediateInfo(OBDef);
      break;
    case LLVMContext::OB_oss_microtask:
      gatherMicrotaskInfo(OBDef);
      break;
    default:
      llvm_unreachable("unknown ompss-2 bundle id");
    }
  }
}

void OmpSsRegionAnalysis::print_verbose(
    Instruction *Cur, int Depth, int PrintSpaceMultiplier) const {
  if (Cur) {
    const DirectiveAnalysisInfo &AnalysisInfo = DEntryToDAnalysisInfo.lookup(Cur);
    const DirectiveInfo *Info = DEntryToDInfo.find(Cur)->second.get();
    const DirectiveEnvironment &DirEnv = Info->DirEnv;
    dbgs() << std::string(Depth*PrintSpaceMultiplier, ' ') << "[" << Depth << "] ";
    dbgs() << DirEnv.getDirectiveNameAsStr();
    dbgs() << " ";
    Cur->printAsOperand(dbgs(), false);

    std::string SpaceMultiplierStr = std::string((Depth + 1) * PrintSpaceMultiplier, ' ');
    if (PrintVerboseLevel == PV_Uses) {
      for (auto *V : AnalysisInfo.UsesBeforeEntry) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr
               << "[Before] ";
        V->printAsOperand(dbgs(), false);
      }
      for (auto *V : AnalysisInfo.UsesAfterExit) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr
               << "[After] ";
        V->printAsOperand(dbgs(), false);
      }
    }
    else if (PrintVerboseLevel == PV_DsaMissing) {
      for (auto *V : AnalysisInfo.UsesBeforeEntry) {
        if (!DirEnv.valueInDSABundles(V)) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr;
          V->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_DsaVLADimsMissing) {
      // Count VLAs and DSAs, Well-formed VLA must have a DSA and dimensions.
      // That is, it must have a frequency of 2
      std::map<const Value *, size_t> DSAVLADimsFreqMap;
      for (const auto &Pair : DirEnv.DSAInfo.Shared) DSAVLADimsFreqMap[Pair.first]++;
      for (const auto &Pair : DirEnv.DSAInfo.Private) DSAVLADimsFreqMap[Pair.first]++;
      for (const auto &Pair : DirEnv.DSAInfo.Firstprivate) DSAVLADimsFreqMap[Pair.first]++;

      for (const auto &VLAWithDimsMap : DirEnv.VLADimsInfo) {
        DSAVLADimsFreqMap[VLAWithDimsMap.first]++;
      }
      for (const auto &Pair : DSAVLADimsFreqMap) {
        // It's expected to have only two VLA bundles, the DSA and dimensions
        if (Pair.second != 2) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr;
          Pair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_VLADimsCaptureMissing) {
      for (const auto &VLAWithDimsMap : DirEnv.VLADimsInfo) {
        for (auto *V : VLAWithDimsMap.second) {
          if (!DirEnv.valueInCapturedBundle(V)) {
            dbgs() << "\n";
            dbgs() << SpaceMultiplierStr;
            V->printAsOperand(dbgs(), false);
          }
        }
      }
    }
    else if (PrintVerboseLevel == PV_NonPODDSAMissing) {
      for (const auto &InitsPair : DirEnv.NonPODsInfo.Inits) {
        if (!DirEnv.DSAInfo.Private.count(InitsPair.first)) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Init] ";
          InitsPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (const auto &CopiesPair : DirEnv.NonPODsInfo.Copies) {
        if (!DirEnv.DSAInfo.Firstprivate.count(CopiesPair.first)) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Copy] ";
          CopiesPair.first->printAsOperand(dbgs(), false);
        }
      }
      for (const auto &DeinitsPair : DirEnv.NonPODsInfo.Deinits) {
        if (!DirEnv.DSAInfo.Private.count(DeinitsPair.first)
            && !DirEnv.DSAInfo.Firstprivate.count(DeinitsPair.first)) {
          dbgs() << "\n";
          dbgs() << SpaceMultiplierStr
                 << "[Deinit] ";
          DeinitsPair.first->printAsOperand(dbgs(), false);
        }
      }
    }
    else if (PrintVerboseLevel == PV_ReductionInitsCombiners) {
      for (const auto &RedInfo : DirEnv.ReductionsInitCombInfo) {
        dbgs() << "\n";
        dbgs() << SpaceMultiplierStr;
        RedInfo.first->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Init->printAsOperand(dbgs(), false);
        dbgs() << " ";
        RedInfo.second.Comb->printAsOperand(dbgs(), false);
      }
    }
    dbgs() << "\n";
  }
  for (auto *II : DirectivesTree.lookup(Cur)) {
    print_verbose(II, Depth + 1, PrintSpaceMultiplier);
  }
}

DirectiveFunctionInfo& OmpSsRegionAnalysis::getFuncInfo() { return DirectiveFuncInfo; }

void OmpSsRegionAnalysis::print(raw_ostream &OS) const {
  print_verbose(nullptr, -1, PrintSpaceMultiplier);
}

// child directives will be placed before its parent directives
void OmpSsRegionAnalysis::convertDirectivesTreeToVectorImpl(
  Instruction *Cur, SmallVectorImpl<Instruction *> &Stack) {

  if (Cur)
    if (auto *II = dyn_cast<IntrinsicInst>(Cur))
      if (II->getIntrinsicID() == Intrinsic::directive_region_entry)
        Stack.push_back(Cur);

  // TODO: Why using operator[] does weird things?
  // for (auto II : DirectivesTree[Cur]) {
  for (auto II : DirectivesTree.lookup(Cur)) {
    convertDirectivesTreeToVectorImpl(II, Stack);
  }
  if (Cur) {
    DirectiveInfo *DI = DEntryToDInfo[Cur].get();

    if (auto *II = dyn_cast<IntrinsicInst>(Cur)) {
      if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
        Stack.pop_back();

        for (Instruction *I : Stack) {
          // Annotate the current directive as inner of all directives in stack
          DirectiveInfo *DIStack = DEntryToDInfo[I].get();
          DIStack->InnerDirectiveInfos.push_back(DI);
        }
      }
    }
    DirectiveFuncInfo.PostOrder.push_back(DI);
  }
}

// child directives will be placed before its parent directives
void OmpSsRegionAnalysis::convertDirectivesTreeToVector() {
  SmallVector<Instruction *, 4> Stack;
  convertDirectivesTreeToVectorImpl(nullptr, Stack);
}

OmpSsRegionAnalysis::OmpSsRegionAnalysis(Function &F, DominatorTree &DT) {

  MapVector<BasicBlock *, SmallVector<Instruction *, 4>> BBDirectiveStacks;
  SmallVector<BasicBlock*, 8> Worklist;
  SmallPtrSet<BasicBlock*, 8> Visited;

  Worklist.push_back(&F.getEntryBlock());
  Visited.insert(&F.getEntryBlock());
  while (!Worklist.empty()) {
    auto WIt = Worklist.begin();
    BasicBlock *BB = *WIt;
    Worklist.erase(WIt);

    SmallVectorImpl<Instruction *> &Stack = BBDirectiveStacks[BB];

    for (Instruction &I : *BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == Intrinsic::directive_region_entry) {
          assert(II->hasOneUse() && "Directive entry has more than one user.");

          Instruction *Exit = cast<Instruction>(II->user_back());
          // This should not happen because it will crash before this pass
          assert(orderedInstructions(DT, II, Exit) && "Directive entry does not dominate exit.");

          // directive.region pushes into the stack
          if (Stack.empty()) {
            // outer directive, insert into nullptr
            DirectivesTree[nullptr].push_back(II);
          } else {
            DirectivesTree[Stack.back()].push_back(II);
          }
          Stack.push_back(II);

          auto Dir = std::make_unique<DirectiveInfo>(II, Exit);
          if (!DisableChecks)
            Dir->DirEnv.verify();

          DEntryToDInfo.insert({II, std::move(Dir)});

        } else if (II->getIntrinsicID() == Intrinsic::directive_region_exit) {
          if (Stack.empty())
            llvm_unreachable("Directive exit hit without and entry.");

          Instruction *StackEntry = Stack.back();
          Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
          assert(StackExit == II && "unexpected directive exit instr.");

          Stack.pop_back();
        } else if (II->getIntrinsicID() == Intrinsic::directive_marker) {

          // directive_marker does not push into the stack
          if (Stack.empty()) {
            // outer directive, insert into nullptr
            DirectivesTree[nullptr].push_back(II);
          } else {
            DirectivesTree[Stack.back()].push_back(II);
          }

          auto Dir = std::make_unique<DirectiveInfo>(II);
          if (!DisableChecks)
            Dir->DirEnv.verify();

          DEntryToDInfo.insert({II, std::move(Dir)});
        }
      } else if (!Stack.empty()) {
        Instruction *StackEntry = Stack.back();
        Instruction *StackExit = cast<Instruction>(StackEntry->user_back());
        DirectiveAnalysisInfo &DAI = DEntryToDAnalysisInfo[StackEntry];
        const DirectiveInfo *DI = DEntryToDInfo[StackEntry].get();
        const DirectiveEnvironment &DirEnv = DI->DirEnv;
        for (Use &U : I.operands()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U.get())) {
            if (orderedInstructions(DT, I2, StackEntry)) {
              DAI.UsesBeforeEntry.insert(I2);
            if (!DisableChecks
                && !DirEnv.valueInDSABundles(I2)
                && !DirEnv.valueInCapturedBundle(I2)) {
                llvm_unreachable("Value supposed to be inside directive entry "
                                 "OperandBundle not found.");
              }
            }
          } else if (Argument *A = dyn_cast<Argument>(U.get())) {
            DAI.UsesBeforeEntry.insert(A);
            if (!DisableChecks
                && !DirEnv.valueInDSABundles(A)
                && !DirEnv.valueInCapturedBundle(A)) {
              llvm_unreachable("Value supposed to be inside directive entry "
                               "OperandBundle not found.");
            }
          }
        }
        for (User *U : I.users()) {
          if (Instruction *I2 = dyn_cast<Instruction>(U)) {
            if (orderedInstructions(DT, StackExit, I2)) {
              DAI.UsesAfterExit.insert(&I);
              if (!DisableChecks) {
                llvm_unreachable("Value inside the directive body used after it.");
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
          // invalidated after BBDirectiveStacks[*It].
          StackCopy.reset(
              new std::vector<Instruction *>(Stack.begin(), Stack.end()));
        }

        BBDirectiveStacks[*It].append(StackCopy->begin(), StackCopy->end());
      }
    }
  }

  convertDirectivesTreeToVector();

}

// OmpSsRegionAnalysisLegacyPass
//
bool OmpSsRegionAnalysisLegacyPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  ORA = OmpSsRegionAnalysis(F, DT);

  return false;
}

void OmpSsRegionAnalysisLegacyPass::releaseMemory() {
  ORA = OmpSsRegionAnalysis();
}

void OmpSsRegionAnalysisLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
}

char OmpSsRegionAnalysisLegacyPass::ID = 0;

OmpSsRegionAnalysisLegacyPass::OmpSsRegionAnalysisLegacyPass() : FunctionPass(ID) {
  initializeOmpSsRegionAnalysisLegacyPassPass(*PassRegistry::getPassRegistry());
}

void OmpSsRegionAnalysisLegacyPass::print(raw_ostream &OS, const Module *M) const {
  ORA.print(OS);
}

OmpSsRegionAnalysis& OmpSsRegionAnalysisLegacyPass::getResult() { return ORA; }

INITIALIZE_PASS_BEGIN(OmpSsRegionAnalysisLegacyPass, "ompss-2-regions",
                      "Classify OmpSs-2 inside region uses", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(OmpSsRegionAnalysisLegacyPass, "ompss-2-regions",
                    "Classify OmpSs-2 inside region uses", false, true)


// OmpSsRegionAnalysisPass
//
AnalysisKey OmpSsRegionAnalysisPass::Key;

OmpSsRegionAnalysis OmpSsRegionAnalysisPass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  auto *DT = &FAM.getResult<DominatorTreeAnalysis>(F);
  return OmpSsRegionAnalysis(F, *DT);
}

// OmpSsRegionPrinterPass
//
OmpSsRegionPrinterPass::OmpSsRegionPrinterPass(raw_ostream &OS) : OS(OS) {}

PreservedAnalyses OmpSsRegionPrinterPass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  FAM.getResult<OmpSsRegionAnalysisPass>(F).print(OS);

  return PreservedAnalyses::all();
}
