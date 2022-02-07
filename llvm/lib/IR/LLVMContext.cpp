//===-- LLVMContext.cpp - Implement LLVMContext ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <utility>

using namespace llvm;

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) {
  // Create the fixed metadata kinds. This is done in the same order as the
  // MD_* enum values so that they correspond.
  std::pair<unsigned, StringRef> MDKinds[] = {
#define LLVM_FIXED_MD_KIND(EnumID, Name, Value) {EnumID, Name},
#include "llvm/IR/FixedMetadataKinds.def"
#undef LLVM_FIXED_MD_KIND
  };

  for (auto &MDKind : MDKinds) {
    unsigned ID = getMDKindID(MDKind.second);
    assert(ID == MDKind.first && "metadata kind id drifted");
    (void)ID;
  }

  auto *DeoptEntry = pImpl->getOrInsertBundleTag("deopt");
  assert(DeoptEntry->second == LLVMContext::OB_deopt &&
         "deopt operand bundle id drifted!");
  (void)DeoptEntry;

  auto *FuncletEntry = pImpl->getOrInsertBundleTag("funclet");
  assert(FuncletEntry->second == LLVMContext::OB_funclet &&
         "funclet operand bundle id drifted!");
  (void)FuncletEntry;

  auto *GCTransitionEntry = pImpl->getOrInsertBundleTag("gc-transition");
  assert(GCTransitionEntry->second == LLVMContext::OB_gc_transition &&
         "gc-transition operand bundle id drifted!");
  (void)GCTransitionEntry;

  auto *CFGuardTargetEntry = pImpl->getOrInsertBundleTag("cfguardtarget");
  assert(CFGuardTargetEntry->second == LLVMContext::OB_cfguardtarget &&
         "cfguardtarget operand bundle id drifted!");
  (void)CFGuardTargetEntry;

  auto *PreallocatedEntry = pImpl->getOrInsertBundleTag("preallocated");
  assert(PreallocatedEntry->second == LLVMContext::OB_preallocated &&
         "preallocated operand bundle id drifted!");
  (void)PreallocatedEntry;

  auto *GCLiveEntry = pImpl->getOrInsertBundleTag("gc-live");
  assert(GCLiveEntry->second == LLVMContext::OB_gc_live &&
         "gc-transition operand bundle id drifted!");
  (void)GCLiveEntry;

  auto *ClangAttachedCall =
      pImpl->getOrInsertBundleTag("clang.arc.attachedcall");
  assert(ClangAttachedCall->second == LLVMContext::OB_clang_arc_attachedcall &&
         "clang.arc.attachedcall operand bundle id drifted!");
  (void)ClangAttachedCall;

  // OmpSs IDs
  auto *OSSDirEntry = pImpl->getOrInsertBundleTag("DIR.OSS");
  assert(OSSDirEntry->second == LLVMContext::OB_oss_dir &&
         "oss_dir operand bundle id drifted!");
  (void)OSSDirEntry;

  auto *OSSSharedEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.SHARED");
  assert(OSSSharedEntry->second == LLVMContext::OB_oss_shared &&
         "oss_shared operand bundle id drifted!");
  (void)OSSSharedEntry;

  auto *OSSPrivateEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.PRIVATE");
  assert(OSSPrivateEntry->second == LLVMContext::OB_oss_private &&
         "oss_private operand bundle id drifted!");
  (void)OSSPrivateEntry;

  auto *OSSFirstprivateEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.FIRSTPRIVATE");
  assert(OSSFirstprivateEntry->second == LLVMContext::OB_oss_firstprivate &&
         "oss_firstprivate operand bundle id drifted!");
  (void)OSSFirstprivateEntry;

  auto *OSSVLADimsEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.VLA.DIMS");
  assert(OSSVLADimsEntry->second == LLVMContext::OB_oss_vla_dims &&
         "oss_vla_dims operand bundle id drifted!");
  (void)OSSVLADimsEntry;

  auto *OSSDepInEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.IN");
  assert(OSSDepInEntry->second == LLVMContext::OB_oss_dep_in &&
         "oss_dep_in operand bundle id drifted!");
  (void)OSSDepInEntry;

  auto *OSSDepOutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.OUT");
  assert(OSSDepOutEntry->second == LLVMContext::OB_oss_dep_out &&
         "oss_dep_out operand bundle id drifted!");
  (void)OSSDepOutEntry;

  auto *OSSDepInoutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.INOUT");
  assert(OSSDepInoutEntry->second == LLVMContext::OB_oss_dep_inout &&
         "oss_dep_inout operand bundle id drifted!");
  (void)OSSDepInoutEntry;

  auto *OSSDepConcurrentEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.CONCURRENT");
  assert(OSSDepConcurrentEntry->second == LLVMContext::OB_oss_dep_concurrent &&
         "oss_dep_concurrent operand bundle id drifted!");
  (void)OSSDepConcurrentEntry;

  auto *OSSDepCommutativeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.COMMUTATIVE");
  assert(OSSDepCommutativeEntry->second == LLVMContext::OB_oss_dep_commutative &&
         "oss_dep_commutative operand bundle id drifted!");
  (void)OSSDepCommutativeEntry;

  auto *OSSDepWeakInEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKIN");
  assert(OSSDepWeakInEntry->second == LLVMContext::OB_oss_dep_weakin &&
         "oss_dep_weakin operand bundle id drifted!");
  (void)OSSDepWeakInEntry;

  auto *OSSDepWeakOutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKOUT");
  assert(OSSDepWeakOutEntry->second == LLVMContext::OB_oss_dep_weakout &&
         "oss_dep_weakout operand bundle id drifted!");
  (void)OSSDepWeakOutEntry;

  auto *OSSDepWeakInoutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKINOUT");
  assert(OSSDepWeakInoutEntry->second == LLVMContext::OB_oss_dep_weakinout &&
         "oss_dep_weakinout operand bundle id drifted!");
  (void)OSSDepWeakInoutEntry;

  auto *OSSDepWeakConcurrentEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKCONCURRENT");
  assert(OSSDepWeakConcurrentEntry->second == LLVMContext::OB_oss_dep_weakconcurrent &&
         "oss_dep_weakconcurrent operand bundle id drifted!");
  (void)OSSDepWeakConcurrentEntry;

  auto *OSSDepWeakCommutativeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKCOMMUTATIVE");
  assert(OSSDepWeakCommutativeEntry->second == LLVMContext::OB_oss_dep_weakcommutative &&
         "oss_dep_weakcommutative operand bundle id drifted!");
  (void)OSSDepWeakCommutativeEntry;

  auto *OSSDepReductionEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.REDUCTION");
  assert(OSSDepReductionEntry->second == LLVMContext::OB_oss_dep_reduction &&
         "oss_dep_reduction operand bundle id drifted!");
  (void)OSSDepReductionEntry;

  auto *OSSDepWeakReductionEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.WEAKREDUCTION");
  assert(OSSDepWeakReductionEntry->second == LLVMContext::OB_oss_dep_weakreduction &&
         "oss_dep_weakreduction operand bundle id drifted!");
  (void)OSSDepWeakReductionEntry;

  auto *OSSDepReductionInitEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.REDUCTION.INIT");
  assert(OSSDepReductionInitEntry->second == LLVMContext::OB_oss_reduction_init &&
         "oss_dep_reduction operand bundle id drifted!");
  (void)OSSDepReductionInitEntry;

  auto *OSSDepReductionCombEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEP.REDUCTION.COMBINE");
  assert(OSSDepReductionCombEntry->second == LLVMContext::OB_oss_reduction_comb &&
         "oss_reduction_comb operand bundle id drifted!");
  (void)OSSDepReductionCombEntry;

  auto *OSSFinalEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.FINAL");
  assert(OSSFinalEntry->second == LLVMContext::OB_oss_final &&
         "oss_final operand bundle id drifted!");
  (void)OSSFinalEntry;

  auto *OSSIfEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.IF");
  assert(OSSIfEntry->second == LLVMContext::OB_oss_if &&
         "oss_if operand bundle id drifted!");
  (void)OSSIfEntry;

  auto *OSSCostEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.COST");
  assert(OSSCostEntry->second == LLVMContext::OB_oss_cost &&
         "oss_cost operand bundle id drifted!");
  (void)OSSCostEntry;

  auto *OSSPriorityEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.PRIORITY");
  assert(OSSPriorityEntry->second == LLVMContext::OB_oss_priority &&
         "oss_priority operand bundle id drifted!");
  (void)OSSPriorityEntry;

  auto *OSSLabelEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LABEL");
  assert(OSSLabelEntry->second == LLVMContext::OB_oss_label &&
         "oss_label operand bundle id drifted!");
  (void)OSSLabelEntry;

  auto *OSSWaitEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.WAIT");
  assert(OSSWaitEntry->second == LLVMContext::OB_oss_wait &&
         "oss_wait operand bundle id drifted!");
  (void)OSSWaitEntry;

  auto *OSSCapturedEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.CAPTURED");
  assert(OSSCapturedEntry->second == LLVMContext::OB_oss_captured &&
         "oss_captured operand bundle id drifted!");
  (void)OSSCapturedEntry;

  auto *OSSInitEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.INIT");
  assert(OSSInitEntry->second == LLVMContext::OB_oss_init &&
         "oss_init operand bundle id drifted!");
  (void)OSSInitEntry;

  auto *OSSDeinitEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DEINIT");
  assert(OSSDeinitEntry->second == LLVMContext::OB_oss_deinit &&
         "oss_deinit operand bundle id drifted!");
  (void)OSSDeinitEntry;

  auto *OSSCopyEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.COPY");
  assert(OSSCopyEntry->second == LLVMContext::OB_oss_copy &&
         "oss_copy operand bundle id drifted!");
  (void)OSSCopyEntry;

  auto *OSSLoopTypeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.TYPE");
  assert(OSSLoopTypeEntry->second == LLVMContext::OB_oss_loop_type &&
         "oss_loop_type operand bundle id drifted!");
  (void)OSSLoopTypeEntry;

  auto *OSSLoopIndVarEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.IND.VAR");
  assert(OSSLoopIndVarEntry->second == LLVMContext::OB_oss_loop_ind_var &&
         "oss_loop_ind_var operand bundle id drifted!");
  (void)OSSLoopIndVarEntry;

  auto *OSSLoopLowerBoundEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.LOWER.BOUND");
  assert(OSSLoopLowerBoundEntry->second == LLVMContext::OB_oss_loop_lower_bound &&
         "oss_loop_lower_bound operand bundle id drifted!");
  (void)OSSLoopLowerBoundEntry;

  auto *OSSLoopUpperBoundEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.UPPER.BOUND");
  assert(OSSLoopUpperBoundEntry->second == LLVMContext::OB_oss_loop_upper_bound &&
         "oss_loop_upper_bound operand bundle id drifted!");
  (void)OSSLoopUpperBoundEntry;

  auto *OSSLoopStepEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.STEP");
  assert(OSSLoopStepEntry->second == LLVMContext::OB_oss_loop_step &&
         "oss_loop_step operand bundle id drifted!");
  (void)OSSLoopStepEntry;

  auto *OSSLoopChunksizeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.CHUNKSIZE");
  assert(OSSLoopChunksizeEntry->second == LLVMContext::OB_oss_loop_chunksize &&
         "oss_loop_chunksize operand bundle id drifted!");
  (void)OSSLoopChunksizeEntry;

  auto *OSSLoopGrainsizeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.LOOP.GRAINSIZE");
  assert(OSSLoopGrainsizeEntry->second == LLVMContext::OB_oss_loop_grainsize &&
         "oss_loop_grainsize operand bundle id drifted!");
  (void)OSSLoopGrainsizeEntry;

  auto *OSSMultiDepRangeInEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.IN");
  assert(OSSMultiDepRangeInEntry->second == LLVMContext::OB_oss_multidep_range_in &&
         "oss_multidep_range_in operand bundle id drifted!");
  (void)OSSMultiDepRangeInEntry;

  auto *OSSMultiDepRangeOutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.OUT");
  assert(OSSMultiDepRangeOutEntry->second == LLVMContext::OB_oss_multidep_range_out &&
         "oss_multidep_range_out operand bundle id drifted!");
  (void)OSSMultiDepRangeOutEntry;

  auto *OSSMultiDepRangeInoutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.INOUT");
  assert(OSSMultiDepRangeInoutEntry->second == LLVMContext::OB_oss_multidep_range_inout &&
         "oss_multidep_range_inout operand bundle id drifted!");
  (void)OSSMultiDepRangeInoutEntry;

  auto *OSSMultiDepRangeConcurrentEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.CONCURRENT");
  assert(OSSMultiDepRangeConcurrentEntry->second == LLVMContext::OB_oss_multidep_range_concurrent &&
         "oss_multidep_range_concurrent operand bundle id drifted!");
  (void)OSSMultiDepRangeConcurrentEntry;

  auto *OSSMultiDepRangeCommutativeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.COMMUTATIVE");
  assert(OSSMultiDepRangeCommutativeEntry->second == LLVMContext::OB_oss_multidep_range_commutative &&
         "oss_multidep_range_commutative operand bundle id drifted!");
  (void)OSSMultiDepRangeCommutativeEntry;

  auto *OSSMultiDepRangeWeakInEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.WEAKIN");
  assert(OSSMultiDepRangeWeakInEntry->second == LLVMContext::OB_oss_multidep_range_weakin &&
         "oss_multidep_range_weakin operand bundle id drifted!");
  (void)OSSMultiDepRangeWeakInEntry;

  auto *OSSMultiDepRangeWeakOutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.WEAKOUT");
  assert(OSSMultiDepRangeWeakOutEntry->second == LLVMContext::OB_oss_multidep_range_weakout &&
         "oss_multidep_range_weakout operand bundle id drifted!");
  (void)OSSMultiDepRangeWeakOutEntry;

  auto *OSSMultiDepRangeWeakInoutEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.WEAKINOUT");
  assert(OSSMultiDepRangeWeakInoutEntry->second == LLVMContext::OB_oss_multidep_range_weakinout &&
         "oss_multidep_range_weakinout operand bundle id drifted!");
  (void)OSSMultiDepRangeWeakInoutEntry;

  auto *OSSMultiDepRangeWeakConcurrentEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.WEAKCONCURRENT");
  assert(OSSMultiDepRangeWeakConcurrentEntry->second == LLVMContext::OB_oss_multidep_range_weakconcurrent &&
         "oss_multidep_range_weakconcurrent operand bundle id drifted!");
  (void)OSSMultiDepRangeWeakConcurrentEntry;

  auto *OSSMultiDepRangeWeakCommutativeEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.MULTIDEP.RANGE.WEAKCOMMUTATIVE");
  assert(OSSMultiDepRangeWeakCommutativeEntry->second == LLVMContext::OB_oss_multidep_range_weakcommutative &&
         "oss_multidep_range_weakcommutative operand bundle id drifted!");
  (void)OSSMultiDepRangeWeakCommutativeEntry;

  auto *OSSDeclSourceEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.DECL.SOURCE");
  assert(OSSDeclSourceEntry->second == LLVMContext::OB_oss_decl_source &&
         "oss_decl_source operand bundle id drifted!");
  (void)OSSDeclSourceEntry;

  auto *OSSOnreadyEntry = pImpl->getOrInsertBundleTag("QUAL.OSS.ONREADY");
  assert(OSSOnreadyEntry->second == LLVMContext::OB_oss_onready &&
         "oss_onready operand bundle id drifted!");
  (void)OSSOnreadyEntry;
  // END OmpSs IDs

  SyncScope::ID SingleThreadSSID =
      pImpl->getOrInsertSyncScopeID("singlethread");
  assert(SingleThreadSSID == SyncScope::SingleThread &&
         "singlethread synchronization scope ID drifted!");
  (void)SingleThreadSSID;

  SyncScope::ID SystemSSID =
      pImpl->getOrInsertSyncScopeID("");
  assert(SystemSSID == SyncScope::System &&
         "system synchronization scope ID drifted!");
  (void)SystemSSID;
}

LLVMContext::~LLVMContext() { delete pImpl; }

void LLVMContext::addModule(Module *M) {
  pImpl->OwnedModules.insert(M);
}

void LLVMContext::removeModule(Module *M) {
  pImpl->OwnedModules.erase(M);
}

//===----------------------------------------------------------------------===//
// Recoverable Backend Errors
//===----------------------------------------------------------------------===//

void LLVMContext::setDiagnosticHandlerCallBack(
    DiagnosticHandler::DiagnosticHandlerTy DiagnosticHandler,
    void *DiagnosticContext, bool RespectFilters) {
  pImpl->DiagHandler->DiagHandlerCallback = DiagnosticHandler;
  pImpl->DiagHandler->DiagnosticContext = DiagnosticContext;
  pImpl->RespectDiagnosticFilters = RespectFilters;
}

void LLVMContext::setDiagnosticHandler(std::unique_ptr<DiagnosticHandler> &&DH,
                                      bool RespectFilters) {
  pImpl->DiagHandler = std::move(DH);
  pImpl->RespectDiagnosticFilters = RespectFilters;
}

void LLVMContext::setDiagnosticsHotnessRequested(bool Requested) {
  pImpl->DiagnosticsHotnessRequested = Requested;
}
bool LLVMContext::getDiagnosticsHotnessRequested() const {
  return pImpl->DiagnosticsHotnessRequested;
}

void LLVMContext::setDiagnosticsHotnessThreshold(Optional<uint64_t> Threshold) {
  pImpl->DiagnosticsHotnessThreshold = Threshold;
}

uint64_t LLVMContext::getDiagnosticsHotnessThreshold() const {
  return pImpl->DiagnosticsHotnessThreshold.getValueOr(UINT64_MAX);
}

bool LLVMContext::isDiagnosticsHotnessThresholdSetFromPSI() const {
  return !pImpl->DiagnosticsHotnessThreshold.hasValue();
}

remarks::RemarkStreamer *LLVMContext::getMainRemarkStreamer() {
  return pImpl->MainRemarkStreamer.get();
}
const remarks::RemarkStreamer *LLVMContext::getMainRemarkStreamer() const {
  return const_cast<LLVMContext *>(this)->getMainRemarkStreamer();
}
void LLVMContext::setMainRemarkStreamer(
    std::unique_ptr<remarks::RemarkStreamer> RemarkStreamer) {
  pImpl->MainRemarkStreamer = std::move(RemarkStreamer);
}

LLVMRemarkStreamer *LLVMContext::getLLVMRemarkStreamer() {
  return pImpl->LLVMRS.get();
}
const LLVMRemarkStreamer *LLVMContext::getLLVMRemarkStreamer() const {
  return const_cast<LLVMContext *>(this)->getLLVMRemarkStreamer();
}
void LLVMContext::setLLVMRemarkStreamer(
    std::unique_ptr<LLVMRemarkStreamer> RemarkStreamer) {
  pImpl->LLVMRS = std::move(RemarkStreamer);
}

DiagnosticHandler::DiagnosticHandlerTy
LLVMContext::getDiagnosticHandlerCallBack() const {
  return pImpl->DiagHandler->DiagHandlerCallback;
}

void *LLVMContext::getDiagnosticContext() const {
  return pImpl->DiagHandler->DiagnosticContext;
}

void LLVMContext::setYieldCallback(YieldCallbackTy Callback, void *OpaqueHandle)
{
  pImpl->YieldCallback = Callback;
  pImpl->YieldOpaqueHandle = OpaqueHandle;
}

void LLVMContext::yield() {
  if (pImpl->YieldCallback)
    pImpl->YieldCallback(this, pImpl->YieldOpaqueHandle);
}

void LLVMContext::emitError(const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(ErrorStr));
}

void LLVMContext::emitError(const Instruction *I, const Twine &ErrorStr) {
  assert (I && "Invalid instruction");
  diagnose(DiagnosticInfoInlineAsm(*I, ErrorStr));
}

static bool isDiagnosticEnabled(const DiagnosticInfo &DI) {
  // Optimization remarks are selective. They need to check whether the regexp
  // pattern, passed via one of the -pass-remarks* flags, matches the name of
  // the pass that is emitting the diagnostic. If there is no match, ignore the
  // diagnostic and return.
  //
  // Also noisy remarks are only enabled if we have hotness information to sort
  // them.
  if (auto *Remark = dyn_cast<DiagnosticInfoOptimizationBase>(&DI))
    return Remark->isEnabled() &&
           (!Remark->isVerbose() || Remark->getHotness());

  return true;
}

const char *
LLVMContext::getDiagnosticMessagePrefix(DiagnosticSeverity Severity) {
  switch (Severity) {
  case DS_Error:
    return "error";
  case DS_Warning:
    return "warning";
  case DS_Remark:
    return "remark";
  case DS_Note:
    return "note";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

void LLVMContext::diagnose(const DiagnosticInfo &DI) {
  if (auto *OptDiagBase = dyn_cast<DiagnosticInfoOptimizationBase>(&DI))
    if (LLVMRemarkStreamer *RS = getLLVMRemarkStreamer())
      RS->emit(*OptDiagBase);

  // If there is a report handler, use it.
  if (pImpl->DiagHandler &&
      (!pImpl->RespectDiagnosticFilters || isDiagnosticEnabled(DI)) &&
      pImpl->DiagHandler->handleDiagnostics(DI))
    return;

  if (!isDiagnosticEnabled(DI))
    return;

  // Otherwise, print the message with a prefix based on the severity.
  DiagnosticPrinterRawOStream DP(errs());
  errs() << getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
  DI.print(DP);
  errs() << "\n";
  if (DI.getSeverity() == DS_Error)
    exit(1);
}

void LLVMContext::emitError(uint64_t LocCookie, const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(LocCookie, ErrorStr));
}

//===----------------------------------------------------------------------===//
// Metadata Kind Uniquing
//===----------------------------------------------------------------------===//

/// Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  // If this is new, assign it its ID.
  return pImpl->CustomMDKindNames.insert(
                                     std::make_pair(
                                         Name, pImpl->CustomMDKindNames.size()))
      .first->second;
}

/// getHandlerNames - Populate client-supplied smallvector using custom
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size());
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    Names[I->second] = I->first();
}

void LLVMContext::getOperandBundleTags(SmallVectorImpl<StringRef> &Tags) const {
  pImpl->getOperandBundleTags(Tags);
}

StringMapEntry<uint32_t> *
LLVMContext::getOrInsertBundleTag(StringRef TagName) const {
  return pImpl->getOrInsertBundleTag(TagName);
}

uint32_t LLVMContext::getOperandBundleTagID(StringRef Tag) const {
  return pImpl->getOperandBundleTagID(Tag);
}

SyncScope::ID LLVMContext::getOrInsertSyncScopeID(StringRef SSN) {
  return pImpl->getOrInsertSyncScopeID(SSN);
}

void LLVMContext::getSyncScopeNames(SmallVectorImpl<StringRef> &SSNs) const {
  pImpl->getSyncScopeNames(SSNs);
}

void LLVMContext::setGC(const Function &Fn, std::string GCName) {
  auto It = pImpl->GCNames.find(&Fn);

  if (It == pImpl->GCNames.end()) {
    pImpl->GCNames.insert(std::make_pair(&Fn, std::move(GCName)));
    return;
  }
  It->second = std::move(GCName);
}

const std::string &LLVMContext::getGC(const Function &Fn) {
  return pImpl->GCNames[&Fn];
}

void LLVMContext::deleteGC(const Function &Fn) {
  pImpl->GCNames.erase(&Fn);
}

bool LLVMContext::shouldDiscardValueNames() const {
  return pImpl->DiscardValueNames;
}

bool LLVMContext::isODRUniquingDebugTypes() const { return !!pImpl->DITypeMap; }

void LLVMContext::enableDebugTypeODRUniquing() {
  if (pImpl->DITypeMap)
    return;

  pImpl->DITypeMap.emplace();
}

void LLVMContext::disableDebugTypeODRUniquing() { pImpl->DITypeMap.reset(); }

void LLVMContext::setDiscardValueNames(bool Discard) {
  pImpl->DiscardValueNames = Discard;
}

OptPassGate &LLVMContext::getOptPassGate() const {
  return pImpl->getOptPassGate();
}

void LLVMContext::setOptPassGate(OptPassGate& OPG) {
  pImpl->setOptPassGate(OPG);
}

const DiagnosticHandler *LLVMContext::getDiagHandlerPtr() const {
  return pImpl->DiagHandler.get();
}

std::unique_ptr<DiagnosticHandler> LLVMContext::getDiagnosticHandler() {
  return std::move(pImpl->DiagHandler);
}

void LLVMContext::enableOpaquePointers() const {
  assert(pImpl->PointerTypes.empty() && pImpl->ASPointerTypes.empty() &&
         "Must be called before creating any pointer types");
  pImpl->setOpaquePointers(true);
}

bool LLVMContext::supportsTypedPointers() const {
  return !pImpl->getOpaquePointers();
}
