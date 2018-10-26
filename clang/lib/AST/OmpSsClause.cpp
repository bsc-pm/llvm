//===- OmpSsClause.cpp - Classes for OmpSs clauses ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in OmpSsClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OmpSsClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>

using namespace clang;

OSSClause::child_range OSSClause::children() {
  switch (getClauseKind()) {
  default:
    break;
#define OMPSS_CLAUSE(Name, Class)                                             \
  case OSSC_##Name:                                                            \
    return static_cast<Class *>(this)->children();
#include "clang/Basic/OmpSsKinds.def"
  }
  llvm_unreachable("unknown OSSClause");
}

OSSClauseWithPreInit *OSSClauseWithPreInit::get(OSSClause *C) {
  auto *Res = OSSClauseWithPreInit::get(const_cast<const OSSClause *>(C));
  return Res ? const_cast<OSSClauseWithPreInit *>(Res) : nullptr;
}

const OSSClauseWithPreInit *OSSClauseWithPreInit::get(const OSSClause *C) {
  switch (C->getClauseKind()) {
  case OSSC_schedule:
    return static_cast<const OSSScheduleClause *>(C);
  case OSSC_dist_schedule:
    return static_cast<const OSSDistScheduleClause *>(C);
  case OSSC_firstprivate:
    return static_cast<const OSSFirstprivateClause *>(C);
  case OSSC_lastprivate:
    return static_cast<const OSSLastprivateClause *>(C);
  case OSSC_reduction:
    return static_cast<const OSSReductionClause *>(C);
  case OSSC_task_reduction:
    return static_cast<const OSSTaskReductionClause *>(C);
  case OSSC_in_reduction:
    return static_cast<const OSSInReductionClause *>(C);
  case OSSC_linear:
    return static_cast<const OSSLinearClause *>(C);
  case OSSC_if:
    return static_cast<const OSSIfClause *>(C);
  case OSSC_num_threads:
    return static_cast<const OSSNumThreadsClause *>(C);
  case OSSC_num_teams:
    return static_cast<const OSSNumTeamsClause *>(C);
  case OSSC_thread_limit:
    return static_cast<const OSSThreadLimitClause *>(C);
  case OSSC_device:
    return static_cast<const OSSDeviceClause *>(C);
  case OSSC_default:
  case OSSC_proc_bind:
  case OSSC_final:
  case OSSC_safelen:
  case OSSC_simdlen:
  case OSSC_collapse:
  case OSSC_private:
  case OSSC_shared:
  case OSSC_aligned:
  case OSSC_copyin:
  case OSSC_copyprivate:
  case OSSC_ordered:
  case OSSC_nowait:
  case OSSC_untied:
  case OSSC_mergeable:
  case OSSC_threadprivate:
  case OSSC_flush:
  case OSSC_read:
  case OSSC_write:
  case OSSC_update:
  case OSSC_capture:
  case OSSC_seq_cst:
  case OSSC_depend:
  case OSSC_threads:
  case OSSC_simd:
  case OSSC_map:
  case OSSC_priority:
  case OSSC_grainsize:
  case OSSC_nogroup:
  case OSSC_num_tasks:
  case OSSC_hint:
  case OSSC_defaultmap:
  case OSSC_unknown:
  case OSSC_uniform:
  case OSSC_to:
  case OSSC_from:
  case OSSC_use_device_ptr:
  case OSSC_is_device_ptr:
    break;
  }

  return nullptr;
}

OSSClauseWithPostUpdate *OSSClauseWithPostUpdate::get(OSSClause *C) {
  auto *Res = OSSClauseWithPostUpdate::get(const_cast<const OSSClause *>(C));
  return Res ? const_cast<OSSClauseWithPostUpdate *>(Res) : nullptr;
}

const OSSClauseWithPostUpdate *OSSClauseWithPostUpdate::get(const OSSClause *C) {
  switch (C->getClauseKind()) {
  case OSSC_lastprivate:
    return static_cast<const OSSLastprivateClause *>(C);
  case OSSC_reduction:
    return static_cast<const OSSReductionClause *>(C);
  case OSSC_task_reduction:
    return static_cast<const OSSTaskReductionClause *>(C);
  case OSSC_in_reduction:
    return static_cast<const OSSInReductionClause *>(C);
  case OSSC_linear:
    return static_cast<const OSSLinearClause *>(C);
  case OSSC_schedule:
  case OSSC_dist_schedule:
  case OSSC_firstprivate:
  case OSSC_default:
  case OSSC_proc_bind:
  case OSSC_if:
  case OSSC_final:
  case OSSC_num_threads:
  case OSSC_safelen:
  case OSSC_simdlen:
  case OSSC_collapse:
  case OSSC_private:
  case OSSC_shared:
  case OSSC_aligned:
  case OSSC_copyin:
  case OSSC_copyprivate:
  case OSSC_ordered:
  case OSSC_nowait:
  case OSSC_untied:
  case OSSC_mergeable:
  case OSSC_threadprivate:
  case OSSC_flush:
  case OSSC_read:
  case OSSC_write:
  case OSSC_update:
  case OSSC_capture:
  case OSSC_seq_cst:
  case OSSC_depend:
  case OSSC_device:
  case OSSC_threads:
  case OSSC_simd:
  case OSSC_map:
  case OSSC_num_teams:
  case OSSC_thread_limit:
  case OSSC_priority:
  case OSSC_grainsize:
  case OSSC_nogroup:
  case OSSC_num_tasks:
  case OSSC_hint:
  case OSSC_defaultmap:
  case OSSC_unknown:
  case OSSC_uniform:
  case OSSC_to:
  case OSSC_from:
  case OSSC_use_device_ptr:
  case OSSC_is_device_ptr:
    break;
  }

  return nullptr;
}

OSSOrderedClause *OSSOrderedClause::Create(const ASTContext &C, Expr *Num,
                                           unsigned NumLoops,
                                           SourceLocation StartLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation EndLoc) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * NumLoops));
  auto *Clause =
      new (Mem) OSSOrderedClause(Num, NumLoops, StartLoc, LParenLoc, EndLoc);
  for (unsigned I = 0; I < NumLoops; ++I) {
    Clause->setLoopNumIterations(I, nullptr);
    Clause->setLoopCounter(I, nullptr);
  }
  return Clause;
}

OSSOrderedClause *OSSOrderedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumLoops) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * NumLoops));
  auto *Clause = new (Mem) OSSOrderedClause(NumLoops);
  for (unsigned I = 0; I < NumLoops; ++I) {
    Clause->setLoopNumIterations(I, nullptr);
    Clause->setLoopCounter(I, nullptr);
  }
  return Clause;
}

void OSSOrderedClause::setLoopNumIterations(unsigned NumLoop,
                                            Expr *NumIterations) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  getTrailingObjects<Expr *>()[NumLoop] = NumIterations;
}

ArrayRef<Expr *> OSSOrderedClause::getLoopNumIterations() const {
  return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumberOfLoops);
}

void OSSOrderedClause::setLoopCounter(unsigned NumLoop, Expr *Counter) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop] = Counter;
}

Expr *OSSOrderedClause::getLoopCounter(unsigned NumLoop) {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  return getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop];
}

const Expr *OSSOrderedClause::getLoopCounter(unsigned NumLoop) const {
  assert(NumLoop < NumberOfLoops && "out of loops number.");
  return getTrailingObjects<Expr *>()[NumberOfLoops + NumLoop];
}

void OSSPrivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

OSSPrivateClause *
OSSPrivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation EndLoc,
                         ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL) {
  // Allocate space for private variables and initializer expressions.
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * VL.size()));
  OSSPrivateClause *Clause =
      new (Mem) OSSPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OSSPrivateClause *OSSPrivateClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * N));
  return new (Mem) OSSPrivateClause(N);
}

void OSSFirstprivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OSSFirstprivateClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OSSFirstprivateClause *
OSSFirstprivateClause::Create(const ASTContext &C, SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc,
                              ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL,
                              ArrayRef<Expr *> InitVL, Stmt *PreInit) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * VL.size()));
  OSSFirstprivateClause *Clause =
      new (Mem) OSSFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  Clause->setPreInitStmt(PreInit);
  return Clause;
}

OSSFirstprivateClause *OSSFirstprivateClause::CreateEmpty(const ASTContext &C,
                                                          unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * N));
  return new (Mem) OSSFirstprivateClause(N);
}

void OSSLastprivateClause::setPrivateCopies(ArrayRef<Expr *> PrivateCopies) {
  assert(PrivateCopies.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(PrivateCopies.begin(), PrivateCopies.end(), varlist_end());
}

void OSSLastprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), getPrivateCopies().end());
}

void OSSLastprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OSSLastprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OSSLastprivateClause *OSSLastprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OSSLastprivateClause *Clause =
      new (Mem) OSSLastprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OSSLastprivateClause *OSSLastprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OSSLastprivateClause(N);
}

OSSSharedClause *OSSSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSSharedClause *Clause =
      new (Mem) OSSSharedClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSSharedClause *OSSSharedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSSharedClause(N);
}

void OSSLinearClause::setPrivates(ArrayRef<Expr *> PL) {
  assert(PL.size() == varlist_size() &&
         "Number of privates is not the same as the preallocated buffer");
  std::copy(PL.begin(), PL.end(), varlist_end());
}

void OSSLinearClause::setInits(ArrayRef<Expr *> IL) {
  assert(IL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(IL.begin(), IL.end(), getPrivates().end());
}

void OSSLinearClause::setUpdates(ArrayRef<Expr *> UL) {
  assert(UL.size() == varlist_size() &&
         "Number of updates is not the same as the preallocated buffer");
  std::copy(UL.begin(), UL.end(), getInits().end());
}

void OSSLinearClause::setFinals(ArrayRef<Expr *> FL) {
  assert(FL.size() == varlist_size() &&
         "Number of final updates is not the same as the preallocated buffer");
  std::copy(FL.begin(), FL.end(), getUpdates().end());
}

OSSLinearClause *OSSLinearClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    OmpSsLinearClauseKind Modifier, SourceLocation ModifierLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
    ArrayRef<Expr *> PL, ArrayRef<Expr *> IL, Expr *Step, Expr *CalcStep,
    Stmt *PreInit, Expr *PostUpdate) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size() + 2));
  OSSLinearClause *Clause = new (Mem) OSSLinearClause(
      StartLoc, LParenLoc, Modifier, ModifierLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivates(PL);
  Clause->setInits(IL);
  // Fill update and final expressions with zeroes, they are provided later,
  // after the directive construction.
  std::fill(Clause->getInits().end(), Clause->getInits().end() + VL.size(),
            nullptr);
  std::fill(Clause->getUpdates().end(), Clause->getUpdates().end() + VL.size(),
            nullptr);
  Clause->setStep(Step);
  Clause->setCalcStep(CalcStep);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OSSLinearClause *OSSLinearClause::CreateEmpty(const ASTContext &C,
                                              unsigned NumVars) {
  // Allocate space for 4 lists (Vars, Inits, Updates, Finals) and 2 expressions
  // (Step and CalcStep).
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * NumVars + 2));
  return new (Mem) OSSLinearClause(NumVars);
}

OSSAlignedClause *
OSSAlignedClause::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation LParenLoc, SourceLocation ColonLoc,
                         SourceLocation EndLoc, ArrayRef<Expr *> VL, Expr *A) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size() + 1));
  OSSAlignedClause *Clause = new (Mem)
      OSSAlignedClause(StartLoc, LParenLoc, ColonLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setAlignment(A);
  return Clause;
}

OSSAlignedClause *OSSAlignedClause::CreateEmpty(const ASTContext &C,
                                                unsigned NumVars) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(NumVars + 1));
  return new (Mem) OSSAlignedClause(NumVars);
}

void OSSCopyinClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OSSCopyinClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OSSCopyinClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OSSCopyinClause *OSSCopyinClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * VL.size()));
  OSSCopyinClause *Clause =
      new (Mem) OSSCopyinClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OSSCopyinClause *OSSCopyinClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * N));
  return new (Mem) OSSCopyinClause(N);
}

void OSSCopyprivateClause::setSourceExprs(ArrayRef<Expr *> SrcExprs) {
  assert(SrcExprs.size() == varlist_size() && "Number of source expressions is "
                                              "not the same as the "
                                              "preallocated buffer");
  std::copy(SrcExprs.begin(), SrcExprs.end(), varlist_end());
}

void OSSCopyprivateClause::setDestinationExprs(ArrayRef<Expr *> DstExprs) {
  assert(DstExprs.size() == varlist_size() && "Number of destination "
                                              "expressions is not the same as "
                                              "the preallocated buffer");
  std::copy(DstExprs.begin(), DstExprs.end(), getSourceExprs().end());
}

void OSSCopyprivateClause::setAssignmentOps(ArrayRef<Expr *> AssignmentOps) {
  assert(AssignmentOps.size() == varlist_size() &&
         "Number of assignment expressions is not the same as the preallocated "
         "buffer");
  std::copy(AssignmentOps.begin(), AssignmentOps.end(),
            getDestinationExprs().end());
}

OSSCopyprivateClause *OSSCopyprivateClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> VL, ArrayRef<Expr *> SrcExprs,
    ArrayRef<Expr *> DstExprs, ArrayRef<Expr *> AssignmentOps) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * VL.size()));
  OSSCopyprivateClause *Clause =
      new (Mem) OSSCopyprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setSourceExprs(SrcExprs);
  Clause->setDestinationExprs(DstExprs);
  Clause->setAssignmentOps(AssignmentOps);
  return Clause;
}

OSSCopyprivateClause *OSSCopyprivateClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(4 * N));
  return new (Mem) OSSCopyprivateClause(N);
}

void OSSReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OSSReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OSSReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OSSReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OSSReductionClause *OSSReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OSSReductionClause *Clause = new (Mem) OSSReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OSSReductionClause *OSSReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OSSReductionClause(N);
}

void OSSTaskReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OSSTaskReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OSSTaskReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OSSTaskReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of task reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OSSTaskReductionClause *OSSTaskReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps, Stmt *PreInit,
    Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OSSTaskReductionClause *Clause = new (Mem) OSSTaskReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OSSTaskReductionClause *OSSTaskReductionClause::CreateEmpty(const ASTContext &C,
                                                            unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OSSTaskReductionClause(N);
}

void OSSInReductionClause::setPrivates(ArrayRef<Expr *> Privates) {
  assert(Privates.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(Privates.begin(), Privates.end(), varlist_end());
}

void OSSInReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), getPrivates().end());
}

void OSSInReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OSSInReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of in reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

void OSSInReductionClause::setTaskgroupDescriptors(
    ArrayRef<Expr *> TaskgroupDescriptors) {
  assert(TaskgroupDescriptors.size() == varlist_size() &&
         "Number of in reduction descriptors is not the same as the "
         "preallocated buffer");
  std::copy(TaskgroupDescriptors.begin(), TaskgroupDescriptors.end(),
            getReductionOps().end());
}

OSSInReductionClause *OSSInReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> Privates, ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps,
    ArrayRef<Expr *> TaskgroupDescriptors, Stmt *PreInit, Expr *PostUpdate) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(6 * VL.size()));
  OSSInReductionClause *Clause = new (Mem) OSSInReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo);
  Clause->setVarRefs(VL);
  Clause->setPrivates(Privates);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->setTaskgroupDescriptors(TaskgroupDescriptors);
  Clause->setPreInitStmt(PreInit);
  Clause->setPostUpdateExpr(PostUpdate);
  return Clause;
}

OSSInReductionClause *OSSInReductionClause::CreateEmpty(const ASTContext &C,
                                                        unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(6 * N));
  return new (Mem) OSSInReductionClause(N);
}

OSSFlushClause *OSSFlushClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size() + 1));
  OSSFlushClause *Clause =
      new (Mem) OSSFlushClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSFlushClause *OSSFlushClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSFlushClause(N);
}

OSSDependClause *
OSSDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        const SmallVector<OmpSsDependClauseKind, 2>& DepKinds, SourceLocation DepLoc,
                        SourceLocation ColonLoc, ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSDependClause *Clause = new (Mem)
      OSSDependClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setDependencyKinds(DepKinds);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  return Clause;
}

OSSDependClause *OSSDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSDependClause(N);
}

unsigned OSSClauseMappableExprCommon::getComponentsTotalNumber(
    MappableExprComponentListsRef ComponentLists) {
  unsigned TotalNum = 0u;
  for (auto &C : ComponentLists)
    TotalNum += C.size();
  return TotalNum;
}

unsigned OSSClauseMappableExprCommon::getUniqueDeclarationsTotalNumber(
    ArrayRef<const ValueDecl *> Declarations) {
  unsigned TotalNum = 0u;
  llvm::SmallPtrSet<const ValueDecl *, 8> Cache;
  for (const ValueDecl *D : Declarations) {
    const ValueDecl *VD = D ? cast<ValueDecl>(D->getCanonicalDecl()) : nullptr;
    if (Cache.count(VD))
      continue;
    ++TotalNum;
    Cache.insert(VD);
  }
  return TotalNum;
}

OSSMapClause *
OSSMapClause::Create(const ASTContext &C, SourceLocation StartLoc,
                     SourceLocation LParenLoc, SourceLocation EndLoc,
                     ArrayRef<Expr *> Vars, ArrayRef<ValueDecl *> Declarations,
                     MappableExprComponentListsRef ComponentLists,
                     OmpSsMapClauseKind TypeModifier, OmpSsMapClauseKind Type,
                     bool TypeIsImplicit, SourceLocation TypeLoc) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  OSSMapClause *Clause = new (Mem) OSSMapClause(
      TypeModifier, Type, TypeIsImplicit, TypeLoc, StartLoc, LParenLoc, EndLoc,
      NumVars, NumUniqueDeclarations, NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  Clause->setMapTypeModifier(TypeModifier);
  Clause->setMapType(Type);
  Clause->setMapLoc(TypeLoc);
  return Clause;
}

OSSMapClause *OSSMapClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                        unsigned NumUniqueDeclarations,
                                        unsigned NumComponentLists,
                                        unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OSSMapClause(NumVars, NumUniqueDeclarations,
                                NumComponentLists, NumComponents);
}

OSSToClause *OSSToClause::Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> Vars,
                                 ArrayRef<ValueDecl *> Declarations,
                                 MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OSSToClause *Clause = new (Mem)
      OSSToClause(StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
                  NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OSSToClause *OSSToClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                      unsigned NumUniqueDeclarations,
                                      unsigned NumComponentLists,
                                      unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OSSToClause(NumVars, NumUniqueDeclarations,
                               NumComponentLists, NumComponents);
}

OSSFromClause *
OSSFromClause::Create(const ASTContext &C, SourceLocation StartLoc,
                      SourceLocation LParenLoc, SourceLocation EndLoc,
                      ArrayRef<Expr *> Vars, ArrayRef<ValueDecl *> Declarations,
                      MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OSSFromClause *Clause = new (Mem)
      OSSFromClause(StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
                    NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OSSFromClause *OSSFromClause::CreateEmpty(const ASTContext &C, unsigned NumVars,
                                          unsigned NumUniqueDeclarations,
                                          unsigned NumComponentLists,
                                          unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OSSFromClause(NumVars, NumUniqueDeclarations,
                                 NumComponentLists, NumComponents);
}

void OSSUseDevicePtrClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OSSUseDevicePtrClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OSSUseDevicePtrClause *OSSUseDevicePtrClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, ArrayRef<Expr *> Vars, ArrayRef<Expr *> PrivateVars,
    ArrayRef<Expr *> Inits, ArrayRef<ValueDecl *> Declarations,
    MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // 3 x NumVars x Expr* - we have an original list expression for each clause
  // list entry and an equal number of private copies and inits.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          3 * NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OSSUseDevicePtrClause *Clause = new (Mem) OSSUseDevicePtrClause(
      StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
      NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setPrivateCopies(PrivateVars);
  Clause->setInits(Inits);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OSSUseDevicePtrClause *OSSUseDevicePtrClause::CreateEmpty(
    const ASTContext &C, unsigned NumVars, unsigned NumUniqueDeclarations,
    unsigned NumComponentLists, unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          3 * NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OSSUseDevicePtrClause(NumVars, NumUniqueDeclarations,
                                         NumComponentLists, NumComponents);
}

OSSIsDevicePtrClause *
OSSIsDevicePtrClause::Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc,
                             ArrayRef<Expr *> Vars,
                             ArrayRef<ValueDecl *> Declarations,
                             MappableExprComponentListsRef ComponentLists) {
  unsigned NumVars = Vars.size();
  unsigned NumUniqueDeclarations =
      getUniqueDeclarationsTotalNumber(Declarations);
  unsigned NumComponentLists = ComponentLists.size();
  unsigned NumComponents = getComponentsTotalNumber(ComponentLists);

  // We need to allocate:
  // NumVars x Expr* - we have an original list expression for each clause list
  // entry.
  // NumUniqueDeclarations x ValueDecl* - unique base declarations associated
  // with each component list.
  // (NumUniqueDeclarations + NumComponentLists) x unsigned - we specify the
  // number of lists for each unique declaration and the size of each component
  // list.
  // NumComponents x MappableComponent - the total of all the components in all
  // the lists.
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));

  OSSIsDevicePtrClause *Clause = new (Mem) OSSIsDevicePtrClause(
      StartLoc, LParenLoc, EndLoc, NumVars, NumUniqueDeclarations,
      NumComponentLists, NumComponents);

  Clause->setVarRefs(Vars);
  Clause->setClauseInfo(Declarations, ComponentLists);
  return Clause;
}

OSSIsDevicePtrClause *OSSIsDevicePtrClause::CreateEmpty(
    const ASTContext &C, unsigned NumVars, unsigned NumUniqueDeclarations,
    unsigned NumComponentLists, unsigned NumComponents) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Expr *, ValueDecl *, unsigned,
                       OSSClauseMappableExprCommon::MappableComponent>(
          NumVars, NumUniqueDeclarations,
          NumUniqueDeclarations + NumComponentLists, NumComponents));
  return new (Mem) OSSIsDevicePtrClause(NumVars, NumUniqueDeclarations,
                                        NumComponentLists, NumComponents);
}
