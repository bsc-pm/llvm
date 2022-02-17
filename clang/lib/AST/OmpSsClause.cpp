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

OSSLabelClause *OSSLabelClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSLabelClause *Clause =
      new (Mem) OSSLabelClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSLabelClause *OSSLabelClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSLabelClause(N);
}

void OSSPrivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

OSSPrivateClause *OSSPrivateClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL,
                                         ArrayRef<Expr *> PrivateVL) {
  // Info of item 'i' is in VL[i], PrivateVL[i]
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * VL.size()));
  OSSPrivateClause *Clause =
      new (Mem) OSSPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OSSPrivateClause *OSSPrivateClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
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

OSSFirstprivateClause *OSSFirstprivateClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL,
                                         ArrayRef<Expr *> PrivateVL,
                                         ArrayRef<Expr *> InitVL) {
  // Info of item 'i' is in VL[i], PrivateVL[i] and InitVL[i]
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * VL.size()));
  OSSFirstprivateClause *Clause =
      new (Mem) OSSFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  return Clause;
}

OSSFirstprivateClause *OSSFirstprivateClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSFirstprivateClause(N);
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

OSSDependClause *
OSSDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        ArrayRef<OmpSsDependClauseKind> DepKinds,
                        ArrayRef<OmpSsDependClauseKind> DepKindsOrdered,
                        SourceLocation DepLoc, SourceLocation ColonLoc,
                        ArrayRef<Expr *> VL, bool OSSSyntax) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSDependClause *Clause = new (Mem)
      OSSDependClause(StartLoc, LParenLoc, EndLoc, VL.size(), OSSSyntax);
  Clause->setVarRefs(VL);
  Clause->setDependencyKinds(DepKinds);
  Clause->setDependencyKindsOrdered(DepKindsOrdered);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  return Clause;
}

OSSDependClause *OSSDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSDependClause(N);
}

void OSSReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), varlist_end());
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
    ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps,
    ArrayRef<BinaryOperatorKind> ReductionKinds, bool IsWeak) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OSSReductionClause *Clause = new (Mem) OSSReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo, IsWeak);
  Clause->setVarRefs(VL);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->ReductionKinds.append(ReductionKinds.begin(), ReductionKinds.end());
  return Clause;
}

OSSReductionClause *OSSReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OSSReductionClause(N);
}

OSSNdrangeClause *OSSNdrangeClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSNdrangeClause *Clause =
      new (Mem) OSSNdrangeClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSNdrangeClause *OSSNdrangeClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSNdrangeClause(N);
}
