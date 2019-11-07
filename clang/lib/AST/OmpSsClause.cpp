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
                        ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
                        SourceLocation ColonLoc, ArrayRef<Expr *> VL,
                        bool OSSSyntax) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSDependClause *Clause = new (Mem)
      OSSDependClause(StartLoc, LParenLoc, EndLoc, VL.size(), OSSSyntax);
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

