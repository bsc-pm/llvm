//===--- StmtOmpSs.cpp - Classes for OmpSs directives -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in StmtOmpSs.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtOmpSs.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

void OSSExecutableDirective::setClauses(ArrayRef<OSSClause *> Clauses) {
  assert(Clauses.size() == getNumClauses() &&
         "Number of clauses is not the same as the preallocated buffer");
  std::copy(Clauses.begin(), Clauses.end(), getClauses().begin());
}

OSSTaskwaitDirective *OSSTaskwaitDirective::Create(const ASTContext &C,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OSSTaskwaitDirective));
  OSSTaskwaitDirective *Dir = new (Mem) OSSTaskwaitDirective(StartLoc, EndLoc);
  return Dir;
}

OSSTaskwaitDirective *OSSTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(sizeof(OSSTaskwaitDirective));
  return new (Mem) OSSTaskwaitDirective();
}

OSSTaskDirective *
OSSTaskDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * Clauses.size() + sizeof(Stmt *));
  OSSTaskDirective *Dir =
      new (Mem) OSSTaskDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OSSTaskDirective *OSSTaskDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OSSTaskDirective(NumClauses);
}
