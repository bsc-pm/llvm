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

OSSTaskwaitDirective *
OSSTaskwaitDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskwaitDirective), alignof(OSSClause *));
  void *Mem = C.Allocate(Size + sizeof(OSSClause *) * Clauses.size());
  OSSTaskwaitDirective *Dir =
      new (Mem) OSSTaskwaitDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  return Dir;
}

OSSTaskwaitDirective *OSSTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskwaitDirective), alignof(OSSClause *));
  void *Mem = C.Allocate(Size + sizeof(OSSClause *) * NumClauses);
  return new (Mem) OSSTaskwaitDirective(NumClauses);
}

OSSTaskDirective *
OSSTaskDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * Clauses.size() + sizeof(Stmt *));
  OSSTaskDirective *Dir =
      new (Mem) OSSTaskDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AStmt);
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

OSSTaskForDirective *
OSSTaskForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskForDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * Clauses.size() + sizeof(Stmt *));
  OSSTaskForDirective *Dir =
      new (Mem) OSSTaskForDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AStmt);
  Dir->setIterationVariable(Exprs.IndVar);
  Dir->setLowerBound(Exprs.LB);
  Dir->setUpperBound(Exprs.UB);
  Dir->setStep(Exprs.Step);
  Dir->setIsLessOp(Exprs.TestIsLessOp);
  Dir->setIsStrictOp(Exprs.TestIsStrictOp);
  return Dir;
}

OSSTaskForDirective *OSSTaskForDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskForDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OSSTaskForDirective(NumClauses);
}

OSSTaskLoopDirective *
OSSTaskLoopDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskLoopDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * Clauses.size() + sizeof(Stmt *));
  OSSTaskLoopDirective *Dir =
      new (Mem) OSSTaskLoopDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AStmt);
  Dir->setIterationVariable(Exprs.IndVar);
  Dir->setLowerBound(Exprs.LB);
  Dir->setUpperBound(Exprs.UB);
  Dir->setStep(Exprs.Step);
  Dir->setIsLessOp(Exprs.TestIsLessOp);
  Dir->setIsStrictOp(Exprs.TestIsStrictOp);
  return Dir;
}

OSSTaskLoopDirective *OSSTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskLoopDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OSSTaskLoopDirective(NumClauses);
}

OSSTaskLoopForDirective *
OSSTaskLoopForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const HelperExprs &Exprs) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskLoopForDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * Clauses.size() + sizeof(Stmt *));
  OSSTaskLoopForDirective *Dir =
      new (Mem) OSSTaskLoopForDirective(StartLoc, EndLoc, Clauses.size());
  Dir->setClauses(Clauses);
  Dir->setAssociatedStmt(AStmt);
  Dir->setIterationVariable(Exprs.IndVar);
  Dir->setLowerBound(Exprs.LB);
  Dir->setUpperBound(Exprs.UB);
  Dir->setStep(Exprs.Step);
  Dir->setIsLessOp(Exprs.TestIsLessOp);
  Dir->setIsStrictOp(Exprs.TestIsStrictOp);
  return Dir;
}

OSSTaskLoopForDirective *OSSTaskLoopForDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  unsigned Size = llvm::alignTo(sizeof(OSSTaskLoopForDirective), alignof(OSSClause *));
  void *Mem =
      C.Allocate(Size + sizeof(OSSClause *) * NumClauses + sizeof(Stmt *));
  return new (Mem) OSSTaskLoopForDirective(NumClauses);
}
