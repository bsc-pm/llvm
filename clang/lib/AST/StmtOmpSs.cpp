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

using namespace clang;

size_t OSSChildren::size(unsigned NumClauses, bool HasAssociatedStmt,
                         unsigned NumChildren) {
  return llvm::alignTo(
      totalSizeToAlloc<OSSClause *, Stmt *>(
          NumClauses, NumChildren + (HasAssociatedStmt ? 1 : 0)),
      alignof(OSSChildren));
}

void OSSChildren::setClauses(ArrayRef<OSSClause *> Clauses) {
  assert(Clauses.size() == NumClauses &&
         "Number of clauses is not the same as the preallocated buffer");
  llvm::copy(Clauses, getTrailingObjects<OSSClause *>());
}

MutableArrayRef<Stmt *> OSSChildren::getChildren() {
  return llvm::makeMutableArrayRef(getTrailingObjects<Stmt *>(), NumChildren);
}

OSSChildren *OSSChildren::Create(void *Mem, ArrayRef<OSSClause *> Clauses) {
  auto *Data = CreateEmpty(Mem, Clauses.size());
  Data->setClauses(Clauses);
  return Data;
}

OSSChildren *OSSChildren::Create(void *Mem, ArrayRef<OSSClause *> Clauses,
                                 Stmt *S, unsigned NumChildren) {
  auto *Data = CreateEmpty(Mem, Clauses.size(), S, NumChildren);
  Data->setClauses(Clauses);
  if (S)
    Data->setAssociatedStmt(S);
  return Data;
}

OSSChildren *OSSChildren::CreateEmpty(void *Mem, unsigned NumClauses,
                                      bool HasAssociatedStmt,
                                      unsigned NumChildren) {
  return new (Mem) OSSChildren(NumClauses, NumChildren, HasAssociatedStmt);
}

OSSTaskwaitDirective *
OSSTaskwaitDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses) {
  return createDirective<OSSTaskwaitDirective>(
      C, Clauses, /*AssociatedStmt=*/nullptr, /*NumChildren=*/0, StartLoc,
      EndLoc);
}

OSSTaskwaitDirective *OSSTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  return createEmptyDirective<OSSTaskwaitDirective>(C, NumClauses);
}

OSSReleaseDirective *
OSSReleaseDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                             SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses) {
  return createDirective<OSSReleaseDirective>(
      C, Clauses, /*AssociatedStmt=*/nullptr, /*NumChildren=*/0, StartLoc,
      EndLoc);
}

OSSReleaseDirective *OSSReleaseDirective::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses,
                                                      EmptyShell) {
  return createEmptyDirective<OSSReleaseDirective>(C, NumClauses);
}

OSSTaskDirective *
OSSTaskDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt) {
  return createDirective<OSSTaskDirective>(
      C, Clauses, AStmt, /*NumChildren=*/0, StartLoc, EndLoc);
}

OSSTaskDirective *OSSTaskDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                EmptyShell) {
  return createEmptyDirective<OSSTaskDirective>(
      C, NumClauses, /*HasAssociatedStmt=*/true);
}

OSSCriticalDirective *OSSCriticalDirective::Create(
    const ASTContext &C, const DeclarationNameInfo &Name,
    SourceLocation StartLoc, SourceLocation EndLoc,
    ArrayRef<OSSClause *> Clauses, Stmt *AssociatedStmt) {
  return createDirective<OSSCriticalDirective>(C, Clauses, AssociatedStmt,
                                               /*NumChildren=*/0, Name,
                                               StartLoc, EndLoc);
}

OSSCriticalDirective *OSSCriticalDirective::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses,
                                                        EmptyShell) {
  return createEmptyDirective<OSSCriticalDirective>(C, NumClauses,
                                                    /*HasAssociatedStmt=*/true);
}

OSSTaskForDirective *
OSSTaskForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const SmallVectorImpl<HelperExprs> &Exprs) {
  auto *Dir = createDirective<OSSTaskForDirective>(
      C, Clauses, AStmt, /*NumChildren=*/0, StartLoc, EndLoc, Exprs.size());
  Expr **IndVar;
  Expr **LB;
  Expr **UB;
  Expr **Step;
  llvm::Optional<bool>* TestIsLessOp;
  bool *TestIsStrictOp;
  IndVar = new (C) Expr*[Exprs.size()];
  LB = new (C) Expr*[Exprs.size()];
  UB = new (C) Expr*[Exprs.size()];
  Step = new (C) Expr*[Exprs.size()];
  TestIsLessOp = new (C) llvm::Optional<bool>[Exprs.size()];
  TestIsStrictOp = new (C) bool[Exprs.size()];
  for (size_t i = 0; i < Exprs.size(); ++i) {
    IndVar[i] = Exprs[i].IndVar;
    LB[i] = Exprs[i].LB;
    UB[i] = Exprs[i].UB;
    Step[i] = Exprs[i].Step;
    TestIsLessOp[i] = Exprs[i].TestIsLessOp;
    TestIsStrictOp[i] = Exprs[i].TestIsStrictOp;
  }
  Dir->setIterationVariable(IndVar);
  Dir->setLowerBound(LB);
  Dir->setUpperBound(UB);
  Dir->setStep(Step);
  Dir->setIsLessOp(TestIsLessOp);
  Dir->setIsStrictOp(TestIsStrictOp);
  return Dir;
}

OSSTaskForDirective *OSSTaskForDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned NumCollapses,
                                                EmptyShell) {
  return createEmptyDirective<OSSTaskForDirective>(
      C, NumClauses, /*HasAssociatedStmt=*/true, /*NumChildren=*/0, NumCollapses);
}

OSSTaskIterDirective *
OSSTaskIterDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const SmallVectorImpl<HelperExprs> &Exprs) {
  auto *Dir = createDirective<OSSTaskIterDirective>(
      C, Clauses, AStmt, /*NumChildren=*/0, StartLoc, EndLoc, Exprs.size());
  if (!Exprs.empty()) {
    // taskiter (for)
    Expr **IndVar;
    Expr **LB;
    Expr **UB;
    Expr **Step;
    llvm::Optional<bool>* TestIsLessOp;
    bool *TestIsStrictOp;
    IndVar = new (C) Expr*[Exprs.size()];
    LB = new (C) Expr*[Exprs.size()];
    UB = new (C) Expr*[Exprs.size()];
    Step = new (C) Expr*[Exprs.size()];
    TestIsLessOp = new (C) llvm::Optional<bool>[Exprs.size()];
    TestIsStrictOp = new (C) bool[Exprs.size()];
    for (size_t i = 0; i < Exprs.size(); ++i) {
      IndVar[i] = Exprs[i].IndVar;
      LB[i] = Exprs[i].LB;
      UB[i] = Exprs[i].UB;
      Step[i] = Exprs[i].Step;
      TestIsLessOp[i] = Exprs[i].TestIsLessOp;
      TestIsStrictOp[i] = Exprs[i].TestIsStrictOp;
    }
    Dir->setIterationVariable(IndVar);
    Dir->setLowerBound(LB);
    Dir->setUpperBound(UB);
    Dir->setStep(Step);
    Dir->setIsLessOp(TestIsLessOp);
    Dir->setIsStrictOp(TestIsStrictOp);
  }
  return Dir;
}

OSSTaskIterDirective *OSSTaskIterDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned NumCollapses,
                                                EmptyShell) {
  return createEmptyDirective<OSSTaskIterDirective>(
      C, NumClauses, /*HasAssociatedStmt=*/true, /*NumChildren=*/0, NumCollapses);
}

OSSTaskLoopDirective *
OSSTaskLoopDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const SmallVectorImpl<HelperExprs> &Exprs) {
  auto *Dir = createDirective<OSSTaskLoopDirective>(
      C, Clauses, AStmt, /*NumChildren=*/0, StartLoc, EndLoc, Exprs.size());
  Expr **IndVar;
  Expr **LB;
  Expr **UB;
  Expr **Step;
  llvm::Optional<bool>* TestIsLessOp;
  bool *TestIsStrictOp;
  IndVar = new (C) Expr*[Exprs.size()];
  LB = new (C) Expr*[Exprs.size()];
  UB = new (C) Expr*[Exprs.size()];
  Step = new (C) Expr*[Exprs.size()];
  TestIsLessOp = new (C) llvm::Optional<bool>[Exprs.size()];
  TestIsStrictOp = new (C) bool[Exprs.size()];
  for (size_t i = 0; i < Exprs.size(); ++i) {
    IndVar[i] = Exprs[i].IndVar;
    LB[i] = Exprs[i].LB;
    UB[i] = Exprs[i].UB;
    Step[i] = Exprs[i].Step;
    TestIsLessOp[i] = Exprs[i].TestIsLessOp;
    TestIsStrictOp[i] = Exprs[i].TestIsStrictOp;
  }
  Dir->setIterationVariable(IndVar);
  Dir->setLowerBound(LB);
  Dir->setUpperBound(UB);
  Dir->setStep(Step);
  Dir->setIsLessOp(TestIsLessOp);
  Dir->setIsStrictOp(TestIsStrictOp);
  return Dir;
}

OSSTaskLoopDirective *OSSTaskLoopDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned NumCollapses,
                                                EmptyShell) {
  return createEmptyDirective<OSSTaskLoopDirective>(
      C, NumClauses, /*HasAssociatedStmt=*/true, /*NumChildren=*/0, NumCollapses);
}

OSSTaskLoopForDirective *
OSSTaskLoopForDirective::Create(const ASTContext &C, SourceLocation StartLoc,
                         SourceLocation EndLoc, ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                         const SmallVectorImpl<HelperExprs> &Exprs) {
  auto *Dir = createDirective<OSSTaskLoopForDirective>(
      C, Clauses, AStmt, /*NumChildren=*/0, StartLoc, EndLoc, Exprs.size());
  Expr **IndVar;
  Expr **LB;
  Expr **UB;
  Expr **Step;
  llvm::Optional<bool>* TestIsLessOp;
  bool *TestIsStrictOp;
  IndVar = new (C) Expr*[Exprs.size()];
  LB = new (C) Expr*[Exprs.size()];
  UB = new (C) Expr*[Exprs.size()];
  Step = new (C) Expr*[Exprs.size()];
  TestIsLessOp = new (C) llvm::Optional<bool>[Exprs.size()];
  TestIsStrictOp = new (C) bool[Exprs.size()];
  for (size_t i = 0; i < Exprs.size(); ++i) {
    IndVar[i] = Exprs[i].IndVar;
    LB[i] = Exprs[i].LB;
    UB[i] = Exprs[i].UB;
    Step[i] = Exprs[i].Step;
    TestIsLessOp[i] = Exprs[i].TestIsLessOp;
    TestIsStrictOp[i] = Exprs[i].TestIsStrictOp;
  }
  Dir->setIterationVariable(IndVar);
  Dir->setLowerBound(LB);
  Dir->setUpperBound(UB);
  Dir->setStep(Step);
  Dir->setIsLessOp(TestIsLessOp);
  Dir->setIsStrictOp(TestIsStrictOp);
  return Dir;
}

OSSTaskLoopForDirective *OSSTaskLoopForDirective::CreateEmpty(const ASTContext &C,
                                                unsigned NumClauses,
                                                unsigned NumCollapses,
                                                EmptyShell) {
  return createEmptyDirective<OSSTaskLoopForDirective>(
      C, NumClauses, /*HasAssociatedStmt=*/true, /*NumChildren=*/0, NumCollapses);
}
