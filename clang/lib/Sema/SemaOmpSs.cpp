//===--- SemaOmpSs.cpp - Semantic Analysis for OmpSs constructs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OmpSs directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
using namespace clang;

StmtResult Sema::ActOnOmpSsExecutableDirective(ArrayRef<OSSClause *> Clauses,
    OmpSsDirectiveKind Kind, Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {

  StmtResult Res = StmtError();
  switch (Kind) {
  case OSSD_taskwait:
    Res = ActOnOmpSsTaskwaitDirective(StartLoc, EndLoc);
    break;
  case OSSD_task:
    Res = ActOnOmpSsTaskDirective(Clauses, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_unknown:
    llvm_unreachable("Unknown OmpSs directive");
  }
  return Res;
}

StmtResult Sema::ActOnOmpSsTaskwaitDirective(SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  return OSSTaskwaitDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOmpSsTaskDirective(ArrayRef<OSSClause *> Clauses,
                                         Stmt *AStmt,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return OSSTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

OSSClause *
Sema::ActOnOmpSsDependClause(OmpSsDependClauseKind DepKind,
                              SourceLocation DepLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                              SourceLocation StartLoc,
                              SourceLocation LParenLoc, SourceLocation EndLoc) {

  for (Expr *RefExpr : VarList) {
    SourceLocation ELoc = RefExpr->getExprLoc();
    Expr *SimpleExpr = RefExpr->IgnoreParenCasts();

    auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
    if (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
        (ASE &&
         !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
         !ASE->getBase()->getType().getNonReferenceType()->isArrayType())) {
      Diag(ELoc, diag::err_oss_expected_addressable_lvalue_or_array_item)
          << RefExpr->getSourceRange();
      continue;
    }
  }
  return OSSDependClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                 DepKind, DepLoc, ColonLoc, VarList,
                                 0); // used in ordered clause. useless for now
}

OSSClause *
Sema::ActOnOmpSsVarListClause(
  OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
  SourceLocation StartLoc, SourceLocation LParenLoc,
  SourceLocation ColonLoc, SourceLocation EndLoc,
  OmpSsDependClauseKind DepKind, SourceLocation DepLinMapLoc) {

  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_depend:
    Res = ActOnOmpSsDependClause(DepKind, DepLinMapLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }

  return Res;
}

