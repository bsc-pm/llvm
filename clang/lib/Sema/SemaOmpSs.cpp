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

static std::string
getListOfPossibleValues(OmpSsClauseKind K, unsigned First, unsigned Last,
                        ArrayRef<unsigned> Exclude = llvm::None) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  unsigned Bound = Last >= 2 ? Last - 2 : 0;
  unsigned Skipped = Exclude.size();
  auto S = Exclude.begin(), E = Exclude.end();
  for (unsigned I = First; I < Last; ++I) {
    if (std::find(S, E, I) != E) {
      --Skipped;
      continue;
    }
    Out << "'" << getOmpSsSimpleClauseTypeName(K, I) << "'";
    if (I == Bound - Skipped)
      Out << " or ";
    else if (I != Bound + 1 - Skipped)
      Out << ", ";
  }
  return Out.str();
}

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
Sema::ActOnOmpSsDependClause(const SmallVector<OmpSsDependClauseKind, 2>& DepKinds, SourceLocation DepLoc,
                             SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (DepKinds.size() == 2) {
    auto DepKindIt = std::find(DepKinds.begin(), DepKinds.end(), OSSC_DEPEND_weak);
    int numWeaks = 0;
    int numUnk = 0;
    if (DepKinds[0] == OSSC_DEPEND_weak)
      ++numWeaks;
    else if (DepKinds[0] == OSSC_DEPEND_unknown)
      ++numUnk;
    if (DepKinds[1] == OSSC_DEPEND_weak)
      ++numWeaks;
    else if (DepKinds[1] == OSSC_DEPEND_unknown)
      ++numUnk;

    if (numWeaks == 0) {
      if (numUnk == 0 || numUnk == 1) {
        Diag(DepLoc, diag::err_oss_depend_weak_required);
        return nullptr;
      } else if (numUnk == 2) {
        unsigned Except[] = {OSSC_DEPEND_source, OSSC_DEPEND_sink};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
      }
    } else if ((numWeaks == 1 && numUnk == 1)
               || (numWeaks == 2 && numUnk == 0)) {
        unsigned Except[] = {OSSC_DEPEND_source, OSSC_DEPEND_sink, OSSC_DEPEND_weak};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
    }
  } else {
    if (DepKinds[0] == OSSC_DEPEND_unknown
        || DepKinds[0] == OSSC_DEPEND_weak) {
      unsigned Except[] = {OSSC_DEPEND_source, OSSC_DEPEND_weak, OSSC_DEPEND_sink};
      Diag(DepLoc, diag::err_oss_unexpected_clause_value)
          << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                     /*Last=*/OSSC_DEPEND_unknown, Except)
          << getOmpSsClauseName(OSSC_depend);
      return nullptr;
    }
  }

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
                                 DepKinds, DepLoc, ColonLoc, VarList);
}

OSSClause *
Sema::ActOnOmpSsVarListClause(
  OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
  SourceLocation StartLoc, SourceLocation LParenLoc,
  SourceLocation ColonLoc, SourceLocation EndLoc,
  const SmallVector<OmpSsDependClauseKind, 2>& DepKinds, SourceLocation DepLinMapLoc) {

  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_depend:
    Res = ActOnOmpSsDependClause(DepKinds, DepLinMapLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }

  return Res;
}

