//===--- ParseOmpSs.cpp - OmpSs directives parsing ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements parsing of all OmpSs directives and clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/PointerIntPair.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OmpSs declarative directives.
//===----------------------------------------------------------------------===//

static OmpSsDirectiveKind parseOmpSsDirectiveKind(Parser &P) {
  Token Tok = P.getCurToken();
  unsigned DKind =
      Tok.isAnnotation()
          ? static_cast<unsigned>(OSSD_unknown)
          : getOmpSsDirectiveKind(P.getPreprocessor().getSpelling(Tok));

  return DKind < OSSD_unknown ? static_cast<OmpSsDirectiveKind>(DKind)
                              : OSSD_unknown;
}

/// Parsing of declarative or executable OmpSs directives.
///       executable-directive:
///         annot_pragma_ompss 'taskwait'
///         annot_pragma_ompss 'task'
///         annot_pragma_ompss_end
///
StmtResult Parser::ParseOmpSsDeclarativeOrExecutableDirective(
    AllowedConstructsKind Allowed) {
  assert(Tok.is(tok::annot_pragma_ompss) && "Not an OmpSs directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  unsigned ScopeFlags = Scope::FnScope | Scope::DeclScope |
                        Scope::CompoundStmtScope | Scope::OmpSsDirectiveScope;
  SourceLocation Loc = ConsumeAnnotationToken(), EndLoc;

  SmallVector<OSSClause *, 5> Clauses;
  SmallVector<llvm::PointerIntPair<OSSClause *, 1, bool>, OSSC_unknown + 1>
    FirstClauses(OSSC_unknown + 1);

  OmpSsDirectiveKind DKind = parseOmpSsDirectiveKind(*this);
  StmtResult Directive = StmtError();
  bool HasAssociatedStatement = true;
  switch (DKind) {
  case OSSD_taskwait:
    HasAssociatedStatement = false;
    LLVM_FALLTHROUGH;
  case OSSD_task: {
    ConsumeToken();
    ParseScope OSSDirectiveScope(this, ScopeFlags);

    while (Tok.isNot(tok::annot_pragma_ompss_end)) {
      OmpSsClauseKind CKind = getOmpSsClauseKind(PP.getSpelling(Tok));

      // Actions.StartOpenMPClause(CKind);

      // Track which clauses have appeared so we can throw an error in case
      // a clause cannot appear again
      OSSClause *Clause =
          ParseOmpSsClause(DKind, CKind, !FirstClauses[CKind].getInt());
      FirstClauses[CKind].setInt(true);
      if (Clause) {
        FirstClauses[CKind].setPointer(Clause);
        Clauses.push_back(Clause);
      }

      // Skip ',' if any.
      if (Tok.is(tok::comma))
        ConsumeToken();

      // Actions.EndOpenMPClause();
    }

    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_openmp_end.
    ConsumeAnnotationToken();

    StmtResult AssociatedStmt;
    if (HasAssociatedStatement) {
      AssociatedStmt = (Sema::CompoundScopeRAII(Actions), ParseStatement());
    }

    Directive = Actions.ActOnOmpSsExecutableDirective(Clauses,
                                                      DKind,
                                                      AssociatedStmt.get(),
                                                      Loc,
                                                      EndLoc);

    // Exit scope.
    OSSDirectiveScope.Exit();
    break;

    }

  case OSSD_unknown:
    Diag(Tok, diag::err_oss_unknown_directive);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  }
  return Directive;
}

/// Parsing of OpenMP clauses.
///
///    clause:
///       task-depend-clause
///
OSSClause *Parser::ParseOmpSsClause(OmpSsDirectiveKind DKind,
                                     OmpSsClauseKind CKind, bool FirstClause) {
  OSSClause *Clause = nullptr;
  bool ErrorFound = false;
  bool WrongDirective = false;
  // Check if clause is allowed for the given directive.
  if (CKind != OSSC_unknown && !isAllowedClauseForDirective(DKind, CKind)) {
    Diag(Tok, diag::err_oss_unexpected_clause) << getOmpSsClauseName(CKind)
                                               << getOmpSsDirectiveName(DKind);
    ErrorFound = true;
    WrongDirective = true;
  }

  switch (CKind) {
  case OSSC_grainsize:
    // These clauses cannot appear more than once
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    break;
  case OSSC_depend:
    Clause = ParseOmpSsVarListClause(DKind, CKind, WrongDirective);
    break;
  case OSSC_unknown:
    Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
        << getOmpSsDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
    break;
  }
  return ErrorFound ? nullptr : Clause;
}

/// Parses clauses with list.
bool Parser::ParseOmpSsVarList(OmpSsDirectiveKind DKind,
                                OmpSsClauseKind Kind,
                                SmallVectorImpl<Expr *> &Vars,
                                OmpSsVarListDataTy &Data) {
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsClauseName(Kind)))
    return true;

  if (Kind == OSSC_depend) {
    // Handle dependency type for depend clause.
    ColonProtectionRAIIObject ColonRAII(*this);
    Data.DepKind =
        static_cast<OmpSsDependClauseKind>(getOmpSsSimpleClauseType(
            Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
    Data.DepLinMapLoc = Tok.getLocation();

    if (Data.DepKind == OSSC_DEPEND_unknown) {
      SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_ompss_end,
                StopBeforeMatch);
    } else {
      ConsumeToken();
    }
    if (Tok.is(tok::colon)) {
      Data.ColonLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::warn_pragma_expected_colon) << "dependency type";
    }
  }

  bool IsComma =
      (Kind == OSSC_depend && Data.DepKind != OSSC_DEPEND_unknown);
  while (IsComma || (Tok.isNot(tok::r_paren) && Tok.isNot(tok::colon) &&
                     Tok.isNot(tok::annot_pragma_ompss_end))) {
    // Parse variable
    ExprResult VarExpr =
        Actions.CorrectDelayedTyposInExpr(ParseAssignmentExpression());
    if (VarExpr.isUsable()) {
      Vars.push_back(VarExpr.get());
    } else {
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_ompss_end,
                StopBeforeMatch);
    }
    // Skip ',' if any
    IsComma = Tok.is(tok::comma);
    if (IsComma)
      ConsumeToken();
    else if (Tok.isNot(tok::r_paren) &&
             Tok.isNot(tok::annot_pragma_ompss_end))
      Diag(Tok, diag::err_omp_expected_punc)
          << getOmpSsClauseName(Kind);
  }

  // Parse ')'.
  Data.RLoc = Tok.getLocation();
  if (!T.consumeClose())
    Data.RLoc = T.getCloseLocation();
  return (Kind == OSSC_depend
          && Data.DepKind != OSSC_DEPEND_unknown
          && Vars.empty());
}

/// Parsing of OmpSs
///
///    depend-clause:
///       'depend' '(' in | out | inout : ')'
OSSClause *Parser::ParseOmpSsVarListClause(OmpSsDirectiveKind DKind,
                                            OmpSsClauseKind Kind,
                                            bool ParseOnly) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  SmallVector<Expr *, 4> Vars;
  OmpSsVarListDataTy Data;

  if (ParseOmpSsVarList(DKind, Kind, Vars, Data))
    return nullptr;

  if (ParseOnly)
    return nullptr;
  return Actions.ActOnOmpSsVarListClause(
      Kind, Vars, Loc, LOpen, Data.ColonLoc, Data.RLoc,
      Data.DepKind, Data.DepLinMapLoc);
}

