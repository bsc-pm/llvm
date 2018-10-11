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
// #include "clang/AST/StmtOmpSs.h"
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
///         annot_pragma_openmp 'taskwait'
///         annot_pragma_openmp_end
///
StmtResult Parser::ParseOmpSsDeclarativeOrExecutableDirective(
    AllowedConstructsKind Allowed) {
  assert(Tok.is(tok::annot_pragma_ompss) && "Not an OmpSs directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  unsigned ScopeFlags = Scope::FnScope | Scope::DeclScope |
                        Scope::CompoundStmtScope | Scope::OmpSsDirectiveScope;
  SourceLocation Loc = ConsumeAnnotationToken(), EndLoc;
  OmpSsDirectiveKind DKind = parseOmpSsDirectiveKind(*this);
  StmtResult Directive = StmtError();
  switch (DKind) {
  case OSSD_task:
    Diag(Tok, diag::err_oss_task_no_implemented);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  case OSSD_taskwait: {
    ConsumeToken();
    ParseScope OSSDirectiveScope(this, ScopeFlags);

    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_openmp_end.
    ConsumeAnnotationToken();

    Directive = Actions.ActOnOmpSsExecutableDirective(
        DKind, Loc, EndLoc);

    // Exit scope.
    OSSDirectiveScope.Exit();
    break;

    }

    // Diag(Tok, diag::err_oss_taskwait_no_implemented);
    // SkipUntil(tok::annot_pragma_ompss_end);
    // break;
  case OSSD_unknown:
    Diag(Tok, diag::err_oss_unknown_directive);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  }
  return Directive;
}

      /*
    ConsumeToken();
    ParseScope OMPDirectiveScope(this, ScopeFlags);

    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_openmp_end.
    ConsumeAnnotationToken();
      */
