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

namespace {
// NOTE: This was taken from OpenMP as is

/// RAII that recreates function context for correct parsing of clauses of
/// 'declare simd' construct.
/// OpenMP, 2.8.2 declare simd Construct
/// The expressions appearing in the clauses of this directive are evaluated in
/// the scope of the arguments of the function declaration or definition.
class FNContextRAII final {
  Parser &P;
  Sema::CXXThisScopeRAII *ThisScope;
  Parser::ParseScope *TempScope;
  Parser::ParseScope *FnScope;
  bool HasTemplateScope = false;
  bool HasFunScope = false;
  FNContextRAII() = delete;
  FNContextRAII(const FNContextRAII &) = delete;
  FNContextRAII &operator=(const FNContextRAII &) = delete;

public:
  FNContextRAII(Parser &P, Parser::DeclGroupPtrTy Ptr) : P(P) {
    Decl *D = *Ptr.get().begin();
    NamedDecl *ND = dyn_cast<NamedDecl>(D);
    RecordDecl *RD = dyn_cast_or_null<RecordDecl>(D->getDeclContext());
    Sema &Actions = P.getActions();

    // Allow 'this' within late-parsed attributes.
    ThisScope = new Sema::CXXThisScopeRAII(Actions, RD, Qualifiers(),
                                           ND && ND->isCXXInstanceMember());

    // If the Decl is templatized, add template parameters to scope.
    HasTemplateScope = D->isTemplateDecl();
    TempScope =
        new Parser::ParseScope(&P, Scope::TemplateParamScope, HasTemplateScope);
    if (HasTemplateScope)
      Actions.ActOnReenterTemplateScope(Actions.getCurScope(), D);

    // If the Decl is on a function, add function parameters to the scope.
    HasFunScope = D->isFunctionOrFunctionTemplate();
    FnScope = new Parser::ParseScope(
        &P, Scope::FnScope | Scope::DeclScope | Scope::CompoundStmtScope,
        HasFunScope);
    if (HasFunScope)
      Actions.ActOnReenterFunctionContext(Actions.getCurScope(), D);
  }
  ~FNContextRAII() {
    if (HasFunScope) {
      P.getActions().ActOnExitFunctionContext();
      FnScope->Exit(); // Pop scope, and remove Decls from IdResolver
    }
    if (HasTemplateScope)
      TempScope->Exit();
    delete FnScope;
    delete TempScope;
    delete ThisScope;
  }
};
} // namespace

static ExprResult *getSingleClause(
    OmpSsClauseKind CKind,
    ExprResult &IfRes, ExprResult &FinalRes,
    ExprResult &CostRes, ExprResult &PriorityRes) {

    if (CKind == OSSC_if) return &IfRes;
    if (CKind == OSSC_final) return &FinalRes;
    if (CKind == OSSC_cost) return &CostRes;
    if (CKind == OSSC_priority) return &PriorityRes;
    return nullptr;
}

static SmallVectorImpl<Expr *> *getClauseList(
    OmpSsClauseKind CKind,
    SmallVectorImpl<Expr *> &Ins, SmallVectorImpl<Expr *> &Outs,
    SmallVectorImpl<Expr *> &Inouts, SmallVectorImpl<Expr *> &Concurrents,
    SmallVectorImpl<Expr *> &Commutatives, SmallVectorImpl<Expr *> &WeakIns,
    SmallVectorImpl<Expr *> &WeakOuts, SmallVectorImpl<Expr *> &WeakInouts) {

    if (CKind == OSSC_in) return &Ins;
    if (CKind == OSSC_out) return &Outs;
    if (CKind == OSSC_inout) return &Inouts;
    if (CKind == OSSC_concurrent) return &Concurrents;
    if (CKind == OSSC_commutative) return &Commutatives;
    if (CKind == OSSC_weakin) return &WeakIns;
    if (CKind == OSSC_weakout) return &WeakOuts;
    if (CKind == OSSC_weakinout) return &WeakInouts;
    return nullptr;
}

// This assumes DepKindsOrdered are well-formed. That is, if DepKinds.size() == 2 it assumes
// is weaksomething
static OmpSsClauseKind getOmpSsClauseFromDependKinds(ArrayRef<OmpSsDependClauseKind> DepKindsOrdered) {
  if (DepKindsOrdered.size() == 2) {
    if (DepKindsOrdered[0] == OSSC_DEPEND_in)
      return OSSC_weakin;
    if (DepKindsOrdered[0] == OSSC_DEPEND_out)
      return OSSC_weakout;
    if (DepKindsOrdered[0] == OSSC_DEPEND_inout)
      return OSSC_weakinout;
  }
  else {
    if (DepKindsOrdered[0] == OSSC_DEPEND_in)
      return OSSC_in;
    if (DepKindsOrdered[0] == OSSC_DEPEND_out)
      return OSSC_out;
    if (DepKindsOrdered[0] == OSSC_DEPEND_inout)
      return OSSC_inout;
    if (DepKindsOrdered[0] == OSSC_DEPEND_mutexinoutset)
      return OSSC_concurrent;
    if (DepKindsOrdered[0] == OSSC_DEPEND_inoutset)
      return OSSC_commutative;
  }
  return OSSC_unknown;
}

/// Parses clauses for 'task' declaration directive.
///
///    clause:
///       depend-clause | if-clause | final-clause
///       | cost-clause | priority-clause
///       | default-clause | shared-clause | private-clause
///       | firstprivate-clause | in-clause | out-clause
///       | inout-clause | weakin-clause | weakout-clause
///       | weakinout-clause
static bool parseDeclareTaskClauses(
    Parser &P,
    ExprResult &IfRes, ExprResult &FinalRes,
    ExprResult &CostRes, ExprResult &PriorityRes,
    SmallVectorImpl<Expr *> &Ins, SmallVectorImpl<Expr *> &Outs,
    SmallVectorImpl<Expr *> &Inouts, SmallVectorImpl<Expr *> &Concurrents,
    SmallVectorImpl<Expr *> &Commutatives, SmallVectorImpl<Expr *> &WeakIns,
    SmallVectorImpl<Expr *> &WeakOuts, SmallVectorImpl<Expr *> &WeakInouts,
    SmallVectorImpl<Expr *> &DepIns, SmallVectorImpl<Expr *> &DepOuts,
    SmallVectorImpl<Expr *> &DepInouts, SmallVectorImpl<Expr *> &DepConcurrents,
    SmallVectorImpl<Expr *> &DepCommutatives, SmallVectorImpl<Expr *> &DepWeakIns,
    SmallVectorImpl<Expr *> &DepWeakOuts, SmallVectorImpl<Expr *> &DepWeakInouts) {
  const Token &Tok = P.getCurToken();
  bool IsError = false;

  SmallVector<bool, 4> FirstClauses(OSSC_unknown + 1);

  while (Tok.isNot(tok::annot_pragma_ompss_end)) {
    if (Tok.isNot(tok::identifier)
        && Tok.isNot(tok::kw_default)
        && Tok.isNot(tok::kw_if))
      break;

    SmallVectorImpl<Expr *> *Vars = nullptr;
    ExprResult *SingleClause = nullptr;
    Parser::OmpSsVarListDataTy Data;

    IdentifierInfo *II = Tok.getIdentifierInfo();
    StringRef ClauseName = II->getName();
    OmpSsClauseKind CKind = getOmpSsClauseKind(ClauseName);

    Vars = getClauseList(
      CKind, Ins, Outs, Inouts,
      Concurrents, Commutatives,
      WeakIns, WeakOuts, WeakInouts);

    switch (CKind) {
    case OSSC_if:
    case OSSC_final:
    case OSSC_cost:
    case OSSC_priority: {
      P.ConsumeToken();
      if (FirstClauses[CKind]) {
        P.Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      SourceLocation RLoc;
      SingleClause = getSingleClause(CKind, IfRes, FinalRes,
                                     CostRes, PriorityRes);
      *SingleClause = P.ParseOmpSsParensExpr(getOmpSsClauseName(CKind), RLoc);

      if (SingleClause->isInvalid())
        IsError = true;

      FirstClauses[CKind] = true;
      }
      break;
    case OSSC_default:
      P.Diag(Tok, diag::err_oss_unexpected_clause) << getOmpSsClauseName(CKind)
                                                   << getOmpSsDirectiveName(OSSD_task);
      P.SkipUntil(tok::annot_pragma_ompss_end, P.StopBeforeMatch);
      break;
    case OSSC_shared:
    case OSSC_private:
    case OSSC_firstprivate: {
      P.Diag(Tok, diag::err_oss_unexpected_clause) << getOmpSsClauseName(CKind)
                                                   << getOmpSsDirectiveName(OSSD_task);
      P.SkipUntil(tok::annot_pragma_ompss_end, P.StopBeforeMatch);
      break;
    }
    case OSSC_depend: {
      P.ConsumeToken();

      Sema::AllowShapingsRAII AllowShapings(P.getActions(), []() { return true; });

      SmallVector<Expr *, 4> TmpList;
      SmallVector<OmpSsDependClauseKind, 2> DepKindsOrdered;
      if (P.ParseOmpSsVarList(OSSD_task, CKind, TmpList, Data))
        IsError = true;
      if (!P.getActions().ActOnOmpSsDependKinds(Data.DepKinds, DepKindsOrdered, Data.DepLoc))
        IsError = true;

      if (!IsError) {
        SmallVectorImpl<Expr *> *DepList = getClauseList(
          getOmpSsClauseFromDependKinds(DepKindsOrdered),
          DepIns, DepOuts, DepInouts,
          DepConcurrents, DepCommutatives,
          DepWeakIns, DepWeakOuts, WeakInouts);
        DepList->append(TmpList.begin(), TmpList.end());
      }
      break;
    }
    case OSSC_in:
    case OSSC_out:
    case OSSC_inout:
    case OSSC_concurrent:
    case OSSC_commutative:
    case OSSC_weakin:
    case OSSC_weakout:
    case OSSC_weakinout: {
      P.ConsumeToken();

      Sema::AllowShapingsRAII AllowShapings(P.getActions(), []() { return true; });
      if (P.ParseOmpSsVarList(OSSD_task, CKind, *Vars, Data))
        IsError = true;
      break;
    }
    case OSSC_unknown:
      P.Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
          << getOmpSsDirectiveName(OSSD_task);
      P.SkipUntil(tok::annot_pragma_ompss_end, P.StopBeforeMatch);
      break;
    }

    // Skip ',' if any.
    if (Tok.is(tok::comma))
      P.ConsumeToken();
  }
  return IsError;
}

/// Parse clauses for '#pragma oss task' declaration directive.
Parser::DeclGroupPtrTy
Parser::ParseOSSDeclareTaskClauses(Parser::DeclGroupPtrTy Ptr,
                                   CachedTokens &Toks, SourceLocation Loc) {
  PP.EnterToken(Tok, /*IsReinject*/ true);
  PP.EnterTokenStream(Toks, /*DisableMacroExpansion=*/true,
                      /*IsReinject*/ true);
  // Consume the previously pushed token.
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);

  FNContextRAII FnContext(*this, Ptr);

  ExprResult IfRes;
  ExprResult FinalRes;
  ExprResult CostRes;
  ExprResult PriorityRes;
  SmallVector<Expr *, 4> Ins;
  SmallVector<Expr *, 4> Outs;
  SmallVector<Expr *, 4> Inouts;
  SmallVector<Expr *, 4> Concurrents;
  SmallVector<Expr *, 4> Commutatives;
  SmallVector<Expr *, 4> WeakIns;
  SmallVector<Expr *, 4> WeakOuts;
  SmallVector<Expr *, 4> WeakInouts;
  SmallVector<Expr *, 4> DepIns;
  SmallVector<Expr *, 4> DepOuts;
  SmallVector<Expr *, 4> DepInouts;
  SmallVector<Expr *, 4> DepConcurrents;
  SmallVector<Expr *, 4> DepCommutatives;
  SmallVector<Expr *, 4> DepWeakIns;
  SmallVector<Expr *, 4> DepWeakOuts;
  SmallVector<Expr *, 4> DepWeakInouts;

  bool IsError =
      parseDeclareTaskClauses(*this,
                              IfRes, FinalRes,
                              CostRes, PriorityRes,
                              Ins, Outs, Inouts,
                              Concurrents, Commutatives,
                              WeakIns, WeakOuts, WeakInouts,
                              DepIns, DepOuts, DepInouts,
                              DepConcurrents, DepCommutatives,
                              DepWeakIns, DepWeakOuts, DepWeakInouts);
  // Need to check for extra tokens.
  if (Tok.isNot(tok::annot_pragma_ompss_end)) {
    Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
        << getOmpSsDirectiveName(OSSD_task);
    while (Tok.isNot(tok::annot_pragma_ompss_end))
      ConsumeAnyToken();
  }
  // Skip the last annot_pragma_ompss_end.
  SourceLocation EndLoc = ConsumeAnnotationToken();
  if (IsError)
    return Ptr;
  return Actions.ActOnOmpSsDeclareTaskDirective(
      Ptr,
      IfRes.get(), FinalRes.get(),
      CostRes.get(), PriorityRes.get(),
      Ins, Outs, Inouts,
      Concurrents, Commutatives,
      WeakIns, WeakOuts, WeakInouts,
      DepIns, DepOuts, DepInouts,
      DepConcurrents, DepCommutatives,
      DepWeakIns, DepWeakOuts, DepWeakInouts,
      SourceRange(Loc, EndLoc));
  return Ptr;
}

/// Parsing of declarative OmpSs directives.
///
///       declare-task-directive:
///         annot_pragma_ompss 'task' {<clause> [,]}
///         annot_pragma_ompss_end
///         <function declaration/definition>
///
Parser::DeclGroupPtrTy Parser::ParseOmpSsDeclarativeDirectiveWithExtDecl(
    AccessSpecifier &AS, ParsedAttributesWithRange &Attrs,
    DeclSpec::TST TagType, Decl *Tag) {
  assert(Tok.is(tok::annot_pragma_ompss) && "Not an OmpSs directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  SourceLocation Loc = ConsumeAnnotationToken();
  OmpSsDirectiveKind DKind = parseOmpSsDirectiveKind(*this);

  switch (DKind) {
  case OSSD_task: {
    // The syntax is:
    // { #pragma oss task }
    // <function-declaration-or-definition>
    //
    Actions.StartOmpSsDSABlock(DKind, Actions.getCurScope(), Loc);

    CachedTokens Toks;
    Toks.push_back(Tok);
    ConsumeToken();
    while(Tok.isNot(tok::annot_pragma_ompss_end)) {
      Toks.push_back(Tok);
      ConsumeAnyToken();
    }
    Toks.push_back(Tok);
    ConsumeAnyToken();

    DeclGroupPtrTy Ptr;
    if (Tok.is(tok::annot_pragma_ompss)) {
      Diag(Loc, diag::err_oss_decl_in_task);
      return DeclGroupPtrTy();
    } else if (Tok.isNot(tok::r_brace) && !isEofOrEom()) {
      // Here we expect to see some function declaration.
      if (AS == AS_none) {
        assert(TagType == DeclSpec::TST_unspecified);
        MaybeParseCXX11Attributes(Attrs);
        ParsingDeclSpec PDS(*this);
        Ptr = ParseExternalDeclaration(Attrs, &PDS);
      } else {
        Ptr =
            ParseCXXClassMemberDeclarationWithPragmas(AS, Attrs, TagType, Tag);
      }
    }
    if (!Ptr) {
      Diag(Loc, diag::err_oss_decl_in_task);
      return DeclGroupPtrTy();
    }
    Parser::DeclGroupPtrTy Ret = ParseOSSDeclareTaskClauses(Ptr, Toks, Loc);

    Actions.EndOmpSsDSABlock(nullptr);
    return Ret;
  }
  case OSSD_taskwait:
    Diag(Tok, diag::err_oss_unexpected_directive)
        << 1 << getOmpSsDirectiveName(DKind);
    break;
  case OSSD_unknown:
    Diag(Tok, diag::err_oss_unknown_directive);
    break;
  }
  while (Tok.isNot(tok::annot_pragma_ompss_end))
    ConsumeAnyToken();
  ConsumeAnyToken();
  return nullptr;
}

/// Parsing of declarative or executable OmpSs directives.
///       executable-directive:
///         annot_pragma_ompss 'taskwait'
///         annot_pragma_ompss 'task'
///         annot_pragma_ompss_end
///
StmtResult Parser::ParseOmpSsDeclarativeOrExecutableDirective(
    ParsedStmtContext Allowed) {
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

    Actions.StartOmpSsDSABlock(DKind, Actions.getCurScope(), Loc);
    while (Tok.isNot(tok::annot_pragma_ompss_end)) {
      OmpSsClauseKind CKind = getOmpSsClauseKind(PP.getSpelling(Tok));

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

    }

    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_ompss_end.
    ConsumeAnnotationToken();

    Actions.ActOnOmpSsAfterClauseGathering(Clauses);

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
    Actions.EndOmpSsDSABlock(Directive.get());
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

/// Parsing of OmpSs clauses.
///
///    clause:
///       depend-clause | if-clause | final-clause
///       | cost-clause | priority-clause
///       | default-clause | shared-clause | private-clause
///       | firstprivate-clause | in-clause | out-clause
///       | inout-clause | weakin-clause | weakout-clause
///       | weakinout-clause
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
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsSingleExprClause(CKind, WrongDirective);
    break;
  case OSSC_default:
    // These clauses cannot appear more than once
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsSimpleClause(CKind, WrongDirective);
    break;
  case OSSC_shared:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_depend:
  case OSSC_in:
  case OSSC_out:
  case OSSC_inout:
  case OSSC_concurrent:
  case OSSC_commutative:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
    Clause = ParseOmpSsVarListClause(DKind, CKind, WrongDirective);
    break;
  case OSSC_unknown:
    Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
        << getOmpSsDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_ompss_end, StopBeforeMatch);
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

  OmpSsDependClauseKind DepKind;

  if (Kind == OSSC_depend) {
    // Handle dependency type for depend clause.
    ColonProtectionRAIIObject ColonRAII(*this);
    DepKind =
        static_cast<OmpSsDependClauseKind>(getOmpSsSimpleClauseType(
            Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
    Data.DepLoc = Tok.getLocation();

    Data.DepKinds.push_back(DepKind);

    if (DepKind == OSSC_DEPEND_unknown) {
      tok::TokenKind TokArray[] = {tok::comma, tok::colon, tok::r_paren, tok::annot_pragma_ompss_end};
      SkipUntil(TokArray, StopBeforeMatch);
    } else {
      ConsumeToken();
    }
    if (Tok.is(tok::comma)) {
      ConsumeToken();
      DepKind =
          static_cast<OmpSsDependClauseKind>(getOmpSsSimpleClauseType(
              Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));

      Data.DepKinds.push_back(DepKind);

      if (DepKind == OSSC_DEPEND_unknown) {
        SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_ompss_end,
                  StopBeforeMatch);
      } else {
        ConsumeToken();
      }
    }
    if (Tok.is(tok::colon)) {
      Data.ColonLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::warn_pragma_expected_colon) << "dependency type";
    }
  }

  auto DepKindIt = std::find(Data.DepKinds.begin(),
                             Data.DepKinds.end(),
                             OSSC_DEPEND_unknown);

  // IsComma init determine if we got a well-formed clause
  bool IsComma = (Kind != OSSC_depend)
                 || (Kind == OSSC_depend && DepKindIt == Data.DepKinds.end());
  // We parse the locator-list when:
  // 1. If we found out something that seems a valid item regardless
  //    of the clause validity
  // 2. We got a well-formed clause regardless what comes next.
  // while (IsComma || Tok.looks_like_valid_item)
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
      Diag(Tok, diag::err_oss_expected_punc)
          << getOmpSsClauseName(Kind);
  }

  // Parse ')'.
  Data.RLoc = Tok.getLocation();
  if (!T.consumeClose())
    Data.RLoc = T.getCloseLocation();
  // return Well-formed clause but empty list
  return (Kind == OSSC_depend
          && DepKindIt == Data.DepKinds.end() && Vars.empty())
          || (Kind != OSSC_depend && Vars.empty());
}

/// Parsing of OmpSs
///
///    depend-clause:
///       'depend' '(' in | out | inout [ ,weak ] : ')'
///       'depend' '(' mutexinoutset | inoutset : ')'
///       'depend' '(' [ weak, ] in | out | inout : ')'
///    private-clause:
///       'private' '(' list ')'
///    firstprivate-clause:
///       'firstprivate' '(' list ')'
///    shared-clause:
///       'shared' '(' list ')'
///    in-clause:
///       'in' '(' list ')'
///    out-clause:
///       'out' '(' list ')'
///    inout-clause:
///       'inout' '(' list ')'
///    concurrent-clause:
///       'concurrent' '(' list ')'
///    commutative-clause:
///       'commutative' '(' list ')'
///    weakin-clause:
///       'weakin' '(' list ')'
///    weakout-clause:
///       'weakout' '(' list ')'
///    weakinout-clause:
///       'weakinout' '(' list ')'
OSSClause *Parser::ParseOmpSsVarListClause(OmpSsDirectiveKind DKind,
                                           OmpSsClauseKind Kind,
                                           bool ParseOnly) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  SmallVector<Expr *, 4> Vars;
  OmpSsVarListDataTy Data;

  Sema::AllowShapingsRAII AllowShapings(Actions, [&Kind]() {
    return Kind == OSSC_depend
     || Kind == OSSC_in || Kind == OSSC_out || Kind == OSSC_inout
     || Kind == OSSC_concurrent || Kind == OSSC_commutative
     || Kind == OSSC_weakin || Kind == OSSC_weakout || Kind == OSSC_weakinout;
  });

  if (ParseOmpSsVarList(DKind, Kind, Vars, Data)) {
    return nullptr;
  }

  if (ParseOnly) {
    return nullptr;
  }
  return Actions.ActOnOmpSsVarListClause(
      Kind, Vars, Loc, LOpen, Data.ColonLoc, Data.RLoc,
      Data.DepKinds, Data.DepLoc);
}

/// Parses simple expression in parens for single-expression clauses of OmpSs
/// constructs.
/// \param RLoc Returned location of right paren.
ExprResult Parser::ParseOmpSsParensExpr(StringRef ClauseName,
                                        SourceLocation &RLoc) {
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ClauseName.data()))
    return ExprError();

  SourceLocation ELoc = Tok.getLocation();
  ExprResult LHS(ParseCastExpression(
      /*isUnaryExpression=*/false, /*isAddressOfOperand=*/false, NotTypeCast));
  ExprResult Val(ParseRHSOfBinaryExpression(LHS, prec::Conditional));
  Val = Actions.ActOnFinishFullExpr(Val.get(), ELoc, /*DiscardedValue*/ false);

  // Parse ')'.
  RLoc = Tok.getLocation();
  if (!T.consumeClose())
    RLoc = T.getCloseLocation();

  return Val;
}

/// Parsing of OmpSs clauses with single expressions like 'final' or 'if'
///
///    final-clause:
///      'final' '(' expression ')'
///
///    cost-clause:
///      'cost' '(' expression ')'
///
///    if-clause:
///      'if' '(' expression ')'
OSSClause *Parser::ParseOmpSsSingleExprClause(OmpSsClauseKind Kind,
                                              bool ParseOnly) {
  SourceLocation Loc = ConsumeToken();
  SourceLocation LLoc = Tok.getLocation();
  SourceLocation RLoc;

  ExprResult Val = ParseOmpSsParensExpr(getOmpSsClauseName(Kind), RLoc);

  if (Val.isInvalid())
    return nullptr;

  if (ParseOnly)
    return nullptr;
  return Actions.ActOnOmpSsSingleExprClause(Kind, Val.get(), Loc, LLoc, RLoc);
}

/// Parsing of simple OmpSs clauses like 'default'.
///
///    default-clause:
///         'default' '(' 'none' | 'shared' ')
///
OSSClause *Parser::ParseOmpSsSimpleClause(OmpSsClauseKind Kind,
                                          bool ParseOnly) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsClauseName(Kind)))
    return nullptr;

  unsigned Type = getOmpSsSimpleClauseType(
      Kind, Tok.isAnnotation() ? "" : PP.getSpelling(Tok));
  SourceLocation TypeLoc = Tok.getLocation();
  if (Tok.isNot(tok::r_paren) && Tok.isNot(tok::comma) &&
      Tok.isNot(tok::annot_pragma_ompss_end))
    ConsumeAnyToken();

  // Parse ')'.
  SourceLocation RLoc = Tok.getLocation();
  if (!T.consumeClose())
    RLoc = T.getCloseLocation();

  if (ParseOnly)
    return nullptr;
  return Actions.ActOnOmpSsSimpleClause(Kind, Type, TypeLoc, LOpen, Loc, RLoc);
}

