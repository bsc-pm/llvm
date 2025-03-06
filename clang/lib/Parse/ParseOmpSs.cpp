//===--- ParseOmpSs.cpp - OmpSs directives parsing ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/Sema/SemaCodeCompletion.h"
#include "llvm/ADT/PointerIntPair.h"

using namespace clang;
using namespace llvm::oss;

//===----------------------------------------------------------------------===//
// OmpSs declarative directives.
//===----------------------------------------------------------------------===//

namespace {
// Keep the same order as in OmpSsKinds.h
enum OmpSsDirectiveKindEx {
  OSSD_declare = llvm::oss::Directive_enumSize + 1,
  OSSD_reduction,
  OSSD_for,
};

// Helper to unify the enum class OmpSsDirectiveKind with its extension
// the OmpSsDirectiveKindEx enum which allows to use them together as if they
// are unsigned values.
struct OmpSsDirectiveKindExWrapper {
  OmpSsDirectiveKindExWrapper(unsigned Value) : Value(Value) {}
  OmpSsDirectiveKindExWrapper(OmpSsDirectiveKind DK) : Value(unsigned(DK)) {}
  bool operator==(OmpSsDirectiveKindExWrapper V) const {
    return Value == V.Value;
  }
  bool operator!=(OmpSsDirectiveKindExWrapper V) const {
    return Value != V.Value;
  }
  bool operator==(OmpSsDirectiveKind V) const { return Value == unsigned(V); }
  bool operator!=(OmpSsDirectiveKind V) const { return Value != unsigned(V); }
  bool operator<(OmpSsDirectiveKind V) const { return Value < unsigned(V); }
  operator unsigned() const { return Value; }
  operator OmpSsDirectiveKind() const { return OmpSsDirectiveKind(Value); }
  unsigned Value;
};
} // namespace

// Map token string to extended OSS token kind that are
// OmpSsDirectiveKind + OmpSsDirectiveKindEx.
static unsigned getOmpSsDirectiveKindEx(StringRef S) {
  OmpSsDirectiveKindExWrapper  DKind = getOmpSsDirectiveKind(S);
  if (DKind != OSSD_unknown)
    return DKind;

  return llvm::StringSwitch<OmpSsDirectiveKindExWrapper>(S)
      .Case("declare", OSSD_declare)
      .Case("reduction", OSSD_reduction)
      .Case("for", OSSD_for)
      .Default(OSSD_unknown);
}

static OmpSsDirectiveKind parseOmpSsDirectiveKind(Parser &P) {
  // Array of foldings: F[i][0] F[i][1] ===> F[i][2].
  // E.g.: OSSD_declare OSSD_reduction ===> OSSD_declare_reduction
  // TODO: add other combined directives in topological order.
  static const OmpSsDirectiveKindExWrapper F[][3] = {
      {OSSD_declare, OSSD_reduction, OSSD_declare_reduction},
      {OSSD_task, OSSD_for, OSSD_task_for},
      {OSSD_taskloop, OSSD_for, OSSD_taskloop_for},
  };
  Token Tok = P.getCurToken();
  OmpSsDirectiveKindExWrapper  DKind =
      Tok.isAnnotation()
          ? static_cast<unsigned>(OSSD_unknown)
          : getOmpSsDirectiveKindEx(P.getPreprocessor().getSpelling(Tok));
  if (DKind == OSSD_unknown)
    return OSSD_unknown;

  for (const auto &I : F) {
    if (DKind != I[0])
      continue;

    Tok = P.getPreprocessor().LookAhead(0);
    OmpSsDirectiveKindExWrapper  SDKind =
        Tok.isAnnotation()
            ? static_cast<unsigned>(OSSD_unknown)
            : getOmpSsDirectiveKindEx(P.getPreprocessor().getSpelling(Tok));
    if (SDKind == OSSD_unknown)
      continue;

    if (SDKind == I[1]) {
      P.ConsumeToken();
      DKind = I[2];
    }
  }
  return unsigned(DKind) < llvm::oss::Directive_enumSize
             ? static_cast<OmpSsDirectiveKind>(DKind)
             : OSSD_unknown;
}

static ExprResult *getSingleClause(
    OmpSsClauseKind CKind,
    ExprResult &ImmediateRes, ExprResult &MicrotaskRes,
    ExprResult &IfRes, ExprResult &FinalRes,
    ExprResult &CostRes, ExprResult &PriorityRes,
    ExprResult &ShmemRes, ExprResult &OnreadyRes) {

    if (CKind == OSSC_immediate) return &ImmediateRes;
    if (CKind == OSSC_microtask) return &MicrotaskRes;
    if (CKind == OSSC_if) return &IfRes;
    if (CKind == OSSC_final) return &FinalRes;
    if (CKind == OSSC_cost) return &CostRes;
    if (CKind == OSSC_priority) return &PriorityRes;
    if (CKind == OSSC_shmem) return &ShmemRes;
    if (CKind == OSSC_onready) return &OnreadyRes;
    return nullptr;
}

static SmallVectorImpl<Expr *> *getClauseList(
    OmpSsClauseKind CKind,
    SmallVectorImpl<Expr *> &Ins, SmallVectorImpl<Expr *> &Outs,
    SmallVectorImpl<Expr *> &Inouts, SmallVectorImpl<Expr *> &Concurrents,
    SmallVectorImpl<Expr *> &Commutatives, SmallVectorImpl<Expr *> &WeakIns,
    SmallVectorImpl<Expr *> &WeakOuts, SmallVectorImpl<Expr *> &WeakInouts,
    SmallVectorImpl<Expr *> &WeakConcurrents, SmallVectorImpl<Expr *> &WeakCommutatives) {

    if (CKind == OSSC_in) return &Ins;
    if (CKind == OSSC_out) return &Outs;
    if (CKind == OSSC_inout) return &Inouts;
    if (CKind == OSSC_concurrent) return &Concurrents;
    if (CKind == OSSC_commutative) return &Commutatives;
    if (CKind == OSSC_weakin) return &WeakIns;
    if (CKind == OSSC_weakout) return &WeakOuts;
    if (CKind == OSSC_weakinout) return &WeakInouts;
    if (CKind == OSSC_weakconcurrent) return &WeakConcurrents;
    if (CKind == OSSC_weakcommutative) return &WeakCommutatives;
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
    if (DepKindsOrdered[0] == OSSC_DEPEND_inoutset)
      return OSSC_weakconcurrent;
    if (DepKindsOrdered[0] == OSSC_DEPEND_mutexinoutset)
      return OSSC_weakcommutative;
  }
  else {
    if (DepKindsOrdered[0] == OSSC_DEPEND_in)
      return OSSC_in;
    if (DepKindsOrdered[0] == OSSC_DEPEND_out)
      return OSSC_out;
    if (DepKindsOrdered[0] == OSSC_DEPEND_inout)
      return OSSC_inout;
    if (DepKindsOrdered[0] == OSSC_DEPEND_inoutset)
      return OSSC_concurrent;
    if (DepKindsOrdered[0] == OSSC_DEPEND_mutexinoutset)
      return OSSC_commutative;
  }
  return OSSC_unknown;
}

/// Parses clauses for 'task' declaration directive.
///
///    clause:
///       depend-clause | if-clause | final-clause
///       | cost-clause | priority-clause | label-clause
///       | wait-clause
///       | default-clause | in-clause | out-clause
///       | inout-clause | concurrent-clause | commutative-clause
///       | weakin-clause | weakout-clause | weakinout-clause
///       | weakconcurrent-clause | weakcommutative-clause
bool Parser::ParseDeclareTaskClauses(
    ExprResult &ImmediateRes, ExprResult &MicrotaskRes,
    ExprResult &IfRes, ExprResult &FinalRes,
    ExprResult &CostRes, ExprResult &PriorityRes,
    ExprResult &ShmemRes, ExprResult &OnreadyRes, bool &Wait,
    unsigned &Device, SourceLocation &DeviceLoc,
    SmallVectorImpl<Expr *> &Labels,
    SmallVectorImpl<Expr *> &Ins, SmallVectorImpl<Expr *> &Outs,
    SmallVectorImpl<Expr *> &Inouts, SmallVectorImpl<Expr *> &Concurrents,
    SmallVectorImpl<Expr *> &Commutatives, SmallVectorImpl<Expr *> &WeakIns,
    SmallVectorImpl<Expr *> &WeakOuts, SmallVectorImpl<Expr *> &WeakInouts,
    SmallVectorImpl<Expr *> &WeakConcurrents, SmallVectorImpl<Expr *> &WeakCommutatives,
    SmallVectorImpl<Expr *> &DepIns, SmallVectorImpl<Expr *> &DepOuts,
    SmallVectorImpl<Expr *> &DepInouts, SmallVectorImpl<Expr *> &DepConcurrents,
    SmallVectorImpl<Expr *> &DepCommutatives, SmallVectorImpl<Expr *> &DepWeakIns,
    SmallVectorImpl<Expr *> &DepWeakOuts, SmallVectorImpl<Expr *> &DepWeakInouts,
    SmallVectorImpl<Expr *> &DepWeakConcurrents, SmallVectorImpl<Expr *> &DepWeakCommutatives,
    SmallVectorImpl<unsigned> &ReductionListSizes,
    SmallVectorImpl<Expr *> &Reductions,
    SmallVectorImpl<unsigned> &ReductionClauseType,
    SmallVectorImpl<CXXScopeSpec> &ReductionCXXScopeSpecs,
    SmallVectorImpl<DeclarationNameInfo> &ReductionIds,
    SmallVectorImpl<Expr *> &Ndrange, SourceLocation &NdrangeLoc,
    SmallVectorImpl<Expr *> &Grid, SourceLocation &GridLoc) {
  const Token &Tok = getCurToken();
  bool IsError = false;

  SmallVector<bool, 4> FirstClauses(llvm::oss::Clause_enumSize + 1);

  while (Tok.isNot(tok::annot_pragma_ompss_end)) {
    if (Tok.isNot(tok::identifier)
        && Tok.isNot(tok::kw_default)
        && Tok.isNot(tok::kw_if))
      break;

    SmallVectorImpl<Expr *> *Vars = nullptr;
    ExprResult *SingleClause = nullptr;
    Parser::OmpSsVarListDataTy VarListData;
    Parser::OmpSsSimpleClauseDataTy SimpleData;

    IdentifierInfo *II = Tok.getIdentifierInfo();
    StringRef ClauseName = II->getName();
    OmpSsClauseKind CKind = getOmpSsClauseKind(ClauseName);

    Vars = getClauseList(
      CKind, Ins, Outs, Inouts,
      Concurrents, Commutatives,
      WeakIns, WeakOuts, WeakInouts,
      WeakConcurrents, WeakCommutatives);

    // Check if clause is allowed for the given directive.
    if (CKind != OSSC_unknown && !isAllowedClauseForDirective(OSSD_declare_task, CKind, /*Version=*/1)) {
      Diag(Tok, diag::err_oss_unexpected_clause) << getOmpSsClauseName(CKind)
                                                 << getOmpSsDirectiveName(OSSD_task);
      IsError = true;
    }

    switch (CKind) {
    case OSSC_immediate:
    case OSSC_microtask:
    case OSSC_if:
    case OSSC_final:
    case OSSC_cost:
    case OSSC_priority:
    case OSSC_shmem:
    case OSSC_onready: {
      ConsumeToken();
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      SourceLocation RLoc;
      SingleClause = getSingleClause(
        CKind, ImmediateRes, MicrotaskRes, IfRes, FinalRes,
        CostRes, PriorityRes, ShmemRes, OnreadyRes);
      *SingleClause = ParseOmpSsParensExpr(getOmpSsClauseName(CKind), RLoc);

      if (SingleClause->isInvalid())
        IsError = true;

      FirstClauses[unsigned(CKind)] = true;
      break;
    }
    case OSSC_wait: {
      SourceLocation Loc = Tok.getLocation();
      ConsumeToken();
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Loc, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      Wait = true;
      FirstClauses[unsigned(CKind)] = true;
      break;
    }
    case OSSC_label: {
      ConsumeToken();
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      SourceLocation RLoc;
      if (ParseOmpSsFixedList<2>(OSSD_task, CKind, Labels, RLoc))
        IsError = true;
      FirstClauses[unsigned(CKind)] = true;
      break;
    }
    case OSSC_depend: {
      ConsumeToken();

      SemaOmpSs::AllowShapingsRAII AllowShapings(getActions().OmpSs(), []() { return true; });

      SmallVector<Expr *, 4> TmpList;
      SmallVector<OmpSsDependClauseKind, 2> DepKindsOrdered;
      if (ParseOmpSsVarList(OSSD_task, CKind, TmpList, VarListData))
        IsError = true;
      if (!getActions().OmpSs().ActOnOmpSsDependKinds(VarListData.DepKinds, DepKindsOrdered, VarListData.DepLoc))
        IsError = true;

      if (!IsError) {
        SmallVectorImpl<Expr *> *DepList = getClauseList(
          getOmpSsClauseFromDependKinds(DepKindsOrdered),
          DepIns, DepOuts, DepInouts,
          DepConcurrents, DepCommutatives,
          DepWeakIns, DepWeakOuts, WeakInouts,
          DepWeakConcurrents, DepWeakCommutatives);
        DepList->append(TmpList.begin(), TmpList.end());
      }
      break;
    }
    case OSSC_reduction:
    case OSSC_weakreduction: {
      ConsumeToken();

      SemaOmpSs::AllowShapingsRAII AllowShapings(getActions().OmpSs(), []() { return true; });

      SmallVector<Expr *, 4> TmpList;
      if (ParseOmpSsVarList(OSSD_task, CKind, TmpList, VarListData))
        IsError = true;
      if (!IsError) {
        Reductions.append(TmpList.begin(), TmpList.end());
        ReductionClauseType.push_back(unsigned(CKind));
        ReductionListSizes.push_back(TmpList.size());
        ReductionCXXScopeSpecs.push_back(VarListData.ReductionIdScopeSpec);
        ReductionIds.push_back(VarListData.ReductionId);
      }
      break;
    }
    case OSSC_ndrange:
      NdrangeLoc = Tok.getLocation();
      ConsumeToken();
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      if (ParseOmpSsVarList(OSSD_task, CKind, Ndrange, VarListData))
        IsError = true;
      FirstClauses[unsigned(CKind)] = true;
      break;
    case OSSC_grid:
      GridLoc = Tok.getLocation();
      ConsumeToken();
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      if (ParseOmpSsVarList(OSSD_task, CKind, Grid, VarListData))
        IsError = true;
      FirstClauses[unsigned(CKind)] = true;
      break;
    case OSSC_in:
    case OSSC_out:
    case OSSC_inout:
    case OSSC_concurrent:
    case OSSC_commutative:
    case OSSC_weakin:
    case OSSC_weakout:
    case OSSC_weakinout:
    case OSSC_weakcommutative: {
      ConsumeToken();

      SemaOmpSs::AllowShapingsRAII AllowShapings(getActions().OmpSs(), []() { return true; });
      if (ParseOmpSsVarList(OSSD_task, CKind, *Vars, VarListData))
        IsError = true;
      break;
    }
    case OSSC_unknown:
      Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
          << getOmpSsDirectiveName(OSSD_task);
      SkipUntil(tok::annot_pragma_ompss_end, StopBeforeMatch);
      break;
    case OSSC_device:
      if (FirstClauses[unsigned(CKind)]) {
        Diag(Tok, diag::err_oss_more_one_clause)
            << getOmpSsDirectiveName(OSSD_task) << getOmpSsClauseName(CKind) << 0;
        IsError = true;
      }
      if (ParseOmpSsSimpleClauseImpl(CKind, SimpleData)) {
        IsError = true;
      } else {
        Device = SimpleData.Type;
        DeviceLoc = SimpleData.TypeLoc;
      }
      FirstClauses[unsigned(CKind)] = true;
      break;
    // Not allowed clauses
    case OSSC_default:
      ParseOmpSsSimpleClauseImpl(CKind, SimpleData);
      break;
    case OSSC_chunksize:
    case OSSC_grainsize:
    case OSSC_unroll:
    case OSSC_collapse:
    case OSSC_on:
    case OSSC_weakconcurrent:
    case OSSC_shared:
    case OSSC_private:
    case OSSC_firstprivate: {
      ConsumeToken();

      SemaOmpSs::AllowShapingsRAII AllowShapings(getActions().OmpSs(), []() { return true; });

      SmallVector<Expr *, 4> TmpList;
      ParseOmpSsVarList(OSSD_task, CKind, TmpList, VarListData);
      break;
    }
    case OSSC_update:
    case OSSC_read:
    case OSSC_write:
    case OSSC_capture:
    case OSSC_compare:
    case OSSC_seq_cst:
    case OSSC_acq_rel:
    case OSSC_acquire:
    case OSSC_release:
    case OSSC_relaxed:
      ConsumeToken();
      break;
    }

    // Skip ',' if any.
    if (Tok.is(tok::comma))
      ConsumeToken();
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

  OSSFNContextRAII FnContext(*this, Ptr);

  ExprResult ImmediateRes;
  ExprResult MicrotaskRes;
  ExprResult IfRes;
  ExprResult FinalRes;
  ExprResult CostRes;
  ExprResult PriorityRes;
  ExprResult ShmemRes;
  ExprResult OnreadyRes;
  bool Wait = false;
  // This value means no clause seen
  unsigned Device = OSSC_DEVICE_unknown + 1;
  SourceLocation DeviceLoc;
  SourceLocation NdrangeLoc;
  SourceLocation GridLoc;

  SmallVector<Expr *, 2> Labels;
  SmallVector<Expr *, 4> Ins;
  SmallVector<Expr *, 4> Outs;
  SmallVector<Expr *, 4> Inouts;
  SmallVector<Expr *, 4> Concurrents;
  SmallVector<Expr *, 4> Commutatives;
  SmallVector<Expr *, 4> WeakIns;
  SmallVector<Expr *, 4> WeakOuts;
  SmallVector<Expr *, 4> WeakInouts;
  SmallVector<Expr *, 4> WeakConcurrents;
  SmallVector<Expr *, 4> WeakCommutatives;
  SmallVector<Expr *, 4> DepIns;
  SmallVector<Expr *, 4> DepOuts;
  SmallVector<Expr *, 4> DepInouts;
  SmallVector<Expr *, 4> DepConcurrents;
  SmallVector<Expr *, 4> DepCommutatives;
  SmallVector<Expr *, 4> DepWeakIns;
  SmallVector<Expr *, 4> DepWeakOuts;
  SmallVector<Expr *, 4> DepWeakInouts;
  SmallVector<Expr *, 4> DepWeakConcurrents;
  SmallVector<Expr *, 4> DepWeakCommutatives;
  SmallVector<unsigned, 4> ReductionListSizes;
  SmallVector<Expr *, 4> Reductions;
  SmallVector<unsigned, 4> ReductionClauseType;
  SmallVector<CXXScopeSpec, 4> ReductionCXXScopeSpecs;
  SmallVector<DeclarationNameInfo, 4> ReductionIds;
  SmallVector<Expr *, 4> Ndranges;
  SmallVector<Expr *, 4> Grids;

  bool IsError =
      ParseDeclareTaskClauses(ImmediateRes, MicrotaskRes,
                              IfRes, FinalRes,
                              CostRes, PriorityRes,
                              ShmemRes, OnreadyRes, Wait,
                              Device, DeviceLoc,
                              Labels,
                              Ins, Outs, Inouts,
                              Concurrents, Commutatives,
                              WeakIns, WeakOuts, WeakInouts,
                              WeakConcurrents, WeakCommutatives,
                              DepIns, DepOuts, DepInouts,
                              DepConcurrents, DepCommutatives,
                              DepWeakIns, DepWeakOuts, DepWeakInouts,
                              DepWeakConcurrents, DepWeakCommutatives,
                              ReductionListSizes, Reductions,
                              ReductionClauseType, ReductionCXXScopeSpecs,
                              ReductionIds, Ndranges, NdrangeLoc,
                              Grids, GridLoc);
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
  return Actions.OmpSs().ActOnOmpSsDeclareTaskDirective(
      Ptr,
      ImmediateRes.get(), MicrotaskRes.get(),
      IfRes.get(), FinalRes.get(),
      CostRes.get(), PriorityRes.get(),
      ShmemRes.get(), OnreadyRes.get(), Wait,
      Device, DeviceLoc,
      Labels,
      Ins, Outs, Inouts,
      Concurrents, Commutatives,
      WeakIns, WeakOuts, WeakInouts,
      WeakConcurrents, WeakCommutatives,
      DepIns, DepOuts, DepInouts,
      DepConcurrents, DepCommutatives,
      DepWeakIns, DepWeakOuts, DepWeakInouts,
      DepWeakConcurrents, DepWeakCommutatives,
      ReductionListSizes, Reductions,
      ReductionClauseType, ReductionCXXScopeSpecs,
      ReductionIds, Ndranges, NdrangeLoc,
      Grids, GridLoc,
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
///       declare-reduction-directive:
///        annot_pragma_ompss 'declare' 'reduction' [...]
///        annot_pragma_ompss_end
///
Parser::DeclGroupPtrTy Parser::ParseOmpSsDeclarativeDirectiveWithExtDecl(
    AccessSpecifier &AS, ParsedAttributes &Attrs, bool Delayed,
    DeclSpec::TST TagType, Decl *Tag) {
  assert(Tok.is(tok::annot_pragma_ompss) && "Not an OmpSs directive!");
  ParsingOmpSsDirectiveRAII DirScope(*this);
  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  SourceLocation Loc;
  OmpSsDirectiveKind DKind;
  if (Delayed) {
    TentativeParsingAction TPA(*this);
    Loc = ConsumeAnnotationToken();
    DKind = parseOmpSsDirectiveKind(*this);
    if (DKind == OSSD_declare_reduction) {
      // Need to delay parsing until completion of the parent class.
      TPA.Revert();
      CachedTokens Toks;
      unsigned Cnt = 1;
      Toks.push_back(Tok);
      while (Cnt && Tok.isNot(tok::eof)) {
        (void)ConsumeAnyToken();
        if (Tok.is(tok::annot_pragma_ompss))
          ++Cnt;
        else if (Tok.is(tok::annot_pragma_ompss_end))
          --Cnt;
        Toks.push_back(Tok);
      }
      // Skip last annot_pragma_ompss_end.
      if (Cnt == 0)
        (void)ConsumeAnyToken();
      auto *LP = new LateParsedPragma(this, AS);
      LP->takeToks(Toks);
      getCurrentClass().LateParsedDeclarations.push_back(LP);
      return nullptr;
    }
    TPA.Commit();
  } else {
    Loc = ConsumeAnnotationToken();
    DKind = parseOmpSsDirectiveKind(*this);
  }

  switch (DKind) {
  case OSSD_task: {
    // Only allowed in C++ mode
    if (!getLangOpts().CPlusPlus && getCurScope()->isClassScope())
      break;
    // The syntax is:
    // { #pragma oss task }
    // <function-declaration-or-definition>
    //
    Actions.OmpSs().StartOmpSsDSABlock(DKind, Actions.getCurScope(), Loc);

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
        ParsedAttributes EmptyDeclSpecAttrs(AttrFactory);
        MaybeParseCXX11Attributes(Attrs);
        ParsingDeclSpec PDS(*this);
        Ptr = ParseExternalDeclaration(Attrs, EmptyDeclSpecAttrs, &PDS);
      } else {
        // Some member functions like void foo(int *x)
        // are not late parsed because they do not need it
        // in our case the oss directive could use
        // a later defined class member so just force late parsing
        // Ideally we would check if the directive uses a late defined variable but we're
        // not gonna do it now.
        Parser::ParsingClass::OmpSsForceDelayRAII OmpSsForceDelay(getCurrentClass());
        Ptr =
            ParseCXXClassMemberDeclarationWithPragmas(AS, Attrs, TagType, Tag);
      }
    }
    if (!Ptr) {
      Diag(Loc, diag::err_oss_decl_in_task);
      return DeclGroupPtrTy();
    }

    Parser::DeclGroupPtrTy Ret = Ptr;
    if (AS == AS_none) {
      Ret = ParseOSSDeclareTaskClauses(Ret, Toks, Loc);
    } else {
      // Store cached tokens to be parsed at the end of the class
      getCurrentClass().OmpSsLateParsedToks.DirDGs.push_back(Ptr);
      getCurrentClass().OmpSsLateParsedToks.DirToks.push_back(Toks);
      getCurrentClass().OmpSsLateParsedToks.DirLocs.push_back(Loc);
    }

    Actions.OmpSs().EndOmpSsDSABlock(nullptr);
    return Ret;
  }
  case OSSD_assert: {
    ConsumeToken();

    BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
    if (T.expectAndConsume(
        diag::err_expected_lparen_after, getOmpSsDirectiveName(DKind).data()))
      break;

    bool IsCorrect = true;
    ExprResult Res;
    if (isTokenStringLiteral()) {
      Res = ParseStringLiteralExpression();
    } else {
      Diag(Tok, diag::err_oss_expected_string_literal);
      SkipUntil(tok::r_paren, StopBeforeMatch);
      IsCorrect = false;
    }

    IsCorrect = !T.consumeClose() && Res.isUsable() && IsCorrect;

    if (IsCorrect) {
      // Need to check for extra tokens.
      if (Tok.isNot(tok::annot_pragma_ompss_end)) {
        Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
            << getOmpSsDirectiveName(DKind);
        while (Tok.isNot(tok::annot_pragma_ompss_end))
          ConsumeAnyToken();
      }
      // Skip the last annot_pragma_ompss_end.
      ConsumeAnnotationToken();
      return Actions.OmpSs().ActOnOmpSsAssertDirective(Loc, Res.get());
    }
    break;
  }
  case OSSD_declare_task:
  case OSSD_critical:
  case OSSD_task_for:
  case OSSD_taskiter:
  case OSSD_taskiter_while:
  case OSSD_taskloop:
  case OSSD_taskloop_for:
  case OSSD_taskwait:
  case OSSD_atomic:
  case OSSD_release:
    Diag(Tok, diag::err_oss_unexpected_directive)
        << 1 << getOmpSsDirectiveName(DKind);
    break;
  case OSSD_declare_reduction:
    ConsumeToken();
    if (DeclGroupPtrTy Res = ParseOmpSsDeclareReductionDirective(AS)) {
      // The last seen token is annot_pragma_ompss_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_ompss_end)) {
        Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
            << getOmpSsDirectiveName(OSSD_declare_reduction);
        while (Tok.isNot(tok::annot_pragma_ompss_end))
          ConsumeAnyToken();
      }
      // Skip the last annot_pragma_ompss_end.
      ConsumeAnnotationToken();
      return Res;
    }
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


void Parser::PreParseCollapse() {
  // Parse late clause tokens
  PP.EnterToken(Tok, /*IsReinject*/ true);
  PP.EnterTokenStream(OSSLateParsedToks, /*DisableMacroExpansion=*/true,
                      /*IsReinject*/ true);

  // Consume the previously pushed token.
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);

  while (Tok.isNot(tok::annot_pragma_ompss_end)) {
    OmpSsClauseKind CKind = getOmpSsClauseKind(PP.getSpelling(Tok));
    ConsumeToken();
    if (CKind != OSSC_collapse) {
      SkipUntil(tok::annot_pragma_ompss_end, tok::identifier, StopBeforeMatch);
      continue;
    }

    SourceLocation RLoc;
    // Supress diagnostics here. Errors will be handled later
    Diags.setSuppressAllDiagnostics(true);

    ExprResult Val = ParseOmpSsParensExpr(getOmpSsClauseName(CKind), RLoc);
    if (!Val.isInvalid())
      Actions.OmpSs().VerifyPositiveIntegerConstant(Val.get(), OSSC_collapse, /*StrictlyPositive=*/true);

    Diags.setSuppressAllDiagnostics(false);

    // Skip ',' if any.
    if (Tok.is(tok::comma))
      ConsumeToken();
  }
  ConsumeAnnotationToken();
}

/// Parsing of declarative or executable OmpSs directives.
///       executable-directive:
///         annot_pragma_ompss 'taskwait'
///         annot_pragma_ompss 'task'
///         annot_pragma_ompss 'task for'
///         annot_pragma_ompss 'taskloop'
///         annot_pragma_ompss 'taskloop for'
///         annot_pragma_ompss 'release'
///         annot_pragma_ompss_end
///
StmtResult Parser::ParseOmpSsDeclarativeOrExecutableDirective(
    ParsedStmtContext Allowed) {
  assert(Tok.is(tok::annot_pragma_ompss) && "Not an OmpSs directive!");
  ParsingOmpSsDirectiveRAII DirScope(*this);
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  unsigned ScopeFlags = Scope::FnScope | Scope::DeclScope |
                        Scope::CompoundStmtScope | Scope::OmpSsDirectiveScope;
  SourceLocation Loc = ConsumeAnnotationToken(), EndLoc;

  OSSClauseList Clauses;
  OmpSsDirectiveKind DKind = parseOmpSsDirectiveKind(*this);
  // Name of critical directive.
  DeclarationNameInfo DirName;
  StmtResult Directive = StmtError();
  bool HasAssociatedStatement = true;
  switch (DKind) {
  case OSSD_taskwait:
  case OSSD_release:
    HasAssociatedStatement = false;
    LLVM_FALLTHROUGH;
  case OSSD_task_for:
  case OSSD_taskiter:
  case OSSD_taskiter_while:
  case OSSD_taskloop:
  case OSSD_taskloop_for:
  case OSSD_critical:
  case OSSD_atomic:
  case OSSD_task: {

    if (isOmpSsLoopDirective(DKind))
      ScopeFlags |= Scope::OmpSsLoopDirectiveScope;

    ParseScope OSSDirectiveScope(this, ScopeFlags);

    // TODO: OmpSs-2 workaround for OmpSsLoopDirectiveScope
    // since the enum underlying type is full
    if (isOmpSsLoopDirective(DKind))
      OSSDirectiveScope.setOmpSsLoopDirectiveScope();

    Actions.OmpSs().StartOmpSsDSABlock(DKind, Actions.getCurScope(), Loc);

    if (isOmpSsTaskLoopDirective(DKind)) {
      // User may write this:
      // #pragma oss taskloop collapse(2)
      // for (...) {
      //   #pragma oss taskloop
      //   for (...) {}
      // }
      // Diagnostic will be emited in Sema, so ignore
      // tokens of the previous taskloop
      OSSLateParsedToks.clear();

      // in taskloop we parse clauses later
      OSSLateParsedToks.push_back(Tok);
      ConsumeToken();
      while(Tok.isNot(tok::annot_pragma_ompss_end)) {
        OSSLateParsedToks.push_back(Tok);
        ConsumeAnyToken();
      }
      OSSLateParsedToks.push_back(Tok);
      ConsumeAnyToken();

     // Parse only collapse clauses. Only the first value (if valid)
     // will be recorded in Stack
     PreParseCollapse();
    } else {
      ConsumeToken();

      // Parse directive name of the 'critical' directive if any.
      if (DKind == OSSD_critical) {
        BalancedDelimiterTracker T(*this, tok::l_paren,
                                   tok::annot_pragma_ompss_end);
        if (!T.consumeOpen()) {
          if (Tok.isAnyIdentifier()) {
            DirName =
                DeclarationNameInfo(Tok.getIdentifierInfo(), Tok.getLocation());
            ConsumeAnyToken();
          } else {
            Diag(Tok, diag::err_oss_expected_identifier_for_critical);
          }
          T.consumeClose();
        }
      }

      Clauses = ParseOmpSsClauses(DKind, EndLoc);
    }
    StackClauses.push_back(Clauses);

    // Determine which taskiter is
    if (DKind == OSSD_taskiter && Tok.is(tok::kw_while))
      Actions.OmpSs().SetTaskiterKind(OSSD_taskiter_while);

    StmtResult AssociatedStmt;
    if (HasAssociatedStatement) {
      Actions.OmpSs().ActOnOmpSsExecutableDirectiveStart();
      AssociatedStmt = (Sema::CompoundScopeRAII(Actions), ParseStatement());
      Actions.OmpSs().ActOnOmpSsExecutableDirectiveEnd();
    }

    Clauses = StackClauses.back();
    StackClauses.pop_back();

    Directive = Actions.OmpSs().ActOnOmpSsExecutableDirective(
      Clauses, DirName, DKind, AssociatedStmt.get(), Loc, EndLoc);

    // Exit scope.
    Actions.OmpSs().EndOmpSsDSABlock(Directive.get());
    OSSDirectiveScope.Exit();
    break;

    }
  case OSSD_declare_reduction:
    ConsumeToken();
    if (DeclGroupPtrTy Res =
            ParseOmpSsDeclareReductionDirective(/*AS=*/AS_none)) {
      // The last seen token is annot_pragma_ompss_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_ompss_end)) {
        Diag(Tok, diag::warn_oss_extra_tokens_at_eol)
            << getOmpSsDirectiveName(OSSD_declare_reduction);
        while (Tok.isNot(tok::annot_pragma_ompss_end))
          ConsumeAnyToken();
      }
      ConsumeAnyToken();
      Directive = Actions.ActOnDeclStmt(Res, Loc, Tok.getLocation());
    } else {
      SkipUntil(tok::annot_pragma_ompss_end);
    }
    break;
  case OSSD_unknown:
    Diag(Tok, diag::err_oss_unknown_directive);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  case OSSD_declare_task:
    Diag(Tok, diag::err_oss_unexpected_directive)
        << 1 << getOmpSsDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  case OSSD_assert:
    Diag(Tok, diag::err_oss_invalid_scope) <<
        getOmpSsDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_ompss_end);
    break;
  }
  return Directive;
}

/// Parsing of all OmpSs clauses of a directive.
Parser::OSSClauseList Parser::ParseOmpSsClauses(OmpSsDirectiveKind DKind, SourceLocation &EndLoc) {
  OSSClauseList Clauses;
  SmallVector<llvm::PointerIntPair<OSSClause *, 1, bool>, llvm::oss::Clause_enumSize + 1>
    FirstClauses(llvm::oss::Clause_enumSize + 1);

  while (Tok.isNot(tok::annot_pragma_ompss_end)) {
    OmpSsClauseKind CKind = getOmpSsClauseKind(PP.getSpelling(Tok));

    // Track which clauses have appeared so we can throw an error in case
    // a clause cannot appear again
    OSSClause *Clause =
        ParseOmpSsClause(DKind, CKind, !FirstClauses[unsigned(CKind)].getInt());
    FirstClauses[unsigned(CKind)].setInt(true);
    if (Clause) {
      FirstClauses[unsigned(CKind)].setPointer(Clause);
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

  // 'release' does not need clause analysis
  if (DKind != OSSD_release &&
      DKind != OSSD_critical && DKind != OSSD_atomic)
    Actions.OmpSs().ActOnOmpSsAfterClauseGathering(Clauses);

  return Clauses;
}

/// Parsing of OmpSs clauses.
///
///    clause:
///       depend-clause | if-clause | final-clause
///       | cost-clause | priority-clause | label-clause
///       | wait-clause
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
  if (CKind != OSSC_unknown && !isAllowedClauseForDirective(DKind, CKind, /*Version=*/1)) {
    Diag(Tok, diag::err_oss_unexpected_clause) << getOmpSsClauseName(CKind)
                                               << getOmpSsDirectiveName(DKind);
    ErrorFound = true;
    WrongDirective = true;
  }

  switch (CKind) {
  case OSSC_immediate:
  case OSSC_microtask:
  case OSSC_if:
  case OSSC_final:
  case OSSC_cost:
  case OSSC_priority:
  case OSSC_shmem:
  case OSSC_onready:
  case OSSC_chunksize:
  case OSSC_grainsize:
  case OSSC_unroll:
  case OSSC_collapse:
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsSingleExprClause(CKind, WrongDirective);
    break;
  case OSSC_wait:
  case OSSC_update:
  case OSSC_read:
  case OSSC_write:
  case OSSC_capture:
  case OSSC_compare:
  case OSSC_seq_cst:
  case OSSC_acq_rel:
  case OSSC_acquire:
  case OSSC_release:
  case OSSC_relaxed:
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsClause(CKind, WrongDirective);
    break;
  case OSSC_default:
  case OSSC_device:
    // These clauses cannot appear more than once
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsSimpleClause(CKind, WrongDirective);
    break;
  case OSSC_label:
    if (!FirstClause) {
      Diag(Tok, diag::err_oss_more_one_clause)
          << getOmpSsDirectiveName(DKind) << getOmpSsClauseName(CKind) << 0;
      ErrorFound = true;
    }
    Clause = ParseOmpSsFixedListClause<2>(DKind, CKind, WrongDirective);
    break;
  case OSSC_shared:
  case OSSC_private:
  case OSSC_firstprivate:
  case OSSC_ndrange:
  case OSSC_depend:
  case OSSC_reduction:
  case OSSC_in:
  case OSSC_out:
  case OSSC_inout:
  case OSSC_concurrent:
  case OSSC_commutative:
  case OSSC_on:
  case OSSC_weakin:
  case OSSC_weakout:
  case OSSC_weakinout:
  case OSSC_weakconcurrent:
  case OSSC_weakcommutative:
  case OSSC_weakreduction:
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

static bool ParseReductionId(Parser &P, CXXScopeSpec &ReductionIdScopeSpec,
                             UnqualifiedId &ReductionId) {
  if (ReductionIdScopeSpec.isEmpty()) {
    auto OOK = OO_None;
    switch (P.getCurToken().getKind()) {
    case tok::plus:
      OOK = OO_Plus;
      break;
    case tok::minus:
      OOK = OO_Minus;
      break;
    case tok::star:
      OOK = OO_Star;
      break;
    case tok::amp:
      OOK = OO_Amp;
      break;
    case tok::pipe:
      OOK = OO_Pipe;
      break;
    case tok::caret:
      OOK = OO_Caret;
      break;
    case tok::ampamp:
      OOK = OO_AmpAmp;
      break;
    case tok::pipepipe:
      OOK = OO_PipePipe;
      break;
    default:
      break;
    }
    if (OOK != OO_None) {
      SourceLocation OpLoc = P.ConsumeToken();
      SourceLocation SymbolLocations[] = {OpLoc, OpLoc, SourceLocation()};
      ReductionId.setOperatorFunctionId(OpLoc, OOK, SymbolLocations);
      return false;
    }
  }
  return P.ParseUnqualifiedId(
      ReductionIdScopeSpec, /*ObjectType=*/nullptr,
      /*ObjectHadErrors=*/false, /*EnteringContext*/ false,
      /*AllowDestructorName*/ false,
      /*AllowConstructorName*/ false,
      /*AllowDeductionGuide*/ false, nullptr, ReductionId);
}

static DeclarationName parseOmpSsDeclareReductionId(Parser &P) {
  Token Tok = P.getCurToken();
  Sema &Actions = P.getActions();
  OverloadedOperatorKind OOK = OO_None;
  // Allow to use 'operator' keyword for C++ operators
  bool WithOperator = false;
  if (Tok.is(tok::kw_operator)) {
    P.ConsumeToken();
    Tok = P.getCurToken();
    WithOperator = true;
  }
  switch (Tok.getKind()) {
  case tok::plus: // '+'
    OOK = OO_Plus;
    break;
  case tok::minus: // '-'
    OOK = OO_Minus;
    break;
  case tok::star: // '*'
    OOK = OO_Star;
    break;
  case tok::amp: // '&'
    OOK = OO_Amp;
    break;
  case tok::pipe: // '|'
    OOK = OO_Pipe;
    break;
  case tok::caret: // '^'
    OOK = OO_Caret;
    break;
  case tok::ampamp: // '&&'
    OOK = OO_AmpAmp;
    break;
  case tok::pipepipe: // '||'
    OOK = OO_PipePipe;
    break;
  case tok::identifier: // identifier
    if (!WithOperator)
      break;
    LLVM_FALLTHROUGH;
  default:
    P.Diag(Tok.getLocation(), diag::err_oss_expected_reduction_identifier);
    P.SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_ompss_end,
                Parser::StopBeforeMatch);
    return DeclarationName();
  }
  P.ConsumeToken();
  auto &DeclNames = Actions.getASTContext().DeclarationNames;
  return OOK == OO_None ? DeclNames.getIdentifier(Tok.getIdentifierInfo())
                        : DeclNames.getCXXOperatorName(OOK);
}

/// Parse 'oss declare reduction' construct.
///
///       declare-reduction-directive:
///        annot_pragma_ompss 'declare' 'reduction'
///        '(' <reduction_id> ':' <type> {',' <type>} ':' <expression> ')'
///        ['initializer' '(' ('omp_priv' '=' <expression>)|<function_call> ')']
///        annot_pragma_ompss_end
/// <reduction_id> is either a base language identifier or one of the following
/// operators: '+', '-', '*', '&', '|', '^', '&&' and '||'.
///
Parser::DeclGroupPtrTy
Parser::ParseOmpSsDeclareReductionDirective(AccessSpecifier AS) {
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsDirectiveName(OSSD_declare_reduction).data())) {
    SkipUntil(tok::annot_pragma_ompss_end, StopBeforeMatch);
    return DeclGroupPtrTy();
  }

  DeclarationName Name = parseOmpSsDeclareReductionId(*this);

  // Keep parsing until no more can be done

  if (Name.isEmpty() && Tok.is(tok::annot_pragma_ompss_end))
    return DeclGroupPtrTy();

  // Consume ':'.
  bool IsCorrect = !ExpectAndConsume(tok::colon);

  if (!IsCorrect && Tok.is(tok::annot_pragma_ompss_end))
    return DeclGroupPtrTy();

  IsCorrect = IsCorrect && !Name.isEmpty();

  // Seeing a colon or annot_pragma_ompss_end finishes typename-list parsing
  if (Tok.is(tok::colon) || Tok.is(tok::annot_pragma_ompss_end)) {
    Diag(Tok.getLocation(), diag::err_expected_type);
    IsCorrect = false;
  }

  if (!IsCorrect && Tok.is(tok::annot_pragma_ompss_end))
    return DeclGroupPtrTy();

  SmallVector<std::pair<QualType, SourceLocation>, 8> ReductionTypes;
  // Here we have something valid
  // declare reduction(fun : <token>
  // Parse list of types until ':' token.
  do {
    ColonProtectionRAIIObject ColonRAII(*this);
    SourceRange Range;
    TypeResult TR =
        ParseTypeName(&Range, DeclaratorContext::Prototype, AS);
    if (TR.isUsable()) {
      QualType ReductionType =
          Actions.OmpSs().ActOnOmpSsDeclareReductionType(Range.getBegin(), TR);
      if (!ReductionType.isNull()) {
        ReductionTypes.push_back(
            std::make_pair(ReductionType, Range.getBegin()));
      }
    } else {
      SkipUntil(tok::comma, tok::colon, tok::annot_pragma_ompss_end,
                StopBeforeMatch);
    }

    // Seeing a colon or annot_pragma_ompss_end finishes typename-list parsing
    if (Tok.is(tok::colon) || Tok.is(tok::annot_pragma_ompss_end))
      break;

    // Consume ','.
    if (ExpectAndConsume(tok::comma)) {
      IsCorrect = false;
      if (Tok.is(tok::annot_pragma_ompss_end)) {
        Diag(Tok.getLocation(), diag::err_expected_type);
        return DeclGroupPtrTy();
      }
    }
  } while (Tok.isNot(tok::annot_pragma_ompss_end));

  if (ReductionTypes.empty()) {
    SkipUntil(tok::annot_pragma_ompss_end, StopBeforeMatch);
    return DeclGroupPtrTy();
  }

  // Parsed some type but failed parsing comma and now token is
  // annot_pragma_ompss_end
  // #pragma oss declare reduction(asdf :int. long,
  if (!IsCorrect && Tok.is(tok::annot_pragma_ompss_end))
    return DeclGroupPtrTy();

  // Consume ':'.
  if (ExpectAndConsume(tok::colon))
    IsCorrect = false;

  if (Tok.is(tok::annot_pragma_ompss_end)) {
    Diag(Tok.getLocation(), diag::err_expected_expression);
    return DeclGroupPtrTy();
  }

  DeclGroupPtrTy DRD = Actions.OmpSs().ActOnOmpSsDeclareReductionDirectiveStart(
      getCurScope(), Actions.getCurLexicalContext(), Name, ReductionTypes, AS);

  // Parse <combiner> expression and then parse initializer if any for each
  // correct type.
  unsigned I = 0, E = ReductionTypes.size();
  for (Decl *D : DRD.get()) {
    TentativeParsingAction TPA(*this);
    ExprResult CombinerResult;
    {
      ParseScope OSSDRScope(this,
        Scope::FnScope | Scope::DeclScope | Scope::CompoundStmtScope
        | Scope::OmpSsDirectiveScope);
      // Parse <combiner> expression.
      Actions.OmpSs().ActOnOmpSsDeclareReductionCombinerStart(getCurScope(), D);
      CombinerResult = Actions.ActOnFinishFullExpr(
          ParseExpression().get(), D->getLocation(), /*DiscardedValue*/ false);
      Actions.OmpSs().ActOnOmpSsDeclareReductionCombinerEnd(D, CombinerResult.get());
    }

    if (CombinerResult.isInvalid() && Tok.isNot(tok::r_paren) &&
        Tok.isNot(tok::annot_pragma_ompss_end)) {
      TPA.Commit();
      IsCorrect = false;
      break;
    }
    IsCorrect = !T.consumeClose() && IsCorrect && CombinerResult.isUsable();
    ExprResult InitializerResult;
    if (Tok.isNot(tok::annot_pragma_ompss_end)) {
      // Parse <initializer> expression.
      if (Tok.is(tok::identifier) &&
          Tok.getIdentifierInfo()->isStr("initializer")) {
        ConsumeToken();
      } else {
        Diag(Tok.getLocation(), diag::err_expected) << "'initializer'";
        TPA.Commit();
        IsCorrect = false;
        break;
      }
      // Parse '('.
      BalancedDelimiterTracker T(*this, tok::l_paren,
                                 tok::annot_pragma_ompss_end);
      IsCorrect =
          !T.expectAndConsume(diag::err_expected_lparen_after, "initializer") &&
          IsCorrect;
      if (Tok.isNot(tok::annot_pragma_ompss_end)) {
        ParseScope OSSDRScope(this,
          Scope::FnScope | Scope::DeclScope | Scope::CompoundStmtScope
          | Scope::OmpSsDirectiveScope);
        // Parse expression.
        VarDecl *OmpPrivParm =
            Actions.OmpSs().ActOnOmpSsDeclareReductionInitializerStart(getCurScope(),
                                                               D);
        // Check if initializer is omp_priv <init_expr> or something else.
        if (Tok.is(tok::identifier) &&
            Tok.getIdentifierInfo()->isStr("omp_priv")) {
          ConsumeToken();
          ParseOmpSsReductionInitializerForDecl(OmpPrivParm);
        } else {
          InitializerResult = Actions.ActOnFinishFullExpr(
              ParseAssignmentExpression().get(), D->getLocation(),
              /*DiscardedValue*/ false);
        }
        Actions.OmpSs().ActOnOmpSsDeclareReductionInitializerEnd(
            D, InitializerResult.get(), OmpPrivParm);
        if (InitializerResult.isInvalid() && Tok.isNot(tok::r_paren) &&
            Tok.isNot(tok::annot_pragma_ompss_end)) {
          TPA.Commit();
          IsCorrect = false;
          break;
        }
        IsCorrect =
            !T.consumeClose() && IsCorrect && !InitializerResult.isInvalid();
      }
    }

    ++I;
    // Revert parsing if not the last type, otherwise accept it, we're done with
    // parsing.
    if (I != E)
      TPA.Revert();
    else
      TPA.Commit();
  }
  return Actions.OmpSs().ActOnOmpSsDeclareReductionDirectiveEnd(getCurScope(), DRD,
                                                        IsCorrect);
}

void Parser::ParseOmpSsReductionInitializerForDecl(VarDecl *OmpPrivParm) {
  // Parse declarator '=' initializer.
  // If a '==' or '+=' is found, suggest a fixit to '='.
  if (isTokenEqualOrEqualTypo()) {
    ConsumeToken();

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompletion().CodeCompleteInitializer(getCurScope(), OmpPrivParm);
      Actions.FinalizeDeclaration(OmpPrivParm);
      cutOffParsing();
      return;
    }

    PreferredType.enterVariableInit(Tok.getLocation(), OmpPrivParm);
    ExprResult Init = ParseInitializer();

    if (Init.isInvalid()) {
      SkipUntil(tok::r_paren, tok::annot_pragma_ompss_end, StopBeforeMatch);
      Actions.ActOnInitializerError(OmpPrivParm);
    } else {
      Actions.AddInitializerToDecl(OmpPrivParm, Init.get(),
                                   /*DirectInit=*/false);
    }
  } else if (Tok.is(tok::l_paren)) {
    // Parse C++ direct initializer: '(' expression-list ')'
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();

    ExprVector Exprs;

    SourceLocation LParLoc = T.getOpenLocation();
    auto RunSignatureHelp = [this, OmpPrivParm, LParLoc, &Exprs]() {
      QualType PreferredType = Actions.CodeCompletion().ProduceConstructorSignatureHelp(
          OmpPrivParm->getType()->getCanonicalTypeInternal(),
          OmpPrivParm->getLocation(), Exprs, LParLoc, /*Braced=*/false);
      CalledSignatureHelp = true;
      return PreferredType;
    };
    if (ParseExpressionList(Exprs, [&] {
          PreferredType.enterFunctionArgument(Tok.getLocation(),
                                              RunSignatureHelp);
        })) {
      if (PP.isCodeCompletionReached() && !CalledSignatureHelp)
        RunSignatureHelp();
      Actions.ActOnInitializerError(OmpPrivParm);
      SkipUntil(tok::r_paren, tok::annot_pragma_ompss_end, StopBeforeMatch);
    } else {
      // Match the ')'.
      SourceLocation RLoc = Tok.getLocation();
      if (!T.consumeClose())
        RLoc = T.getCloseLocation();

      ExprResult Initializer =
          Actions.ActOnParenListExpr(T.getOpenLocation(), RLoc, Exprs);
      Actions.AddInitializerToDecl(OmpPrivParm, Initializer.get(),
                                   /*DirectInit=*/true);
    }
  } else if (getLangOpts().CPlusPlus11 && Tok.is(tok::l_brace)) {
    // Parse C++0x braced-init-list.
    Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);

    ExprResult Init(ParseBraceInitializer());

    if (Init.isInvalid()) {
      Actions.ActOnInitializerError(OmpPrivParm);
    } else {
      Actions.AddInitializerToDecl(OmpPrivParm, Init.get(),
                                   /*DirectInit=*/true);
    }
  } else {
    Actions.ActOnUninitializedDecl(OmpPrivParm);
  }
}

/// Parses clauses with list with a max limit of N
template<int N>
bool Parser::ParseOmpSsFixedList(
    OmpSsDirectiveKind DKind, OmpSsClauseKind Kind, SmallVectorImpl<Expr *> &Vars,
    SourceLocation &RLoc) {
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsClauseName(Kind).data()))
    return true;

  bool IsComma = true;
  auto Cond = [&](int i) {
    return i < N &&
      (IsComma
       || (Tok.isNot(tok::r_paren)
         && Tok.isNot(tok::colon)
         && Tok.isNot(tok::annot_pragma_ompss_end)));
  };

  for (int i = 0; Cond(i); ++i) {

    SourceLocation ELoc = Tok.getLocation();

    // 1. Disable diagnostics for label clause. Emit a warning and keep going
    // FIXME: this hides any typo correction diagnostic like:
    // const char *blabla;
    // #pragma oss task label(babla) <- hidden typo correction
    if (i == 0 && Kind == OSSC_label)
      Diags.setSuppressAllDiagnostics(true);

    ExprResult Val =
        Actions.CorrectDelayedTyposInExpr(ParseAssignmentExpression());

    // See 1.
    if (i == 0 && Kind == OSSC_label) {
      Diags.setSuppressAllDiagnostics(false);

      if (Val.isInvalid() || Val.get()->containsErrors()) {
        Diag(ELoc, diag::warn_oss_label_error);
        Val = ExprError();
      }
    }

    if (Val.isUsable()) {
      Vars.push_back(Val.get());
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
  RLoc = Tok.getLocation();
  if (!T.consumeClose())
    RLoc = T.getCloseLocation();
  return Vars.empty();
}

/// Parses clauses with list.
bool Parser::ParseOmpSsVarList(OmpSsDirectiveKind DKind,
                               OmpSsClauseKind Kind,
                               SmallVectorImpl<Expr *> &Vars,
                               OmpSsVarListDataTy &Data) {
  UnqualifiedId UnqualifiedReductionId;
  bool InvalidReductionId = false;

  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsClauseName(Kind).data()))
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
  } else if (Kind == OSSC_reduction || Kind == OSSC_weakreduction) {
    ColonProtectionRAIIObject ColonRAII(*this);
    if (getLangOpts().CPlusPlus)
      ParseOptionalCXXScopeSpecifier(Data.ReductionIdScopeSpec,
                                     /*ObjectType=*/nullptr,
                                     /*ObjectHadErrors=*/false,
                                     /*EnteringContext=*/false);
    InvalidReductionId = ParseReductionId(
        *this, Data.ReductionIdScopeSpec, UnqualifiedReductionId);
    if (InvalidReductionId) {
      SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_ompss_end,
                StopBeforeMatch);
    }
    if (Tok.is(tok::colon))
      Data.ColonLoc = ConsumeToken();
    else
      Diag(Tok, diag::warn_pragma_expected_colon) << "reduction identifier";
    if (!InvalidReductionId)
      Data.ReductionId =
          Actions.GetNameFromUnqualifiedId(UnqualifiedReductionId);
  }

  auto DepKindIt = std::find(Data.DepKinds.begin(),
                             Data.DepKinds.end(),
                             OSSC_DEPEND_unknown);


  // IsComma init determine if we got a well-formed clause
  bool IsComma = (Kind != OSSC_depend && Kind != OSSC_reduction && Kind != OSSC_weakreduction)
                 || ((Kind == OSSC_reduction
                      || Kind == OSSC_weakreduction) && !InvalidReductionId)
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
        Actions.CorrectDelayedTyposInExpr(ParseOSSAssignmentExpression(DKind, Kind));
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
  // Do not pass to Sema when
  //   - shared()
  //       we have nothing to analize
  //   - reduction(invalid : ...)
  //       we cannot analize anything because we need a valid reduction id
  return (Kind != OSSC_depend && Vars.empty()) || InvalidReductionId;
}

/// Parsing of OmpSs
///
///    label-clause:
///      'label' '(' expression ' [, 'expression' ])'
template<int N>
OSSClause *Parser::ParseOmpSsFixedListClause(
    OmpSsDirectiveKind DKind, OmpSsClauseKind Kind, bool ParseOnly) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  SourceLocation RLoc;
  SmallVector<Expr *, 4> Vars;

  if (ParseOmpSsFixedList<N>(DKind, Kind, Vars, RLoc)) {
    return nullptr;
  }

  if (ParseOnly) {
    return nullptr;
  }
  return Actions.OmpSs().ActOnOmpSsFixedListClause(
      Kind, Vars, Loc, LOpen, RLoc);
  return nullptr;
}

/// Parsing of OmpSs
///
///    depend-clause:
///       'depend' '(' in | out | inout | mutexinoutset [ ,weak ] : ')'
///       'depend' '(' inoutset : ')'
///       'depend' '(' [ weak, ] in | out | inout | mutexinoutset : ')'
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

  SemaOmpSs::AllowShapingsRAII AllowShapings(Actions.OmpSs(), [&Kind]() {
    return Kind == OSSC_depend || Kind == OSSC_reduction
     || Kind == OSSC_in || Kind == OSSC_out || Kind == OSSC_inout
     || Kind == OSSC_concurrent || Kind == OSSC_commutative
     || Kind == OSSC_on
     || Kind == OSSC_weakin || Kind == OSSC_weakout || Kind == OSSC_weakinout
     || Kind == OSSC_weakconcurrent || Kind == OSSC_weakcommutative
     || Kind == OSSC_weakreduction;
  });

  if (ParseOmpSsVarList(DKind, Kind, Vars, Data)) {
    return nullptr;
  }

  if (ParseOnly) {
    return nullptr;
  }
  return Actions.OmpSs().ActOnOmpSsVarListClause(
      Kind, Vars, Loc, LOpen, Data.ColonLoc, Data.RLoc,
      Data.DepKinds, Data.DepLoc, Data.ReductionIdScopeSpec,
      Data.ReductionId);
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
      AnyCastExpr, /*isAddressOfOperand=*/false, NotTypeCast));
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
///
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
  return Actions.OmpSs().ActOnOmpSsSingleExprClause(Kind, Val.get(), Loc, LLoc, RLoc);
}

bool Parser::ParseOmpSsSimpleClauseImpl(OmpSsClauseKind Kind,
                                OmpSsSimpleClauseDataTy &Data) {
  Data.Loc = Tok.getLocation();
  Data.LOpen = ConsumeToken();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_ompss_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOmpSsClauseName(Kind).data()))
    return true;

  Data.Type = getOmpSsSimpleClauseType(
      Kind, Tok.isAnnotation() ? "" : PP.getSpelling(Tok));
  Data.TypeLoc = Tok.getLocation();
  if (Tok.isNot(tok::r_paren) && Tok.isNot(tok::comma) &&
      Tok.isNot(tok::annot_pragma_ompss_end))
    ConsumeAnyToken();

  // Parse ')'.
  Data.RLoc = Tok.getLocation();
  if (!T.consumeClose())
    Data.RLoc = T.getCloseLocation();

  return false;
}

/// Parsing of simple OmpSs clauses like 'default'.
///
///    default-clause:
///         'default' '(' 'none' | 'shared' ')
///
///    device-clause:
///         'device' '(' 'smp' | 'cuda' | 'opencl' | 'fpga' ')
///
OSSClause *Parser::ParseOmpSsSimpleClause(OmpSsClauseKind Kind,
                                          bool ParseOnly) {
  OmpSsSimpleClauseDataTy Data;
  if (ParseOmpSsSimpleClauseImpl(Kind, Data))
    return nullptr;

  if (ParseOnly)
    return nullptr;
  return Actions.OmpSs().ActOnOmpSsSimpleClause(Kind, Data.Type, Data.TypeLoc, Data.LOpen, Data.Loc, Data.RLoc);
}

/// Parsing of OmpSs clauses like 'wait'.
///
///    wait-clause:
///         'wait'
///    update-clause:
///         'update'
///    read-clause:
///         'read'
///    write-clause:
///         'write'
///    capture-clause:
///         'capture'
///    compare-clause:
///         'compare'
///    seq_cst-clause:
///         'seq_cst'
///    acq_rel-clause:
///         'acq_rel'
///    acquire-clause:
///         'acquire'
///    release-clause:
///         'release'
///    relaxed-clause:
///         'relaxed'
///
OSSClause *Parser::ParseOmpSsClause(OmpSsClauseKind Kind, bool ParseOnly) {
  SourceLocation Loc = Tok.getLocation();
  ConsumeAnyToken();

  if (ParseOnly)
    return nullptr;
  return Actions.OmpSs().ActOnOmpSsClause(Kind, Loc, Tok.getLocation());
}
