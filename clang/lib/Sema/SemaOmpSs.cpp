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

namespace {
/// Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0, /// Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// Default data sharing attribute 'none'.
  DSA_shared = 1 << 1, /// Default data sharing attribute 'shared'.
};

/// Stack for tracking declarations used in OmpSs directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OmpSsDirectiveKind DKind = OSSD_unknown;
    OmpSsClauseKind CKind = OSSC_unknown;
    const Expr *RefExpr = nullptr;
    bool Ignore = false;
    DSAVarData() = default;
    DSAVarData(OmpSsDirectiveKind DKind, OmpSsClauseKind CKind,
               const Expr *RefExpr, bool Ignore)
        : DKind(DKind), CKind(CKind), RefExpr(RefExpr), Ignore(Ignore)
          {}
  };

private:
  struct DSAInfo {
    OmpSsClauseKind Attributes = OSSC_unknown;
    const Expr * RefExpr;
    bool Ignore = false;
  };
  using DeclSAMapTy = llvm::SmallDenseMap<const ValueDecl *, DSAInfo, 8>;

  // Directive
  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DefaultDataSharingAttributes DefaultAttr = DSA_unspecified;
    SourceLocation DefaultAttrLoc;
    OmpSsDirectiveKind Directive = OSSD_unknown;
    Scope *CurScope = nullptr;
    CXXThisExpr *ThisExpr = nullptr;
    SourceLocation ConstructLoc;
    SharingMapTy(OmpSsDirectiveKind DKind,
                 Scope *CurScope, SourceLocation Loc)
        : Directive(DKind), CurScope(CurScope),
          ConstructLoc(Loc) {}
    SharingMapTy() = default;
  };

  using StackTy = SmallVector<SharingMapTy, 4>;

  /// Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  Sema &SemaRef;

  using iterator = StackTy::const_reverse_iterator;

  DSAVarData getDSA(iterator &Iter, ValueDecl *D) const;

  bool isStackEmpty() const {
    return Stack.empty();
  }

public:
  explicit DSAStackTy(Sema &S) : SemaRef(S) {}

  void push(OmpSsDirectiveKind DKind,
            Scope *CurScope, SourceLocation Loc) {
    Stack.emplace_back(DKind, CurScope, Loc);
  }

  void pop() {
    assert(!Stack.empty() && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  /// Adds explicit data sharing attribute to the specified declaration.
  void addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A, bool Ignore=false);

  /// Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  const DSAVarData getTopDSA(ValueDecl *D, bool FromParent);
  /// Returns data sharing attributes from the current directive for the
  /// specified declaration.
  const DSAVarData getCurrentDSA(ValueDecl *D);
  /// Returns data-sharing attributes for the specified declaration.
  /// Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any directive which matches \a DPred
  /// predicate.
  const DSAVarData
  hasDSA(ValueDecl *D, const llvm::function_ref<bool(OmpSsClauseKind)> CPred,
         const llvm::function_ref<bool(OmpSsDirectiveKind)> DPred,
         bool FromParent) const;
  /// Set default data sharing attribute to none.
  void setDefaultDSANone(SourceLocation Loc) {
    assert(!isStackEmpty());
    Stack.back().DefaultAttr = DSA_none;
    Stack.back().DefaultAttrLoc = Loc;
  }
  /// Set default data sharing attribute to shared.
  void setDefaultDSAShared(SourceLocation Loc) {
    assert(!isStackEmpty());
    Stack.back().DefaultAttr = DSA_shared;
    Stack.back().DefaultAttrLoc = Loc;
  }
  void setThisExpr(CXXThisExpr *ThisE) {
    Stack.back().ThisExpr = ThisE;
  }
  /// Returns currently analyzed directive.
  OmpSsDirectiveKind getCurrentDirective() const {
    return isStackEmpty() ? OSSD_unknown : Stack.back().Directive;
  }
  DefaultDataSharingAttributes getCurrentDefaultDataSharingAttributtes() const {
    return isStackEmpty() ? DSA_unspecified : Stack.back().DefaultAttr;
  }
  CXXThisExpr *getThisExpr() const {
    return isStackEmpty() ? nullptr : Stack.back().ThisExpr;
  }
};

} // namespace

static const ValueDecl *getCanonicalDecl(const ValueDecl *D) {
  const auto *VD = dyn_cast<VarDecl>(D);
  const auto *FD = dyn_cast<FieldDecl>(D);
  if (VD != nullptr) {
    VD = VD->getCanonicalDecl();
    D = VD;
  } else {
    assert(FD);
    FD = FD->getCanonicalDecl();
    D = FD;
  }
  return D;
}

static ValueDecl *getCanonicalDecl(ValueDecl *D) {
  return const_cast<ValueDecl *>(
      getCanonicalDecl(const_cast<const ValueDecl *>(D)));
}

DSAStackTy::DSAVarData DSAStackTy::getDSA(iterator &Iter,
                                          ValueDecl *D) const {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  DVar.DKind = Iter->Directive;
  if (Iter->SharingMap.count(D)) {
    const DSAInfo &Data = Iter->SharingMap.lookup(D);
    DVar.RefExpr = Data.RefExpr;
    DVar.Ignore = Data.Ignore;
    DVar.CKind = Data.Attributes;
    return DVar;
  }

  return DVar;
}

void DSAStackTy::addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A, bool Ignore) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  DSAInfo &Data = Stack.back().SharingMap[D];
  Data.Attributes = A;
  Data.RefExpr = E;
  Data.Ignore = Ignore;
}

const DSAStackTy::DSAVarData DSAStackTy::getTopDSA(ValueDecl *D,
                                                   bool FromParent) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return Dir == OSSD_task; };
  auto &&AnyClause = [](OmpSsClauseKind Clause) { return Clause != OSSC_shared; };
  if (VD) {
    DSAVarData DVarTemp = hasDSA(D, AnyClause, IsTaskDir, FromParent);
    if (DVarTemp.CKind != OSSC_unknown && DVarTemp.RefExpr)
      return DVarTemp;
  }

  return DVar;
}

const DSAStackTy::DSAVarData DSAStackTy::getCurrentDSA(ValueDecl *D) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return Dir == OSSD_task; };
  auto &&AnyClause = [](OmpSsClauseKind Clause) { return true; };
  iterator I = Stack.rbegin();
  iterator EndI = Stack.rend();
  if (VD){
    if (I != EndI) {
      if (IsTaskDir(I->Directive)) {
        DSAVarData DVar = getDSA(I, D);
          if (AnyClause(DVar.CKind))
            return DVar;
      }
    }
  }
  return DVar;
}

const DSAStackTy::DSAVarData
DSAStackTy::hasDSA(ValueDecl *D,
                   const llvm::function_ref<bool(OmpSsClauseKind)> CPred,
                   const llvm::function_ref<bool(OmpSsDirectiveKind)> DPred,
                   bool FromParent) const {
  D = getCanonicalDecl(D);
  iterator I = Stack.rbegin();
  iterator EndI = Stack.rend();
  if (FromParent && I != EndI)
    std::advance(I, 1);
  for (; I != EndI; std::advance(I, 1)) {
    if (!DPred(I->Directive))
      continue;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
  }
  return {};
}

namespace {
class DSAAttrChecker final : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound = false;
  Stmt *CS = nullptr;
  llvm::SmallVector<Expr *, 4> ImplicitShared;
  llvm::SmallVector<Expr *, 4> ImplicitFirstprivate;
  llvm::SmallSet<ValueDecl *, 4> InnerDecls;

  // Walks over all array dimensions looking for VLA size Expr.
  void GetTypeDSAs(QualType T) {
    QualType TmpTy = T;
    while (TmpTy->isArrayType()) {
      if (const ConstantArrayType *BaseArrayTy = SemaRef.Context.getAsConstantArrayType(TmpTy)) {
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = SemaRef.Context.getAsVariableArrayType(TmpTy)) {
        Expr *SizeExpr = BaseArrayTy->getSizeExpr();
        Visit(SizeExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

public:
  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // Add DSA to 'this' if is the first time we see it
    if (!Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
      ImplicitShared.push_back(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();

      // Variables declared inside region don't have DSA
      if (InnerDecls.count(VD))
        return;

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      DSAStackTy::DSAVarData DVarFromParent = Stack->getTopDSA(VD, /*FromParent=*/true);

      bool ExistsParent = DVarFromParent.RefExpr;
      bool ParentIgnore = DVarFromParent.Ignore;

      bool ExistsCurrent = DVarCurrent.RefExpr;

      // Check if the variable has DSA set on the current
      // directive and stop analysis if it so.
      if (ExistsCurrent)
        return;
      // If explicit DSA comes from parent inherit it
      if (ExistsParent && !ParentIgnore) {
          switch (DVarFromParent.CKind) {
          case OSSC_shared:
            ImplicitShared.push_back(E);
            break;
          case OSSC_private:
          case OSSC_firstprivate:
            ImplicitFirstprivate.push_back(E);
            break;
          default:
            llvm_unreachable("unexpected DSA from parent");
          }
      } else {

        QualType Type = VD->getType();
        if (!Type.isPODType(SemaRef.Context)) {
          return;
        }

        OmpSsDirectiveKind DKind = Stack->getCurrentDirective();
        switch (Stack->getCurrentDefaultDataSharingAttributtes()) {
        case DSA_shared:
          // Define implicit data-sharing attributes for task.
          if (isOmpSsTaskingDirective(DKind))
            ImplicitShared.push_back(E);
          // Record DSA as Ignored to avoid making the same node again
          Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/true);
          break;
        case DSA_none:
          if (!DVarCurrent.Ignore) {
            SemaRef.Diag(E->getExprLoc(), diag::err_oss_not_defined_dsa_when_default_none) << E->getDecl();
            // Record DSA as ignored to diagnostic only once
            Stack->addDSA(VD, E, OSSC_unknown, /*Ignore=*/true);
          }
          break;
        case DSA_unspecified:
          if (VD->hasLocalStorage()) {
            // If no default clause is present and the variable was private/local
            // in the context encountering the construct, the variable will
            // be firstprivate

            // Define implicit data-sharing attributes for task.
            if (isOmpSsTaskingDirective(DKind))
              ImplicitFirstprivate.push_back(E);

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_firstprivate, /*Ignore=*/true);
          } else {
            // If no default clause is present and the variable was shared/global
            // in the context encountering the construct, the variable will be shared.

            // Define implicit data-sharing attributes for task.
            if (isOmpSsTaskingDirective(DKind))
              ImplicitShared.push_back(E);

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/true);
          }
        }
      }
    }
  }

  void VisitExpr(Expr *E) {
    for (Stmt *Child : E->children()) {
      if (Child)
        Visit(Child);
    }
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  void VisitDeclStmt(DeclStmt *S) {
    for (Decl *D : S->decls()) {
      if (auto *VD = dyn_cast_or_null<VarDecl>(D)) {
        InnerDecls.insert(VD);
        if (VD->hasInit()) {
          Visit(VD->getInit());
        }
        GetTypeDSAs(VD->getType());
      }
    }
  }

  bool isErrorFound() const { return ErrorFound; }

  ArrayRef<Expr *> getImplicitShared() const {
    return ImplicitShared;
  }

  ArrayRef<Expr *> getImplicitFirstprivate() const {
    return ImplicitFirstprivate;
  }

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef, Stmt *CS)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false), CS(CS) {}
};

class OSSClauseDSAChecker final : public StmtVisitor<OSSClauseDSAChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound = false;
  llvm::SmallVector<Expr *, 4> ImplicitFirstprivate;
  llvm::SmallVector<Expr *, 4> ImplicitShared;
  bool IsArraySubscriptIdx = false;
public:
  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    VisitExpr(E->getBase());
    IsArraySubscriptIdx = true;
    VisitExpr(E->getIdx());
    IsArraySubscriptIdx = false;
  }
  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // Add DSA to 'this' if is the first time we see it
    if (!Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
      ImplicitShared.push_back(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;

    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();
      // inout(x)              | shared(x)        | int x;
      // inout(p[i])           | firstprivate(p)  | int *p;
      // inout(a[i])           | shared(a)        | int a[N];
      // inout(*p)/inout(p[0]) | firstprivate(p)  | int *p;
      // inout(s.x)            | shared(s)        | struct S s;
      // inout(ps->x)          | firstprivate(ps) | struct S *ps;
      OmpSsClauseKind VKind = OSSC_shared;
      if (VD->getType()->isPointerType())
        VKind = OSSC_firstprivate;
      else if (IsArraySubscriptIdx)
        VKind = OSSC_firstprivate;

      SourceLocation ELoc = E->getExprLoc();
      SourceRange ERange = E->getSourceRange();

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      switch (DVarCurrent.CKind) {
      case OSSC_shared:
        if (VKind == OSSC_private || VKind == OSSC_firstprivate) {
          ErrorFound = true;
          SemaRef.Diag(ELoc, diag::err_oss_mismatch_depend_dsa)
            << getOmpSsClauseName(DVarCurrent.CKind)
            << getOmpSsClauseName(VKind) << ERange;
        }
        break;
      case OSSC_private:
      case OSSC_firstprivate:
        if (VKind == OSSC_shared) {
          ErrorFound = true;
          SemaRef.Diag(ELoc, diag::err_oss_mismatch_depend_dsa)
            << getOmpSsClauseName(DVarCurrent.CKind)
            << getOmpSsClauseName(VKind) << ERange;
        }
        break;
      case OSSC_unknown:
        // No DSA explicit
        if (Stack->getCurrentDefaultDataSharingAttributtes() == DSA_none) {
          // KO: but default(none)
          // Record DSA as ignored
          SemaRef.Diag(ELoc, diag::err_oss_not_defined_dsa_when_default_none) << E->getDecl();
          Stack->addDSA(VD, E, OSSC_unknown, /*Ignore=*/true);
        } else {
          // OK: record DSA as explicit
          if (VKind == OSSC_shared)
            ImplicitShared.push_back(E);
          if (VKind == OSSC_firstprivate)
            ImplicitFirstprivate.push_back(E);

          Stack->addDSA(VD, E, VKind);
        }

        break;
      default:
        llvm_unreachable("unexpected DSA");
      }
    }
  }

  void VisitOSSDepend(OSSDependClause *Clause) {
    for (Stmt *Child : Clause->children()) {
      if (Child)
        Visit(Child);
    }
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  bool isErrorFound() const { return ErrorFound; }

  OSSClauseDSAChecker(DSAStackTy *S, Sema &SemaRef)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false) {}

  ArrayRef<Expr *> getImplicitShared() const {
    return ImplicitShared;
  }

  ArrayRef<Expr *> getImplicitFirstprivate() const {
    return ImplicitFirstprivate;
  }
};
} // namespace

void Sema::InitDataSharingAttributesStackOmpSs() {
  VarDataSharingAttributesStackOmpSs = new DSAStackTy(*this);
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStackOmpSs)

void Sema::StartOmpSsDSABlock(OmpSsDirectiveKind DKind,
                               Scope *CurScope, SourceLocation Loc) {
  DSAStack->push(DKind, CurScope, Loc);
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);
}

void Sema::EndOmpSsDSABlock(Stmt *CurDirective) {
  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

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

void Sema::ActOnOmpSsAfterClauseGathering(SmallVectorImpl<OSSClause *>& Clauses) {

  bool ErrorFound = false;

  for (auto *Clause : Clauses) {
    OSSClauseDSAChecker OSSDependChecker(DSAStack, *this);
    if (isa<OSSDependClause>(Clause)) {
      OSSDependChecker.VisitOSSDepend(cast<OSSDependClause> (Clause));

      if (OSSDependChecker.isErrorFound())
        ErrorFound = true;

      SmallVector<Expr *, 4> ImplicitShared(
          OSSDependChecker.getImplicitShared().begin(),
          OSSDependChecker.getImplicitShared().end());

      SmallVector<Expr *, 4> ImplicitFirstprivate(
          OSSDependChecker.getImplicitFirstprivate().begin(),
          OSSDependChecker.getImplicitFirstprivate().end());

      if (!ImplicitShared.empty()) {
        if (OSSClause *Implicit = ActOnOmpSsSharedClause(
                ImplicitShared, SourceLocation(), SourceLocation(),
                SourceLocation(), /*isImplicit=*/true)) {
          Clauses.push_back(Implicit);
          if (cast<OSSSharedClause>(Implicit)->varlist_size() != ImplicitShared.size())
            ErrorFound = true;

        } else {
          ErrorFound = true;
        }
      }

      if (!ImplicitFirstprivate.empty()) {
        if (OSSClause *Implicit = ActOnOmpSsFirstprivateClause(
                ImplicitFirstprivate, SourceLocation(), SourceLocation(),
                SourceLocation())) {
          Clauses.push_back(Implicit);
          if (cast<OSSFirstprivateClause>(Implicit)->varlist_size() != ImplicitFirstprivate.size())
            ErrorFound = true;
        } else {
          ErrorFound = true;
        }
      }
    }
  }
}

StmtResult Sema::ActOnOmpSsExecutableDirective(ArrayRef<OSSClause *> Clauses,
    OmpSsDirectiveKind Kind, Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {

  bool ErrorFound = false;

  llvm::SmallVector<OSSClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt) {
    // Check default data sharing attributes for referenced variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, AStmt);
    Stmt *S = AStmt;
    DSAChecker.Visit(S);
    if (DSAChecker.isErrorFound())
      ErrorFound = true;

    SmallVector<Expr *, 4> ImplicitShared(
        DSAChecker.getImplicitShared().begin(),
        DSAChecker.getImplicitShared().end());

    SmallVector<Expr *, 4> ImplicitFirstprivate(
        DSAChecker.getImplicitFirstprivate().begin(),
        DSAChecker.getImplicitFirstprivate().end());

    if (!ImplicitShared.empty()) {
      if (OSSClause *Implicit = ActOnOmpSsSharedClause(
              ImplicitShared, SourceLocation(), SourceLocation(),
              SourceLocation(), /*isImplicit=*/true)) {
        ClausesWithImplicit.push_back(Implicit);
        if (cast<OSSSharedClause>(Implicit)->varlist_size() != ImplicitShared.size())
          ErrorFound = true;
      } else {
        ErrorFound = true;
      }
    }

    if (!ImplicitFirstprivate.empty()) {
      if (OSSClause *Implicit = ActOnOmpSsFirstprivateClause(
              ImplicitFirstprivate, SourceLocation(), SourceLocation(),
              SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        if (cast<OSSFirstprivateClause>(Implicit)->varlist_size() != ImplicitFirstprivate.size())
          ErrorFound = true;
      } else {
        ErrorFound = true;
      }
    }
  }

  StmtResult Res = StmtError();
  switch (Kind) {
  case OSSD_taskwait:
    Res = ActOnOmpSsTaskwaitDirective(StartLoc, EndLoc);
    break;
  case OSSD_task:
    Res = ActOnOmpSsTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_unknown:
    llvm_unreachable("Unknown OmpSs directive");
  }

  if (ErrorFound)
    return StmtError();

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
  if (!AStmt)
    return StmtError();
  return OSSTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

OSSClause *
Sema::ActOnOmpSsDependClause(const SmallVector<OmpSsDependClauseKind, 2>& DepKinds, SourceLocation DepLoc,
                             SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc) {
  if (DepKinds.size() == 2) {
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
  const SmallVector<OmpSsDependClauseKind, 2>& DepKinds, SourceLocation DepLoc) {

  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_shared:
    Res = ActOnOmpSsSharedClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_private:
    Res = ActOnOmpSsPrivateClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_firstprivate:
    Res = ActOnOmpSsFirstprivateClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_depend:
    Res = ActOnOmpSsDependClause(DepKinds, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }

  return Res;
}

OSSClause *
Sema::ActOnOmpSsSimpleClause(OmpSsClauseKind Kind,
                             unsigned Argument,
                             SourceLocation ArgumentLoc,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc,
                             SourceLocation EndLoc) {
  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_default:
    Res =
    ActOnOmpSsDefaultClause(static_cast<OmpSsDefaultClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsDefaultClause(OmpSsDefaultClauseKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  switch (Kind) {
  case OSSC_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OSSC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared(KindKwLoc);
    break;
  case OSSC_DEFAULT_unknown:
    Diag(KindKwLoc, diag::err_oss_unexpected_clause_value)
        << getListOfPossibleValues(OSSC_default, /*First=*/0,
                                   /*Last=*/OSSC_DEFAULT_unknown)
        << getOmpSsClauseName(OSSC_default);
    return nullptr;
  }
  return new (Context)
      OSSDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

static ValueDecl *
getPrivateItem(Sema &S, Expr *&RefExpr, SourceLocation &ELoc,
               SourceRange &ERange) {
  if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
      RefExpr->containsUnexpandedParameterPack())
    return nullptr;

  RefExpr = RefExpr->IgnoreParens();
  ELoc = RefExpr->getExprLoc();
  ERange = RefExpr->getSourceRange();
  RefExpr = RefExpr->IgnoreParenImpCasts();
  auto *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
  auto *ME = dyn_cast_or_null<MemberExpr>(RefExpr);

  if ((!DE || !isa<VarDecl>(DE->getDecl())) &&
      (S.getCurrentThisType().isNull() || !ME ||
       !isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()) ||
       !isa<FieldDecl>(ME->getMemberDecl()))) {

    S.Diag(ELoc, diag::err_oss_expected_var_name_member_expr)
        << (S.getCurrentThisType().isNull() ? 0 : 1) << ERange;
    return nullptr;
  }

  auto *VD = dyn_cast<VarDecl>(DE->getDecl());
  QualType Type = VD->getType();
  if (!Type.isPODType(S.Context)) {
      S.Diag(ELoc, diag::err_oss_non_pod_not_supported) << ERange;
      return nullptr;
  }
  return getCanonicalDecl(VD);
}

ExprResult Sema::PerformOmpSsImplicitIntegerConversion(SourceLocation Loc,
                                                       Expr *Op) {
  if (!Op)
    return ExprError();

  class IntConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    IntConvertDiagnoser()
        : ICEConvertDiagnoser(/*AllowScopedEnumerations*/ false, false, true) {}
    SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                         QualType T) override {
      return S.Diag(Loc, diag::err_oss_not_integral) << T;
    }
    SemaDiagnosticBuilder diagnoseIncomplete(Sema &S, SourceLocation Loc,
                                             QualType T) override {
      return S.Diag(Loc, diag::err_oss_incomplete_type) << T;
    }
    SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S, SourceLocation Loc,
                                               QualType T,
                                               QualType ConvTy) override {
      return S.Diag(Loc, diag::err_oss_explicit_conversion) << T << ConvTy;
    }
    SemaDiagnosticBuilder noteExplicitConv(Sema &S, CXXConversionDecl *Conv,
                                           QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_oss_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                            QualType T) override {
      return S.Diag(Loc, diag::err_oss_ambiguous_conversion) << T;
    }
    SemaDiagnosticBuilder noteAmbiguous(Sema &S, CXXConversionDecl *Conv,
                                        QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_oss_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseConversion(Sema &, SourceLocation, QualType,
                                             QualType) override {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;
  return PerformContextualImplicitConversion(Loc, Op, ConvertDiagnoser);
}


OSSClause *
Sema::ActOnOmpSsSharedClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc,
                       bool isImplicit) {
  SmallVector<Expr *, 8> ClauseVars;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;
    // Implicit CXXThisExpr generated by the compiler are fine
    if (isImplicit && isa<CXXThisExpr>(RefExpr)) {
      ClauseVars.push_back(RefExpr);
      continue;
    }
    ValueDecl *D = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (!D) {
      continue;
    }
    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_shared &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_shared);
      continue;
    }
    DSAStack->addDSA(D, RefExpr, OSSC_shared);
    ClauseVars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSSharedClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

OSSClause *
Sema::ActOnOmpSsPrivateClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 8> ClauseVars;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;
    ValueDecl *D = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (!D) {
      continue;
    }
    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_private &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_private);
      continue;
    }
    DSAStack->addDSA(D, RefExpr, OSSC_private);
    ClauseVars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

OSSClause *
Sema::ActOnOmpSsFirstprivateClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 8> ClauseVars;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;
    ValueDecl *D = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (!D) {
      continue;
    }
    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_firstprivate &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_firstprivate);
      continue;
    }
    DSAStack->addDSA(D, RefExpr, OSSC_firstprivate);
    ClauseVars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

OSSClause *Sema::ActOnOmpSsIfClause(Expr *Condition,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = MakeFullExpr(Val.get()).get();
  }

  return new (Context) OSSIfClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsFinalClause(Expr *Condition,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  Expr *ValExpr = Condition;
  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return nullptr;

    ValExpr = MakeFullExpr(Val.get()).get();
  }

  return new (Context) OSSFinalClause(ValExpr, StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsSingleExprClause(OmpSsClauseKind Kind, Expr *Expr,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_if:
    Res = ActOnOmpSsIfClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_final:
    Res = ActOnOmpSsFinalClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}
