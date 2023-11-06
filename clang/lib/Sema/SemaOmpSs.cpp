//===--- SemaOmpSs.cpp - Semantic Analysis for OmpSs constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/AST/DeclOmpSs.h"
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
using namespace llvm::oss;

namespace {
/// Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0, /// Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// Default data sharing attribute 'none'.
  DSA_shared = 1 << 1, /// Default data sharing attribute 'shared'.
  DSA_private = 1 << 2, /// Default data sharing attribute 'private'.
  DSA_firstprivate = 1 << 3, /// Default data sharing attribute 'firstprivate'.
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
    bool IsBase = true;
    DSAVarData() = default;
    DSAVarData(OmpSsDirectiveKind DKind, OmpSsClauseKind CKind,
               const Expr *RefExpr, bool Ignore, bool IsBase)
        : DKind(DKind), CKind(CKind), RefExpr(RefExpr),
          Ignore(Ignore), IsBase(IsBase)
          {}
  };
  struct ImplicitDSAs {
    SmallVector<Expr *, 4> Shareds;
    SmallVector<Expr *, 4> Privates;
    SmallVector<Expr *, 4> Firstprivates;
  };

private:
  struct DSAInfo {
    OmpSsClauseKind Attributes = OSSC_unknown;
    const Expr * RefExpr;
    bool Ignore = false;
    bool IsBase = true;
    bool Implicit = true;
  };
  using DeclSAMapTy = llvm::MapVector<const ValueDecl *, DSAInfo>;

  // Directive
  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DefaultDataSharingAttributes DefaultAttr = DSA_unspecified;
    SourceLocation DefaultAttrLoc;
    OmpSsDirectiveKind Directive = OSSD_unknown;
    Scope *CurScope = nullptr;
    CXXThisExpr *ThisExpr = nullptr;
    SmallVector<const ValueDecl *, 2> LoopICVDecls;
    SmallVector<const Expr *, 2> LoopICVExprs;
    unsigned AssociatedLoops = 1;
    unsigned SeenAssociatedLoops = 0;
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

  using iterator = StackTy::const_reverse_iterator;

  DSAVarData getDSA(iterator &Iter, ValueDecl *D) const;

  bool isStackEmpty() const {
    return Stack.empty();
  }

public:
  explicit DSAStackTy() {}

  void push(OmpSsDirectiveKind DKind,
            Scope *CurScope, SourceLocation Loc) {
    Stack.emplace_back(DKind, CurScope, Loc);
  }

  void pop() {
    assert(!Stack.empty() && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  /// Adds explicit data sharing attribute to the specified declaration.
  void addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A,
              bool Ignore, bool IsBase, bool Implicit=true);

  void addLoopControlVariable(const ValueDecl *D, const Expr *E);

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
  void setDefault(
      DefaultDataSharingAttributes DefaultAttr, SourceLocation Loc) {
    assert(!isStackEmpty());
    Stack.back().DefaultAttr = DefaultAttr;
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
  ArrayRef<const ValueDecl *> getCurrentLoopICVDecls() const {
    ArrayRef<const ValueDecl *> Ret;
    if (!isStackEmpty())
      Ret = Stack.back().LoopICVDecls;
    return Ret;
  }

  ArrayRef<const Expr *> getCurrentLoopICVExprs() const {
    ArrayRef<const Expr *> Ret;
    if (!isStackEmpty())
      Ret = Stack.back().LoopICVExprs;
    return Ret;
  }

  void setCurrentDirective(OmpSsDirectiveKind DKind) {
    Stack.back().Directive = DKind;
  }

  /// Set collapse value for the region.
  void setAssociatedLoops(unsigned Val) {
    Stack.back().AssociatedLoops = Val;
  }
  /// Return collapse value for region.
  unsigned getAssociatedLoops() const {
    return isStackEmpty() ? 0 : Stack.back().AssociatedLoops;
  }

  /// Set collapse value for the region.
  void setSeenAssociatedLoops(unsigned Val) {
    Stack.back().SeenAssociatedLoops = Val;
  }
  /// Return collapse value for region.
  unsigned getSeenAssociatedLoops() const {
    return isStackEmpty() ? 0 : Stack.back().SeenAssociatedLoops;
  }

  // Get the current scope. This is null when instantiating templates
  // Used for Reductions
  Scope *getCurScope() const {
    return isStackEmpty() ? nullptr : Stack.back().CurScope;
  }
  ImplicitDSAs getCurImplDSAs() const {
    assert(!isStackEmpty());
    ImplicitDSAs IDSAs;
    for (const auto &p : Stack.back().SharingMap) {
      if (!p.second.Implicit)
        continue;
      switch (p.second.Attributes) {
      case OSSC_shared:
        IDSAs.Shareds.push_back(const_cast<Expr*>(p.second.RefExpr));
        break;
      case OSSC_private:
        IDSAs.Privates.push_back(const_cast<Expr*>(p.second.RefExpr));
        break;
      case OSSC_firstprivate:
        IDSAs.Firstprivates.push_back(const_cast<Expr*>(p.second.RefExpr));
        break;
      // Ignore
      case OSSC_unknown:
        break;
      default:
        llvm_unreachable("unhandled implicit DSA");
      }
    }
    if (getThisExpr())
      IDSAs.Shareds.push_back(getThisExpr());
    return IDSAs;
  }

};

} // namespace

static const Expr *getExprAsWritten(const Expr *E) {
  if (const auto *FE = dyn_cast<FullExpr>(E))
    E = FE->getSubExpr();

  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();

  while (const auto *Binder = dyn_cast<CXXBindTemporaryExpr>(E))
    E = Binder->getSubExpr();

  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  return E->IgnoreParens();
}

static Expr *getExprAsWritten(Expr *E) {
  return const_cast<Expr *>(getExprAsWritten(const_cast<const Expr *>(E)));
}

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
    DVar.IsBase = Data.IsBase;
    DVar.CKind = Data.Attributes;
    return DVar;
  }

  return DVar;
}

void DSAStackTy::addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A,
                        bool Ignore, bool IsBase, bool Implicit) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  DSAInfo &Data = Stack.back().SharingMap[D];
  Data.Attributes = A;
  Data.RefExpr = E;
  Data.Ignore = Ignore;
  Data.IsBase = IsBase;
  Data.Implicit = Implicit;
}

void DSAStackTy::addLoopControlVariable(const ValueDecl *D, const Expr *E) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  Stack.back().LoopICVDecls.push_back(D);
  Stack.back().LoopICVExprs.push_back(E);
}

const DSAStackTy::DSAVarData DSAStackTy::getTopDSA(ValueDecl *D,
                                                   bool FromParent) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return true; };
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

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return true; };
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
  llvm::SmallSet<ValueDecl *, 4> InnerDecls;

  // Walks over all array dimensions looking for VLA size Expr.
  void GetTypeDSAs(QualType T) {
    QualType TmpTy = T;
    // int (**p)[sizex][sizey] -> we need sizex sizey for vla dims
    while (TmpTy->isPointerType())
      TmpTy = TmpTy->getPointeeType();
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

  void VisitOSSMultiDepExpr(OSSMultiDepExpr *E) {
    for (auto *E : E->getDepIterators()) {
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      InnerDecls.insert(VD);
    }
  }

  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // Add DSA to 'this' if is the first time we see it
    if (!Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();

      // Variables declared inside region don't have DSA
      if (InnerDecls.count(VD))
        return;

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      DSAStackTy::DSAVarData DVarFromParent = Stack->getTopDSA(VD, /*FromParent=*/true);

      bool ExistsParent = DVarFromParent.RefExpr;

      bool ExistsCurrent = DVarCurrent.RefExpr;
      bool IsBaseCurrent = DVarCurrent.IsBase;
      bool IgnoreCurrent = DVarCurrent.Ignore;

      // DSA defined already and IsBase, so we cannot promote here.
      if (ExistsCurrent && (IsBaseCurrent || IgnoreCurrent))
        return;
      if (ExistsCurrent && !IsBaseCurrent) {
        if (DVarCurrent.CKind == OSSC_firstprivate && !VD->hasLocalStorage()) {
          // Promote Implicit firstprivate to Implicit shared
          // in decls that are global

          // Rewrite DSA
          Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/false, IsBaseCurrent);
          return;
        }
      }
      // If explicit DSA comes from parent inherit it
      if (ExistsParent) {
          switch (DVarFromParent.CKind) {
          case OSSC_shared:
            Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/false, /*IsBase=*/true);
            break;
          case OSSC_private:
          case OSSC_firstprivate:
            Stack->addDSA(VD, E, OSSC_firstprivate, /*Ignore=*/false, /*IsBase=*/true);
            break;
          default:
            llvm_unreachable("unexpected DSA from parent");
          }
      } else {

        switch (Stack->getCurrentDefaultDataSharingAttributtes()) {
        case DSA_shared:
          // Record DSA as Ignored to avoid making the same node again
          Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/false, /*IsBase=*/true);
          break;
        case DSA_private:
          // Record DSA as Ignored to avoid making the same node again
          Stack->addDSA(VD, E, OSSC_private, /*Ignore=*/false, /*IsBase=*/true);
          break;
        case DSA_firstprivate:
          // Record DSA as Ignored to avoid making the same node again
          Stack->addDSA(VD, E, OSSC_firstprivate, /*Ignore=*/false, /*IsBase=*/true);
          break;
        case DSA_none:
          SemaRef.Diag(E->getExprLoc(), diag::err_oss_not_defined_dsa_when_default_none) << E->getDecl();
          // Record DSA as ignored to diagnostic only once
          Stack->addDSA(VD, E, OSSC_unknown, /*Ignore=*/true, /*IsBase=*/true);
          break;
        case DSA_unspecified:
          if (VD->hasLocalStorage()) {
            // If no default clause is present and the variable was private/local
            // in the context encountering the construct, the variable will
            // be firstprivate

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_firstprivate, /*Ignore=*/false, /*IsBase=*/true);
          } else {
            // If no default clause is present and the variable was shared/global
            // in the context encountering the construct, the variable will be shared.

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/false, /*IsBase=*/true);
          }
        }
      }
    }
  }

  void VisitLambdaExpr(LambdaExpr *LE) {
    CXXMethodDecl *MD = LE->getCallOperator();
    for (ParmVarDecl *param : MD->parameters()) {
      InnerDecls.insert(param);
    }
    for (Expr *E : LE->capture_inits())
      Visit(E);
  }

  void VisitCXXCatchStmt(CXXCatchStmt *Node) {
    InnerDecls.insert(Node->getExceptionDecl());
    Visit(Node->getHandlerBlock());
  }

  void VisitOSSClause(OSSClause *Clause) {
    for (Stmt *C : Clause->children()) {
      Visit(C);
    }
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C) {
        Visit(C);
        if (auto *OSSStmt = dyn_cast<OSSExecutableDirective>(C)) {
          // Visit clauses too.
          for (OSSClause *Clause : OSSStmt->clauses()) {
            VisitOSSClause(Clause);
          }
        }
      }
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

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false) {}
};

// OSSClauseDSAChecker gathers for each expression in a clause
// all implicit data-sharings.
//
// To do so, we classify as firstprivate the base symbol if it's a pointer and is
// dereferenced by a SubscriptExpr, MemberExpr or UnaryOperator.
// Otherwise it's shared.
//
// At the same time, all symbols found inside a SubscriptExpr will be firstprivate.
// NOTE: implicit DSA from other tasks are ignored
class OSSClauseDSAChecker final : public StmtVisitor<OSSClauseDSAChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  OmpSsClauseKind CKind;
  bool ErrorFound = false;
  // Map to record that one variable has been used in a reduction/dependency.
  // Strong restriction (true) is when the variable is the symbol reduced:
  //
  // i.e.
  //     reduction(+ : x)
  // Weak restriction (false) are the other case:
  //
  // i.e. looking at 'x'
  //     reduction(+ : [x]p)
  //     in([x]p)
  //     in(x)
  llvm::DenseMap<const VarDecl *, bool> SeenStrongRestric;
  // We visit the base of all the expressions, so the first
  // decl value is the dep base.
  bool IsFirstDecl = true;
  // This is used to distinguish
  // p → shared
  // from
  // *p, p[0], ps->x → firstprivate
  bool IsPlainDeclRef = true;

public:

  void VisitOSSMultiDepExpr(OSSMultiDepExpr *E) {
    IsPlainDeclRef = false;

    if (Stack) {
      // Ignore iterators
      for (auto *E : E->getDepIterators()) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
        Stack->addDSA(VD, E, OSSC_private, /*Ignore=*/true, /*IsBase=*/true);
      }
    }

    Visit(E->getDepExpr());

    for (size_t i = 0; i < E->getDepInits().size(); ++i) {
      Visit(E->getDepInits()[i]);
      if (E->getDepSizes()[i])
        Visit(E->getDepSizes()[i]);
      if (E->getDepSteps()[i])
        Visit(E->getDepSteps()[i]);
    }
  }

  void VisitOSSArrayShapingExpr(OSSArrayShapingExpr *E) {
    IsPlainDeclRef = false;
    Visit(E->getBase());

    for (Expr *S : E->getShapes())
      Visit(S);
  }

  void VisitOSSArraySectionExpr(OSSArraySectionExpr *E) {
    IsPlainDeclRef = false;
    Visit(E->getBase());

    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
  }

  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    IsPlainDeclRef = false;
    Visit(E->getBase());
    Visit(E->getIdx());
  }

  void VisitUnaryOperator(UnaryOperator *E) {
    IsPlainDeclRef = false;
    Visit(E->getSubExpr());
  }

  void VisitMemberExpr(MemberExpr *E) {
    IsPlainDeclRef = false;
    Visit(E->getBase());
  }

  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // The dep base may use this:
    // out(this->array[i])
    IsFirstDecl = false;

    // Add DSA to 'this' if is the first time we see it
    if (Stack && !Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();

      SourceLocation ELoc = E->getExprLoc();
      SourceRange ERange = E->getSourceRange();

      // inout(x)              | shared(x)        | int x;
      // inout(p[i])           | firstprivate(p)  | int *p;
      // inout(a[i])           | shared(a)        | int a[N];
      // inout(*p)/inout(p[0]) | firstprivate(p)  | int *p;
      // inout(s.x)            | shared(s)        | struct S s;
      // inout(ps->x)          | firstprivate(ps) | struct S *ps;
      // inout([1]p)           | firstprivate(p)  | int *p;
      OmpSsClauseKind VKind = OSSC_shared;

      bool IsBaseDeduced = true;

      bool CurIsFirstDecl = IsFirstDecl;
      IsFirstDecl = false;
      if (CurIsFirstDecl) {
        if (VD->getType()->isPointerType() && !IsPlainDeclRef)
          VKind = OSSC_firstprivate;
      } else {
        VKind = OSSC_firstprivate;
        IsBaseDeduced = false;
      }

      // We have to emit a diagnostic if a strong restriction
      // is used in other places.
      bool IsReduction = false;
      if (CKind == OSSC_reduction
          || CKind == OSSC_weakreduction)
        IsReduction = true;

      bool CurIsStrongRestric = IsReduction && CurIsFirstDecl;
      auto It = SeenStrongRestric.find(VD);
      if (It != SeenStrongRestric.end()) {
        // if (Seen strong red || Seen weak and now we're strong)
        if (It->second || CurIsStrongRestric) {
          ErrorFound = true;
          SemaRef.Diag(ELoc, diag::err_oss_reduction_depend_conflict)
            << E->getDecl();
        }
      }
      SeenStrongRestric[VD] = SeenStrongRestric[VD] || CurIsStrongRestric;

      if (!Stack)
        return;

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      bool ExistsCurrent = DVarCurrent.RefExpr;
      if (!ExistsCurrent) {
        // No DSA in current directive, give assign one and done
        Stack->addDSA(VD, E, VKind, /*Ignore=*/false, IsBaseDeduced);
        return;
      }
      bool IgnoreCurrent = DVarCurrent.Ignore;
      if (IgnoreCurrent) {
        return;
      }

      bool IsBaseCurrent = DVarCurrent.IsBase;
      if (IsBaseDeduced && IsBaseCurrent
          && VKind != DVarCurrent.CKind) {
        // The DSA of the current clause and the one deduced here
        // are base, but don't match
        SemaRef.Diag(ELoc, diag::err_oss_mismatch_depend_dsa)
          << getOmpSsClauseName(DVarCurrent.CKind)
          << getOmpSsClauseName(VKind) << ERange;
      } else if (VKind == OSSC_shared && DVarCurrent.CKind == OSSC_firstprivate) {
        // Promotion
        Stack->addDSA(VD, E, VKind, /*Ignore=*/false, IsBaseDeduced || IsBaseCurrent);
      }
    }
  }

  void VisitClause(OSSClause *Clause) {
    for (Stmt *Child : Clause->children()) {
      reset();
      CKind = Clause->getClauseKind();
      if (Child)
        Visit(Child);
    }
  }

  void VisitClauseExpr(Expr *E, OmpSsClauseKind CKind) {
    reset();
    this->CKind = CKind;
    if (E)
      Visit(E);
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  void reset() {
    CKind = OSSC_unknown;
    IsFirstDecl = true;
    IsPlainDeclRef = true;
  }

  bool isErrorFound() const { return ErrorFound; }

  OSSClauseDSAChecker(DSAStackTy *S, Sema &SemaRef)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false) {}

};

} // namespace

static VarDecl *buildVarDecl(Sema &SemaRef, SourceLocation Loc, QualType Type,
                             StringRef Name, const AttrVec *Attrs = nullptr) {
  DeclContext *DC = SemaRef.CurContext;
  IdentifierInfo *II = &SemaRef.PP.getIdentifierTable().get(Name);
  TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(Type, Loc);
  auto *Decl =
      VarDecl::Create(SemaRef.Context, DC, Loc, Loc, II, Type, TInfo, SC_None);
  if (Attrs) {
    for (specific_attr_iterator<AlignedAttr> I(Attrs->begin()), E(Attrs->end());
         I != E; ++I)
      Decl->addAttr(*I);
  }
  Decl->setImplicit();
  return Decl;
}

static DeclRefExpr *buildDeclRefExpr(Sema &S, VarDecl *D, QualType Ty,
                                     SourceLocation Loc,
                                     bool RefersToCapture = false) {
  D->setReferenced();
  D->markUsed(S.Context);
  return DeclRefExpr::Create(S.getASTContext(), NestedNameSpecifierLoc(),
                             SourceLocation(), D, RefersToCapture, Loc, Ty,
                             VK_LValue);
}

void Sema::InitDataSharingAttributesStackOmpSs() {
  VarDataSharingAttributesStackOmpSs = new DSAStackTy();
  // TODO: use another function
  AllowShapings = false;
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStackOmpSs)

void Sema::DestroyDataSharingAttributesStackOmpSs() { delete DSAStack; }

OmpSsDirectiveKind Sema::GetCurrentOmpSsDirective() const {
  return DSAStack->getCurrentDirective();
}

void Sema::SetTaskiterKind(OmpSsDirectiveKind DKind) {
  DSAStack->setCurrentDirective(DKind);
}

bool Sema::IsEndOfTaskloop() const {
  unsigned AssociatedLoops = DSAStack->getAssociatedLoops();
  unsigned SeenAssociatedLoops = DSAStack->getSeenAssociatedLoops();
  return AssociatedLoops == SeenAssociatedLoops;
}

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
                        ArrayRef<unsigned> Exclude = std::nullopt) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  unsigned Skipped = Exclude.size();
  auto S = Exclude.begin(), E = Exclude.end();
  for (unsigned I = First; I < Last; ++I) {
    if (std::find(S, E, I) != E) {
      --Skipped;
      continue;
    }
    Out << "'" << getOmpSsSimpleClauseTypeName(K, I) << "'";
    if (I + Skipped + 2 == Last)
      Out << " or ";
    else if (I + Skipped + 1 != Last)
      Out << ", ";
  }
  return std::string(Out.str());
}

void Sema::ActOnOmpSsAfterClauseGathering(SmallVectorImpl<OSSClause *>& Clauses) {

  if (CurContext->isDependentContext())
    return;

  bool ErrorFound = false;
  (void)ErrorFound;

  OSSClauseDSAChecker OSSClauseChecker(DSAStack, *this);
  for (auto *Clause : Clauses) {
    if (isa<OSSDependClause>(Clause) || isa<OSSReductionClause>(Clause)) {
      OSSClauseChecker.VisitClause(Clause);
    }
    // FIXME: how to handle an error?
    if (OSSClauseChecker.isErrorFound())
      ErrorFound = true;
  }

  // TODO: Can we just put this in the previous
  // loop over clauses?
  // I think no, here is an example:
  //   void foo(int a) {
  //     #pragma oss task cost(a) in(a)
  //   }
  // Here we expect 'a' to be shared because of the dependency
  // but as we find cost before we register firstprivate
  for (auto *Clause : Clauses) {
    if (isa<OSSCostClause>(Clause) || isa<OSSPriorityClause>(Clause)
        || isa<OSSOnreadyClause>(Clause)) {
      DSAAttrChecker DSAChecker(DSAStack, *this);
      DSAChecker.VisitOSSClause(Clause);
      // FIXME: how to handle an error?
      if (DSAChecker.isErrorFound())
        ErrorFound = true;
    }
  }

}

Expr *Sema::ActOnOmpSsMultiDepIterator(Scope *S, StringRef Name, SourceLocation Loc) {
  VarDecl *MultiDepItDecl =
      buildVarDecl(*this, Loc, Context.IntTy, Name);
  OSSCheckShadow(S, MultiDepItDecl);
  // TODO: what about templates?
  if (S) {
    PushOnScopeChains(MultiDepItDecl, S);
  }
  Expr *MultiDepItE = buildDeclRefExpr(*this, MultiDepItDecl, Context.IntTy, Loc);
  return MultiDepItE;
}

ExprResult Sema::ActOnOmpSsMultiDepIteratorInitListExpr(InitListExpr *InitList) {
  if (InitList->getNumInits() == 0) {
      Diag(InitList->getBeginLoc(), diag::err_oss_multidep_discrete_empty);
      return ExprError();
  }
  // int [] = InitList
  unsigned NumInits = InitList->getNumInits();
  ExprResult Res = ActOnIntegerConstant(SourceLocation(), NumInits);

  QualType Ty = BuildArrayType(Context.IntTy, ArraySizeModifier::Normal, Res.get(), /*Quals=*/0,
                        SourceRange(), DeclarationName());
  VarDecl *DiscreteArrayDecl =
      buildVarDecl(*this, SourceLocation(), Ty, "discrete.array");
  AddInitializerToDecl(DiscreteArrayDecl, InitList, /*DirectInit=*/false);

  Expr *DiscreteArrayE = buildDeclRefExpr(*this, DiscreteArrayDecl, Ty, SourceLocation());

  if (DiscreteArrayDecl->hasInit())
    return DiscreteArrayE;
  return ExprError();
}

ExprResult Sema::ActOnOSSMultiDepExpression(
    SourceLocation Loc, SourceLocation RLoc, ArrayRef<Expr *> MultiDepIterators,
    ArrayRef<Expr *> MultiDepInits, ArrayRef<Expr *> MultiDepSizes,
    ArrayRef<Expr *> MultiDepSteps, ArrayRef<bool> MultiDepSizeOrSection,
    Expr *DepExpr) {
  assert((MultiDepIterators.size() == MultiDepInits.size() &&
          MultiDepIterators.size() == MultiDepSizes.size() &&
          MultiDepIterators.size() == MultiDepSteps.size() &&
          MultiDepIterators.size() == MultiDepSizeOrSection.size())
         && "Multidep info lists do not have the same size");

  bool IsDependent = DepExpr &&
    (DepExpr->isTypeDependent() || DepExpr->isValueDependent());
  bool IsError = !DepExpr;
  for (size_t i = 0; i < MultiDepIterators.size(); ++i) {
    if (MultiDepInits[i]) {
      if (!isa<InitListExpr>(MultiDepInits[i]) && !MultiDepSizes[i]) {
        IsError = true;
      }
    } else {
      IsError = true;
    }
    IsDependent = IsDependent ||
      (MultiDepInits[i] && (MultiDepInits[i]->isTypeDependent()
       || MultiDepInits[i]->isValueDependent()));
    IsDependent = IsDependent ||
      (MultiDepSizes[i] && (MultiDepSizes[i]->isTypeDependent()
       || MultiDepSizes[i]->isValueDependent()));
    IsDependent = IsDependent ||
      (MultiDepSteps[i] && (MultiDepSteps[i]->isTypeDependent()
       || MultiDepSteps[i]->isValueDependent()));
  }

  SmallVector<Expr *, 4> Inits;
  SmallVector<Expr *, 4> Sizes;
  SmallVector<Expr *, 4> Steps;
  SmallVector<Expr *, 4> DiscreteArrays;

  // This errors are from Parser, which should have emitted
  // some diagnostics
  if (IsError)
    return ExprError();

  if (IsDependent)
    // Analyze later
    return OSSMultiDepExpr::Create(
      Context,
      Context.DependentTy, VK_LValue, OK_Ordinary, DepExpr,
      MultiDepIterators, MultiDepInits, MultiDepSizes,
      MultiDepSteps, DiscreteArrays, MultiDepSizeOrSection, Loc, RLoc);

  // This class emits an error if the iterator
  // is used in the init/size/step
  // { v[i], i=i;i:i }
  class MultidepIterUseChecker final
      : public ConstStmtVisitor<MultidepIterUseChecker, bool> {
  public:
    enum Type {
      LBound = 0,
      Size,
      UBound,
      Step
    };
  private:
    Sema &SemaRef;
    const VarDecl *IterVD;
    enum Type Ty;
    bool checkDecl(const Expr *E, const ValueDecl *VD) {
      if (getCanonicalDecl(VD) == getCanonicalDecl(IterVD)) {
        SemaRef.Diag(E->getExprLoc(), diag::err_oss_multidep_iterator_scope)
          << Ty;
        return true;
      }
      return false;
    }

  public:
    bool VisitDeclRefExpr(const DeclRefExpr *E) {
      const ValueDecl *VD = E->getDecl();
      if (isa<VarDecl>(VD))
        return checkDecl(E, VD);
      return false;
    }
    bool VisitStmt(const Stmt *S) {
      bool Res = false;
      for (const Stmt *Child : S->children())
        Res = (Child && Visit(Child)) || Res;
      return Res;
    }
    explicit MultidepIterUseChecker(
      Sema &SemaRef, const VarDecl *IterVD, enum Type Ty)
        : SemaRef(SemaRef), IterVD(IterVD), Ty(Ty)
          {}
  };

  for (size_t i = 0; i < MultiDepIterators.size(); ++i) {
    Expr *ItE = MultiDepIterators[i];
    VarDecl *ItVD = cast<VarDecl>(cast<DeclRefExpr>(ItE)->getDecl());

    Expr *InitExpr = MultiDepInits[i];
    if (MultidepIterUseChecker(
        *this, ItVD, MultidepIterUseChecker::Type::LBound).Visit(InitExpr))
      IsError = true;
    Expr *DiscreteArrExpr = nullptr;
    if (InitListExpr *InitList = dyn_cast<InitListExpr>(InitExpr)) {
      // Initialize the iterator to a valid number so it can
      // be used to index the discrete array.
      AddInitializerToDecl(
          ItVD, ActOnIntegerConstant(SourceLocation(), 0).get(), /*DirectInit=*/false);
      ExprResult Res = ActOnOmpSsMultiDepIteratorInitListExpr(InitList);
      if (Res.isInvalid())
        IsError = true;
      DiscreteArrExpr = Res.get();
    } else {
      AddInitializerToDecl(ItVD, InitExpr, /*DirectInit=*/false);
      InitExpr = ItVD->getInit();
      if (!InitExpr)
        IsError = true;
    }
    Inits.push_back(InitExpr);
    DiscreteArrays.push_back(DiscreteArrExpr);

    Expr *SizeExpr = MultiDepSizes[i];
    if (SizeExpr) {
      if (MultidepIterUseChecker(*this, ItVD,
          MultiDepSizeOrSection[i] ? MultidepIterUseChecker::Type::Size
            : MultidepIterUseChecker::Type::UBound).Visit(SizeExpr))
        IsError = true;
      ExprResult Res = PerformImplicitConversion(SizeExpr, Context.IntTy, AA_Converting);
      if (Res.isInvalid()) {
        IsError = true;
      } else {
        SizeExpr = Res.get();
      }
    }
    Sizes.push_back(SizeExpr);

    Expr *StepExpr = MultiDepSteps[i];
    if (StepExpr) {
      if (MultidepIterUseChecker(
          *this, ItVD, MultidepIterUseChecker::Type::Step).Visit(StepExpr))
        IsError = true;
      ExprResult Res = PerformImplicitConversion(StepExpr, Context.IntTy, AA_Converting);
      if (Res.isInvalid()) {
        IsError = true;
      } else {
        StepExpr = Res.get();
      }
    }
    Steps.push_back(StepExpr);
  }

  if (IsError)
    return ExprError();

  return OSSMultiDepExpr::Create(
    Context,
    DepExpr->getType(), VK_LValue, OK_Ordinary, DepExpr,
    MultiDepIterators, Inits, Sizes, Steps, DiscreteArrays,
    MultiDepSizeOrSection, Loc, RLoc);
}

Sema::DeclGroupPtrTy Sema::ActOnOmpSsAssertDirective(SourceLocation Loc, Expr *E) {
  OSSAssertDecl *D = nullptr;
  if (!CurContext->isFileContext()) {
    Diag(Loc, diag::err_oss_invalid_scope) << "assert";
  } else {
    D = OSSAssertDecl::Create(
      Context, getCurLexicalContext(), Loc, {E});
    CurContext->addDecl(D);
  }
  return DeclGroupPtrTy::make(DeclGroupRef(D));
}

QualType Sema::ActOnOmpSsDeclareReductionType(SourceLocation TyLoc,
                                              TypeResult ParsedType) {
  assert(ParsedType.isUsable());

  QualType ReductionType = GetTypeFromParser(ParsedType.get());
  if (ReductionType.isNull())
    return QualType();

  // [OpenMP 4.0], 2.15 declare reduction Directive, Restrictions, C\C++
  // A type name in a declare reduction directive cannot be a function type, an
  // array type, a reference type, or a type qualified with const, volatile or
  // restrict.
  if (ReductionType.hasQualifiers()) {
    Diag(TyLoc, diag::err_oss_reduction_wrong_type) << 0;
    return QualType();
  }

  if (ReductionType->isFunctionType()) {
    Diag(TyLoc, diag::err_oss_reduction_wrong_type) << 1;
    return QualType();
  }
  if (ReductionType->isReferenceType()) {
    Diag(TyLoc, diag::err_oss_reduction_wrong_type) << 2;
    return QualType();
  }
  if (ReductionType->isArrayType()) {
    Diag(TyLoc, diag::err_oss_reduction_wrong_type) << 3;
    return QualType();
  }
  // [OmpSs] cannot be a POD, but here we cannot do the check.
  // Example
  //
  // template <typename T> struct A; // incomplete
  // #pragma omp declare reduction(foo : A<int>)
  // template <typename T> struct A { }; // from here complete
  return ReductionType;
}

Sema::DeclGroupPtrTy Sema::ActOnOmpSsDeclareReductionDirectiveStart(
    Scope *S, DeclContext *DC, DeclarationName Name,
    ArrayRef<std::pair<QualType, SourceLocation>> ReductionTypes,
    AccessSpecifier AS, Decl *PrevDeclInScope) {
  SmallVector<Decl *, 8> Decls;
  Decls.reserve(ReductionTypes.size());

  LookupResult Lookup(*this, Name, SourceLocation(), LookupOSSReductionName,
                      forRedeclarationInCurContext());
  // [OpenMP 4.0], 2.15 declare reduction Directive, Restrictions
  // A reduction-identifier may not be re-declared in the current scope for the
  // same type or for a type that is compatible according to the base language
  // rules.
  llvm::DenseMap<QualType, SourceLocation> PreviousRedeclTypes;
  OSSDeclareReductionDecl *PrevDRD = nullptr;
  bool InCompoundScope = true;
  // S == nullptr for templates
  // and PrevDeclInScope is the Decl without instantiate, if any
  if (S) {
    // Find previous declaration with the same name not referenced in other
    // declarations.
    FunctionScopeInfo *ParentFn = getEnclosingFunction();
    InCompoundScope = ParentFn && !ParentFn->CompoundScopes.empty();
    LookupName(Lookup, S);
    FilterLookupForScope(Lookup, DC, S, /*ConsiderLinkage=*/false,
                         /*AllowInlineNamespace=*/false);
    llvm::DenseMap<OSSDeclareReductionDecl *, bool> UsedAsPrevious;
    LookupResult::Filter Filter = Lookup.makeFilter();
    while (Filter.hasNext()) {
      auto *PrevDecl = cast<OSSDeclareReductionDecl>(Filter.next());
      if (InCompoundScope) {
        auto I = UsedAsPrevious.find(PrevDecl);
        // Build the Decl previous chain
        // NOTE: Is this used because we do not trust Filter order?
        // Example:
        // declare reduction -> int, char
        // declare reduction -> char (Current)
        // This is translated in three separated decls
        // int <- char <- char (Current)
        // We may find the 'int' version before than the char version
        // This ensures we will build the chain: int <- char <- char and
        // not char <- int <- char
        if (I == UsedAsPrevious.end())
          UsedAsPrevious[PrevDecl] = false;
        if (OSSDeclareReductionDecl *D = PrevDecl->getPrevDeclInScope())
          UsedAsPrevious[D] = true;
      }
      // Record types of previous declare reductions with that name
      PreviousRedeclTypes[PrevDecl->getType().getCanonicalType()] =
          PrevDecl->getLocation();
    }
    Filter.done();
    if (InCompoundScope) {
      for (const auto &PrevData : UsedAsPrevious) {
        if (!PrevData.second) {
          PrevDRD = PrevData.first;
          break;
        }
      }
    }
  } else if (PrevDeclInScope) {
    // Since we have only the immediate previous decl, loop over all
    // previous decls
    auto *PrevDRDInScope = PrevDRD =
        cast<OSSDeclareReductionDecl>(PrevDeclInScope);
    do {
      PreviousRedeclTypes[PrevDRDInScope->getType().getCanonicalType()] =
          PrevDRDInScope->getLocation();
      PrevDRDInScope = PrevDRDInScope->getPrevDeclInScope();
    } while (PrevDRDInScope);
  }
  for (const auto &TyData : ReductionTypes) {
    const auto I = PreviousRedeclTypes.find(TyData.first.getCanonicalType());
    bool Invalid = false;
    // Check for every type of the current declare reduction if there is
    // a previous declaration of it
    if (I != PreviousRedeclTypes.end()) {
      Diag(TyData.second, diag::err_oss_declare_reduction_redefinition)
          << TyData.first;
      Diag(I->second, diag::note_previous_definition);
      Invalid = true;
    }
    PreviousRedeclTypes[TyData.first.getCanonicalType()] = TyData.second;
    // Create an OSSDeclareReductionDecl for each type and set previous
    // declare to the one created before
    auto *DRD = OSSDeclareReductionDecl::Create(Context, DC, TyData.second,
                                                Name, TyData.first, PrevDRD);
    DC->addDecl(DRD);
    DRD->setAccess(AS);
    Decls.push_back(DRD);
    if (Invalid)
      DRD->setInvalidDecl();
    else
      PrevDRD = DRD;
  }

  return DeclGroupPtrTy::make(
      DeclGroupRef::Create(Context, Decls.begin(), Decls.size()));
}

void Sema::ActOnOmpSsDeclareReductionCombinerStart(Scope *S, Decl *D) {
  auto *DRD = cast<OSSDeclareReductionDecl>(D);

  // Enter new function scope.
  PushFunctionScope();
  setFunctionHasBranchProtectedScope();
  getCurFunction()->setHasOSSDeclareReductionCombiner();

  if (S)
    PushDeclContext(S, DRD);
  else // Template instantiation
    CurContext = DRD;

  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  QualType ReductionType = DRD->getType();
  // Create 'T* omp_parm;T omp_in;'. All references to 'omp_in' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_in'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_in;' variable.
  VarDecl *OmpInParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_in");
  // Create 'T* omp_parm;T omp_out;'. All references to 'omp_out' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_out'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_out;' variable.
  VarDecl *OmpOutParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_out");
  if (S) {
    PushOnScopeChains(OmpInParm, S);
    PushOnScopeChains(OmpOutParm, S);
  } else {
    DRD->addDecl(OmpInParm);
    DRD->addDecl(OmpOutParm);
  }
  Expr *InE =
      ::buildDeclRefExpr(*this, OmpInParm, ReductionType, D->getLocation());
  Expr *OutE =
      ::buildDeclRefExpr(*this, OmpOutParm, ReductionType, D->getLocation());
  DRD->setCombinerData(InE, OutE);
}

void Sema::ActOnOmpSsDeclareReductionCombinerEnd(Decl *D, Expr *Combiner) {
  auto *DRD = cast<OSSDeclareReductionDecl>(D);
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  PopDeclContext();
  PopFunctionScopeInfo();

  if (Combiner)
    DRD->setCombiner(Combiner);
  else
    DRD->setInvalidDecl();
}

VarDecl *Sema::ActOnOmpSsDeclareReductionInitializerStart(Scope *S, Decl *D) {
  auto *DRD = cast<OSSDeclareReductionDecl>(D);

  // Enter new function scope.
  PushFunctionScope();
  setFunctionHasBranchProtectedScope();

  if (S)
    PushDeclContext(S, DRD);
  else // Template instantiation
    CurContext = DRD;

  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  QualType ReductionType = DRD->getType();
  // Create 'T* omp_parm;T omp_priv;'. All references to 'omp_priv' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_priv'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_priv;' variable.
  VarDecl *OmpPrivParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_priv");
  // Create 'T* omp_parm;T omp_orig;'. All references to 'omp_orig' will
  // be replaced by '*omp_parm' during codegen. This required because 'omp_orig'
  // uses semantics of argument handles by value, but it should be passed by
  // reference. C lang does not support references, so pass all parameters as
  // pointers.
  // Create 'T omp_orig;' variable.
  VarDecl *OmpOrigParm =
      buildVarDecl(*this, D->getLocation(), ReductionType, "omp_orig");
  if (S) {
    PushOnScopeChains(OmpPrivParm, S);
    PushOnScopeChains(OmpOrigParm, S);
  } else {
    DRD->addDecl(OmpPrivParm);
    DRD->addDecl(OmpOrigParm);
  }
  Expr *OrigE =
      ::buildDeclRefExpr(*this, OmpOrigParm, ReductionType, D->getLocation());
  Expr *PrivE =
      ::buildDeclRefExpr(*this, OmpPrivParm, ReductionType, D->getLocation());
  DRD->setInitializerData(OrigE, PrivE);
  return OmpPrivParm;
}

void Sema::ActOnOmpSsDeclareReductionInitializerEnd(Decl *D, Expr *Initializer,
                                                    VarDecl *OmpPrivParm) {
  auto *DRD = cast<OSSDeclareReductionDecl>(D);
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  PopDeclContext();
  PopFunctionScopeInfo();

  if (Initializer) {
    DRD->setInitializer(Initializer, OSSDeclareReductionDecl::CallInit);
  } else if (OmpPrivParm->hasInit()) {
    DRD->setInitializer(OmpPrivParm->getInit(),
                        OmpPrivParm->isDirectInit()
                            ? OSSDeclareReductionDecl::DirectInit
                            : OSSDeclareReductionDecl::CopyInit);
  } else {
    DRD->setInvalidDecl();
  }
}

Sema::DeclGroupPtrTy Sema::ActOnOmpSsDeclareReductionDirectiveEnd(
    Scope *S, DeclGroupPtrTy DeclReductions, bool IsValid) {
  for (Decl *D : DeclReductions.get()) {
    if (IsValid) {
      if (S)
        PushOnScopeChains(cast<OSSDeclareReductionDecl>(D), S,
                          /*AddToContext=*/false);
    } else {
      D->setInvalidDecl();
    }
  }
  return DeclReductions;
}

void Sema::ActOnOmpSsExecutableDirectiveStart() {
  // Enter new function scope.
  PushFunctionScope();
  setFunctionHasBranchProtectedScope();
  getCurFunction()->setHasOSSExecutableDirective();
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);
}

void Sema::ActOnOmpSsExecutableDirectiveEnd() {
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  PopFunctionScopeInfo();
}

StmtResult Sema::ActOnOmpSsExecutableDirective(ArrayRef<OSSClause *> Clauses,
    const DeclarationNameInfo &DirName, OmpSsDirectiveKind Kind, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {

  bool ErrorFound = false;

  llvm::SmallVector<OSSClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt && !CurContext->isDependentContext() &&
      Kind != OSSD_critical && Kind != OSSD_atomic) {
    // Check default data sharing attributes for referenced variables.
    DSAAttrChecker DSAChecker(DSAStack, *this);

    DSAChecker.Visit(AStmt);
    if (DSAChecker.isErrorFound())
      ErrorFound = true;

  }

  DSAStackTy::ImplicitDSAs IDSAs = DSAStack->getCurImplDSAs();

  if (!IDSAs.Shareds.empty()) {
    if (OSSClause *Implicit = ActOnOmpSsSharedClause(
            IDSAs.Shareds, SourceLocation(), SourceLocation(),
            SourceLocation(), /*isImplicit=*/true)) {
      ClausesWithImplicit.push_back(Implicit);
      if (cast<OSSSharedClause>(Implicit)->varlist_size() != IDSAs.Shareds.size())
        ErrorFound = true;
    } else {
      ErrorFound = true;
    }
  }

  if (!IDSAs.Privates.empty()) {
    if (OSSClause *Implicit = ActOnOmpSsPrivateClause(
            IDSAs.Privates, SourceLocation(), SourceLocation(),
            SourceLocation())) {
      ClausesWithImplicit.push_back(Implicit);
      if (cast<OSSPrivateClause>(Implicit)->varlist_size() != IDSAs.Privates.size())
        ErrorFound = true;
    } else {
      ErrorFound = true;
    }
  }

  if (!IDSAs.Firstprivates.empty()) {
    if (OSSClause *Implicit = ActOnOmpSsFirstprivateClause(
            IDSAs.Firstprivates, SourceLocation(), SourceLocation(),
            SourceLocation())) {
      ClausesWithImplicit.push_back(Implicit);
      if (cast<OSSFirstprivateClause>(Implicit)->varlist_size() != IDSAs.Firstprivates.size())
        ErrorFound = true;
    } else {
      ErrorFound = true;
    }
  }

  StmtResult Res = StmtError();
  switch (Kind) {
  case OSSD_taskwait:
    Res = ActOnOmpSsTaskwaitDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OSSD_release:
    Res = ActOnOmpSsReleaseDirective(ClausesWithImplicit, StartLoc, EndLoc);
    break;
  case OSSD_task:
    Res = ActOnOmpSsTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_critical:
    Res = ActOnOmpSsCriticalDirective(DirName, ClausesWithImplicit, AStmt,
                                      StartLoc, EndLoc);
    break;
  case OSSD_task_for:
    Res = ActOnOmpSsTaskForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_taskiter:
  case OSSD_taskiter_while:
    Res = ActOnOmpSsTaskIterDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_taskloop:
    Res = ActOnOmpSsTaskLoopDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_taskloop_for:
    Res = ActOnOmpSsTaskLoopForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_atomic:
    Res = ActOnOmpSsAtomicDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_declare_task:
  case OSSD_declare_reduction:
  case OSSD_assert:
  case OSSD_unknown:
    llvm_unreachable("Unknown OmpSs directive");
  }

  ErrorFound = ErrorFound || Res.isInvalid();

  if (ErrorFound)
    return StmtError();

  return Res;
}

StmtResult Sema::ActOnOmpSsTaskwaitDirective(ArrayRef<OSSClause *> Clauses,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  return OSSTaskwaitDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOmpSsReleaseDirective(ArrayRef<OSSClause *> Clauses,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  return OSSReleaseDirective::Create(Context, StartLoc, EndLoc, Clauses);
}

StmtResult Sema::ActOnOmpSsTaskDirective(ArrayRef<OSSClause *> Clauses,
                                         Stmt *AStmt,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();
  setFunctionHasBranchProtectedScope();
  return OSSTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

StmtResult Sema::ActOnOmpSsCriticalDirective(
    const DeclarationNameInfo &DirName, ArrayRef<OSSClause *> Clauses,
    Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();
  setFunctionHasBranchProtectedScope();
  return OSSCriticalDirective::Create(Context, DirName, StartLoc, EndLoc,
                                      Clauses, AStmt);
}

namespace {
// NOTE: Port from OpenMP
/// Helper class for checking canonical form of the OmpSs loops and
/// extracting iteration space of each loop in the loop nest, that will be used
/// for IR generation.
class OmpSsIterationSpaceChecker {
  /// Reference to Sema.
  Sema &SemaRef;
  /// A location for diagnostics (when there is no some better location).
  SourceLocation DefaultLoc;
  /// A location for diagnostics (when increment is not compatible).
  SourceLocation ConditionLoc;
  /// A source location for referring to loop init later.
  SourceRange InitSrcRange;
  /// A source location for referring to condition later.
  SourceRange ConditionSrcRange;
  /// A source location for referring to increment later.
  SourceRange IncrementSrcRange;
  /// Loop variable.
  ValueDecl *LCDecl = nullptr;
  /// Reference to loop variable.
  Expr *LCRef = nullptr;
  /// Lower bound (initializer for the var).
  Expr *LB = nullptr;
  /// Upper bound.
  Expr *UB = nullptr;
  /// Loop step (increment).
  Expr *Step = nullptr;
  /// This flag is true when condition is one of:
  ///   Var <  UB
  ///   Var <= UB
  ///   UB  >  Var
  ///   UB  >= Var
  /// This will have no value when the condition is !=
  std::optional<bool> TestIsLessOp;
  /// This flag is true when condition is strict ( < or > ).
  bool TestIsStrictOp = false;
  /// Checks if the provide statement depends on the loop counter.
  // true if depends on the loop counter
  bool doesDependOnLoopCounter(const Stmt *S, bool IsInitializer);

public:
  OmpSsIterationSpaceChecker(Sema &SemaRef, SourceLocation DefaultLoc)
      : SemaRef(SemaRef), DefaultLoc(DefaultLoc), ConditionLoc(DefaultLoc) {}
  /// Check init-expr for canonical loop form and save loop counter
  /// variable - #Var and its initialization value - #LB.
  bool checkAndSetInit(Stmt *S, bool EmitDiags = true);
  /// Check test-expr for canonical form, save upper-bound (#UB), flags
  /// for less/greater and for strict/non-strict comparison.
  bool checkAndSetCond(Expr *S);
  /// Check incr-expr for canonical loop form and return true if it
  /// does not conform, otherwise save loop step (#Step).
  bool checkAndSetInc(Expr *S);
  /// Return the loop counter variable.
  ValueDecl *getLoopDecl() const { return LCDecl; }
  /// Return the reference expression to loop counter variable.
  Expr *getLoopDeclRefExpr() const { return LCRef; }
  /// Return the Lower bound expression
  Expr *getLoopLowerBoundExpr() const { return LB; }
  /// Return the Upper bound expression
  Expr *getLoopUpperBoundExpr() const { return UB; }
  /// Return the Step expression
  Expr *getLoopStepExpr() const { return Step; }
  /// Return true if < or <=, false if >= or >. No value means !=
  std::optional<bool> getLoopIsLessOp() const { return TestIsLessOp; }
  /// Return true is strict comparison
  bool getLoopIsStrictOp() const { return TestIsStrictOp; }
  /// Source range of the loop init.
  SourceRange getInitSrcRange() const { return InitSrcRange; }
  /// Source range of the loop condition.
  SourceRange getConditionSrcRange() const { return ConditionSrcRange; }
  /// Source range of the loop increment.
  SourceRange getIncrementSrcRange() const { return IncrementSrcRange; }
  /// True, if the compare operator is strict (<, > or !=).
  bool isStrictTestOp() const { return TestIsStrictOp; }
  /// Return true if any expression is dependent.
  bool dependent() const;

private:
  /// Check the right-hand side of an assignment in the increment
  /// expression.
  bool checkAndSetIncRHS(Expr *RHS);
  /// Helper to set loop counter variable and its initializer.
  bool setLCDeclAndLB(ValueDecl *NewLCDecl, Expr *NewDeclRefExpr, Expr *NewLB,
                      bool EmitDiags);
  /// Helper to set upper bound.
  bool setUB(Expr *NewUB, std::optional<bool> LessOp, bool StrictOp,
             SourceRange SR, SourceLocation SL);
  /// Helper to set loop increment.
  bool setStep(Expr *NewStep, bool Subtract);
};

bool OmpSsIterationSpaceChecker::dependent() const {
  if (!LCDecl) {
    assert(!LB && !UB && !Step);
    return false;
  }
  return LCDecl->getType()->isDependentType() ||
         (LB && LB->isValueDependent()) || (UB && UB->isValueDependent()) ||
         (Step && Step->isValueDependent());
}

bool OmpSsIterationSpaceChecker::setLCDeclAndLB(ValueDecl *NewLCDecl,
                                                 Expr *NewLCRefExpr,
                                                 Expr *NewLB, bool EmitDiags) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl == nullptr && LB == nullptr && LCRef == nullptr &&
         UB == nullptr && Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewLCDecl || !NewLB)
    return true;
  LCDecl = getCanonicalDecl(NewLCDecl);
  LCRef = NewLCRefExpr;
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(NewLB))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        NewLB = CE->getArg(0)->IgnoreParenImpCasts();
  LB = NewLB;
  if (EmitDiags)
    doesDependOnLoopCounter(LB, /*IsInitializer=*/true);
  return false;
}

bool OmpSsIterationSpaceChecker::setUB(Expr *NewUB,
                                        std::optional<bool> LessOp,
                                        bool StrictOp, SourceRange SR,
                                        SourceLocation SL) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && UB == nullptr &&
         Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewUB)
    return true;
  UB = NewUB;
  TestIsLessOp = LessOp;
  TestIsStrictOp = StrictOp;
  ConditionSrcRange = SR;
  ConditionLoc = SL;
  doesDependOnLoopCounter(UB, /*IsInitializer=*/false);
  return false;
}

bool OmpSsIterationSpaceChecker::setStep(Expr *NewStep, bool Subtract) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && Step == nullptr);
  if (!NewStep)
    return true;
  if (!NewStep->isValueDependent()) {
    // Check that the step is integer expression.
    SourceLocation StepLoc = NewStep->getBeginLoc();
    ExprResult Val = SemaRef.PerformOmpSsImplicitIntegerConversion(
        StepLoc, getExprAsWritten(NewStep));
    if (Val.isInvalid())
      return true;
    NewStep = Val.get();

    // OmpSs [2.6, Canonical Loop Form, Restrictions]
    //  If test-expr is of form var relational-op b and relational-op is < or
    //  <= then incr-expr must cause var to increase on each iteration of the
    //  loop. If test-expr is of form var relational-op b and relational-op is
    //  > or >= then incr-expr must cause var to decrease on each iteration of
    //  the loop.
    //  If test-expr is of form b relational-op var and relational-op is < or
    //  <= then incr-expr must cause var to decrease on each iteration of the
    //  loop. If test-expr is of form b relational-op var and relational-op is
    //  > or >= then incr-expr must cause var to increase on each iteration of
    //  the loop.
    std::optional<llvm::APSInt> Result =
        NewStep->getIntegerConstantExpr(SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        Result && Result->isSigned() && (Subtract != Result->isNegative());
    bool IsConstPos =
        Result && Result->isSigned() && (Subtract == Result->isNegative());
    bool IsConstZero = Result && !Result->getBoolValue();

    if (UB && (IsConstZero ||
               (TestIsLessOp.value() ?
                  (IsConstNeg || (IsUnsigned && Subtract)) :
                  (IsConstPos || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_oss_loop_incr_not_compatible)
          << LCDecl << TestIsLessOp.value() << NewStep->getSourceRange();
      SemaRef.Diag(ConditionLoc,
                   diag::note_oss_loop_cond_requres_compatible_incr)
          << TestIsLessOp.value() << ConditionSrcRange;
      return true;
    }
    if (Subtract) {
      NewStep =
          SemaRef.CreateBuiltinUnaryOp(NewStep->getExprLoc(), UO_Minus, NewStep)
              .get();
    }
  }

  Step = NewStep;
  return false;
}

namespace {
/// Checker for the non-rectangular loops. Checks if the initializer or
/// condition expression references loop counter variable.
class LoopCounterRefChecker final
    : public ConstStmtVisitor<LoopCounterRefChecker, bool> {
  Sema &SemaRef;
  const ValueDecl *CurLCDecl = nullptr;
  bool IsInitializer = true;
  bool EmitDiags = true;
  bool checkDecl(const Expr *E, const ValueDecl *VD) {
    if (getCanonicalDecl(VD) == getCanonicalDecl(CurLCDecl)) {
      if (EmitDiags) {
        SemaRef.Diag(E->getExprLoc(), diag::err_oss_stmt_depends_on_loop_counter)
            << (IsInitializer ? 0 : 1);
      }
      return true;
    }
    return false;
  }

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const ValueDecl *VD = E->getDecl();
    if (isa<VarDecl>(VD))
      return checkDecl(E, VD);
    return false;
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (isa<CXXThisExpr>(E->getBase()->IgnoreParens())) {
      const ValueDecl *VD = E->getMemberDecl();
      if (isa<VarDecl>(VD) || isa<FieldDecl>(VD))
        return checkDecl(E, VD);
    }
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    bool Res = false;
    for (const Stmt *Child : S->children())
      Res = (Child && Visit(Child)) || Res;
    return Res;
  }
  explicit LoopCounterRefChecker(Sema &SemaRef, const ValueDecl *CurLCDecl,
                                 bool IsInitializer, bool EmitDiags)
      : SemaRef(SemaRef), CurLCDecl(CurLCDecl), IsInitializer(IsInitializer),
        EmitDiags(EmitDiags) {}
};
} // namespace

bool
OmpSsIterationSpaceChecker::doesDependOnLoopCounter(const Stmt *S,
                                                     bool IsInitializer) {
  // Check for the non-rectangular loops.
  LoopCounterRefChecker LoopStmtChecker(
    SemaRef, LCDecl, IsInitializer, /*EmitDiags=*/true);
  return LoopStmtChecker.Visit(S);
}

bool OmpSsIterationSpaceChecker::checkAndSetInit(Stmt *S, bool EmitDiags) {
  // Check init-expr for canonical loop form and save loop counter
  // variable - #Var and its initialization value - #LB.
  // OmpSs [2.6] Canonical loop form. init-expr may be one of the following:
  //   var = lb
  //   integer-type var = lb
  //
  if (!S) {
    if (EmitDiags)
      SemaRef.Diag(DefaultLoc, diag::err_oss_loop_not_canonical_init);
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  InitSrcRange = S->getSourceRange();
  bool IsThisMemberExpr = false;
  if (Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign) {
      Expr *LHS = BO->getLHS()->IgnoreParens();
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        return setLCDeclAndLB(DRE->getDecl(), DRE, BO->getRHS(),
                              EmitDiags);
      }
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          IsThisMemberExpr = true;
      }
    }
  } else if (auto *DS = dyn_cast<DeclStmt>(S)) {
    if (DS->isSingleDecl()) {
      if (auto *Var = dyn_cast_or_null<VarDecl>(DS->getSingleDecl())) {
        if (Var->hasInit() && !Var->getType()->isReferenceType()) {
          // Accept non-canonical init form here (i.e. int i{0}) but emit ext. warning.
          if (Var->getInitStyle() != VarDecl::CInit && EmitDiags)
            SemaRef.Diag(S->getBeginLoc(),
                         diag::ext_oss_loop_not_canonical_init)
                << S->getSourceRange();
          return setLCDeclAndLB(
              Var,
              buildDeclRefExpr(SemaRef, Var,
                               Var->getType().getNonReferenceType(),
                               DS->getBeginLoc()),
              Var->getInit(), EmitDiags);
        }
      }
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getOperator() == OO_Equal) {
      Expr *LHS = CE->getArg(0);
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        return setLCDeclAndLB(DRE->getDecl(), DRE, CE->getArg(1), EmitDiags);
      }
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          IsThisMemberExpr = true;
      }
    }
  }

  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  if (EmitDiags) {
    SemaRef.Diag(S->getBeginLoc(), diag::err_oss_loop_not_canonical_init)
        << S->getSourceRange();
    if (IsThisMemberExpr)
      SemaRef.Diag(S->getBeginLoc(), diag::err_oss_loop_this_expr_init)
          << S->getSourceRange();
  }
  return true;
}

/// Ignore parenthesizes, implicit casts, copy constructor and return the
/// variable (which may be the loop variable) if possible.
static const ValueDecl *getInitLCDecl(const Expr *E) {
  if (!E)
    return nullptr;
  E = getExprAsWritten(E);
  if (const auto *CE = dyn_cast_or_null<CXXConstructExpr>(E))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        E = CE->getArg(0)->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast_or_null<DeclRefExpr>(E)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return getCanonicalDecl(VD);
  }
  if (const auto *ME = dyn_cast_or_null<MemberExpr>(E))
    if (ME->isArrow() && isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
      return getCanonicalDecl(ME->getMemberDecl());
  return nullptr;
}

bool OmpSsIterationSpaceChecker::checkAndSetCond(Expr *S) {
  // Check test-expr for canonical form, save upper-bound UB, flags for
  // less/greater and for strict/non-strict comparison.
  // OmpSs [2.9] Canonical loop form. Test-expr may be one of the following:
  //   var relational-op b
  //   b relational-op var
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_oss_loop_not_canonical_cond)
        << LCDecl;
    return true;
  }
  S = getExprAsWritten(S);
  SourceLocation CondLoc = S->getBeginLoc();
  if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->isRelationalOp()) {
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setUB(BO->getRHS(),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_LE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
      if (getInitLCDecl(BO->getRHS()) == LCDecl)
        return setUB(BO->getLHS(),
                     (BO->getOpcode() == BO_GT || BO->getOpcode() == BO_GE),
                     (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT),
                     BO->getSourceRange(), BO->getOperatorLoc());
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getNumArgs() == 2) {
      auto Op = CE->getOperator();
      switch (Op) {
      case OO_Greater:
      case OO_GreaterEqual:
      case OO_Less:
      case OO_LessEqual:
        if (getInitLCDecl(CE->getArg(0)) == LCDecl)
          return setUB(CE->getArg(1), Op == OO_Less || Op == OO_LessEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        if (getInitLCDecl(CE->getArg(1)) == LCDecl)
          return setUB(CE->getArg(0), Op == OO_Greater || Op == OO_GreaterEqual,
                       Op == OO_Less || Op == OO_Greater, CE->getSourceRange(),
                       CE->getOperatorLoc());
        break;
      default:
        break;
      }
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(CondLoc, diag::err_oss_loop_not_canonical_cond)
      << S->getSourceRange() << LCDecl;
  return true;
}

bool OmpSsIterationSpaceChecker::checkAndSetIncRHS(Expr *RHS) {
  // RHS of canonical loop form increment can be:
  //   var + incr
  //   incr + var
  //   var - incr
  //
  RHS = RHS->IgnoreParenImpCasts();
  if (auto *BO = dyn_cast<BinaryOperator>(RHS)) {
    if (BO->isAdditiveOp()) {
      bool IsAdd = BO->getOpcode() == BO_Add;
      // The following cases are handled here
      // var + incr
      // var + -incr
      // var - incr
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), !IsAdd);
      // The following cases are handled here
      // incr + var
      // -incr + var
      if (IsAdd && getInitLCDecl(BO->getRHS()) == LCDecl)
        return setStep(BO->getLHS(), /*Subtract=*/false);
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    bool IsAdd = CE->getOperator() == OO_Plus;
    if ((IsAdd || CE->getOperator() == OO_Minus) && CE->getNumArgs() == 2) {
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), !IsAdd);
      if (IsAdd && getInitLCDecl(CE->getArg(1)) == LCDecl)
        return setStep(CE->getArg(0), /*Subtract=*/false);
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  // The following cases are handled here
  //  incr - var
  //  -incr - var
  SemaRef.Diag(RHS->getBeginLoc(), diag::err_oss_loop_not_canonical_incr)
      << RHS->getSourceRange() << LCDecl;
  return true;
}

bool OmpSsIterationSpaceChecker::checkAndSetInc(Expr *S) {
  // Check incr-expr for canonical loop form and return true if it
  // does not conform.
  // OmpSs [2.6] Canonical loop form. Test-expr may be one of the following:
  //   ++var
  //   var++
  //   --var
  //   var--
  //   var += incr
  //   var -= incr
  //   var = var + incr
  //   var = incr + var
  //   var = var - incr
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_oss_loop_not_canonical_incr) << LCDecl;
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  IncrementSrcRange = S->getSourceRange();
  S = S->IgnoreParens();
  if (auto *UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementDecrementOp() &&
        getInitLCDecl(UO->getSubExpr()) == LCDecl)
      return setStep(SemaRef
                         .ActOnIntegerConstant(UO->getBeginLoc(),
                                               (UO->isDecrementOp() ? -1 : 1))
                         .get(),
                     /*Subtract=*/false);
  } else if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), BO->getOpcode() == BO_SubAssign);
      break;
    case BO_Assign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return checkAndSetIncRHS(BO->getRHS());
      break;
    default:
      break;
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    switch (CE->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(SemaRef
                           .ActOnIntegerConstant(
                               CE->getBeginLoc(),
                               ((CE->getOperator() == OO_MinusMinus) ? -1 : 1))
                           .get(),
                       /*Subtract=*/false);
      break;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), CE->getOperator() == OO_MinusEqual);
      break;
    case OO_Equal:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return checkAndSetIncRHS(CE->getArg(1));
      break;
    default:
      break;
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(S->getBeginLoc(), diag::err_oss_loop_not_canonical_incr)
      << S->getSourceRange() << LCDecl;
  return true;
}

} // namespace

void Sema::ActOnOmpSsLoopInitialization(SourceLocation ForLoc, Stmt *Init) {
  assert(getLangOpts().OmpSs && "OmpSs is not active.");
  assert(Init && "Expected loop in canonical form.");

  OmpSsDirectiveKind DKind = DSAStack->getCurrentDirective();
  unsigned AssociatedLoops = DSAStack->getAssociatedLoops();
  unsigned SeenAssociatedLoops = DSAStack->getSeenAssociatedLoops();
  if (AssociatedLoops > SeenAssociatedLoops &&
      isOmpSsLoopDirective(DKind)) {
    OmpSsIterationSpaceChecker ISC(*this, ForLoc);
    if (!ISC.checkAndSetInit(Init, /*EmitDiags=*/false)) {
      if (ValueDecl *D = ISC.getLoopDecl()) {
        const Expr *E = ISC.getLoopDeclRefExpr();
        auto *VD = dyn_cast<VarDecl>(D);

        OmpSsClauseKind CKind = OSSC_private;
        if (DSAStack->getCurrentDirective() == OSSD_taskiter)
          CKind = OSSC_firstprivate;

        DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
        if (DVar.CKind != OSSC_unknown && DVar.CKind != CKind &&
            DVar.RefExpr) {
          Diag(E->getExprLoc(), diag::err_oss_wrong_dsa)
            << getOmpSsClauseName(DVar.CKind)
            << getOmpSsClauseName(CKind);
            return;
        }

        // Register loop control variable
        if (!CurContext->isDependentContext()) {
          DSAStack->addLoopControlVariable(VD, E);
          DSAStack->addDSA(VD, E, CKind, /*Ignore=*/true, /*IsBase=*/true);
        }
      }
    }
    DSAStack->setSeenAssociatedLoops(SeenAssociatedLoops + 1);
  }
}

namespace {
class OSSClauseLoopIterUse final : public StmtVisitor<OSSClauseLoopIterUse, void> {
  Sema &SemaRef;
  ArrayRef<OSSLoopDirective::HelperExprs> B;
  bool ErrorFound = false;

  bool CurEmitDiags = false;
  unsigned CurDiagID = 0;
public:

  void VisitOSSMultiDepExpr(OSSMultiDepExpr *E) {
    Visit(E->getDepExpr());

    for (size_t i = 0; i < E->getDepInits().size(); ++i) {
      Visit(E->getDepInits()[i]);
      if (E->getDepSizes()[i])
        Visit(E->getDepSizes()[i]);
      if (E->getDepSteps()[i])
        Visit(E->getDepSteps()[i]);
    }
  }

  void VisitOSSArrayShapingExpr(OSSArrayShapingExpr *E) {
    Visit(E->getBase());

    for (Expr *S : E->getShapes())
      Visit(S);
  }

  void VisitOSSArraySectionExpr(OSSArraySectionExpr *E) {
    Visit(E->getBase());

    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
  }

  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    Visit(E->getBase());
    Visit(E->getIdx());
  }

  void VisitUnaryOperator(UnaryOperator *E) {
    Visit(E->getSubExpr());
  }

  void VisitMemberExpr(MemberExpr *E) {
    Visit(E->getBase());
  }

  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();

      for (size_t i = 0; i < B.size(); ++i) {
        ValueDecl *IndVarVD = cast<DeclRefExpr>(B[i].IndVar)->getDecl();
        if (CurEmitDiags && VD == IndVarVD) {
          SemaRef.Diag(E->getBeginLoc(), CurDiagID);
          ErrorFound = true;
        }
      }
    }
  }

  void VisitClause(OSSClause *Clause, bool EmitDiags, unsigned DiagID) {
    CurEmitDiags = EmitDiags;
    CurDiagID = DiagID;
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

  OSSClauseLoopIterUse(Sema &SemaRef, const SmallVectorImpl<OSSLoopDirective::HelperExprs> &B)
      : SemaRef(SemaRef), B(B), ErrorFound(false)
      { }

};
} // namespace

static bool checkOmpSsLoop(
    OmpSsDirectiveKind DKind, Stmt *AStmt,
    Sema &SemaRef, DSAStackTy &Stack,
    SmallVectorImpl<OSSLoopDirective::HelperExprs> &B) {

  bool HasErrors = false;
  bool IsDependent = false;

  QualType LoopTy;

  // OmpSs [2.9.1, Canonical Loop Form]
  //   for (init-expr; test-expr; incr-expr) structured-block
  auto *For = dyn_cast_or_null<ForStmt>(AStmt);
  unsigned TotalLoops = Stack.getAssociatedLoops();
  for (unsigned i = 0; i < TotalLoops; ++i) {
    if (!For) {
      SemaRef.Diag(AStmt->getBeginLoc(), diag::err_oss_not_for)
          << (TotalLoops != 1) << getOmpSsDirectiveName(DKind)
          << TotalLoops << i;
      return true;
    }
    // Perfect nesting check.
    //   More than one child and one of them is not a ForStmt
    if (i + 1 < TotalLoops) {
      const Stmt *NotFor = nullptr;
      int j = 0;
      for (const Stmt *Child  : For->getBody()->children()) {
        if (!NotFor && !isa<ForStmt>(Child)) {
          NotFor = Child;
        }
        ++j;
      }
      if (NotFor && j > 1) {
        SemaRef.Diag(NotFor->getBeginLoc(), diag::err_oss_perfect_for_nesting);
        return true;
      }
    }
    OmpSsIterationSpaceChecker ISC(SemaRef, For->getForLoc());

    // Check init.
    Stmt *Init = For->getInit();
    if (ISC.checkAndSetInit(Init))
      return true;

    // Check loop variable's type.
    if (ValueDecl *LCDecl = ISC.getLoopDecl()) {
      // OmpSs [2.6, Canonical Loop Form]
      // Var is one of the following:
      //   A variable of signed or unsigned integer type.
      QualType VarType = LCDecl->getType().getNonReferenceType();
      if (!VarType->isDependentType() && !VarType->isIntegerType()) {
        SemaRef.Diag(Init->getBeginLoc(), diag::err_oss_loop_variable_type);
        HasErrors = true;
      }

      if (!LCDecl->getType()->isDependentType()) {
        if (LoopTy.isNull()) {
          LoopTy = LCDecl->getType().getCanonicalType();
        } else if (i > 0 && LoopTy != LCDecl->getType().getCanonicalType()) {
          SemaRef.Diag(LCDecl->getBeginLoc(), diag::err_oss_collapse_same_loop_type)
            << LoopTy << LCDecl->getType().getCanonicalType();
          HasErrors = true;
        }
      }

      // Check test-expr.
      HasErrors |= ISC.checkAndSetCond(For->getCond());

      // Check incr-expr.
      HasErrors |= ISC.checkAndSetInc(For->getInc());
    }

    B[i].IndVar = ISC.getLoopDeclRefExpr();
    B[i].LB = ISC.getLoopLowerBoundExpr();
    B[i].UB = ISC.getLoopUpperBoundExpr();
    B[i].Step = ISC.getLoopStepExpr();
    B[i].TestIsLessOp = ISC.getLoopIsLessOp();
    B[i].TestIsStrictOp = ISC.getLoopIsStrictOp();

    if (ISC.dependent() || SemaRef.CurContext->isDependentContext())
      IsDependent = true;

    // Try to find the next For
    ForStmt *TmpFor = For;
    For = nullptr;
    for (Stmt *Child : TmpFor->getBody()->children()) {
      if ((For = dyn_cast_or_null<ForStmt>(Child)))
        break;
    }
  }

  if (IsDependent || HasErrors)
    return HasErrors;
  return false;
}

static bool checkNonRectangular(Sema &SemaRef, DSAStackTy *Stack, const SmallVectorImpl<OSSLoopDirective::HelperExprs> &B) {
  for (size_t i = 1; i < B.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      ValueDecl *VD = cast<DeclRefExpr>(B[j].IndVar)->getDecl();
      if (LoopCounterRefChecker(
          SemaRef, VD, /*IsInitializer=*/true, /*EmitDiags=*/false).Visit(B[i].LB)) {
        return true;
      }
      if (LoopCounterRefChecker(
          SemaRef, VD, /*IsInitializer=*/true, /*EmitDiags=*/false).Visit(B[i].UB)) {
        return true;
      }
      if (LoopCounterRefChecker(
          SemaRef, VD, /*IsInitializer=*/true, /*EmitDiags=*/false).Visit(B[i].Step)) {
        // TODO: error here. it is not valid
      }
    }
  }
  return false;
}

StmtResult Sema::ActOnOmpSsTaskForDirective(
    ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  SmallVector<OSSLoopDirective::HelperExprs> B(DSAStack->getAssociatedLoops());
  if (checkOmpSsLoop(OSSD_task_for, AStmt, *this, *DSAStack, B))
    return StmtError();

  setFunctionHasBranchProtectedScope();
  return OSSTaskForDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOmpSsTaskIterDirective(
    ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();
  SmallVector<OSSLoopDirective::HelperExprs> B(DSAStack->getAssociatedLoops());
  if (isa<ForStmt>(AStmt)) {
    // taskiter (for)
    if (checkOmpSsLoop(OSSD_taskiter, AStmt, *this, *DSAStack, B))
      return StmtError();
  } else if (isa<WhileStmt>(AStmt)) {
    // taskiter (while)
    // do nothing
  } else {
    Diag(AStmt->getBeginLoc(), diag::err_oss_taskiter_for_while)
        << getOmpSsDirectiveName(OSSD_taskiter);
  }

  setFunctionHasBranchProtectedScope();
  return OSSTaskIterDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOmpSsTaskLoopDirective(
    ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  SmallVector<OSSLoopDirective::HelperExprs> B(DSAStack->getAssociatedLoops());
  if (checkOmpSsLoop(OSSD_taskloop, AStmt, *this, *DSAStack, B))
    return StmtError();

  bool IsNonRectangular = checkNonRectangular(*this, DSAStack, B);
  OSSClauseLoopIterUse OSSLoopIterUse(*this, B);
  for (auto *Clause : Clauses) {
    if (isa<OSSDependClause>(Clause) || isa<OSSReductionClause>(Clause)) {
      OSSLoopIterUse.VisitClause(
        Clause, IsNonRectangular, diag::err_oss_nonrectangular_loop_iter_dep);
      if (OSSLoopIterUse.isErrorFound())
        return StmtError();
    } else if (isa<OSSIfClause>(Clause) ||
               isa<OSSFinalClause>(Clause) ||
               isa<OSSCostClause>(Clause) ||
               isa<OSSPriorityClause>(Clause) ||
               isa<OSSOnreadyClause>(Clause) ||
               isa<OSSChunksizeClause>(Clause) ||
               isa<OSSGrainsizeClause>(Clause)) {
      OSSLoopIterUse.VisitClause(Clause, /*EmitDiags=*/true, diag::err_oss_loop_iter_no_dep);
      if (OSSLoopIterUse.isErrorFound())
        return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OSSTaskLoopDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
}

StmtResult Sema::ActOnOmpSsTaskLoopForDirective(
    ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
    SourceLocation StartLoc, SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  SmallVector<OSSLoopDirective::HelperExprs> B(DSAStack->getAssociatedLoops());
  if (checkOmpSsLoop(OSSD_taskloop_for, AStmt, *this, *DSAStack, B))
    return StmtError();

  bool IsNonRectangular = checkNonRectangular(*this, DSAStack, B);
  OSSClauseLoopIterUse OSSLoopIterUse(*this, B);
  for (auto *Clause : Clauses) {
    if (isa<OSSDependClause>(Clause) || isa<OSSReductionClause>(Clause)) {
      OSSLoopIterUse.VisitClause(
        Clause, IsNonRectangular, diag::err_oss_nonrectangular_loop_iter_dep);
      if (OSSLoopIterUse.isErrorFound())
        return StmtError();
    } else if (isa<OSSIfClause>(Clause) ||
               isa<OSSFinalClause>(Clause) ||
               isa<OSSCostClause>(Clause) ||
               isa<OSSPriorityClause>(Clause) ||
               isa<OSSOnreadyClause>(Clause) ||
               isa<OSSChunksizeClause>(Clause) ||
               isa<OSSGrainsizeClause>(Clause)) {
      OSSLoopIterUse.VisitClause(Clause, /*EmitDiags=*/true, diag::err_oss_loop_iter_no_dep);
      if (OSSLoopIterUse.isErrorFound())
        return StmtError();
    }
  }

  setFunctionHasBranchProtectedScope();
  return OSSTaskLoopForDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
}

namespace {
/// Helper class for checking expression in 'omp atomic [update]'
/// construct.
class OmpSsAtomicUpdateChecker {
  /// Error results for atomic update expressions.
  enum ExprAnalysisErrorCode {
    /// A statement is not an expression statement.
    NotAnExpression,
    /// Expression is not builtin binary or unary operation.
    NotABinaryOrUnaryExpression,
    /// Unary operation is not post-/pre- increment/decrement operation.
    NotAnUnaryIncDecExpression,
    /// An expression is not of scalar type.
    NotAScalarType,
    /// A binary operation is not an assignment operation.
    NotAnAssignmentOp,
    /// RHS part of the binary operation is not a binary expression.
    NotABinaryExpression,
    /// RHS part is not additive/multiplicative/shift/biwise binary
    /// expression.
    NotABinaryOperator,
    /// RHS binary operation does not have reference to the updated LHS
    /// part.
    NotAnUpdateExpression,
    /// No errors is found.
    NoError
  };
  /// Reference to Sema.
  Sema &SemaRef;
  /// A location for note diagnostics (when error is found).
  SourceLocation NoteLoc;
  /// 'x' lvalue part of the source atomic expression.
  Expr *X;
  /// 'expr' rvalue part of the source atomic expression.
  Expr *E;
  /// Helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *UpdateExpr;
  /// Is 'x' a LHS in a RHS part of full update expression. It is
  /// important for non-associative operations.
  bool IsXLHSInRHSPart;
  BinaryOperatorKind Op;
  SourceLocation OpLoc;
  /// true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool IsPostfixUpdate;

public:
  OmpSsAtomicUpdateChecker(Sema &SemaRef)
      : SemaRef(SemaRef), X(nullptr), E(nullptr), UpdateExpr(nullptr),
        IsXLHSInRHSPart(false), Op(BO_PtrMemD), IsPostfixUpdate(false) {}
  /// Check specified statement that it is suitable for 'atomic update'
  /// constructs and extract 'x', 'expr' and Operation from the original
  /// expression. If DiagId and NoteId == 0, then only check is performed
  /// without error notification.
  /// \param DiagId Diagnostic which should be emitted if error is found.
  /// \param NoteId Diagnostic note for the main error message.
  /// \return true if statement is not an update expression, false otherwise.
  bool checkStatement(Stmt *S, unsigned DiagId = 0, unsigned NoteId = 0);
  /// Return the 'x' lvalue part of the source atomic expression.
  Expr *getX() const { return X; }
  /// Return the 'expr' rvalue part of the source atomic expression.
  Expr *getExpr() const { return E; }
  /// Return the update expression used in calculation of the updated
  /// value. Always has form 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *getUpdateExpr() const { return UpdateExpr; }
  /// Return true if 'x' is LHS in RHS part of full update expression,
  /// false otherwise.
  bool isXLHSInRHSPart() const { return IsXLHSInRHSPart; }

  /// true if the source expression is a postfix unary operation, false
  /// if it is a prefix unary operation.
  bool isPostfixUpdate() const { return IsPostfixUpdate; }

private:
  bool checkBinaryOperation(BinaryOperator *AtomicBinOp, unsigned DiagId = 0,
                            unsigned NoteId = 0);
};

bool OmpSsAtomicUpdateChecker::checkBinaryOperation(
    BinaryOperator *AtomicBinOp, unsigned DiagId, unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x = x binop expr;
  //  x = expr binop x;
  if (AtomicBinOp->getOpcode() == BO_Assign) {
    X = AtomicBinOp->getLHS();
    if (const auto *AtomicInnerBinOp = dyn_cast<BinaryOperator>(
            AtomicBinOp->getRHS()->IgnoreParenImpCasts())) {
      if (AtomicInnerBinOp->isMultiplicativeOp() ||
          AtomicInnerBinOp->isAdditiveOp() || AtomicInnerBinOp->isShiftOp() ||
          AtomicInnerBinOp->isBitwiseOp()) {
        Op = AtomicInnerBinOp->getOpcode();
        OpLoc = AtomicInnerBinOp->getOperatorLoc();
        Expr *LHS = AtomicInnerBinOp->getLHS();
        Expr *RHS = AtomicInnerBinOp->getRHS();
        llvm::FoldingSetNodeID XId, LHSId, RHSId;
        X->IgnoreParenImpCasts()->Profile(XId, SemaRef.getASTContext(),
                                          /*Canonical=*/true);
        LHS->IgnoreParenImpCasts()->Profile(LHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        RHS->IgnoreParenImpCasts()->Profile(RHSId, SemaRef.getASTContext(),
                                            /*Canonical=*/true);
        if (XId == LHSId) {
          E = RHS;
          IsXLHSInRHSPart = true;
        } else if (XId == RHSId) {
          E = LHS;
          IsXLHSInRHSPart = false;
        } else {
          ErrorLoc = AtomicInnerBinOp->getExprLoc();
          ErrorRange = AtomicInnerBinOp->getSourceRange();
          NoteLoc = X->getExprLoc();
          NoteRange = X->getSourceRange();
          ErrorFound = NotAnUpdateExpression;
        }
      } else {
        ErrorLoc = AtomicInnerBinOp->getExprLoc();
        ErrorRange = AtomicInnerBinOp->getSourceRange();
        NoteLoc = AtomicInnerBinOp->getOperatorLoc();
        NoteRange = SourceRange(NoteLoc, NoteLoc);
        ErrorFound = NotABinaryOperator;
      }
    } else {
      NoteLoc = ErrorLoc = AtomicBinOp->getRHS()->getExprLoc();
      NoteRange = ErrorRange = AtomicBinOp->getRHS()->getSourceRange();
      ErrorFound = NotABinaryExpression;
    }
  } else {
    ErrorLoc = AtomicBinOp->getExprLoc();
    ErrorRange = AtomicBinOp->getSourceRange();
    NoteLoc = AtomicBinOp->getOperatorLoc();
    NoteRange = SourceRange(NoteLoc, NoteLoc);
    ErrorFound = NotAnAssignmentOp;
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  }
  if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  return ErrorFound != NoError;
}

bool OmpSsAtomicUpdateChecker::checkStatement(Stmt *S, unsigned DiagId,
                                               unsigned NoteId) {
  ExprAnalysisErrorCode ErrorFound = NoError;
  SourceLocation ErrorLoc, NoteLoc;
  SourceRange ErrorRange, NoteRange;
  // Allowed constructs are:
  //  x++;
  //  x--;
  //  ++x;
  //  --x;
  //  x binop= expr;
  //  x = x binop expr;
  //  x = expr binop x;
  if (auto *AtomicBody = dyn_cast<Expr>(S)) {
    AtomicBody = AtomicBody->IgnoreParenImpCasts();
    if (AtomicBody->getType()->isScalarType() ||
        AtomicBody->isInstantiationDependent()) {
      if (const auto *AtomicCompAssignOp = dyn_cast<CompoundAssignOperator>(
              AtomicBody->IgnoreParenImpCasts())) {
        // Check for Compound Assignment Operation
        Op = BinaryOperator::getOpForCompoundAssignment(
            AtomicCompAssignOp->getOpcode());
        OpLoc = AtomicCompAssignOp->getOperatorLoc();
        E = AtomicCompAssignOp->getRHS();
        X = AtomicCompAssignOp->getLHS()->IgnoreParens();
        IsXLHSInRHSPart = true;
      } else if (auto *AtomicBinOp = dyn_cast<BinaryOperator>(
                     AtomicBody->IgnoreParenImpCasts())) {
        // Check for Binary Operation
        if (checkBinaryOperation(AtomicBinOp, DiagId, NoteId))
          return true;
      } else if (const auto *AtomicUnaryOp = dyn_cast<UnaryOperator>(
                     AtomicBody->IgnoreParenImpCasts())) {
        // Check for Unary Operation
        if (AtomicUnaryOp->isIncrementDecrementOp()) {
          IsPostfixUpdate = AtomicUnaryOp->isPostfix();
          Op = AtomicUnaryOp->isIncrementOp() ? BO_Add : BO_Sub;
          OpLoc = AtomicUnaryOp->getOperatorLoc();
          X = AtomicUnaryOp->getSubExpr()->IgnoreParens();
          E = SemaRef.ActOnIntegerConstant(OpLoc, /*uint64_t Val=*/1).get();
          IsXLHSInRHSPart = true;
        } else {
          ErrorFound = NotAnUnaryIncDecExpression;
          ErrorLoc = AtomicUnaryOp->getExprLoc();
          ErrorRange = AtomicUnaryOp->getSourceRange();
          NoteLoc = AtomicUnaryOp->getOperatorLoc();
          NoteRange = SourceRange(NoteLoc, NoteLoc);
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotABinaryOrUnaryExpression;
        NoteLoc = ErrorLoc = AtomicBody->getExprLoc();
        NoteRange = ErrorRange = AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAScalarType;
      NoteLoc = ErrorLoc = AtomicBody->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
  } else {
    ErrorFound = NotAnExpression;
    NoteLoc = ErrorLoc = S->getBeginLoc();
    NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
  }
  if (ErrorFound != NoError && DiagId != 0 && NoteId != 0) {
    SemaRef.Diag(ErrorLoc, DiagId) << ErrorRange;
    SemaRef.Diag(NoteLoc, NoteId) << ErrorFound << NoteRange;
    return true;
  }
  if (SemaRef.CurContext->isDependentContext())
    E = X = UpdateExpr = nullptr;
  if (ErrorFound == NoError && E && X) {
    // Build an update expression of form 'OpaqueValueExpr(x) binop
    // OpaqueValueExpr(expr)' or 'OpaqueValueExpr(expr) binop
    // OpaqueValueExpr(x)' and then cast it to the type of the 'x' expression.
    auto *OVEX = new (SemaRef.getASTContext())
        OpaqueValueExpr(X->getExprLoc(), X->getType(), VK_PRValue);
    auto *OVEExpr = new (SemaRef.getASTContext())
        OpaqueValueExpr(E->getExprLoc(), E->getType(), VK_PRValue);
    ExprResult Update =
        SemaRef.CreateBuiltinBinOp(OpLoc, Op, IsXLHSInRHSPart ? OVEX : OVEExpr,
                                   IsXLHSInRHSPart ? OVEExpr : OVEX);
    if (Update.isInvalid())
      return true;
    Update = SemaRef.PerformImplicitConversion(Update.get(), X->getType(),
                                               Sema::AA_Casting);
    if (Update.isInvalid())
      return true;
    UpdateExpr = Update.get();
  }
  return ErrorFound != NoError;
}

/// Get the node id of the fixed point of an expression \a S.
llvm::FoldingSetNodeID getNodeId(ASTContext &Context, const Expr *S) {
  llvm::FoldingSetNodeID Id;
  S->IgnoreParenImpCasts()->Profile(Id, Context, true);
  return Id;
}

/// Check if two expressions are same.
bool checkIfTwoExprsAreSame(ASTContext &Context, const Expr *LHS,
                            const Expr *RHS) {
  return getNodeId(Context, LHS) == getNodeId(Context, RHS);
}

class OmpSsAtomicCompareChecker {
public:
  /// All kinds of errors that can occur in `atomic compare`
  enum ErrorTy {
    /// Empty compound statement.
    NoStmt = 0,
    /// More than one statement in a compound statement.
    MoreThanOneStmt,
    /// Not an assignment binary operator.
    NotAnAssignment,
    /// Not a conditional operator.
    NotCondOp,
    /// Wrong false expr. According to the spec, 'x' should be at the false
    /// expression of a conditional expression.
    WrongFalseExpr,
    /// The condition of a conditional expression is not a binary operator.
    NotABinaryOp,
    /// Invalid binary operator (not <, >, or ==).
    InvalidBinaryOp,
    /// Invalid comparison (not x == e, e == x, x ordop expr, or expr ordop x).
    InvalidComparison,
    /// X is not a lvalue.
    XNotLValue,
    /// Not a scalar.
    NotScalar,
    /// Not an integer.
    NotInteger,
    /// 'else' statement is not expected.
    UnexpectedElse,
    /// Not an equality operator.
    NotEQ,
    /// Invalid assignment (not v == x).
    InvalidAssignment,
    /// Not if statement
    NotIfStmt,
    /// More than two statements in a compund statement.
    MoreThanTwoStmts,
    /// Not a compound statement.
    NotCompoundStmt,
    /// No else statement.
    NoElse,
    /// Not 'if (r)'.
    InvalidCondition,
    /// No error.
    NoError,
  };

  struct ErrorInfoTy {
    ErrorTy Error;
    SourceLocation ErrorLoc;
    SourceRange ErrorRange;
    SourceLocation NoteLoc;
    SourceRange NoteRange;
  };

  OmpSsAtomicCompareChecker(Sema &S) : ContextRef(S.getASTContext()) {}

  /// Check if statement \a S is valid for <tt>atomic compare</tt>.
  bool checkStmt(Stmt *S, ErrorInfoTy &ErrorInfo);

  Expr *getX() const { return X; }
  Expr *getE() const { return E; }
  Expr *getD() const { return D; }
  Expr *getCond() const { return C; }
  bool isXBinopExpr() const { return IsXBinopExpr; }

protected:
  /// Reference to ASTContext
  ASTContext &ContextRef;
  /// 'x' lvalue part of the source atomic expression.
  Expr *X = nullptr;
  /// 'expr' or 'e' rvalue part of the source atomic expression.
  Expr *E = nullptr;
  /// 'd' rvalue part of the source atomic expression.
  Expr *D = nullptr;
  /// 'cond' part of the source atomic expression. It is in one of the following
  /// forms:
  /// expr ordop x
  /// x ordop expr
  /// x == e
  /// e == x
  Expr *C = nullptr;
  /// True if the cond expr is in the form of 'x ordop expr'.
  bool IsXBinopExpr = true;

  /// Check if it is a valid conditional update statement (cond-update-stmt).
  bool checkCondUpdateStmt(IfStmt *S, ErrorInfoTy &ErrorInfo);

  /// Check if it is a valid conditional expression statement (cond-expr-stmt).
  bool checkCondExprStmt(Stmt *S, ErrorInfoTy &ErrorInfo);

  /// Check if all captured values have right type.
  bool checkType(ErrorInfoTy &ErrorInfo) const;

  static bool CheckValue(const Expr *E, ErrorInfoTy &ErrorInfo,
                         bool ShouldBeLValue, bool ShouldBeInteger = false) {
    if (E->isInstantiationDependent())
      return true;

    if (ShouldBeLValue && !E->isLValue()) {
      ErrorInfo.Error = ErrorTy::XNotLValue;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = E->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = E->getSourceRange();
      return false;
    }

    QualType QTy = E->getType();
    if (!QTy->isScalarType()) {
      ErrorInfo.Error = ErrorTy::NotScalar;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = E->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = E->getSourceRange();
      return false;
    }
    if (ShouldBeInteger && !QTy->isIntegerType()) {
      ErrorInfo.Error = ErrorTy::NotInteger;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = E->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = E->getSourceRange();
      return false;
    }

    return true;
  }
  };

bool OmpSsAtomicCompareChecker::checkCondUpdateStmt(IfStmt *S,
                                                     ErrorInfoTy &ErrorInfo) {
  auto *Then = S->getThen();
  if (auto *CS = dyn_cast<CompoundStmt>(Then)) {
    if (CS->body_empty()) {
      ErrorInfo.Error = ErrorTy::NoStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }
    if (CS->size() > 1) {
      ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getSourceRange();
      return false;
    }
    Then = CS->body_front();
  }

  auto *BO = dyn_cast<BinaryOperator>(Then);
  if (!BO) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Then->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Then->getSourceRange();
    return false;
  }
  if (BO->getOpcode() != BO_Assign) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = BO->getExprLoc();
    ErrorInfo.NoteLoc = BO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
    return false;
  }

  X = BO->getLHS();

  auto *Cond = dyn_cast<BinaryOperator>(S->getCond());
  if (!Cond) {
    ErrorInfo.Error = ErrorTy::NotABinaryOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getCond()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getCond()->getSourceRange();
    return false;
  }

  switch (Cond->getOpcode()) {
  case BO_EQ: {
    C = Cond;
    D = BO->getRHS();
    if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getLHS())) {
      E = Cond->getRHS();
    } else if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getRHS())) {
      E = Cond->getLHS();
    } else {
      ErrorInfo.Error = ErrorTy::InvalidComparison;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
      return false;
    }
    break;
  }
  case BO_LT:
  case BO_GT: {
    E = BO->getRHS();
    if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getLHS()) &&
        checkIfTwoExprsAreSame(ContextRef, E, Cond->getRHS())) {
      C = Cond;
    } else if (checkIfTwoExprsAreSame(ContextRef, E, Cond->getLHS()) &&
               checkIfTwoExprsAreSame(ContextRef, X, Cond->getRHS())) {
      C = Cond;
      IsXBinopExpr = false;
    } else {
      ErrorInfo.Error = ErrorTy::InvalidComparison;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
      return false;
    }
    break;
  }
  default:
    ErrorInfo.Error = ErrorTy::InvalidBinaryOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
    return false;
  }

  if (S->getElse()) {
    ErrorInfo.Error = ErrorTy::UnexpectedElse;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getElse()->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getElse()->getSourceRange();
    return false;
  }

  return true;
}

bool OmpSsAtomicCompareChecker::checkCondExprStmt(Stmt *S,
                                                   ErrorInfoTy &ErrorInfo) {
  auto *BO = dyn_cast<BinaryOperator>(S);
  if (!BO) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getSourceRange();
    return false;
  }
  if (BO->getOpcode() != BO_Assign) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = BO->getExprLoc();
    ErrorInfo.NoteLoc = BO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
    return false;
  }

  X = BO->getLHS();

  auto *CO = dyn_cast<ConditionalOperator>(BO->getRHS()->IgnoreParenImpCasts());
  if (!CO) {
    ErrorInfo.Error = ErrorTy::NotCondOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = BO->getRHS()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getRHS()->getSourceRange();
    return false;
  }

  if (!checkIfTwoExprsAreSame(ContextRef, X, CO->getFalseExpr())) {
    ErrorInfo.Error = ErrorTy::WrongFalseExpr;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CO->getFalseExpr()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange =
        CO->getFalseExpr()->getSourceRange();
    return false;
  }

  auto *Cond = dyn_cast<BinaryOperator>(CO->getCond());
  if (!Cond) {
    ErrorInfo.Error = ErrorTy::NotABinaryOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CO->getCond()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange =
        CO->getCond()->getSourceRange();
    return false;
  }

  switch (Cond->getOpcode()) {
  case BO_EQ: {
    C = Cond;
    D = CO->getTrueExpr();
    if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getLHS())) {
      E = Cond->getRHS();
    } else if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getRHS())) {
      E = Cond->getLHS();
    } else {
      ErrorInfo.Error = ErrorTy::InvalidComparison;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
      return false;
    }
    break;
  }
  case BO_LT:
  case BO_GT: {
    E = CO->getTrueExpr();
    if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getLHS()) &&
        checkIfTwoExprsAreSame(ContextRef, E, Cond->getRHS())) {
      C = Cond;
    } else if (checkIfTwoExprsAreSame(ContextRef, E, Cond->getLHS()) &&
               checkIfTwoExprsAreSame(ContextRef, X, Cond->getRHS())) {
      C = Cond;
      IsXBinopExpr = false;
    } else {
      ErrorInfo.Error = ErrorTy::InvalidComparison;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
      return false;
    }
    break;
  }
  default:
    ErrorInfo.Error = ErrorTy::InvalidBinaryOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
    return false;
  }

  return true;
}

bool OmpSsAtomicCompareChecker::checkType(ErrorInfoTy &ErrorInfo) const {
  // 'x' and 'e' cannot be nullptr
  assert(X && E && "X and E cannot be nullptr");

  if (!CheckValue(X, ErrorInfo, true))
    return false;

  if (!CheckValue(E, ErrorInfo, false))
    return false;

  if (D && !CheckValue(D, ErrorInfo, false))
    return false;

  return true;
}

bool OmpSsAtomicCompareChecker::checkStmt(
    Stmt *S, OmpSsAtomicCompareChecker::ErrorInfoTy &ErrorInfo) {
  auto *CS = dyn_cast<CompoundStmt>(S);
  if (CS) {
    if (CS->body_empty()) {
      ErrorInfo.Error = ErrorTy::NoStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }

    if (CS->size() != 1) {
      ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }
    S = CS->body_front();
  }

  auto Res = false;

  if (auto *IS = dyn_cast<IfStmt>(S)) {
    // Check if the statement is in one of the following forms
    // (cond-update-stmt):
    // if (expr ordop x) { x = expr; }
    // if (x ordop expr) { x = expr; }
    // if (x == e) { x = d; }
    Res = checkCondUpdateStmt(IS, ErrorInfo);
  } else {
    // Check if the statement is in one of the following forms (cond-expr-stmt):
    // x = expr ordop x ? expr : x;
    // x = x ordop expr ? expr : x;
    // x = x == e ? d : x;
    Res = checkCondExprStmt(S, ErrorInfo);
  }

  if (!Res)
    return false;

  return checkType(ErrorInfo);
}

class OmpSsAtomicCompareCaptureChecker final
    : public OmpSsAtomicCompareChecker {
public:
  OmpSsAtomicCompareCaptureChecker(Sema &S) : OmpSsAtomicCompareChecker(S) {}

  Expr *getV() const { return V; }
  Expr *getR() const { return R; }
  bool isFailOnly() const { return IsFailOnly; }
  bool isPostfixUpdate() const { return IsPostfixUpdate; }

  /// Check if statement \a S is valid for <tt>atomic compare capture</tt>.
  bool checkStmt(Stmt *S, ErrorInfoTy &ErrorInfo);

private:
  bool checkType(ErrorInfoTy &ErrorInfo);

  // NOTE: Form 3, 4, 5 in the following comments mean the 3rd, 4th, and 5th
  // form of 'conditional-update-capture-atomic' structured block on the v5.2
  // spec p.p. 82:
  // (1) { v = x; cond-update-stmt }
  // (2) { cond-update-stmt v = x; }
  // (3) if(x == e) { x = d; } else { v = x; }
  // (4) { r = x == e; if(r) { x = d; } }
  // (5) { r = x == e; if(r) { x = d; } else { v = x; } }

  /// Check if it is valid 'if(x == e) { x = d; } else { v = x; }' (form 3)
  bool checkForm3(IfStmt *S, ErrorInfoTy &ErrorInfo);

  /// Check if it is valid '{ r = x == e; if(r) { x = d; } }',
  /// or '{ r = x == e; if(r) { x = d; } else { v = x; } }' (form 4 and 5)
  bool checkForm45(Stmt *S, ErrorInfoTy &ErrorInfo);

  /// 'v' lvalue part of the source atomic expression.
  Expr *V = nullptr;
  /// 'r' lvalue part of the source atomic expression.
  Expr *R = nullptr;
  /// If 'v' is only updated when the comparison fails.
  bool IsFailOnly = false;
  /// If original value of 'x' must be stored in 'v', not an updated one.
  bool IsPostfixUpdate = false;
};

bool OmpSsAtomicCompareCaptureChecker::checkType(ErrorInfoTy &ErrorInfo) {
  if (!OmpSsAtomicCompareChecker::checkType(ErrorInfo))
    return false;

  if (V && !CheckValue(V, ErrorInfo, true))
    return false;

  if (R && !CheckValue(R, ErrorInfo, true, true))
    return false;

  return true;
}

bool OmpSsAtomicCompareCaptureChecker::checkForm3(IfStmt *S,
                                                   ErrorInfoTy &ErrorInfo) {
  IsFailOnly = true;

  auto *Then = S->getThen();
  if (auto *CS = dyn_cast<CompoundStmt>(Then)) {
    if (CS->body_empty()) {
      ErrorInfo.Error = ErrorTy::NoStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }
    if (CS->size() > 1) {
      ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }
    Then = CS->body_front();
  }

  auto *BO = dyn_cast<BinaryOperator>(Then);
  if (!BO) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Then->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Then->getSourceRange();
    return false;
  }
  if (BO->getOpcode() != BO_Assign) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = BO->getExprLoc();
    ErrorInfo.NoteLoc = BO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
    return false;
  }

  X = BO->getLHS();
  D = BO->getRHS();

  auto *Cond = dyn_cast<BinaryOperator>(S->getCond());
  if (!Cond) {
    ErrorInfo.Error = ErrorTy::NotABinaryOp;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getCond()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getCond()->getSourceRange();
    return false;
  }
  if (Cond->getOpcode() != BO_EQ) {
    ErrorInfo.Error = ErrorTy::NotEQ;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
    return false;
  }

  if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getLHS())) {
    E = Cond->getRHS();
  } else if (checkIfTwoExprsAreSame(ContextRef, X, Cond->getRHS())) {
    E = Cond->getLHS();
  } else {
    ErrorInfo.Error = ErrorTy::InvalidComparison;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Cond->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Cond->getSourceRange();
    return false;
  }

  C = Cond;

  if (!S->getElse()) {
    ErrorInfo.Error = ErrorTy::NoElse;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getSourceRange();
    return false;
  }

  auto *Else = S->getElse();
  if (auto *CS = dyn_cast<CompoundStmt>(Else)) {
    if (CS->body_empty()) {
      ErrorInfo.Error = ErrorTy::NoStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
      return false;
    }
    if (CS->size() > 1) {
      ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getSourceRange();
      return false;
    }
    Else = CS->body_front();
  }

  auto *ElseBO = dyn_cast<BinaryOperator>(Else);
  if (!ElseBO) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Else->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Else->getSourceRange();
    return false;
  }
  if (ElseBO->getOpcode() != BO_Assign) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ElseBO->getExprLoc();
    ErrorInfo.NoteLoc = ElseBO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ElseBO->getSourceRange();
    return false;
  }

  if (!checkIfTwoExprsAreSame(ContextRef, X, ElseBO->getRHS())) {
    ErrorInfo.Error = ErrorTy::InvalidAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = ElseBO->getRHS()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange =
        ElseBO->getRHS()->getSourceRange();
    return false;
  }

  V = ElseBO->getLHS();

  return checkType(ErrorInfo);
}

bool OmpSsAtomicCompareCaptureChecker::checkForm45(Stmt *S,
                                                    ErrorInfoTy &ErrorInfo) {
  // We don't check here as they should be already done before call this
  // function.
  auto *CS = cast<CompoundStmt>(S);
  assert(CS->size() == 2 && "CompoundStmt size is not expected");
  auto *S1 = cast<BinaryOperator>(CS->body_front());
  auto *S2 = cast<IfStmt>(CS->body_back());
  assert(S1->getOpcode() == BO_Assign && "unexpected binary operator");

  if (!checkIfTwoExprsAreSame(ContextRef, S1->getLHS(), S2->getCond())) {
    ErrorInfo.Error = ErrorTy::InvalidCondition;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S2->getCond()->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S1->getLHS()->getSourceRange();
    return false;
  }

  R = S1->getLHS();

  auto *Then = S2->getThen();
  if (auto *ThenCS = dyn_cast<CompoundStmt>(Then)) {
    if (ThenCS->body_empty()) {
      ErrorInfo.Error = ErrorTy::NoStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = ThenCS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ThenCS->getSourceRange();
      return false;
    }
    if (ThenCS->size() > 1) {
      ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = ThenCS->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ThenCS->getSourceRange();
      return false;
    }
    Then = ThenCS->body_front();
  }

  auto *ThenBO = dyn_cast<BinaryOperator>(Then);
  if (!ThenBO) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S2->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S2->getSourceRange();
    return false;
  }
  if (ThenBO->getOpcode() != BO_Assign) {
    ErrorInfo.Error = ErrorTy::NotAnAssignment;
    ErrorInfo.ErrorLoc = ThenBO->getExprLoc();
    ErrorInfo.NoteLoc = ThenBO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ThenBO->getSourceRange();
    return false;
  }

  X = ThenBO->getLHS();
  D = ThenBO->getRHS();

  auto *BO = cast<BinaryOperator>(S1->getRHS()->IgnoreImpCasts());
  if (BO->getOpcode() != BO_EQ) {
    ErrorInfo.Error = ErrorTy::NotEQ;
    ErrorInfo.ErrorLoc = BO->getExprLoc();
    ErrorInfo.NoteLoc = BO->getOperatorLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
    return false;
  }

  C = BO;

  if (checkIfTwoExprsAreSame(ContextRef, X, BO->getLHS())) {
    E = BO->getRHS();
  } else if (checkIfTwoExprsAreSame(ContextRef, X, BO->getRHS())) {
    E = BO->getLHS();
  } else {
    ErrorInfo.Error = ErrorTy::InvalidComparison;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = BO->getExprLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
    return false;
  }

  if (S2->getElse()) {
    IsFailOnly = true;

    auto *Else = S2->getElse();
    if (auto *ElseCS = dyn_cast<CompoundStmt>(Else)) {
      if (ElseCS->body_empty()) {
        ErrorInfo.Error = ErrorTy::NoStmt;
        ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = ElseCS->getBeginLoc();
        ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ElseCS->getSourceRange();
        return false;
      }
      if (ElseCS->size() > 1) {
        ErrorInfo.Error = ErrorTy::MoreThanOneStmt;
        ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = ElseCS->getBeginLoc();
        ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ElseCS->getSourceRange();
        return false;
      }
      Else = ElseCS->body_front();
    }

    auto *ElseBO = dyn_cast<BinaryOperator>(Else);
    if (!ElseBO) {
      ErrorInfo.Error = ErrorTy::NotAnAssignment;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = Else->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = Else->getSourceRange();
      return false;
    }
    if (ElseBO->getOpcode() != BO_Assign) {
      ErrorInfo.Error = ErrorTy::NotAnAssignment;
      ErrorInfo.ErrorLoc = ElseBO->getExprLoc();
      ErrorInfo.NoteLoc = ElseBO->getOperatorLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange = ElseBO->getSourceRange();
      return false;
    }
    if (!checkIfTwoExprsAreSame(ContextRef, X, ElseBO->getRHS())) {
      ErrorInfo.Error = ErrorTy::InvalidAssignment;
      ErrorInfo.ErrorLoc = ElseBO->getRHS()->getExprLoc();
      ErrorInfo.NoteLoc = X->getExprLoc();
      ErrorInfo.ErrorRange = ElseBO->getRHS()->getSourceRange();
      ErrorInfo.NoteRange = X->getSourceRange();
      return false;
    }

    V = ElseBO->getLHS();
  }

  return checkType(ErrorInfo);
}

bool OmpSsAtomicCompareCaptureChecker::checkStmt(Stmt *S,
                                                  ErrorInfoTy &ErrorInfo) {
  // if(x == e) { x = d; } else { v = x; }
  if (auto *IS = dyn_cast<IfStmt>(S))
    return checkForm3(IS, ErrorInfo);

  auto *CS = dyn_cast<CompoundStmt>(S);
  if (!CS) {
    ErrorInfo.Error = ErrorTy::NotCompoundStmt;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = S->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = S->getSourceRange();
    return false;
  }
  if (CS->body_empty()) {
    ErrorInfo.Error = ErrorTy::NoStmt;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
    return false;
  }

  // { if(x == e) { x = d; } else { v = x; } }
  if (CS->size() == 1) {
    auto *IS = dyn_cast<IfStmt>(CS->body_front());
    if (!IS) {
      ErrorInfo.Error = ErrorTy::NotIfStmt;
      ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->body_front()->getBeginLoc();
      ErrorInfo.ErrorRange = ErrorInfo.NoteRange =
          CS->body_front()->getSourceRange();
      return false;
    }

    return checkForm3(IS, ErrorInfo);
  } else if (CS->size() == 2) {
    auto *S1 = CS->body_front();
    auto *S2 = CS->body_back();

    Stmt *UpdateStmt = nullptr;
    Stmt *CondUpdateStmt = nullptr;
    Stmt *CondExprStmt = nullptr;

    if (auto *BO = dyn_cast<BinaryOperator>(S1)) {
      // It could be one of the following cases:
      // { v = x; cond-update-stmt }
      // { v = x; cond-expr-stmt }
      // { cond-expr-stmt; v = x; }
      // form 45
      if (isa<BinaryOperator>(BO->getRHS()->IgnoreImpCasts()) ||
          isa<ConditionalOperator>(BO->getRHS()->IgnoreImpCasts())) {
        // check if form 45
        if (isa<IfStmt>(S2))
          return checkForm45(CS, ErrorInfo);
        // { cond-expr-stmt; v = x; }
        CondExprStmt = S1;
        UpdateStmt = S2;
      } else {
        IsPostfixUpdate = true;
        UpdateStmt = S1;
        if (isa<IfStmt>(S2)) {
          // { v = x; cond-update-stmt }
          CondUpdateStmt = S2;
        } else {
          // { v = x; cond-expr-stmt }
          CondExprStmt = S2;
        }
      }
    } else {
      // { cond-update-stmt v = x; }
      UpdateStmt = S2;
      CondUpdateStmt = S1;
    }

    auto CheckCondUpdateStmt = [this, &ErrorInfo](Stmt *CUS) {
      auto *IS = dyn_cast<IfStmt>(CUS);
      if (!IS) {
        ErrorInfo.Error = ErrorTy::NotIfStmt;
        ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CUS->getBeginLoc();
        ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CUS->getSourceRange();
        return false;
      }

      return checkCondUpdateStmt(IS, ErrorInfo);
    };

    // CheckUpdateStmt has to be called *after* CheckCondUpdateStmt.
    auto CheckUpdateStmt = [this, &ErrorInfo](Stmt *US) {
      auto *BO = dyn_cast<BinaryOperator>(US);
      if (!BO) {
        ErrorInfo.Error = ErrorTy::NotAnAssignment;
        ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = US->getBeginLoc();
        ErrorInfo.ErrorRange = ErrorInfo.NoteRange = US->getSourceRange();
        return false;
      }
      if (BO->getOpcode() != BO_Assign) {
        ErrorInfo.Error = ErrorTy::NotAnAssignment;
        ErrorInfo.ErrorLoc = BO->getExprLoc();
        ErrorInfo.NoteLoc = BO->getOperatorLoc();
        ErrorInfo.ErrorRange = ErrorInfo.NoteRange = BO->getSourceRange();
        return false;
      }
      if (!checkIfTwoExprsAreSame(ContextRef, this->X, BO->getRHS())) {
        ErrorInfo.Error = ErrorTy::InvalidAssignment;
        ErrorInfo.ErrorLoc = BO->getRHS()->getExprLoc();
        ErrorInfo.NoteLoc = this->X->getExprLoc();
        ErrorInfo.ErrorRange = BO->getRHS()->getSourceRange();
        ErrorInfo.NoteRange = this->X->getSourceRange();
        return false;
      }

      this->V = BO->getLHS();

      return true;
    };

    if (CondUpdateStmt && !CheckCondUpdateStmt(CondUpdateStmt))
      return false;
    if (CondExprStmt && !checkCondExprStmt(CondExprStmt, ErrorInfo))
      return false;
    if (!CheckUpdateStmt(UpdateStmt))
      return false;
  } else {
    ErrorInfo.Error = ErrorTy::MoreThanTwoStmts;
    ErrorInfo.ErrorLoc = ErrorInfo.NoteLoc = CS->getBeginLoc();
    ErrorInfo.ErrorRange = ErrorInfo.NoteRange = CS->getSourceRange();
    return false;
  }

  return checkType(ErrorInfo);
}
} // namespace

StmtResult Sema::ActOnOmpSsAtomicDirective(ArrayRef<OSSClause *> Clauses,
                                            Stmt *AStmt,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();

  // 1.2.2 OmpSs-2 Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  OmpSsClauseKind AtomicKind = OSSC_unknown;
  SourceLocation AtomicKindLoc;
  OmpSsClauseKind MemOrderKind = OSSC_unknown;
  SourceLocation MemOrderLoc;
  bool MutexClauseEncountered = false;
  llvm::SmallSet<OmpSsClauseKind, 2> EncounteredAtomicKinds;
  for (const OSSClause *C : Clauses) {
    switch (C->getClauseKind()) {
    case OSSC_read:
    case OSSC_write:
    case OSSC_update:
      MutexClauseEncountered = true;
      [[fallthrough]];
    case OSSC_capture:
    case OSSC_compare: {
      if (AtomicKind != OSSC_unknown && MutexClauseEncountered) {
        Diag(C->getBeginLoc(), diag::err_oss_atomic_several_clauses)
            << SourceRange(C->getBeginLoc(), C->getEndLoc());
        Diag(AtomicKindLoc, diag::note_oss_previous_mem_order_clause)
            << getOmpSsClauseName(AtomicKind);
      } else {
        AtomicKind = C->getClauseKind();
        AtomicKindLoc = C->getBeginLoc();
        if (!EncounteredAtomicKinds.insert(C->getClauseKind()).second) {
          Diag(C->getBeginLoc(), diag::err_oss_atomic_several_clauses)
              << SourceRange(C->getBeginLoc(), C->getEndLoc());
          Diag(AtomicKindLoc, diag::note_oss_previous_mem_order_clause)
              << getOmpSsClauseName(AtomicKind);
        }
      }
      break;
    }
    case OSSC_seq_cst:
    case OSSC_acq_rel:
    case OSSC_acquire:
    case OSSC_release:
    case OSSC_relaxed: {
      if (MemOrderKind != OSSC_unknown) {
        Diag(C->getBeginLoc(), diag::err_oss_several_mem_order_clauses)
            << getOmpSsDirectiveName(OSSD_atomic) << 0
            << SourceRange(C->getBeginLoc(), C->getEndLoc());
        Diag(MemOrderLoc, diag::note_oss_previous_mem_order_clause)
            << getOmpSsClauseName(MemOrderKind);
      } else {
        MemOrderKind = C->getClauseKind();
        MemOrderLoc = C->getBeginLoc();
      }
      break;
    }
    default:
      llvm_unreachable("unknown clause is encountered");
    }
  }
  bool IsCompareCapture = false;
  if (EncounteredAtomicKinds.contains(OSSC_compare) &&
      EncounteredAtomicKinds.contains(OSSC_capture)) {
    IsCompareCapture = true;
    AtomicKind = OSSC_compare;
  }
  // OmpSs-2, 2.17.7 atomic Construct, Restrictions
  // If atomic-clause is read then memory-order-clause must not be acq_rel or
  // release.
  // If atomic-clause is write then memory-order-clause must not be acq_rel or
  // acquire.
  // If atomic-clause is update or not present then memory-order-clause must not
  // be acq_rel or acquire.
  if ((AtomicKind == OSSC_read &&
       (MemOrderKind == OSSC_acq_rel || MemOrderKind == OSSC_release)) ||
      ((AtomicKind == OSSC_write || AtomicKind == OSSC_update ||
        AtomicKind == OSSC_unknown) &&
       (MemOrderKind == OSSC_acq_rel || MemOrderKind == OSSC_acquire))) {
    SourceLocation Loc = AtomicKindLoc;
    if (AtomicKind == OSSC_unknown)
      Loc = StartLoc;
    Diag(Loc, diag::err_oss_atomic_incompatible_mem_order_clause)
        << getOmpSsClauseName(AtomicKind)
        << (AtomicKind == OSSC_unknown ? 1 : 0)
        << getOmpSsClauseName(MemOrderKind);
    Diag(MemOrderLoc, diag::note_oss_previous_mem_order_clause)
        << getOmpSsClauseName(MemOrderKind);
  }

  Stmt *Body = AStmt;
  if (auto *EWC = dyn_cast<ExprWithCleanups>(Body))
    Body = EWC->getSubExpr();

  Expr *X = nullptr;
  Expr *V = nullptr;
  Expr *E = nullptr;
  Expr *UE = nullptr;
  Expr *D = nullptr;
  Expr *CE = nullptr;
  Expr *R = nullptr;
  bool IsXLHSInRHSPart = false;
  bool IsPostfixUpdate = false;
  bool IsFailOnly = false;
  // OmpSs-2 [2.12.6, atomic Construct]
  // In the next expressions:
  // * x and v (as applicable) are both l-value expressions with scalar type.
  // * During the execution of an atomic region, multiple syntactic
  // occurrences of x must designate the same storage location.
  // * Neither of v and expr (as applicable) may access the storage location
  // designated by x.
  // * Neither of x and expr (as applicable) may access the storage location
  // designated by v.
  // * expr is an expression with scalar type.
  // * binop is one of +, *, -, /, &, ^, |, <<, or >>.
  // * binop, binop=, ++, and -- are not overloaded operators.
  // * The expression x binop expr must be numerically equivalent to x binop
  // (expr). This requirement is satisfied if the operators in expr have
  // precedence greater than binop, or by using parentheses around expr or
  // subexpressions of expr.
  // * The expression expr binop x must be numerically equivalent to (expr)
  // binop x. This requirement is satisfied if the operators in expr have
  // precedence equal to or greater than binop, or by using parentheses around
  // expr or subexpressions of expr.
  // * For forms that allow multiple occurrences of x, the number of times
  // that x is evaluated is unspecified.
  if (AtomicKind == OSSC_read) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is read:
    //  v = x;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        V = AtomicBinOp->getLHS()->IgnoreParenImpCasts();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (V->isInstantiationDependent() || V->getType()->isScalarType())) {
          if (!X->isLValue() || !V->isLValue()) {
            const Expr *NotLValueExpr = X->isLValue() ? V : X;
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = NotLValueExpr->getExprLoc();
            NoteRange = NotLValueExpr->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !V->isInstantiationDependent()) {
          const Expr *NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? V
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_oss_atomic_read_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_oss_atomic_read_write)
          << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      V = X = nullptr;
  } else if (AtomicKind == OSSC_write) {
    enum {
      NotAnExpression,
      NotAnAssignmentOp,
      NotAScalarType,
      NotAnLValue,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    // If clause is write:
    //  x = expr;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        X = AtomicBinOp->getLHS();
        E = AtomicBinOp->getRHS();
        if ((X->isInstantiationDependent() || X->getType()->isScalarType()) &&
            (E->isInstantiationDependent() || E->getType()->isScalarType())) {
          if (!X->isLValue()) {
            ErrorFound = NotAnLValue;
            ErrorLoc = AtomicBinOp->getExprLoc();
            ErrorRange = AtomicBinOp->getSourceRange();
            NoteLoc = X->getExprLoc();
            NoteRange = X->getSourceRange();
          }
        } else if (!X->isInstantiationDependent() ||
                   !E->isInstantiationDependent()) {
          const Expr *NotScalarExpr =
              (X->isInstantiationDependent() || X->getType()->isScalarType())
                  ? E
                  : X;
          ErrorFound = NotAScalarType;
          ErrorLoc = AtomicBinOp->getExprLoc();
          ErrorRange = AtomicBinOp->getSourceRange();
          NoteLoc = NotScalarExpr->getExprLoc();
          NoteRange = NotScalarExpr->getSourceRange();
        }
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorFound = NotAnAssignmentOp;
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
      }
    } else {
      ErrorFound = NotAnExpression;
      NoteLoc = ErrorLoc = Body->getBeginLoc();
      NoteRange = ErrorRange = SourceRange(NoteLoc, NoteLoc);
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_oss_atomic_write_not_expression_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_oss_atomic_read_write)
          << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      E = X = nullptr;
  } else if (AtomicKind == OSSC_update || AtomicKind == OSSC_unknown) {
    // If clause is update:
    //  x++;
    //  x--;
    //  ++x;
    //  --x;
    //  x binop= expr;
    //  x = x binop expr;
    //  x = expr binop x;
    OmpSsAtomicUpdateChecker Checker(*this);
    if (Checker.checkStatement(
            Body,
            (AtomicKind == OSSC_update)
                ? diag::err_oss_atomic_update_not_expression_statement
                : diag::err_oss_atomic_not_expression_statement,
            diag::note_oss_atomic_update))
      return StmtError();
    if (!CurContext->isDependentContext()) {
      E = Checker.getExpr();
      X = Checker.getX();
      UE = Checker.getUpdateExpr();
      IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
    }
  } else if (AtomicKind == OSSC_capture) {
    enum {
      NotAnAssignmentOp,
      NotACompoundStatement,
      NotTwoSubstatements,
      NotASpecificExpression,
      NoError
    } ErrorFound = NoError;
    SourceLocation ErrorLoc, NoteLoc;
    SourceRange ErrorRange, NoteRange;
    if (const auto *AtomicBody = dyn_cast<Expr>(Body)) {
      // If clause is a capture:
      //  v = x++;
      //  v = x--;
      //  v = ++x;
      //  v = --x;
      //  v = x binop= expr;
      //  v = x = x binop expr;
      //  v = x = expr binop x;
      const auto *AtomicBinOp =
          dyn_cast<BinaryOperator>(AtomicBody->IgnoreParenImpCasts());
      if (AtomicBinOp && AtomicBinOp->getOpcode() == BO_Assign) {
        V = AtomicBinOp->getLHS();
        Body = AtomicBinOp->getRHS()->IgnoreParenImpCasts();
        OmpSsAtomicUpdateChecker Checker(*this);
        if (Checker.checkStatement(
                Body, diag::err_oss_atomic_capture_not_expression_statement,
                diag::note_oss_atomic_update))
          return StmtError();
        E = Checker.getExpr();
        X = Checker.getX();
        UE = Checker.getUpdateExpr();
        IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
        IsPostfixUpdate = Checker.isPostfixUpdate();
      } else if (!AtomicBody->isInstantiationDependent()) {
        ErrorLoc = AtomicBody->getExprLoc();
        ErrorRange = AtomicBody->getSourceRange();
        NoteLoc = AtomicBinOp ? AtomicBinOp->getOperatorLoc()
                              : AtomicBody->getExprLoc();
        NoteRange = AtomicBinOp ? AtomicBinOp->getSourceRange()
                                : AtomicBody->getSourceRange();
        ErrorFound = NotAnAssignmentOp;
      }
      if (ErrorFound != NoError) {
        Diag(ErrorLoc, diag::err_oss_atomic_capture_not_expression_statement)
            << ErrorRange;
        Diag(NoteLoc, diag::note_oss_atomic_capture) << ErrorFound << NoteRange;
        return StmtError();
      }
      if (CurContext->isDependentContext())
        UE = V = E = X = nullptr;
    } else {
      // If clause is a capture:
      //  { v = x; x = expr; }
      //  { v = x; x++; }
      //  { v = x; x--; }
      //  { v = x; ++x; }
      //  { v = x; --x; }
      //  { v = x; x binop= expr; }
      //  { v = x; x = x binop expr; }
      //  { v = x; x = expr binop x; }
      //  { x++; v = x; }
      //  { x--; v = x; }
      //  { ++x; v = x; }
      //  { --x; v = x; }
      //  { x binop= expr; v = x; }
      //  { x = x binop expr; v = x; }
      //  { x = expr binop x; v = x; }
      if (auto *CS = dyn_cast<CompoundStmt>(Body)) {
        // Check that this is { expr1; expr2; }
        if (CS->size() == 2) {
          Stmt *First = CS->body_front();
          Stmt *Second = CS->body_back();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(First))
            First = EWC->getSubExpr()->IgnoreParenImpCasts();
          if (auto *EWC = dyn_cast<ExprWithCleanups>(Second))
            Second = EWC->getSubExpr()->IgnoreParenImpCasts();
          // Need to find what subexpression is 'v' and what is 'x'.
          OmpSsAtomicUpdateChecker Checker(*this);
          bool IsUpdateExprFound = !Checker.checkStatement(Second);
          BinaryOperator *BinOp = nullptr;
          if (IsUpdateExprFound) {
            BinOp = dyn_cast<BinaryOperator>(First);
            IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
          }
          if (IsUpdateExprFound && !CurContext->isDependentContext()) {
            //  { v = x; x++; }
            //  { v = x; x--; }
            //  { v = x; ++x; }
            //  { v = x; --x; }
            //  { v = x; x binop= expr; }
            //  { v = x; x = x binop expr; }
            //  { v = x; x = expr binop x; }
            // Check that the first expression has form v = x.
            Expr *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
            llvm::FoldingSetNodeID XId, PossibleXId;
            Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
            PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
            IsUpdateExprFound = XId == PossibleXId;
            if (IsUpdateExprFound) {
              V = BinOp->getLHS();
              X = Checker.getX();
              E = Checker.getExpr();
              UE = Checker.getUpdateExpr();
              IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
              IsPostfixUpdate = true;
            }
          }
          if (!IsUpdateExprFound) {
            IsUpdateExprFound = !Checker.checkStatement(First);
            BinOp = nullptr;
            if (IsUpdateExprFound) {
              BinOp = dyn_cast<BinaryOperator>(Second);
              IsUpdateExprFound = BinOp && BinOp->getOpcode() == BO_Assign;
            }
            if (IsUpdateExprFound && !CurContext->isDependentContext()) {
              //  { x++; v = x; }
              //  { x--; v = x; }
              //  { ++x; v = x; }
              //  { --x; v = x; }
              //  { x binop= expr; v = x; }
              //  { x = x binop expr; v = x; }
              //  { x = expr binop x; v = x; }
              // Check that the second expression has form v = x.
              Expr *PossibleX = BinOp->getRHS()->IgnoreParenImpCasts();
              llvm::FoldingSetNodeID XId, PossibleXId;
              Checker.getX()->Profile(XId, Context, /*Canonical=*/true);
              PossibleX->Profile(PossibleXId, Context, /*Canonical=*/true);
              IsUpdateExprFound = XId == PossibleXId;
              if (IsUpdateExprFound) {
                V = BinOp->getLHS();
                X = Checker.getX();
                E = Checker.getExpr();
                UE = Checker.getUpdateExpr();
                IsXLHSInRHSPart = Checker.isXLHSInRHSPart();
                IsPostfixUpdate = false;
              }
            }
          }
          if (!IsUpdateExprFound) {
            //  { v = x; x = expr; }
            auto *FirstExpr = dyn_cast<Expr>(First);
            auto *SecondExpr = dyn_cast<Expr>(Second);
            if (!FirstExpr || !SecondExpr ||
                !(FirstExpr->isInstantiationDependent() ||
                  SecondExpr->isInstantiationDependent())) {
              auto *FirstBinOp = dyn_cast<BinaryOperator>(First);
              if (!FirstBinOp || FirstBinOp->getOpcode() != BO_Assign) {
                ErrorFound = NotAnAssignmentOp;
                NoteLoc = ErrorLoc = FirstBinOp ? FirstBinOp->getOperatorLoc()
                                                : First->getBeginLoc();
                NoteRange = ErrorRange = FirstBinOp
                                             ? FirstBinOp->getSourceRange()
                                             : SourceRange(ErrorLoc, ErrorLoc);
              } else {
                auto *SecondBinOp = dyn_cast<BinaryOperator>(Second);
                if (!SecondBinOp || SecondBinOp->getOpcode() != BO_Assign) {
                  ErrorFound = NotAnAssignmentOp;
                  NoteLoc = ErrorLoc = SecondBinOp
                                           ? SecondBinOp->getOperatorLoc()
                                           : Second->getBeginLoc();
                  NoteRange = ErrorRange =
                      SecondBinOp ? SecondBinOp->getSourceRange()
                                  : SourceRange(ErrorLoc, ErrorLoc);
                } else {
                  Expr *PossibleXRHSInFirst =
                      FirstBinOp->getRHS()->IgnoreParenImpCasts();
                  Expr *PossibleXLHSInSecond =
                      SecondBinOp->getLHS()->IgnoreParenImpCasts();
                  llvm::FoldingSetNodeID X1Id, X2Id;
                  PossibleXRHSInFirst->Profile(X1Id, Context,
                                               /*Canonical=*/true);
                  PossibleXLHSInSecond->Profile(X2Id, Context,
                                                /*Canonical=*/true);
                  IsUpdateExprFound = X1Id == X2Id;
                  if (IsUpdateExprFound) {
                    V = FirstBinOp->getLHS();
                    X = SecondBinOp->getLHS();
                    E = SecondBinOp->getRHS();
                    UE = nullptr;
                    IsXLHSInRHSPart = false;
                    IsPostfixUpdate = true;
                  } else {
                    ErrorFound = NotASpecificExpression;
                    ErrorLoc = FirstBinOp->getExprLoc();
                    ErrorRange = FirstBinOp->getSourceRange();
                    NoteLoc = SecondBinOp->getLHS()->getExprLoc();
                    NoteRange = SecondBinOp->getRHS()->getSourceRange();
                  }
                }
              }
            }
          }
        } else {
          NoteLoc = ErrorLoc = Body->getBeginLoc();
          NoteRange = ErrorRange =
              SourceRange(Body->getBeginLoc(), Body->getBeginLoc());
          ErrorFound = NotTwoSubstatements;
        }
      } else {
        NoteLoc = ErrorLoc = Body->getBeginLoc();
        NoteRange = ErrorRange =
            SourceRange(Body->getBeginLoc(), Body->getBeginLoc());
        ErrorFound = NotACompoundStatement;
      }
    }
    if (ErrorFound != NoError) {
      Diag(ErrorLoc, diag::err_oss_atomic_capture_not_compound_statement)
          << ErrorRange;
      Diag(NoteLoc, diag::note_oss_atomic_capture) << ErrorFound << NoteRange;
      return StmtError();
    }
    if (CurContext->isDependentContext())
      UE = V = E = X = nullptr;
  } else if (AtomicKind == OSSC_compare) {
    if (IsCompareCapture) {
      OmpSsAtomicCompareCaptureChecker::ErrorInfoTy ErrorInfo;
      OmpSsAtomicCompareCaptureChecker Checker(*this);
      if (!Checker.checkStmt(Body, ErrorInfo)) {
        Diag(ErrorInfo.ErrorLoc, diag::err_oss_atomic_compare_capture)
            << ErrorInfo.ErrorRange;
        Diag(ErrorInfo.NoteLoc, diag::note_oss_atomic_compare)
            << ErrorInfo.Error << ErrorInfo.NoteRange;
        return StmtError();
      }
      X = Checker.getX();
      E = Checker.getE();
      D = Checker.getD();
      CE = Checker.getCond();
      V = Checker.getV();
      R = Checker.getR();
      // We reuse IsXLHSInRHSPart to tell if it is in the form 'x ordop expr'.
      IsXLHSInRHSPart = Checker.isXBinopExpr();
      IsFailOnly = Checker.isFailOnly();
      IsPostfixUpdate = Checker.isPostfixUpdate();
    } else {
      OmpSsAtomicCompareChecker::ErrorInfoTy ErrorInfo;
      OmpSsAtomicCompareChecker Checker(*this);
      if (!Checker.checkStmt(Body, ErrorInfo)) {
        Diag(ErrorInfo.ErrorLoc, diag::err_oss_atomic_compare)
            << ErrorInfo.ErrorRange;
        Diag(ErrorInfo.NoteLoc, diag::note_oss_atomic_compare)
          << ErrorInfo.Error << ErrorInfo.NoteRange;
        return StmtError();
      }
      X = Checker.getX();
      E = Checker.getE();
      D = Checker.getD();
      CE = Checker.getCond();
      // We reuse IsXLHSInRHSPart to tell if it is in the form 'x ordop expr'.
      IsXLHSInRHSPart = Checker.isXBinopExpr();
    }
  }

  setFunctionHasBranchProtectedScope();

  return OSSAtomicDirective::Create(
      Context, StartLoc, EndLoc, Clauses, AStmt,
      {X, V, R, E, UE, D, CE, IsXLHSInRHSPart, IsPostfixUpdate, IsFailOnly});
}

namespace {
// This visitor looks for global variables in a expression
// and gives an error
class OSSGlobalFinderVisitor
  : public ConstStmtVisitor<OSSGlobalFinderVisitor, void> {
  Sema &S;
  bool ErrorFound = false;

public:
  OSSGlobalFinderVisitor(Sema &S)
    : S(S)
      {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void VisitStmt(const Stmt *S) {
    for (const Stmt *C : S->children()) {
      if (C) {
        Visit(C);
      }
    }
  }

  void VisitOSSMultiDepExpr(const OSSMultiDepExpr *E) {
    Visit(E->getDepExpr());

    for (size_t i = 0; i < E->getDepInits().size(); ++i) {
      Visit(E->getDepInits()[i]);
      if (E->getDepSizes()[i])
        Visit(E->getDepSizes()[i]);
      if (E->getDepSteps()[i])
        Visit(E->getDepSteps()[i]);
    }
  }

  void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
    Visit(E->getBase());

    for (const Expr *S : E->getShapes())
      Visit(S);
  }

  void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
    Visit(E->getBase());

    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    Visit(E->getBase());
    Visit(E->getIdx());
  }

  void VisitUnaryOperator(const UnaryOperator *E) {
    Visit(E->getSubExpr());
  }

  void VisitMemberExpr(const MemberExpr *E) {
    Visit(E->getBase());
  }

  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;
    if (const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->hasGlobalStorage()) {
        S.Diag(E->getExprLoc(), diag::err_oss_global_variable)
            << E->getSourceRange();
        ErrorFound = true;
      }
    }
  }

  bool isErrorFound() const { return ErrorFound; }
};
} // end namespace

static bool checkDependency(Sema &S, Expr *RefExpr, bool OSSSyntax, bool Outline) {
  SourceLocation ELoc = RefExpr->getExprLoc();
  Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
  if (RefExpr->containsUnexpandedParameterPack()) {
    S.Diag(RefExpr->getExprLoc(), diag::err_oss_variadic_templates_not_clause_allowed);
    return false;
  } else if (RefExpr->isTypeDependent() || RefExpr->isValueDependent()) {
    // It will be analyzed later.
    return true;
  }

  if (S.RequireCompleteExprType(RefExpr, diag::err_oss_incomplete_type))
    return false;

  auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
  // Allow only LValues, forbid ArraySubscripts over things
  // that are not an array like:
  //   typedef float V __attribute__((vector_size(16)));
  //   V a;
  //   #pragma oss task in(a[3])
  // and functions:
  //   void foo() { #pragma oss task in(foo) {} }
  if (RefExpr->IgnoreParenImpCasts()->getType()->isFunctionType() ||
      !RefExpr->IgnoreParenImpCasts()->isLValue() ||
      (ASE &&
       !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
       !ASE->getBase()->getType().getNonReferenceType()->isArrayType())) {
    if (!Outline)
      S.Diag(ELoc, diag::err_oss_expected_addressable_lvalue_or_array_item)
          << RefExpr->getSourceRange();
    else
      S.Diag(ELoc, diag::err_oss_expected_lvalue_reference_or_global_or_dereference_or_array_item)
          << 0 << RefExpr->getSourceRange();
    return false;
  }

  if (Outline) {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(RefExpr->IgnoreParenImpCasts())) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (!(VD->getType()->isReferenceType() || VD->hasGlobalStorage())) {
          S.Diag(ELoc, diag::err_oss_expected_lvalue_reference_or_global_or_dereference_or_array_item)
              << 0 << DRE->getSourceRange();
          return false;
        }
      }
    }
  }

  class CheckCallExpr
      : public ConstStmtVisitor<CheckCallExpr, bool> {
  // This Visitor checks the base of the
  // dependency is over a CallExpr, which is error.
  // int *get();
  // auto l = []() -> int * {...};
  // #pragma oss task in(get()[1], l()[3])
  public:
    bool VisitOSSMultiDepExpr(const OSSMultiDepExpr *E) {
      return Visit(E->getDepExpr());
    }

    bool VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
      return Visit(E->getBase());
    }

    bool VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
      return Visit(E->getBase());
    }

    bool VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
      return Visit(E->getBase());
    }

    bool VisitUnaryOperator(const UnaryOperator *E) {
      return Visit(E->getSubExpr());
    }

    bool VisitMemberExpr(const MemberExpr *E) {
      return Visit(E->getBase());
    }

    bool VisitCallExpr(const CallExpr *E) {
      return true;
    }
  };
  CheckCallExpr CCE;
  if (CCE.Visit(RefExpr)) {
    S.Diag(ELoc, diag::err_oss_call_expr_support)
        << RefExpr->getSourceRange();
    return false;
  }

  bool InvalidArraySection = false;
  while (auto *OASE = dyn_cast<OSSArraySectionExpr>(SimpleExpr)) {
    if (!OASE->isColonForm() && !OSSSyntax) {
      S.Diag(OASE->getColonLoc(), diag::err_oss_section_invalid_form)
          << RefExpr->getSourceRange();
      // Only diagnose the first error
      InvalidArraySection = true;
      break;
    }
    SimpleExpr = OASE->getBase()->IgnoreParenImpCasts();
  }
  if (InvalidArraySection)
    return false;
  return true;
}

static bool checkNdrange(
    Sema &S, SourceLocation Loc, ArrayRef<Expr *> VL,
    SmallVectorImpl<Expr *> &ClauseVars, bool Outline) {

  ClauseVars.append(VL.begin(), VL.end());

  Expr *NumDimsE = ClauseVars[0];
  // The parameter of the collapse clause must be a constant
  // positive integer expression.
  ExprResult NumDimsResult =
      S.VerifyPositiveIntegerConstant(NumDimsE, OSSC_ndrange, /*StrictlyPositive=*/true);
  if (NumDimsResult.isInvalid())
    return false;
  NumDimsE = NumDimsResult.get();
  if (!isa<ConstantExpr>(NumDimsE))
    return true;

  uint64_t NumDims = cast<ConstantExpr>(NumDimsE)->getResultAsAPSInt().getExtValue();
  if (!(NumDims >= 1 && NumDims <= 3)) {
    S.Diag(NumDimsE->getExprLoc(), diag::err_oss_clause_expect_constant_between)
        << 1 << 3 << getOmpSsClauseName(OSSC_ndrange)
        << NumDimsE->getSourceRange();
    return false;
  }
  if (NumDims + 1 != ClauseVars.size() &&
      NumDims*2 + 1 != ClauseVars.size()) {
    S.Diag(Loc, diag::err_oss_ndrange_expect_nelems)
      << NumDims << NumDims << NumDims*2 << ClauseVars.size() - 1;
    return false;
  }
  ClauseVars[0] = NumDimsE;
  bool ErrorFound = false;
  for (size_t i = 1; i < ClauseVars.size(); ++i) {
    // TODO: check global[i] >= local[i]
    ExprResult Res = S.CheckNonNegativeIntegerValue(
      ClauseVars[i], OSSC_ndrange, /*StrictlyPositive=*/true, Outline);
    if (Res.isInvalid())
      ErrorFound = true;
    ClauseVars[i] = Res.get();
  }
  return !ErrorFound;
}

namespace {
/// Data for the reduction-based clauses.
struct ReductionData {
  /// List of original reduction items.
  SmallVector<Expr *, 8> Vars;
  /// LHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> LHSs;
  /// RHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> RHSs;
  /// Reduction operation expression.
  SmallVector<Expr *, 8> ReductionOps;
  /// Reduction operation kind. BO_Comma stands for UDR
  SmallVector<BinaryOperatorKind, 8> ReductionKinds;
  ReductionData() = delete;
  /// Reserves required memory for the reduction data.
  ReductionData(unsigned Size) {
    Vars.reserve(Size);
    LHSs.reserve(Size);
    RHSs.reserve(Size);
    ReductionOps.reserve(Size);
    ReductionKinds.reserve(Size);
  }
  /// Stores reduction item and reduction operation only (required for dependent
  /// reduction item).
  void push(Expr *Item, Expr *ReductionOp) {
    Vars.emplace_back(Item);
    LHSs.emplace_back(nullptr);
    RHSs.emplace_back(nullptr);
    ReductionOps.emplace_back(ReductionOp);
    ReductionKinds.emplace_back(BO_Comma);
  }
  /// Stores reduction data.
  void push(Expr *Item, Expr *LHS, Expr *RHS, Expr *ReductionOp,
            BinaryOperatorKind BOK) {
    Vars.emplace_back(Item);
    LHSs.emplace_back(LHS);
    RHSs.emplace_back(RHS);
    ReductionOps.emplace_back(ReductionOp);
    ReductionKinds.emplace_back(BOK);
  }
};
} // namespace

// Fwd declaration
static bool actOnOSSReductionKindClause(
    Sema &S, DSAStackTy *Stack, OmpSsClauseKind ClauseKind,
    ArrayRef<Expr *> VarList, CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId, ArrayRef<Expr *> UnresolvedReductions,
    ReductionData &RD, bool Outline);

Sema::DeclGroupPtrTy Sema::ActOnOmpSsDeclareTaskDirective(
    DeclGroupPtrTy DG,
    Expr *If, Expr *Final, Expr *Cost, Expr *Priority,
    Expr *Shmem, Expr *Onready, bool Wait,
    unsigned Device, SourceLocation DeviceLoc,
    ArrayRef<Expr *> Labels,
    ArrayRef<Expr *> Ins, ArrayRef<Expr *> Outs, ArrayRef<Expr *> Inouts,
    ArrayRef<Expr *> Concurrents, ArrayRef<Expr *> Commutatives,
    ArrayRef<Expr *> WeakIns, ArrayRef<Expr *> WeakOuts,
    ArrayRef<Expr *> WeakInouts,
    ArrayRef<Expr *> WeakConcurrents, ArrayRef<Expr *> WeakCommutatives,
    ArrayRef<Expr *> DepIns, ArrayRef<Expr *> DepOuts, ArrayRef<Expr *> DepInouts,
    ArrayRef<Expr *> DepConcurrents, ArrayRef<Expr *> DepCommutatives,
    ArrayRef<Expr *> DepWeakIns, ArrayRef<Expr *> DepWeakOuts,
    ArrayRef<Expr *> DepWeakInouts,
    ArrayRef<Expr *> DepWeakConcurrents, ArrayRef<Expr *> DepWeakCommutatives,
    ArrayRef<unsigned> ReductionListSizes,
    ArrayRef<Expr *> Reductions,
    ArrayRef<unsigned> ReductionClauseType,
    ArrayRef<CXXScopeSpec> ReductionCXXScopeSpecs,
    ArrayRef<DeclarationNameInfo> ReductionIds,
    ArrayRef<Expr *> Ndranges, SourceLocation NdrangeLoc,
    SourceRange SR,
    ArrayRef<Expr *> UnresolvedReductions) {
  if (!DG || DG.get().isNull())
    return DeclGroupPtrTy();

  if (!DG.get().isSingleDecl()) {
    Diag(SR.getBegin(), diag::err_oss_single_decl_in_task);
    return DG;
  }
  Decl *ADecl = DG.get().getSingleDecl();
  if (auto *FTD = dyn_cast<FunctionTemplateDecl>(ADecl))
    ADecl = FTD->getTemplatedDecl();

  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_oss_function_expected);
    return DeclGroupPtrTy();
  }
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isVirtual() || isa<CXXConstructorDecl>(MD)
        || isa<CXXDestructorDecl>(MD)
        || MD->isOverloadedOperator()) {
      Diag(ADecl->getLocation(), diag::err_oss_function_expected) << 1;
      return DeclGroupPtrTy();
    }
    // Member tasks outlines with device != smp are
    // not supported
    const unsigned DeviceNotSeen = OSSC_DEVICE_unknown + 1;
    if (!(Device == OSSC_DEVICE_smp
        || Device == DeviceNotSeen)) {
      Diag(DeviceLoc, diag::err_oss_member_device_no_smp)
        << getOmpSsSimpleClauseTypeName(OSSC_device, Device);
      return DeclGroupPtrTy();
    }
  }
  if (FD->getReturnType() != Context.VoidTy) {
    Diag(ADecl->getLocation(), diag::err_oss_non_void_task);
    return DeclGroupPtrTy();
  }

  FunctionDecl *PrevFD = FD->getPreviousDecl();
  if (PrevFD && PrevFD != FD) {
    bool IsPrevTask = PrevFD->hasAttr<OSSTaskDeclAttr>();

    Diag(ADecl->getLocation(), diag::warn_oss_task_redeclaration)
      << IsPrevTask;
    Diag(PrevFD->getLocation(), diag::note_previous_decl)
      << PrevFD;
  }

  auto ParI = FD->param_begin();
  while (ParI != FD->param_end()) {
    QualType Type = (*ParI)->getType();
    if (!Type->isDependentType()
        && !Type.isPODType(Context)
        && !Type->isReferenceType()) {
      Diag((*ParI)->getBeginLoc(), diag::err_oss_non_pod_parm_task);
    }
    ++ParI;
  }

  // Add dummy to catch potential instantiations of the class that contains us
  // or our associated function
  ADecl->addAttr(OSSTaskDeclSentinelAttr::CreateImplicit(Context, SR));

  ExprResult IfRes, FinalRes, CostRes, PriorityRes, ShmemRes, OnreadyRes;
  SmallVector<Expr *, 2> LabelsRes;
  SmallVector<Expr *, 4> NdrangesRes;
  OSSTaskDeclAttr::DeviceType DevType = OSSTaskDeclAttr::DeviceType::Unknown;
  if (If) {
    IfRes = VerifyBooleanConditionWithCleanups(If, If->getExprLoc());
  }
  if (Final) {
    FinalRes = VerifyBooleanConditionWithCleanups(Final, Final->getExprLoc());
  }
  if (Cost) {
    CostRes = CheckNonNegativeIntegerValue(
      Cost, OSSC_cost, /*StrictlyPositive=*/false, /*Outline=*/true);
  }
  if (Priority) {
    PriorityRes = CheckSignedIntegerValue(Priority, /*Outline=*/true);
  }
  if (!Labels.empty()) {
    LabelsRes.push_back(
      CheckIsConstCharPtrConvertibleExpr(Labels[0], /*ConstConstraint=*/true).get());
    if (Labels.size() == 2)
      LabelsRes.push_back(
        CheckIsConstCharPtrConvertibleExpr(Labels[1], /*ConstConstraint=*/false).get());
  }
  if (Shmem) {
    ShmemRes = CheckNonNegativeIntegerValue(
      Shmem, OSSC_shmem, /*StrictlyPositive=*/false, /*Outline=*/true);
  }
  if (Onready) {
    OnreadyRes = Onready;
  }
  switch (Device) {
  case OSSC_DEVICE_smp:
    DevType = OSSTaskDeclAttr::DeviceType::Smp;
    break;
  case OSSC_DEVICE_cuda:
    DevType = OSSTaskDeclAttr::DeviceType::Cuda;
    break;
  case OSSC_DEVICE_opencl:
    DevType = OSSTaskDeclAttr::DeviceType::Opencl;
    break;
  case OSSC_DEVICE_fpga:
    DevType = OSSTaskDeclAttr::DeviceType::Fpga;
    break;
  case OSSC_DEVICE_unknown:
    Diag(DeviceLoc, diag::err_oss_unexpected_clause_value)
        << getListOfPossibleValues(OSSC_device, /*First=*/0,
                                   /*Last=*/OSSC_DEVICE_unknown)
        << getOmpSsClauseName(OSSC_device);
  }
  OSSClauseDSAChecker OSSClauseChecker(/*Stack=*/nullptr, *this);
  for (Expr *RefExpr : Ins) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_in);
  }
  for (Expr *RefExpr : Outs) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_out);
  }
  for (Expr *RefExpr : Inouts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_inout);
  }
  for (Expr *RefExpr : Concurrents) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_concurrent);
  }
  for (Expr *RefExpr : Commutatives) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_commutative);
  }
  for (Expr *RefExpr : WeakIns) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakin);
  }
  for (Expr *RefExpr : WeakOuts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakout);
  }
  for (Expr *RefExpr : WeakInouts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakinout);
  }
  for (Expr *RefExpr : WeakConcurrents) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakconcurrent);
  }
  for (Expr *RefExpr : WeakCommutatives) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/true, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakcommutative);
  }
  for (Expr *RefExpr : DepIns) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_in);
  }
  for (Expr *RefExpr : DepOuts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_out);
  }
  for (Expr *RefExpr : DepInouts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_inout);
  }
  for (Expr *RefExpr : DepConcurrents) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_concurrent);
  }
  for (Expr *RefExpr : DepCommutatives) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_commutative);
  }
  for (Expr *RefExpr : DepWeakIns) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakin);
  }
  for (Expr *RefExpr : DepWeakOuts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakout);
  }
  for (Expr *RefExpr : DepWeakInouts) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakinout);
  }
  for (Expr *RefExpr : DepWeakConcurrents) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakconcurrent);
  }
  for (Expr *RefExpr : DepWeakCommutatives) {
    checkDependency(*this, RefExpr, /*OSSSyntax=*/false, /*Outline=*/true);
    OSSClauseChecker.VisitClauseExpr(RefExpr, OSSC_weakcommutative);
  }
  ReductionData RD(Reductions.size());
  SmallVector<NestedNameSpecifierLoc, 4> ReductionNSLoc;
  auto UnresolvedReductions_it = UnresolvedReductions.begin();
  auto Reductions_it = Reductions.begin();
  for (size_t i = 0; i < ReductionListSizes.size(); ++i) {
    ArrayRef<Expr *> TmpList(Reductions_it, Reductions_it + ReductionListSizes[i]);
    OmpSsClauseKind CKind = (OmpSsClauseKind)ReductionClauseType[i];
    CXXScopeSpec ScopeSpec = ReductionCXXScopeSpecs[i];
    // UnresolvedReductions is std::nullopt when parsing the first time. Pass
    // std::nullopt.
    // In instantiation we will get an array with all the info for the reductions, build
    // the subarray associated to each reduction list (like with TmpList
    if (UnresolvedReductions.empty()) {
      actOnOSSReductionKindClause(
        *this, DSAStack, CKind, TmpList, ScopeSpec, ReductionIds[i],
        std::nullopt, RD, /*Outline=*/true);
    } else {
      actOnOSSReductionKindClause(
        *this, DSAStack, CKind, TmpList, ScopeSpec, ReductionIds[i],
        ArrayRef<Expr *>(UnresolvedReductions_it, UnresolvedReductions_it + ReductionListSizes[i]),
        RD, /*Outline=*/true);
    }

    for (Expr *RefExpr : TmpList)
      OSSClauseChecker.VisitClauseExpr(RefExpr, CKind);

    ReductionNSLoc.push_back(ReductionCXXScopeSpecs[i].getWithLocInContext(Context));
    Reductions_it += ReductionListSizes[i];
    UnresolvedReductions_it += ReductionListSizes[i];
  }
  if (!Ndranges.empty()) {
    if (!(DevType == OSSTaskDeclAttr::DeviceType::Cuda
        || DevType == OSSTaskDeclAttr::DeviceType::Opencl))
      Diag(DeviceLoc, diag::err_oss_ndrange_incompatible_device);

    checkNdrange(*this, NdrangeLoc, Ndranges, NdrangesRes, /*Outline=*/true);
  } else if (Shmem) {
    // It is an error to specify shmem without ndrange
    Diag(Shmem->getExprLoc(), diag::err_oss_shmem_without_ndrange);
  }

  // FIXME: the specs says the underlying type of a enum
  // is implementation defined. I do this to be able to compile
  // but it has to be done in a better way. TableGen does not
  // have a BinaryOperatorKind type.
  SmallVector<unsigned, 4> TmpReductionKinds;
  for (BinaryOperatorKind &b : RD.ReductionKinds)
    TmpReductionKinds.push_back(b);

  auto *NewAttr = OSSTaskDeclAttr::CreateImplicit(
    Context,
    IfRes.get(), FinalRes.get(), CostRes.get(), PriorityRes.get(),
    ShmemRes.get(), Wait, DevType,
    OnreadyRes.get(),
    const_cast<Expr **>(LabelsRes.data()), LabelsRes.size(),
    const_cast<Expr **>(Ins.data()), Ins.size(),
    const_cast<Expr **>(Outs.data()), Outs.size(),
    const_cast<Expr **>(Inouts.data()), Inouts.size(),
    const_cast<Expr **>(Concurrents.data()), Concurrents.size(),
    const_cast<Expr **>(Commutatives.data()), Commutatives.size(),
    const_cast<Expr **>(WeakIns.data()), WeakIns.size(),
    const_cast<Expr **>(WeakOuts.data()), WeakOuts.size(),
    const_cast<Expr **>(WeakInouts.data()), WeakInouts.size(),
    const_cast<Expr **>(WeakConcurrents.data()), WeakConcurrents.size(),
    const_cast<Expr **>(WeakCommutatives.data()), WeakCommutatives.size(),
    const_cast<Expr **>(DepIns.data()), DepIns.size(),
    const_cast<Expr **>(DepOuts.data()), DepOuts.size(),
    const_cast<Expr **>(DepInouts.data()), DepInouts.size(),
    const_cast<Expr **>(DepConcurrents.data()), DepConcurrents.size(),
    const_cast<Expr **>(DepCommutatives.data()), DepCommutatives.size(),
    const_cast<Expr **>(DepWeakIns.data()), DepWeakIns.size(),
    const_cast<Expr **>(DepWeakOuts.data()), DepWeakOuts.size(),
    const_cast<Expr **>(DepWeakInouts.data()), DepWeakInouts.size(),
    const_cast<Expr **>(DepWeakConcurrents.data()), DepWeakConcurrents.size(),
    const_cast<Expr **>(DepWeakCommutatives.data()), DepWeakCommutatives.size(),
    const_cast<unsigned *>(ReductionListSizes.data()), ReductionListSizes.size(),
    const_cast<Expr **>(RD.Vars.data()), RD.Vars.size(),
    const_cast<Expr **>(RD.LHSs.data()), RD.LHSs.size(),
    const_cast<Expr **>(RD.RHSs.data()), RD.RHSs.size(),
    const_cast<Expr **>(RD.ReductionOps.data()), RD.ReductionOps.size(),
    const_cast<unsigned *>(TmpReductionKinds.data()), RD.ReductionKinds.size(),
    const_cast<unsigned *>(ReductionClauseType.data()), ReductionClauseType.size(),
    const_cast<NestedNameSpecifierLoc *>(ReductionNSLoc.data()), ReductionNSLoc.size(),
    const_cast<DeclarationNameInfo *>(ReductionIds.data()), ReductionIds.size(),
    const_cast<Expr **>(NdrangesRes.data()), NdrangesRes.size(),
    SR);
  ADecl->dropAttr<OSSTaskDeclSentinelAttr>();
  ADecl->addAttr(NewAttr);
  return DG;
}

// the boolean marks if it's a template
static std::pair<ValueDecl *, bool>
getPrivateItem(Sema &S, Expr *&RefExpr, SourceLocation &ELoc,
               SourceRange &ERange,
               bool AllowArrayShaping = false) {
  if (RefExpr->containsUnexpandedParameterPack()) {
    S.Diag(RefExpr->getExprLoc(), diag::err_oss_variadic_templates_not_clause_allowed);
    return std::make_pair(nullptr, false);
  } else if (RefExpr->isTypeDependent() || RefExpr->isValueDependent()) {
    return std::make_pair(nullptr, true);
  }

  RefExpr = RefExpr->IgnoreParens();
  bool IsArrayShaping = false;
  if (AllowArrayShaping) {
    // We do not allow shaping expr of a subscript/section
    if (auto OASE = dyn_cast_or_null<OSSArrayShapingExpr>(RefExpr)) {
      Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempOASE = dyn_cast<OSSArrayShapingExpr>(Base))
        Base = TempOASE->getBase()->IgnoreParenImpCasts();
      RefExpr = Base;
      IsArrayShaping = true;
    }
  }

  ELoc = RefExpr->getExprLoc();
  ERange = RefExpr->getSourceRange();
  RefExpr = RefExpr->IgnoreParenImpCasts();
  auto *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
  auto *ME = dyn_cast_or_null<MemberExpr>(RefExpr);

  // Only allow VarDecl from DeclRefExpr
  // and VarDecl implicits from MemberExpr // (i.e. static members without 'this')
  if ((!DE || !isa<VarDecl>(DE->getDecl())) &&
      (S.getCurrentThisType().isNull() || !ME ||
       !isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()) ||
       !cast<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts())->isImplicit() ||
       !isa<VarDecl>(ME->getMemberDecl()))) {
    if (IsArrayShaping) {
      // int *get();
      // reduction(+ : [3](get()))
      // reduction(+ : [3](p[4]))
      S.Diag(ELoc, diag::err_oss_expected_base_var_name) << ERange;
    } else {
      S.Diag(ELoc,
             AllowArrayShaping
                 ? diag::err_oss_expected_var_name_member_expr_or_array_shaping
                 : diag::err_oss_expected_var_name_member_expr)
          << (S.getCurrentThisType().isNull() ? 0 : 1) << ERange;
    }
    return std::make_pair(nullptr, false);
  }

  auto *VD = cast<VarDecl>(DE ? DE->getDecl() : ME->getMemberDecl());

  return std::make_pair(getCanonicalDecl(VD), false);
}

bool Sema::ActOnOmpSsDependKinds(ArrayRef<OmpSsDependClauseKind> DepKinds,
                                 SmallVectorImpl<OmpSsDependClauseKind> &DepKindsOrdered,
                                 SourceLocation DepLoc) {
  bool HasTwoKinds = DepKinds.size() == 2;

  int WeakCnt = DepKinds[0] == OSSC_DEPEND_weak;
  if (HasTwoKinds)
    WeakCnt += DepKinds[1] == OSSC_DEPEND_weak;

  bool HasConcurrent =
    HasTwoKinds ? DepKinds[0] == OSSC_DEPEND_inoutset
                  || DepKinds[1] == OSSC_DEPEND_inoutset
               : DepKinds[0] == OSSC_DEPEND_inoutset;

  int UnknownCnt = (DepKinds[0] == OSSC_DEPEND_unknown);
  if (HasTwoKinds)
    UnknownCnt += DepKinds[1] == OSSC_DEPEND_unknown;

  int InOutInoutCnt = DepKinds[0] == OSSC_DEPEND_in;
  InOutInoutCnt = InOutInoutCnt + (DepKinds[0] == OSSC_DEPEND_out);
  InOutInoutCnt = InOutInoutCnt + (DepKinds[0] == OSSC_DEPEND_inout);
  if (HasTwoKinds) {
    InOutInoutCnt = InOutInoutCnt + (DepKinds[1] == OSSC_DEPEND_in);
    InOutInoutCnt = InOutInoutCnt + (DepKinds[1] == OSSC_DEPEND_out);
    InOutInoutCnt = InOutInoutCnt + (DepKinds[1] == OSSC_DEPEND_inout);
  }

  if (isOmpSsTaskingDirective(DSAStack->getCurrentDirective())) {
    if (HasTwoKinds) {
      if (HasConcurrent) {
        // concurrent (inoutset) cannot be combined with other modifiers
        SmallString<256> Buffer;
        llvm::raw_svector_ostream Out(Buffer);
        Out << "'" << getOmpSsSimpleClauseTypeName(OSSC_depend, OSSC_DEPEND_inoutset) << "'";
        Diag(DepLoc, diag::err_oss_depend_no_weak_compatible)
          << Out.str() << 1;
        return false;
      }
      if ((WeakCnt == 1 && UnknownCnt == 1) || (WeakCnt == 2)) {
        // depend(weak, asdf:
        // depend(weak, weak:
        unsigned Except[] = {OSSC_DEPEND_weak, OSSC_DEPEND_inoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
      if (WeakCnt == 0 && UnknownCnt <= 1) {
        // depend(in, in:
        // depend(in, asdf:
        Diag(DepLoc, diag::err_oss_depend_weak_required);
        return false;
      }
      if (UnknownCnt == 2) {
        // depend(asdf, asdf:
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
    } else {
      if (WeakCnt == 1 || UnknownCnt == 1) {
        // depend(weak:
        // depend(asdf:
        unsigned Except[] = {OSSC_DEPEND_weak};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
    }
  } else if (DSAStack->getCurrentDirective() == OSSD_taskwait) {
    // Taskwait
    // Only allow in/out/inout
    if (HasTwoKinds || !InOutInoutCnt) {
      unsigned Except[] = {OSSC_DEPEND_weak, OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
      Diag(DepLoc, diag::err_oss_unexpected_clause_value)
          << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                     /*Last=*/OSSC_DEPEND_unknown, Except)
          << getOmpSsClauseName(OSSC_depend);
      return false;
    }
  } else if (DSAStack->getCurrentDirective() == OSSD_release) {
    // Release
    if (HasTwoKinds) {
      if ((WeakCnt == 1 && InOutInoutCnt != 1) || (WeakCnt == 2)) {
        // depend(weak, asdf:
        // depend(weak, weak:
        unsigned Except[] = {OSSC_DEPEND_weak, OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
      if (WeakCnt == 0 && InOutInoutCnt >= 1) {
        // depend(in, in:
        // depend(in, asdf:
        Diag(DepLoc, diag::err_oss_depend_weak_required);
        return false;
      }
      if (InOutInoutCnt == 0) {
        // depend(asdf, asdf:
        unsigned Except[] = {OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
    } else {
      if (WeakCnt == 1 || InOutInoutCnt == 0) {
        // depend(weak:
        // depend(asdf:
        unsigned Except[] = {OSSC_DEPEND_weak, OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
    }
  }
  // Here we have three cases:
  // { OSSC_DEPEND_in }
  // { OSSC_DEPEND_weak, OSSC_DEPEND_in }
  // { OSSC_DEPEND_in, OSSC_DEPEND_weak }
  if (DepKinds[0] == OSSC_DEPEND_weak) {
    DepKindsOrdered.push_back(DepKinds[1]);
    DepKindsOrdered.push_back(DepKinds[0]);
  } else {
    DepKindsOrdered.push_back(DepKinds[0]);
    if (DepKinds.size() == 2)
      DepKindsOrdered.push_back(DepKinds[1]);
  }
  return true;
}

static bool isConstNotMutableType(Sema &SemaRef, QualType Type,
                                  bool AcceptIfMutable = true,
                                  bool *IsClassType = nullptr) {
  ASTContext &Context = SemaRef.getASTContext();
  Type = Type.getNonReferenceType().getCanonicalType();
  bool IsConstant = Type.isConstant(Context);
  Type = Context.getBaseElementType(Type);
  const CXXRecordDecl *RD = AcceptIfMutable && SemaRef.getLangOpts().CPlusPlus
                                ? Type->getAsCXXRecordDecl()
                                : nullptr;
  if (const auto *CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(RD))
    if (const ClassTemplateDecl *CTD = CTSD->getSpecializedTemplate())
      RD = CTD->getTemplatedDecl();
  if (IsClassType)
    *IsClassType = RD;
  return IsConstant && !(SemaRef.getLangOpts().CPlusPlus && RD &&
                         RD->hasDefinition() && RD->hasMutableFields());
}

static bool rejectConstNotMutableType(Sema &SemaRef, const ValueDecl *D,
                                      QualType Type, OmpSsClauseKind CKind,
                                      SourceLocation ELoc,
                                      bool AcceptIfMutable = true,
                                      bool ListItemNotVar = false) {
  ASTContext &Context = SemaRef.getASTContext();
  bool IsClassType;
  if (isConstNotMutableType(SemaRef, Type, AcceptIfMutable, &IsClassType)) {
    unsigned Diag = ListItemNotVar
                        ? diag::err_oss_const_list_item
                        : IsClassType ? diag::err_oss_const_not_mutable_variable
                                      : diag::err_oss_const_variable;
    SemaRef.Diag(ELoc, Diag) << getOmpSsClauseName(CKind);
    if (!ListItemNotVar && D) {
      const VarDecl *VD = dyn_cast<VarDecl>(D);
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      SemaRef.Diag(D->getLocation(),
                   IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
    }
    return true;
  }
  return false;
}


template <typename T, typename U>
static T filterLookupForUDReductionAndMapper(
    SmallVectorImpl<U> &Lookups, const llvm::function_ref<T(ValueDecl *)> Gen) {
  for (U &Set : Lookups) {
    for (auto *D : Set) {
      if (T Res = Gen(cast<ValueDecl>(D)))
        return Res;
    }
  }
  return T();
}

static NamedDecl *findAcceptableDecl(Sema &SemaRef, NamedDecl *D) {
  assert(!LookupResult::isVisible(SemaRef, D) && "not in slow case");

  for (auto RD : D->redecls()) {
    // Don't bother with extra checks if we already know this one isn't visible.
    if (RD == D)
      continue;

    auto ND = cast<NamedDecl>(RD);
    if (LookupResult::isVisible(SemaRef, ND))
      return ND;
  }

  return nullptr;
}

// Perform ADL https://en.cppreference.com/w/cpp/language/adl
// http://eel.is/c++draft/over.match.oper
// http://eel.is/c++draft/basic.lookup.argdep
// but instead of looking for functions look for pragmas
static void
argumentDependentLookup(Sema &SemaRef, const DeclarationNameInfo &Id,
                        SourceLocation Loc, QualType Ty,
                        SmallVectorImpl<UnresolvedSet<8>> &Lookups) {
  // Find all of the associated namespaces and classes based on the
  // arguments we have.
  Sema::AssociatedNamespaceSet AssociatedNamespaces;
  Sema::AssociatedClassSet AssociatedClasses;
  OpaqueValueExpr OVE(Loc, Ty, VK_LValue);
  SemaRef.FindAssociatedClassesAndNamespaces(Loc, &OVE, AssociatedNamespaces,
                                             AssociatedClasses);

  // C++ [basic.lookup.argdep]p3:
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains [...] then Y is
  //   empty. Otherwise Y is the set of declarations found in the
  //   namespaces associated with the argument types as described
  //   below. The set of declarations found by the lookup of the name
  //   is the union of X and Y.
  //
  // Here, we compute Y and add its members to the overloaded
  // candidate set.
  for (auto *NS : AssociatedNamespaces) {
    //   When considering an associated namespace, the lookup is the
    //   same as the lookup performed when the associated namespace is
    //   used as a qualifier (3.4.3.2) except that:
    //
    //     -- Any using-directives in the associated namespace are
    //        ignored.
    //
    //     -- Any namespace-scope friend functions declared in
    //        associated classes are visible within their respective
    //        namespaces even if they are not visible during an ordinary
    //        lookup (11.4).
    DeclContext::lookup_result R = NS->lookup(Id.getName());
    for (auto *D : R) {
      auto *Underlying = D;
      if (auto *USD = dyn_cast<UsingShadowDecl>(D))
        Underlying = USD->getTargetDecl();

      if (!isa<OSSDeclareReductionDecl>(Underlying))
        continue;

      if (!SemaRef.isVisible(D)) {
        D = findAcceptableDecl(SemaRef, D);
        if (!D)
          continue;
        if (auto *USD = dyn_cast<UsingShadowDecl>(D))
          Underlying = USD->getTargetDecl();
      }
      Lookups.emplace_back();
      Lookups.back().addDecl(Underlying);
    }
  }
}

static ExprResult
buildDeclareReductionRef(Sema &SemaRef, SourceLocation Loc, SourceRange Range,
                         Scope *S, CXXScopeSpec &ReductionIdScopeSpec,
                         const DeclarationNameInfo &ReductionId, QualType Ty,
                         CXXCastPath &BasePath, Expr *UnresolvedReduction) {
  if (ReductionIdScopeSpec.isInvalid())
    return ExprError();
  SmallVector<UnresolvedSet<8>, 4> Lookups;
  if (S) {
    LookupResult Lookup(SemaRef, ReductionId, Sema::LookupOSSReductionName);
    // NOTE: OpenMP does this but we are not able to trigger an
    // unexpected diagnostic disabling it
    // Lookup.suppressDiagnostics();

    // LookupParsedName fails when trying to lookup this code
    //
    // template <class T>
    // class Class1 {
    //  T a;
    // public:
    //   Class1() : a() {}
    //   #pragma omp declare reduction(fun : T : temp)    // Error
    // };
    //
    //
    // template <class T>
    // class Class2 : public Class1<T> {
    // #pragma omp declare reduction(fun : T : omp_out += omp_in)
    // };
    //
    // int main() {
    //     int i;
    //     #pragma omp parallel reduction (::Class2<int>::fun : i) // Error
    //     {}
    // }
    //
    // When that happens, ReductionIdScopeSpec is unset so we
    // end up returning ExprEmpty()
    while (S && SemaRef.LookupParsedName(Lookup, S, &ReductionIdScopeSpec)) {
      NamedDecl *D = Lookup.getRepresentativeDecl();
      do {
        S = S->getParent();
      } while (S && !S->isDeclScope(D));
      if (S)
        S = S->getParent();
      Lookups.emplace_back();
      Lookups.back().append(Lookup.begin(), Lookup.end());
      Lookup.clear();
    }
  } else if (auto *ULE =
                 cast_or_null<UnresolvedLookupExpr>(UnresolvedReduction)) {
    Lookups.push_back(UnresolvedSet<8>());
    Decl *PrevD = nullptr;
    for (NamedDecl *D : ULE->decls()) {
      // 1.
      if (D == PrevD)
        Lookups.push_back(UnresolvedSet<8>());
      else if (auto *DRD = dyn_cast<OSSDeclareReductionDecl>(D))
        Lookups.back().addDecl(DRD);
      PrevD = D;
    }
  }
  if (SemaRef.CurContext->isDependentContext() || Ty->isDependentType() ||
      Ty->isInstantiationDependentType() ||
      Ty->containsUnexpandedParameterPack() ||
      filterLookupForUDReductionAndMapper<bool>(Lookups, [](ValueDecl *D) {
        return !D->isInvalidDecl() &&
               (D->getType()->isDependentType() ||
                D->getType()->isInstantiationDependentType() ||
                D->getType()->containsUnexpandedParameterPack());
      })) {
    UnresolvedSet<8> ResSet;
    for (const UnresolvedSet<8> &Set : Lookups) {
      if (Set.empty())
        continue;
      ResSet.append(Set.begin(), Set.end());
      // The last item marks the end of all declarations at the specified scope.
      // This is used becase here we're merging Sets, and we want to separate them      // in instantiation
      // See 1.
      ResSet.addDecl(Set[Set.size() - 1]);
    }
    return UnresolvedLookupExpr::Create(
        SemaRef.Context, /*NamingClass=*/nullptr,
        ReductionIdScopeSpec.getWithLocInContext(SemaRef.Context), ReductionId,
        /*ADL=*/true, /*Overloaded=*/true, ResSet.begin(), ResSet.end());
  }
  // Lookup inside the classes.
  // C++ [over.match.oper]p3:
  //   For a unary operator @ with an operand of a type whose
  //   cv-unqualified version is T1, and for a binary operator @ with
  //   a left operand of a type whose cv-unqualified version is T1 and
  //   a right operand of a type whose cv-unqualified version is T2,
  //   three sets of candidate functions, designated member
  //   candidates, non-member candidates and built-in candidates, are
  //   constructed as follows:
  //     -- If T1 is a complete class type or a class currently being
  //        defined, the set of member candidates is the result of the
  //        qualified lookup of T1::operator@ (13.3.1.1.1); otherwise,
  //        the set of member candidates is empty.
  LookupResult Lookup(SemaRef, ReductionId, Sema::LookupOSSReductionName);
  // NOTE: OpenMP does this but we are not able to trigger an
  // unexpected diagnostic disabling it
  // Lookup.suppressDiagnostics();
  if (const auto *TyRec = Ty->getAs<RecordType>()) {
    // Complete the type if it can be completed.
    // If the type is neither complete nor being defined, bail out now.
    if (SemaRef.isCompleteType(Loc, Ty) || TyRec->isBeingDefined() ||
        TyRec->getDecl()->getDefinition()) {
      Lookup.clear();
      SemaRef.LookupQualifiedName(Lookup, TyRec->getDecl());
      if (Lookup.empty()) {
        Lookups.emplace_back();
        Lookups.back().append(Lookup.begin(), Lookup.end());
      }
    }
  }
  // Perform ADL.
  if (SemaRef.getLangOpts().CPlusPlus)
    argumentDependentLookup(SemaRef, ReductionId, Loc, Ty, Lookups);
  if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
          Lookups, [&SemaRef, Ty](ValueDecl *D) -> ValueDecl * {
            if (!D->isInvalidDecl() &&
                SemaRef.Context.hasSameType(D->getType(), Ty))
              return D;
            return nullptr;
          }))
    return SemaRef.BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(),
                                    VK_LValue, Loc);
  // If the type is a derived class, then any reduction-identifier that matches its base classes is also a
  // match, if there is no specific match for the type.
  if (SemaRef.getLangOpts().CPlusPlus) {
    if (auto *VD = filterLookupForUDReductionAndMapper<ValueDecl *>(
            Lookups, [&SemaRef, Ty, Loc](ValueDecl *D) -> ValueDecl * {
              if (!D->isInvalidDecl() &&
                  SemaRef.IsDerivedFrom(Loc, Ty, D->getType()) &&
                  !Ty.isMoreQualifiedThan(D->getType()))
                return D;
              return nullptr;
            })) {
      CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                         /*DetectVirtual=*/false);
      if (SemaRef.IsDerivedFrom(Loc, Ty, VD->getType(), Paths)) {
        if (!Paths.isAmbiguous(SemaRef.Context.getCanonicalType(
                VD->getType().getUnqualifiedType()))) {
          if (SemaRef.CheckBaseClassAccess(
                  Loc, VD->getType(), Ty, Paths.front(),
                  /*DiagID=*/0) != Sema::AR_inaccessible) {
            SemaRef.BuildBasePathArray(Paths, BasePath);
            return SemaRef.BuildDeclRefExpr(
                VD, VD->getType().getNonReferenceType(), VK_LValue, Loc);
          }
        }
      }
    }
  }
  if (ReductionIdScopeSpec.isSet()) {
    SemaRef.Diag(Loc, diag::err_oss_not_resolved_reduction_identifier) << Ty << Range;
    return ExprError();
  }
  return ExprEmpty();
}

static bool actOnOSSReductionKindClause(
    Sema &S, DSAStackTy *Stack, OmpSsClauseKind ClauseKind,
    ArrayRef<Expr *> VarList, CXXScopeSpec &ReductionIdScopeSpec,
    const DeclarationNameInfo &ReductionId, ArrayRef<Expr *> UnresolvedReductions,
    ReductionData &RD, bool Outline) {
  DeclarationName DN = ReductionId.getName();
  OverloadedOperatorKind OOK = DN.getCXXOverloadedOperator();
  BinaryOperatorKind BOK = BO_Comma;

  ASTContext &Context = S.Context;
  // OpenMP [2.14.3.6, reduction clause]
  // C
  // reduction-identifier is either an identifier or one of the following
  // operators: +, -, *,  &, |, ^, && and ||
  // C++
  // reduction-identifier is either an id-expression or one of the following
  // operators: +, -, *, &, |, ^, && and ||
  switch (OOK) {
  case OO_Plus:
  case OO_Minus:
    BOK = BO_Add;
    break;
  case OO_Star:
    BOK = BO_Mul;
    break;
  case OO_Amp:
    BOK = BO_And;
    break;
  case OO_Pipe:
    BOK = BO_Or;
    break;
  case OO_Caret:
    BOK = BO_Xor;
    break;
  case OO_AmpAmp:
    BOK = BO_LAnd;
    break;
  case OO_PipePipe:
    BOK = BO_LOr;
    break;
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Slash:
  case OO_Percent:
  case OO_Tilde:
  case OO_Exclaim:
  case OO_Equal:
  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
  case OO_PlusEqual:
  case OO_MinusEqual:
  case OO_StarEqual:
  case OO_SlashEqual:
  case OO_PercentEqual:
  case OO_CaretEqual:
  case OO_AmpEqual:
  case OO_PipeEqual:
  case OO_LessLess:
  case OO_GreaterGreater:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_EqualEqual:
  case OO_ExclaimEqual:
  case OO_Spaceship:
  case OO_PlusPlus:
  case OO_MinusMinus:
  case OO_Comma:
  case OO_ArrowStar:
  case OO_Arrow:
  case OO_Call:
  case OO_Subscript:
  case OO_Conditional:
  case OO_Coawait:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Unexpected reduction identifier");
  case OO_None:
    if (IdentifierInfo *II = DN.getAsIdentifierInfo()) {
      if (II->isStr("max"))
        BOK = BO_GT;
      else if (II->isStr("min"))
        BOK = BO_LT;
    }
    break;
  }
  SourceRange ReductionIdRange;
  if (ReductionIdScopeSpec.isValid())
    ReductionIdRange.setBegin(ReductionIdScopeSpec.getBeginLoc());
  else
    ReductionIdRange.setBegin(ReductionId.getBeginLoc());
  ReductionIdRange.setEnd(ReductionId.getEndLoc());

  auto IR = UnresolvedReductions.begin(), ER = UnresolvedReductions.end();
  bool FirstIter = true;
  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "nullptr expr in OmpSs reduction clause.");
    // OpenMP [2.1, C/C++]
    //  A list item is a variable or array section, subject to the restrictions
    //  specified in Section 2.4 on page 42 and in each of the sections
    // describing clauses and directives for which a list appears.
    // OpenMP  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    if (!FirstIter && IR != ER)
      ++IR;
    FirstIter = false;
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange,
                              /*AllowArrayShaping=*/true);
    if (Res.second) {
      // Try to find 'declare reduction' corresponding construct before using
      // builtin/overloaded operators.
      QualType Type = Context.DependentTy;
      CXXCastPath BasePath;
      ExprResult DeclareReductionRef = buildDeclareReductionRef(
          S, ELoc, ERange, Stack->getCurScope(), ReductionIdScopeSpec,
          ReductionId, Type, BasePath, IR == ER ? nullptr : *IR);
      Expr *ReductionOp = nullptr;
      if (S.CurContext->isDependentContext() &&
          (DeclareReductionRef.isUnset() ||
           isa<UnresolvedLookupExpr>(DeclareReductionRef.get())))
        ReductionOp = DeclareReductionRef.get();
      // It will be analyzed later.
      RD.push(RefExpr, ReductionOp);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    // QualType Type = D->getType().getNonReferenceType();
    QualType Type = Context.getBaseElementType(RefExpr->getType().getNonReferenceType());
    auto *VD = dyn_cast<VarDecl>(D);

    // OpenMP [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (S.RequireCompleteType(ELoc, D->getType(),
                              diag::err_oss_incomplete_type))
      continue;
    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // A list item that appears in a reduction clause must not be
    // const-qualified.
    if (rejectConstNotMutableType(S, D, Type, ClauseKind, ELoc,
                                  /*AcceptIfMutable*/ false, /*ListItemNotVar=ASE || OASE*/ false))
      continue;

    // Non-POD and refs to Non-POD are not allowed in reductions
    if (!Type.isPODType(S.Context)) {
      S.Diag(ELoc, diag::err_oss_non_pod_reduction);
      continue;
    }

    if (Outline) {
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(RefExpr->IgnoreParenImpCasts())) {
        if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->hasGlobalStorage() || !VD->getType()->isReferenceType()) {
            S.Diag(ELoc, diag::err_oss_expected_lvalue_reference_or_global_or_dereference_or_array_item)
              << 1 << DRE->getSourceRange();
            return false;
          }
        }
      }
    }

    // Try to find 'declare reduction' corresponding construct before using
    // builtin/overloaded operators.
    CXXCastPath BasePath;
    ExprResult DeclareReductionRef = buildDeclareReductionRef(
        S, ELoc, ERange, Stack->getCurScope(), ReductionIdScopeSpec,
        ReductionId, Type, BasePath, IR == ER ? nullptr : *IR);
    // DeclareReductionRef.isInvalid() -> There was an error
    // DeclareReductionRef.isUnset()   -> No declare reduction found
    // DeclareReductionRef.isUsable()  -> declare reduction found
    if (DeclareReductionRef.isInvalid())
      continue;
    if (S.CurContext->isDependentContext() &&
        (DeclareReductionRef.isUnset() ||
         isa<UnresolvedLookupExpr>(DeclareReductionRef.get()))) {
      RD.push(RefExpr, DeclareReductionRef.get());
      continue;
    }
    if (BOK == BO_Comma && DeclareReductionRef.isUnset()) {
      // Not allowed reduction identifier is found.
      S.Diag(ReductionId.getBeginLoc(),
             diag::err_oss_unknown_reduction_identifier)
          << Type << ReductionIdRange;
      continue;
    }

    // OpenMP [2.14.3.6, reduction clause, Restrictions]
    // The type of a list item that appears in a reduction clause must be valid
    // for the reduction-identifier. For a max or min reduction in C, the type
    // of the list item must be an allowed arithmetic data type: char, int,
    // float, double, or _Bool, possibly modified with long, short, signed, or
    // unsigned. For a max or min reduction in C++, the type of the list item
    // must be an allowed arithmetic data type: char, wchar_t, int, float,
    // double, or bool, possibly modified with long, short, signed, or unsigned.
    if (DeclareReductionRef.isUnset()) {
      if ((BOK == BO_GT || BOK == BO_LT) &&
          !(Type->isScalarType() ||
            (S.getLangOpts().CPlusPlus && Type->isArithmeticType()))) {
        S.Diag(ELoc, diag::err_oss_clause_not_arithmetic_type_arg)
            << getOmpSsClauseName(ClauseKind) << S.getLangOpts().CPlusPlus;
        continue;
      }
      if ((BOK == BO_OrAssign || BOK == BO_AndAssign || BOK == BO_XorAssign) &&
          !S.getLangOpts().CPlusPlus && Type->isFloatingType()) {
        S.Diag(ELoc, diag::err_oss_clause_floating_type_arg)
            << getOmpSsClauseName(ClauseKind);
        continue;
      }
    }

    Type = Type.getNonLValueExprType(Context).getUnqualifiedType();
    VarDecl *LHSVD = buildVarDecl(S, ELoc, Type, ".reduction.lhs",
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);
    VarDecl *RHSVD = buildVarDecl(S, ELoc, Type, D->getName(),
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);

    // Add initializer for private variable.
    Expr *Init = nullptr;
    DeclRefExpr *LHSDRE = buildDeclRefExpr(S, LHSVD, Type, ELoc);
    DeclRefExpr *RHSDRE = buildDeclRefExpr(S, RHSVD, Type, ELoc);
    if (DeclareReductionRef.isUsable()) {
      auto *DRDRef = DeclareReductionRef.getAs<DeclRefExpr>();
      auto *DRD = cast<OSSDeclareReductionDecl>(DRDRef->getDecl());
      if (DRD->getInitializer()) {
        Init = DRDRef;
        RHSVD->setInit(DRDRef);
        RHSVD->setInitStyle(VarDecl::CallInit);
      }
    } else {
      switch (BOK) {
      case BO_Add:
      case BO_Xor:
      case BO_Or:
      case BO_LOr:
        // '+', '-', '^', '|', '||' reduction ops - initializer is '0'.
        if (Type->isScalarType() || Type->isAnyComplexType())
          Init = S.ActOnIntegerConstant(ELoc, /*Val=*/0).get();
        break;
      case BO_Mul:
      case BO_LAnd:
        if (Type->isScalarType() || Type->isAnyComplexType()) {
          // '*' and '&&' reduction ops - initializer is '1'.
          Init = S.ActOnIntegerConstant(ELoc, /*Val=*/1).get();
        }
        break;
      case BO_And: {
        // '&' reduction op - initializer is '~0'.
        QualType OrigType = Type;
        if (auto *ComplexTy = OrigType->getAs<ComplexType>())
          Type = ComplexTy->getElementType();
        if (Type->isRealFloatingType()) {
          llvm::APFloat InitValue = llvm::APFloat::getAllOnesValue(
              Context.getFloatTypeSemantics(Type));
          Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                         Type, ELoc);
        } else if (Type->isScalarType()) {
          uint64_t Size = Context.getTypeSize(Type);
          QualType IntTy = Context.getIntTypeForBitwidth(Size, /*Signed=*/0);
          llvm::APInt InitValue = llvm::APInt::getAllOnes(Size);
          Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
        }
        if (Init && OrigType->isAnyComplexType()) {
          // Init = 0xFFFF + 0xFFFFi;
          auto *Im = new (Context) ImaginaryLiteral(Init, OrigType);
          Init = S.CreateBuiltinBinOp(ELoc, BO_Add, Init, Im).get();
        }
        Type = OrigType;
        break;
      }
      case BO_LT:
      case BO_GT: {
        // 'min' reduction op - initializer is 'Largest representable number in
        // the reduction list item type'.
        // 'max' reduction op - initializer is 'Least representable number in
        // the reduction list item type'.
        if (Type->isIntegerType() || Type->isPointerType()) {
          bool IsSigned = Type->hasSignedIntegerRepresentation();
          uint64_t Size = Context.getTypeSize(Type);
          QualType IntTy =
              Context.getIntTypeForBitwidth(Size, /*Signed=*/IsSigned);
          llvm::APInt InitValue =
              (BOK != BO_LT) ? IsSigned ? llvm::APInt::getSignedMinValue(Size)
                                        : llvm::APInt::getMinValue(Size)
                             : IsSigned ? llvm::APInt::getSignedMaxValue(Size)
                                        : llvm::APInt::getMaxValue(Size);
          Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
          if (Type->isPointerType()) {
            // Cast to pointer type.
            ExprResult CastExpr = S.BuildCStyleCastExpr(
                ELoc, Context.getTrivialTypeSourceInfo(Type, ELoc), ELoc, Init);
            if (CastExpr.isInvalid())
              continue;
            Init = CastExpr.get();
          }
        } else if (Type->isRealFloatingType()) {
          llvm::APFloat InitValue = llvm::APFloat::getLargest(
              Context.getFloatTypeSemantics(Type), BOK != BO_LT);
          Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                         Type, ELoc);
        }
        break;
      }
      case BO_PtrMemD:
      case BO_PtrMemI:
      case BO_MulAssign:
      case BO_Div:
      case BO_Rem:
      case BO_Sub:
      case BO_Shl:
      case BO_Shr:
      case BO_LE:
      case BO_GE:
      case BO_EQ:
      case BO_NE:
      case BO_Cmp:
      case BO_AndAssign:
      case BO_XorAssign:
      case BO_OrAssign:
      case BO_Assign:
      case BO_AddAssign:
      case BO_SubAssign:
      case BO_DivAssign:
      case BO_RemAssign:
      case BO_ShlAssign:
      case BO_ShrAssign:
      case BO_Comma:
        llvm_unreachable("Unexpected reduction operation");
      }
    }
    if (Init && DeclareReductionRef.isUnset())
      S.AddInitializerToDecl(RHSVD, Init, /*DirectInit=*/false);
    else if (!Init)
      S.ActOnUninitializedDecl(RHSVD);
    if (RHSVD->isInvalidDecl())
      continue;
    if (!RHSVD->hasInit() &&
        (DeclareReductionRef.isUnset() || !S.LangOpts.CPlusPlus)) {
      // C structs do not have initializer
      S.Diag(ELoc, diag::err_oss_reduction_id_not_compatible)
          << Type << ReductionIdRange;
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      S.Diag(D->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }
    ExprResult ReductionOp;
    if (DeclareReductionRef.isUsable()) {
      ReductionOp = DeclareReductionRef;
    } else {
      ReductionOp = S.BuildBinOp(
          Stack->getCurScope(), ReductionId.getBeginLoc(), BOK, LHSDRE, RHSDRE);
      if (ReductionOp.isUsable()) {
        if (BOK != BO_LT && BOK != BO_GT) {
          ReductionOp =
              S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(),
                           BO_Assign, LHSDRE, ReductionOp.get());
        } else {
          auto *ConditionalOp = new (Context)
              ConditionalOperator(ReductionOp.get(), ELoc, LHSDRE, ELoc, RHSDRE,
                                  Type, VK_LValue, OK_Ordinary);
          ReductionOp =
              S.BuildBinOp(Stack->getCurScope(), ReductionId.getBeginLoc(),
                           BO_Assign, LHSDRE, ConditionalOp);
        }
        if (ReductionOp.isUsable())
          ReductionOp = S.ActOnFinishFullExpr(ReductionOp.get(),
                                              /*DiscardedValue*/ false);
      }
      if (!ReductionOp.isUsable())
        continue;
    }

    RD.push(RefExpr, LHSDRE, RHSDRE, ReductionOp.get(), BOK);
  }
  return RD.Vars.empty();
}

OSSClause *
Sema::ActOnOmpSsReductionClause(OmpSsClauseKind Kind, ArrayRef<Expr *> VarList,
                       SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation ColonLoc,
                       SourceLocation EndLoc,
                       CXXScopeSpec &ReductionIdScopeSpec,
                       const DeclarationNameInfo &ReductionId,
                       ArrayRef<Expr *> UnresolvedReductions) {
  ReductionData RD(VarList.size());
  if (actOnOSSReductionKindClause(*this, DSAStack, Kind, VarList,
                                  ReductionIdScopeSpec, ReductionId,
                                  UnresolvedReductions, RD, /*Outline=*/false))
    return nullptr;
  return OSSReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, RD.Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId,
      RD.LHSs, RD.RHSs, RD.ReductionOps, RD.ReductionKinds,
      Kind == OSSC_weakreduction);
}

OSSClause *
Sema::ActOnOmpSsDependClause(ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
                             SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc,
                             bool OSSSyntax) {
  SmallVector<OmpSsDependClauseKind, 2> DepKindsOrdered;
  SmallVector<Expr *, 8> ClauseVars;
  if (!ActOnOmpSsDependKinds(DepKinds, DepKindsOrdered, DepLoc))
    return nullptr;

  for (Expr *RefExpr : VarList) {
    if (checkDependency(*this, RefExpr, OSSSyntax, /*Outline=*/false))
      ClauseVars.push_back(RefExpr->IgnoreParenImpCasts());
  }
  return OSSDependClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                 DepKinds, DepKindsOrdered,
                                 DepLoc, ColonLoc, ClauseVars,
                                 OSSSyntax);
}

OSSClause *
Sema::ActOnOmpSsVarListClause(
  OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
  SourceLocation StartLoc, SourceLocation LParenLoc,
  SourceLocation ColonLoc, SourceLocation EndLoc,
  ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
  CXXScopeSpec &ReductionIdScopeSpec,
  DeclarationNameInfo &ReductionId) {

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
  case OSSC_ndrange:
    Res = ActOnOmpSsNdrangeClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_depend:
    Res = ActOnOmpSsDependClause(DepKinds, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_reduction:
  case OSSC_weakreduction:
    Res = ActOnOmpSsReductionClause(Kind, Vars, StartLoc, LParenLoc, ColonLoc, EndLoc,
                                    ReductionIdScopeSpec, ReductionId);
    break;
  case OSSC_in:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_in }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_out:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_out }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_inout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inout }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_concurrent:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inoutset }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_commutative:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_mutexinoutset }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_on:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inout }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakin:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_in, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_out, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakinout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inout, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakconcurrent:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inoutset, OSSC_DEPEND_weak }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakcommutative:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_mutexinoutset, OSSC_DEPEND_weak }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }

  return Res;
}

OSSClause *
Sema::ActOnOmpSsFixedListClause(
  OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
  SourceLocation StartLoc, SourceLocation LParenLoc,
  SourceLocation EndLoc) {

  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_label:
    Res = ActOnOmpSsLabelClause(Vars, StartLoc, LParenLoc, EndLoc);
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
    ActOnOmpSsDefaultClause(static_cast<llvm::oss::DefaultKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_device:
    Res =
    ActOnOmpSsDeviceClause(static_cast<OmpSsDeviceClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsDefaultClause(llvm::oss::DefaultKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  if (Kind == OSS_DEFAULT_unknown) {
    Diag(KindKwLoc, diag::err_oss_unexpected_clause_value)
        << getListOfPossibleValues(OSSC_default, /*First=*/0,
                                   /*Last=*/unsigned(OSS_DEFAULT_unknown))
        << getOmpSsClauseName(OSSC_default);
    return nullptr;
  }

  switch (Kind) {
  case OSS_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OSS_DEFAULT_shared:
    DSAStack->setDefault(DSA_shared, KindKwLoc);
    break;
  case OSS_DEFAULT_private:
    DSAStack->setDefault(DSA_private, KindKwLoc);
    break;
  case OSS_DEFAULT_firstprivate:
    DSAStack->setDefault(DSA_firstprivate, KindKwLoc);
    break;
  default:
    llvm_unreachable("DSA unexpected in OmpSs-2 default clause");
  }
  return new (Context)
      OSSDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsDeviceClause(OmpSsDeviceClauseKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  switch (Kind) {
  case OSSC_DEVICE_smp:
  case OSSC_DEVICE_opencl:
  case OSSC_DEVICE_cuda:
    break;
  case OSSC_DEVICE_fpga:
    Diag(KindKwLoc, diag::err_oss_inline_device_not_supported)
      << getOmpSsSimpleClauseTypeName(OSSC_device, Kind);
    break;
  case OSSC_DEVICE_unknown:
    Diag(KindKwLoc, diag::err_oss_unexpected_clause_value)
        << getListOfPossibleValues(OSSC_device, /*First=*/0,
                                   /*Last=*/OSSC_DEVICE_unknown)
        << getOmpSsClauseName(OSSC_device);
    return nullptr;
  }
  return new (Context)
      OSSDeviceClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
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
  ExprResult Ex = PerformContextualImplicitConversion(Loc, Op, ConvertDiagnoser);
  if (Ex.isInvalid())
    return ExprError();
  QualType Type = Ex.get()->getType();
  if (!ConvertDiagnoser.match(Type))
    // FIXME: PerformContextualImplicitConversion should return ExprError
    //        itself in this case.
    // Case: int [10]
    return ExprError();
  return Ex;
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

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
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
    DSAStack->addDSA(D, RefExpr, OSSC_shared, /*Ignore=*/false, /*IsBase=*/true, /*Implicit=*/false);
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
  SmallVector<Expr *, 8> PrivateCopies;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D) {
      continue;
    }

    if (RequireCompleteType(ELoc, D->getType(),
                            diag::err_oss_incomplete_type))
      continue;

    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_private &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_private);
      continue;
    }

    QualType Type = D->getType().getUnqualifiedType().getNonReferenceType();
    if (Type->isArrayType())
      Type = Context.getBaseElementType(Type).getCanonicalType();

    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OmpSs-2 region uses
    // original variable for proper diagnostics and variable capturing.

    // Build DSA Copy
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    ActOnUninitializedDecl(VDPrivate);

    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, Type, RefExpr->getExprLoc());

    DSAStack->addDSA(D, RefExpr, OSSC_private, /*Ignore=*/false, /*IsBase=*/true, /*Implicit=*/false);
    ClauseVars.push_back(RefExpr);
    PrivateCopies.push_back(VDPrivateRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars, PrivateCopies);
}

OSSClause *
Sema::ActOnOmpSsFirstprivateClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 8> ClauseVars;
  SmallVector<Expr *, 8> PrivateCopies;
  SmallVector<Expr *, 8> Inits;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D) {
      continue;
    }

    if (RequireCompleteType(ELoc, D->getType(),
                            diag::err_oss_incomplete_type))
      continue;

    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_firstprivate &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_firstprivate);
      continue;
    }

    QualType Type = D->getType().getUnqualifiedType().getNonReferenceType();
    if (Type->isArrayType())
      Type = Context.getBaseElementType(Type).getCanonicalType();

    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OmpSs-2 region uses
    // original variable for proper diagnostics and variable capturing.

    // Build DSA clone
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    Expr *VDInitRefExpr = nullptr;
    // Build a temp variable to use it as initializer
    VarDecl *VDInit = buildVarDecl(*this, RefExpr->getExprLoc(), Type,
                                   ".firstprivate.temp");
    VDInitRefExpr = buildDeclRefExpr(*this, VDInit, Type,
                                     RefExpr->getExprLoc());
    // Set temp variable as initializer of DSA clone
    AddInitializerToDecl(VDPrivate, VDInitRefExpr,
                         /*DirectInit=*/false);

    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, Type, RefExpr->getExprLoc());

    DSAStack->addDSA(D, RefExpr, OSSC_firstprivate, /*Ignore=*/false, /*IsBase=*/true, /*Implicit=*/false);
    ClauseVars.push_back(RefExpr);
    PrivateCopies.push_back(VDPrivateRefExpr);
    Inits.push_back(VDInitRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       ClauseVars, PrivateCopies, Inits);
}

OSSClause *
Sema::ActOnOmpSsNdrangeClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 4> ClauseVars;
  if (!checkNdrange(*this, StartLoc, Vars, ClauseVars, /*Outline=*/false))
    return nullptr;

  return OSSNdrangeClause::Create(
      Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

ExprResult Sema::CheckNonNegativeIntegerValue(Expr *ValExpr,
                                      OmpSsClauseKind CKind,
                                      bool StrictlyPositive,
                                      bool Outline) {
  ExprResult Res = CheckSignedIntegerValue(ValExpr, Outline);
  if (Res.isInvalid())
    return ExprError();

  ValExpr = Res.get();

  if (ValExpr->containsErrors())
    return Res.get();

  // The expression must evaluate to a non-negative integer value.
  if (std::optional<llvm::APSInt> Result =
          ValExpr->getIntegerConstantExpr(Context)) {
    if (Result->isSigned() &&
        !((!StrictlyPositive && Result->isNonNegative()) ||
          (StrictlyPositive && Result->isStrictlyPositive()))) {
      Diag(ValExpr->getExprLoc(), diag::err_oss_negative_expression_in_clause)
          << getOmpSsClauseName(CKind) << (StrictlyPositive ? 1 : 0)
          << ValExpr->getSourceRange();
      return ExprError();
    }
  }
  return ValExpr;
}

ExprResult Sema::VerifyBooleanConditionWithCleanups(
    Expr *Condition,
    SourceLocation StartLoc) {

  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return ExprError();

    return MakeFullExpr(Val.get()).get();
  }
  return Condition;
}

ExprResult Sema::CheckIsConstCharPtrConvertibleExpr(Expr *E, bool ConstConstraint) {
  const QualType &ConstCharPtrTy =
      Context.getPointerType(Context.CharTy.withConst());

  if (!E->isValueDependent() && !E->isTypeDependent() &&
      !E->isInstantiationDependent() &&
      !E->containsUnexpandedParameterPack()) {

    VarDecl *LabelVD =
        buildVarDecl(*this, E->getExprLoc(), ConstCharPtrTy, ".tmp.label");
    AddInitializerToDecl(LabelVD, E,
                         /*DirectInit=*/false);
    if (!LabelVD->hasInit())
      return ExprError();

    if (ConstConstraint) {
      if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
        if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          bool IsClassType;
          // Only allow these cases
          // const char* const k1 = "hola";
          // const char k2[] = "hola";
          if (!(VD->hasGlobalStorage()
              && isConstNotMutableType(*this, VD->getType(), /*AcceptIfMutable*/ false, &IsClassType))
              || !VD->hasInit() || !isa<StringLiteral>(VD->getInit()->IgnoreParenImpCasts())) {
            Diag(E->getExprLoc(), diag::err_oss_non_const_variable);
            return ExprError();
          }
        }
      }
    }

    return LabelVD->getInit();
  }
  return E;
}

ExprResult Sema::VerifyPositiveIntegerConstant(
    Expr *E, OmpSsClauseKind CKind, bool StrictlyPositive) {
  if (!E)
    return ExprError();
  if (E->isValueDependent() || E->isTypeDependent() ||
      E->isInstantiationDependent() || E->containsUnexpandedParameterPack())
    return E;
  llvm::APSInt Result;
  ExprResult ICE =
      VerifyIntegerConstantExpression(E, &Result, /*FIXME*/ AllowFold);
  if (ICE.isInvalid())
    return ExprError();
  if ((StrictlyPositive && !Result.isStrictlyPositive()) ||
      (!StrictlyPositive && !Result.isNonNegative())) {
    Diag(E->getExprLoc(), diag::err_oss_negative_expression_in_clause)
        << getOmpSsClauseName(CKind) << (StrictlyPositive ? 1 : 0)
        << E->getSourceRange();
    return ExprError();
  }
  if (CKind == OSSC_collapse && DSAStack->getAssociatedLoops() == 1)
    DSAStack->setAssociatedLoops(Result.getExtValue());
  return ICE;
}

OSSClause *Sema::ActOnOmpSsIfClause(Expr *Condition,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc) {
  ExprResult Res = VerifyBooleanConditionWithCleanups(Condition, StartLoc);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSIfClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsFinalClause(Expr *Condition,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  ExprResult Res = VerifyBooleanConditionWithCleanups(Condition, StartLoc);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSFinalClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsCostClause(Expr *E,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc) {
  // The parameter of the cost() clause must be > 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_cost, /*StrictlyPositive=*/false, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSCostClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::CheckSignedIntegerValue(Expr *ValExpr, bool Outline) {
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent() &&
      !ValExpr->containsUnexpandedParameterPack()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformOmpSsImplicitIntegerConversion(Loc, ValExpr);
    if (Value.isInvalid())
      return ExprError();
    if (Outline) {
      OSSGlobalFinderVisitor GlobalFinderVisitor(*this);
      GlobalFinderVisitor.Visit(ValExpr);
      if (GlobalFinderVisitor.isErrorFound())
        return ExprError();
    }
    return Value.get();
  }
  return ValExpr;
}

OSSClause *Sema::ActOnOmpSsPriorityClause(Expr *E,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc) {
  // The parameter of the priority() clause must be integer signed
  // expression.
  ExprResult Res = CheckSignedIntegerValue(E, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSPriorityClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsShmemClause(Expr *E,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  // The parameter of the shmem() clause must be >= 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_shmem, /*StrictlyPositive=*/false, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSShmemClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsLabelClause(ArrayRef<Expr *> VarList,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  SmallVector<Expr *, 2> ClauseVars;
  ExprResult LabelRes;
  ExprResult InstLabelRes;

  LabelRes = CheckIsConstCharPtrConvertibleExpr(
    VarList[0], /*ConstConstraint=*/true);
  ClauseVars.push_back(LabelRes.get());

  if (VarList.size() == 2) {
    InstLabelRes = CheckIsConstCharPtrConvertibleExpr(
      VarList[1], /*ConstConstraint=*/false);
    ClauseVars.push_back(InstLabelRes.get());
  }

  if (LabelRes.isInvalid() || InstLabelRes.isInvalid())
    return nullptr;

  return OSSLabelClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

OSSClause *Sema::ActOnOmpSsOnreadyClause(Expr *E,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc) {
  if (!E)
    return nullptr;

  return new (Context) OSSOnreadyClause(E, StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsChunksizeClause(
    Expr *E, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation EndLoc) {
  // The parameter of the chunksize() clause must be > 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_chunksize, /*StrictlyPositive=*/true, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSChunksizeClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsGrainsizeClause(
    Expr *E, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation EndLoc) {
  // The parameter of the grainsize() clause must be > 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_grainsize, /*StrictlyPositive=*/true, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSGrainsizeClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsUnrollClause(
    Expr *E, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation EndLoc) {
  // The parameter of the unroll() clause must be > 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_unroll, /*StrictlyPositive=*/true, /*Outline=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSUnrollClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsCollapseClause(
    Expr *E, SourceLocation StartLoc,
    SourceLocation LParenLoc, SourceLocation EndLoc) {
  // The parameter of the collapse clause must be a constant
  // positive integer expression.
  ExprResult NumForLoopsResult =
      VerifyPositiveIntegerConstant(E, OSSC_collapse, /*StrictlyPositive=*/true);
  if (NumForLoopsResult.isInvalid())
    return nullptr;
  return new (Context)
      OSSCollapseClause(NumForLoopsResult.get(), StartLoc, LParenLoc, EndLoc);
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
  case OSSC_cost:
    Res = ActOnOmpSsCostClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_priority:
    Res = ActOnOmpSsPriorityClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_shmem:
    Res = ActOnOmpSsShmemClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_onready:
    Res = ActOnOmpSsOnreadyClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_chunksize:
    Res = ActOnOmpSsChunksizeClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_grainsize:
    Res = ActOnOmpSsGrainsizeClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_unroll:
    Res = ActOnOmpSsUnrollClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_collapse:
    Res = ActOnOmpSsCollapseClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsClause(OmpSsClauseKind Kind,
                                  SourceLocation StartLoc,
                                  SourceLocation EndLoc) {
  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_wait:
    Res = ActOnOmpSsWaitClause(StartLoc, EndLoc);
    break;
  case OSSC_update:
    Res = ActOnOmpSsUpdateClause(StartLoc, EndLoc);
    break;
  case OSSC_read:
    Res = ActOnOmpSsReadClause(StartLoc, EndLoc);
    break;
  case OSSC_write:
    Res = ActOnOmpSsWriteClause(StartLoc, EndLoc);
    break;
  case OSSC_capture:
    Res = ActOnOmpSsCaptureClause(StartLoc, EndLoc);
    break;
  case OSSC_compare:
    Res = ActOnOmpSsCompareClause(StartLoc, EndLoc);
    break;
  case OSSC_seq_cst:
    Res = ActOnOmpSsSeqCstClause(StartLoc, EndLoc);
    break;
  case OSSC_acq_rel:
    Res = ActOnOmpSsAcqRelClause(StartLoc, EndLoc);
    break;
  case OSSC_acquire:
    Res = ActOnOmpSsAcquireClause(StartLoc, EndLoc);
    break;
  case OSSC_release:
    Res = ActOnOmpSsReleaseClause(StartLoc, EndLoc);
    break;
  case OSSC_relaxed:
    Res = ActOnOmpSsRelaxedClause(StartLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsWaitClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc) {
  return new (Context) OSSWaitClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsUpdateClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  return new (Context) OSSUpdateClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsReadClause(SourceLocation StartLoc,
                                       SourceLocation EndLoc) {
  return new (Context) OSSReadClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsWriteClause(SourceLocation StartLoc,
                                        SourceLocation EndLoc) {
  return new (Context) OSSWriteClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsCaptureClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OSSCaptureClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsCompareClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OSSCompareClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsSeqCstClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OSSSeqCstClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsAcqRelClause(SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  return new (Context) OSSAcqRelClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsAcquireClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OSSAcquireClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsReleaseClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OSSReleaseClause(StartLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsRelaxedClause(SourceLocation StartLoc,
                                          SourceLocation EndLoc) {
  return new (Context) OSSRelaxedClause(StartLoc, EndLoc);
}

