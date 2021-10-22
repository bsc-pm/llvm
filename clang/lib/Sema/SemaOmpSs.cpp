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

  void VisitCXXCatchStmt(CXXCatchStmt *Node) {
    InnerDecls.insert(Node->getExceptionDecl());
    Visit(Node->getHandlerBlock());
  }

  void VisitExpr(Expr *E) {
    for (Stmt *Child : E->children()) {
      if (Child)
        Visit(Child);
    }
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
  OSSClause *CurClause;
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
    // Ignore iterators
    for (auto *E : E->getDepIterators()) {
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      Stack->addDSA(VD, E, OSSC_private, /*Ignore=*/true, /*IsBase=*/true);
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

      SourceLocation ELoc = E->getExprLoc();
      SourceRange ERange = E->getSourceRange();

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);

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
      if (CurClause->getClauseKind() == OSSC_reduction
          || CurClause->getClauseKind() == OSSC_weakreduction)
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
      CurClause = Clause;
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

  void reset() {
    CurClause = nullptr;
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
                        ArrayRef<unsigned> Exclude = llvm::None) {
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

  QualType Ty = BuildArrayType(Context.IntTy, ArrayType::Normal, Res.get(), /*Quals=*/0,
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
    OmpSsDirectiveKind Kind, Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {

  bool ErrorFound = false;

  llvm::SmallVector<OSSClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt && !CurContext->isDependentContext()) {
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
  case OSSD_task_for:
    Res = ActOnOmpSsTaskForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_taskloop:
    Res = ActOnOmpSsTaskLoopDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_taskloop_for:
    Res = ActOnOmpSsTaskLoopForDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
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
  return OSSTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
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
  llvm::Optional<bool> TestIsLessOp;
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
  llvm::Optional<bool> getLoopIsLessOp() const { return TestIsLessOp; }
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
  bool setUB(Expr *NewUB, llvm::Optional<bool> LessOp, bool StrictOp,
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
                                        llvm::Optional<bool> LessOp,
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
    Optional<llvm::APSInt> Result =
        NewStep->getIntegerConstantExpr(SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        Result && Result->isSigned() && (Subtract != Result->isNegative());
    bool IsConstPos =
        Result && Result->isSigned() && (Subtract == Result->isNegative());
    bool IsConstZero = Result && !Result->getBoolValue();

    if (UB && (IsConstZero ||
               (TestIsLessOp.getValue() ?
                  (IsConstNeg || (IsUnsigned && Subtract)) :
                  (IsConstPos || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_oss_loop_incr_not_compatible)
          << LCDecl << TestIsLessOp.getValue() << NewStep->getSourceRange();
      SemaRef.Diag(ConditionLoc,
                   diag::note_oss_loop_cond_requres_compatible_incr)
          << TestIsLessOp.getValue() << ConditionSrcRange;
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

        DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
        if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_private &&
            DVar.RefExpr) {
          Diag(E->getExprLoc(), diag::err_oss_wrong_dsa)
            << getOmpSsClauseName(DVar.CKind)
            << getOmpSsClauseName(OSSC_private);
            return;
        }

        // Register loop control variable
        if (!CurContext->isDependentContext()) {
          DSAStack->addLoopControlVariable(VD, E);
          DSAStack->addDSA(VD, E, OSSC_private, /*Ignore=*/true, /*IsBase=*/true);
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

  return OSSTaskForDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
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

  return OSSTaskLoopForDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt, B);
}

static void checkOutlineDependency(Sema &S, Expr *RefExpr, bool OSSSyntax=false) {
  SourceLocation ELoc = RefExpr->getExprLoc();
  Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
  if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
      RefExpr->containsUnexpandedParameterPack()) {
    // It will be analyzed later.
    return;
  }
  auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
  if (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
      (ASE &&
       !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
       !ASE->getBase()->getType().getNonReferenceType()->isArrayType())) {
    S.Diag(ELoc, diag::err_oss_expected_dereference_or_array_item)
        << RefExpr->getSourceRange();
    return;
  }
  if (isa<DeclRefExpr>(SimpleExpr) || isa<MemberExpr>(SimpleExpr)) {
    S.Diag(ELoc, diag::err_oss_expected_dereference_or_array_item)
        << RefExpr->getSourceRange();
    return;
  }
  while (auto *OASE = dyn_cast<OSSArraySectionExpr>(SimpleExpr)) {
    if (!OASE->isColonForm() && !OSSSyntax) {
      S.Diag(OASE->getColonLoc(), diag::err_oss_section_invalid_form)
          << RefExpr->getSourceRange();
      return;
    }
    SimpleExpr = OASE->getBase()->IgnoreParenCasts();
  }
}

Sema::DeclGroupPtrTy Sema::ActOnOmpSsDeclareTaskDirective(
    DeclGroupPtrTy DG,
    Expr *If, Expr *Final, Expr *Cost, Expr *Priority,
    Expr *Label, Expr *Onready,
    bool Wait,
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
    SourceRange SR) {
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

  ExprResult IfRes, FinalRes, CostRes, PriorityRes, LabelRes, OnreadyRes;
  if (If) {
    IfRes = VerifyBooleanConditionWithCleanups(If, If->getExprLoc());
  }
  if (Final) {
    FinalRes = VerifyBooleanConditionWithCleanups(Final, Final->getExprLoc());
  }
  if (Cost) {
    CostRes = CheckNonNegativeIntegerValue(
      Cost, OSSC_cost, /*StrictlyPositive=*/false);
  }
  if (Priority) {
    PriorityRes = CheckSignedIntegerValue(Priority);
  }
  if (Label) {
    LabelRes = CheckIsConstCharPtrConvertibleExpr(Label);
  }
  if (Onready) {
    OnreadyRes = Onready;
  }
  for (Expr *RefExpr : Ins) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Outs) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Inouts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Concurrents) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Commutatives) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakIns) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakOuts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakInouts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakConcurrents) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakCommutatives) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : DepIns) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepOuts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepInouts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepConcurrents) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepCommutatives) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakIns) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakOuts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakInouts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakConcurrents) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakCommutatives) {
    checkOutlineDependency(*this, RefExpr);
  }

  auto *NewAttr = OSSTaskDeclAttr::CreateImplicit(
    Context,
    IfRes.get(), FinalRes.get(), CostRes.get(), PriorityRes.get(),
    LabelRes.get(),
    Wait,
    OnreadyRes.get(),
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
    SR);
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

static bool actOnOSSReductionKindClause(
    Sema &S, DSAStackTy *Stack, OmpSsClauseKind ClauseKind,
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ArrayRef<Expr *> UnresolvedReductions, ReductionData &RD) {
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
          llvm::APInt InitValue = llvm::APInt::getAllOnesValue(Size);
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
                                  StartLoc, LParenLoc, ColonLoc, EndLoc,
                                  ReductionIdScopeSpec, ReductionId,
                                  UnresolvedReductions, RD))
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
    SourceLocation ELoc = RefExpr->getExprLoc();
    Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
    if (RefExpr->containsUnexpandedParameterPack()) {
      Diag(RefExpr->getExprLoc(), diag::err_oss_variadic_templates_not_clause_allowed);
      continue;
    } else if (RefExpr->isTypeDependent() || RefExpr->isValueDependent()) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      continue;
    }

    if (RequireCompleteExprType(RefExpr, diag::err_oss_incomplete_type))
      continue;

    // TODO: check with OSSArraySectionExpr
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
      Diag(ELoc, diag::err_oss_expected_addressable_lvalue_or_array_item)
          << RefExpr->getSourceRange();
      continue;
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
      Diag(ELoc, diag::err_oss_call_expr_support)
          << RefExpr->getSourceRange();
      continue;
    }

    bool InvalidArraySection = false;
    while (auto *OASE = dyn_cast<OSSArraySectionExpr>(SimpleExpr)) {
      if (!OASE->isColonForm() && !OSSSyntax) {
        Diag(OASE->getColonLoc(), diag::err_oss_section_invalid_form)
            << RefExpr->getSourceRange();
        // Only diagnose the first error
        InvalidArraySection = true;
        break;
      }
      SimpleExpr = OASE->getBase()->IgnoreParenImpCasts();
    }
    if (InvalidArraySection)
      continue;
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

ExprResult Sema::CheckNonNegativeIntegerValue(Expr *ValExpr,
                                      OmpSsClauseKind CKind,
                                      bool StrictlyPositive) {
  ExprResult Res = CheckSignedIntegerValue(ValExpr);
  if (Res.isInvalid())
    return ExprError();

  ValExpr = Res.get();

  if (ValExpr->containsErrors())
    return Res.get();

  // The expression must evaluate to a non-negative integer value.
  if (Optional<llvm::APSInt> Result =
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

ExprResult Sema::CheckIsConstCharPtrConvertibleExpr(Expr *E) {
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
  if (CKind == OSSC_collapse && DSAStack->getAssociatedLoops() == 1) {
    DSAStack->setAssociatedLoops(Result.getExtValue());
    DSAStack->setSeenAssociatedLoops(0);
  }
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
    E, OSSC_cost, /*StrictlyPositive=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSCostClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::CheckSignedIntegerValue(Expr *ValExpr) {
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent() &&
      !ValExpr->containsUnexpandedParameterPack()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformOmpSsImplicitIntegerConversion(Loc, ValExpr);
    if (Value.isInvalid())
      return ExprError();
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
  ExprResult Res = CheckSignedIntegerValue(E);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSPriorityClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsLabelClause(Expr *E,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  ExprResult Res = CheckIsConstCharPtrConvertibleExpr(E);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSLabelClause(Res.get(), StartLoc, LParenLoc, EndLoc);
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
    E, OSSC_chunksize, /*StrictlyPositive=*/false);
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
    E, OSSC_grainsize, /*StrictlyPositive=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSGrainsizeClause(Res.get(), StartLoc, LParenLoc, EndLoc);
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
  case OSSC_label:
    Res = ActOnOmpSsLabelClause(Expr, StartLoc, LParenLoc, EndLoc);
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
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsWaitClause(SourceLocation StartLoc,
                                      SourceLocation EndLoc) {
  return new (Context) OSSWaitClause(StartLoc, EndLoc);
}
