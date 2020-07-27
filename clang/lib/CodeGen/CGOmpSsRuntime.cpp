//===----- CGOmpSsRuntime.cpp - Interface to OmpSs Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OmpSs runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGOmpSsRuntime.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "ConstantEmitter.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/BitmaskEnum.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsOmpSs.h"

using namespace clang;
using namespace CodeGen;

namespace {

enum OmpSsBundleKind {
  OSSB_directive,
  OSSB_task,
  OSSB_task_for,
  OSSB_taskloop,
  OSSB_taskloop_for,
  OSSB_taskwait,
  OSSB_shared,
  OSSB_private,
  OSSB_firstprivate,
  OSSB_if,
  OSSB_final,
  OSSB_cost,
  OSSB_priority,
  OSSB_label,
  OSSB_chunksize,
  OSSB_grainsize,
  OSSB_wait,
  OSSB_in,
  OSSB_out,
  OSSB_inout,
  OSSB_concurrent,
  OSSB_commutative,
  OSSB_reduction,
  OSSB_redinit,
  OSSB_redcomb,
  OSSB_weakin,
  OSSB_weakout,
  OSSB_weakinout,
  OSSB_weakconcurrent,
  OSSB_weakcommutative,
  OSSB_weakreduction,
  OSSB_init,
  OSSB_copy,
  OSSB_deinit,
  OSSB_vladims,
  OSSB_captured,
  OSSB_loop_type,
  OSSB_loop_indvar,
  OSSB_loop_lowerbound,
  OSSB_loop_upperbound,
  OSSB_loop_step,
  OSSB_unknown
};

const char *getBundleStr(OmpSsBundleKind Kind) {
  switch (Kind) {
  case OSSB_directive:
    return "DIR.OSS";
  case OSSB_task:
    return "TASK";
  case OSSB_task_for:
    return "TASK.FOR";
  case OSSB_taskloop:
    return "TASKLOOP";
  case OSSB_taskloop_for:
    return "TASKLOOP.FOR";
  case OSSB_taskwait:
    return "TASKWAIT";
  case OSSB_shared:
    return "QUAL.OSS.SHARED";
  case OSSB_private:
    return "QUAL.OSS.PRIVATE";
  case OSSB_firstprivate:
    return "QUAL.OSS.FIRSTPRIVATE";
  case OSSB_if:
    return "QUAL.OSS.IF";
  case OSSB_final:
    return "QUAL.OSS.FINAL";
  case OSSB_cost:
    return "QUAL.OSS.COST";
  case OSSB_priority:
    return "QUAL.OSS.PRIORITY";
  case OSSB_label:
    return "QUAL.OSS.LABEL";
  case OSSB_chunksize:
    return "QUAL.OSS.LOOP.CHUNKSIZE";
  case OSSB_grainsize:
    return "QUAL.OSS.LOOP.GRAINSIZE";
  case OSSB_wait:
    return "QUAL.OSS.WAIT";
  case OSSB_in:
    return "QUAL.OSS.DEP.IN";
  case OSSB_out:
    return "QUAL.OSS.DEP.OUT";
  case OSSB_inout:
    return "QUAL.OSS.DEP.INOUT";
  case OSSB_concurrent:
    return "QUAL.OSS.DEP.CONCURRENT";
  case OSSB_commutative:
    return "QUAL.OSS.DEP.COMMUTATIVE";
  case OSSB_reduction:
    return "QUAL.OSS.DEP.REDUCTION";
  case OSSB_redinit:
    return "QUAL.OSS.DEP.REDUCTION.INIT";
  case OSSB_redcomb:
    return "QUAL.OSS.DEP.REDUCTION.COMBINE";
  case OSSB_weakin:
    return "QUAL.OSS.DEP.WEAKIN";
  case OSSB_weakout:
    return "QUAL.OSS.DEP.WEAKOUT";
  case OSSB_weakinout:
    return "QUAL.OSS.DEP.WEAKINOUT";
  case OSSB_weakconcurrent:
    return "QUAL.OSS.DEP.WEAKCONCURRENT";
  case OSSB_weakcommutative:
    return "QUAL.OSS.DEP.WEAKCOMMUTATIVE";
  case OSSB_weakreduction:
    return "QUAL.OSS.DEP.WEAKREDUCTION";
  case OSSB_init:
    return "QUAL.OSS.INIT";
  case OSSB_copy:
    return "QUAL.OSS.COPY";
  case OSSB_deinit:
    return "QUAL.OSS.DEINIT";
  case OSSB_vladims:
    return "QUAL.OSS.VLA.DIMS";
  case OSSB_captured:
    return "QUAL.OSS.CAPTURED";
  case OSSB_loop_type:
    return "QUAL.OSS.LOOP.TYPE";
  case OSSB_loop_indvar:
    return "QUAL.OSS.LOOP.IND.VAR";
  case OSSB_loop_lowerbound:
    return "QUAL.OSS.LOOP.LOWER.BOUND";
  case OSSB_loop_upperbound:
    return "QUAL.OSS.LOOP.UPPER.BOUND";
  case OSSB_loop_step:
    return "QUAL.OSS.LOOP.STEP";
  default:
    llvm_unreachable("Invalid OmpSs bundle kind");
  }
}

} // namespace

namespace {
// This class gathers all the info needed for OSSDependVisitor to emit a dependency in
// compute_dep.
// The visitor walks and annotates the number of dimensions in the same way that OSSDependVisitor
// NOTE: keep synchronized
//
// Additionally it gathers:
// 1. all the shape expressions involved variables. Since a shaping expression may be a VLA, it's dimensions
// have not been built yet. We need the involved variable to be able to emit them in compute_dep
// 2. all the vla size expressions. We may have 'vla[sizeof(vla)]' which needs all the dimensions for example
// 3. 'this'
class OSSDependInfoGathering
  : public ConstStmtVisitor<OSSDependInfoGathering, void> {

  CodeGenFunction &CGF;
  QualType OSSArgTy;

  // List of vars (not references or globals)
  // involved in the dependency
  llvm::MapVector<const VarDecl *, LValue> ExprInvolvedVarList;
  // Map of VLASizeExpr involved in the dependency.
  llvm::MapVector<const Expr *, llvm::Value *> VLASizeInvolvedMap;
  // Map of captures involved in the dependency.
  // i.e. references and global variables
  llvm::DenseMap<const VarDecl *, Address> CaptureInvolvedMap;

  Address CXXThisAddress = Address::invalid();

  SmallVector<QualType, 4> RetTypes;

  llvm::Value *Base = nullptr;

  void AddDimStartEnd() {
    RetTypes.push_back(OSSArgTy);
    RetTypes.push_back(OSSArgTy);
    RetTypes.push_back(OSSArgTy);
  }

public:

  OSSDependInfoGathering(CodeGenFunction &CGF)
    : CGF(CGF), OSSArgTy(CGF.getContext().LongTy)
      {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  // This is used for DeclRefExpr, MemberExpr and Unary Deref
  void FillBaseExprDims(const Expr *E) {
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType() || !TmpTy->isArrayType()) {
      // T * || T
      AddDimStartEnd();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // This is used in the innermost Expr * in ArraySubscripts and OSSArraySection. Also in OSSArrayShaping
  void FillDimsFromInnermostExpr(const Expr *E) {
    // Go through the expression which may be a DeclRefExpr or MemberExpr or OSSArrayShapingExpr
    E = E->IgnoreParenImpCasts();
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType()) {
      // T *
      // We have added section dimension before
      if (RetTypes.empty())
        AddDimStartEnd();
      TmpTy = TmpTy->getPointeeType();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // Walk down into the type and look for VLA expressions.
  void FillTypeVLASizes(const Expr *E) {
    QualType type = E->getType();
    while (type->isVariablyModifiedType()) {
      if (type->isPointerType()) {
        type = type->getPointeeType();
      } else if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(type)) {
        type = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(type)) {
        const Expr *VLASizeExpr = BaseArrayTy->getSizeExpr();
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        VLASizeInvolvedMap[VLASizeExpr] = VlaSize.NumElts;

        type = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
    if (RetTypes.empty()) {
      // We want the original type, not the shaping expr. type
      RetTypes.push_back(E->getBase()->getType());

      // Annotate dims
      FillDimsFromInnermostExpr(E);
    }

    Visit(E->getBase());

    // Annotate all the variables involved in shape expr. We will
    // emit its VLASizeExpr in compute_dep context
    for (const Expr *S : E->getShapes())
      Visit(S);
  }

  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;

    CodeGenFunction::ConstantEmission CE = CGF.tryEmitAsConstant(const_cast<DeclRefExpr*>(E));
    if (CE && !CE.isReference())
      // Constant value, no need to annotate it.
      return;

    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }
    FillTypeVLASizes(E);

    llvm::Value *PossibleBase = nullptr;
    const VarDecl *VD = cast<VarDecl>(E->getDecl());
    if (VD->getType()->isReferenceType()
        || E->refersToEnclosingVariableOrCapture()) {
      // Reuse the ref addr. got in DSA
      LValue LV = CGF.EmitDeclRefLValue(E);
      PossibleBase = LV.getPointer(CGF);
      CaptureInvolvedMap.try_emplace(VD, LV.getAddress(CGF));
    } else if (VD->hasLinkage() || VD->isStaticDataMember()) {
      PossibleBase = CGF.CGM.GetAddrOfGlobalVar(VD);
      CharUnits Alignment = CGF.getContext().getDeclAlign(VD);
      Address Addr(PossibleBase, Alignment);
      CaptureInvolvedMap.try_emplace(VD, Addr);
    } else {
      LValue LV = CGF.EmitDeclRefLValue(E);
      PossibleBase = LV.getPointer(CGF);
      ExprInvolvedVarList[VD] = LV;
    }
    // Since we prioritize visiting the base of the expression
    // the first DeclRefExpr is always the Base
    if (!Base)
      Base = PossibleBase;
  }

  void VisitCXXThisExpr(const CXXThisExpr *ThisE) {
    CXXThisAddress = CGF.LoadCXXThisAddress();
    if (!Base)
      Base = CXXThisAddress.getPointer();
  }

  void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
    if (RetTypes.empty()) {
      // Get the inner expr
      const Expr *TmpE = E;
      // First come OSSArraySection
      while (const OSSArraySectionExpr *ASE = dyn_cast<OSSArraySectionExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // If we see a Pointer we must to add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          AddDimStartEnd();
          break;
        }
      }
      while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // If we see a Pointer we must to add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          AddDimStartEnd();
          break;
        }
      }
      RetTypes.insert(RetTypes.begin(), TmpE->getType());
      FillDimsFromInnermostExpr(TmpE);
    }

    Visit(E->getBase());

    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    if (RetTypes.empty()) {
      // Get the inner expr
      const Expr *TmpE = E;
      while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // If we see a Pointer we must to add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          AddDimStartEnd();
          break;
        }
      }

      RetTypes.insert(RetTypes.begin(), TmpE->getType());
      FillDimsFromInnermostExpr(TmpE);
    }

    Visit(E->getBase());
    Visit(E->getIdx());
  }

  void VisitMemberExpr(const MemberExpr *E) {
    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }

    Visit(E->getBase());
  }

  void VisitUnaryDeref(const UnaryOperator *E) {
    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }

    Visit(E->getSubExpr());
  }

  void VisitStmt(const Stmt *S) {
    for (const Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  const llvm::MapVector<const VarDecl *, LValue> &getInvolvedVarList() const { return ExprInvolvedVarList; }
  const llvm::MapVector<const Expr *, llvm::Value *> &getVLASizeInvolvedMap() const { return VLASizeInvolvedMap; }
  const llvm::DenseMap<const VarDecl *, Address> &getCaptureInvolvedMap() const { return CaptureInvolvedMap; }
  const Address getThisAddress() const { return CXXThisAddress; }
  ArrayRef<QualType> getRetTypes() const { return RetTypes; }
  llvm::Value *getBaseValue() const { assert(Base); return Base; }

};

} // namespace

namespace {
class OSSDependVisitor
  : public ConstStmtVisitor<OSSDependVisitor, void> {
  CodeGenFunction &CGF;
  bool OSSSyntax;

  llvm::Type *OSSArgTy;

  llvm::Value *Ptr;
  SmallVector<llvm::Value *, 4> Starts;
  SmallVector<llvm::Value *, 4> Ends;
  SmallVector<llvm::Value *, 4> Dims;
  QualType BaseElementTy;

public:

  OSSDependVisitor(CodeGenFunction &CGF, bool OSSSyntax)
    : CGF(CGF), OSSSyntax(OSSSyntax),
      OSSArgTy(CGF.ConvertType(CGF.getContext().LongTy))
      {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  QualType GetInnermostElementType(const QualType &Q) {
    if (Q->isArrayType()) {
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(Q)) {
        return CGF.getContext().getBaseElementType(Q);
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(Q)) {
        return CGF.getContext().getBaseElementType(Q);
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
    return Q;
  }

  void FillBaseExprDimsAndType(const Expr *E) {
    BaseElementTy = GetInnermostElementType(E->getType());
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType() || !TmpTy->isArrayType()) {
      // T * || T
      Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, DimSize));
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        Dims.push_back(DimExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // This is used in the innermost Expr * in ArraySubscripts and OSSArraySection
  void FillDimsFromInnermostExpr(const Expr *E) {
    // Go through the expression which may be a DeclRefExpr or MemberExpr or OSSArrayShapingExpr
    E = E->IgnoreParenImpCasts();
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType()) {
      // T *
      // We have added section dimension before
      if (Dims.empty())
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
      TmpTy = TmpTy->getPointeeType();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, DimSize));
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        Dims.push_back(DimExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(const Expr *E) {
    ConstStmtVisitor<OSSDependVisitor, void>::Visit(E);
  }

  void VisitStmt(const Stmt *S) {
    llvm_unreachable("Unhandled stmt");
  }

  void VisitExpr(const Expr *E) {
    llvm_unreachable("Unhandled expr");
  }

  void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
    BaseElementTy = GetInnermostElementType(E->getType());
    Ptr = CGF.EmitLValue(E).getPointer(CGF);
    if (E->getType()->isVariablyModifiedType()) {
      CGF.EmitVariablyModifiedType(E->getType());
    }
    FillDimsFromInnermostExpr(E);
  }

  // l-values.
  void VisitDeclRefExpr(const DeclRefExpr *E) {
    Ptr = CGF.EmitDeclRefLValue(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
    // Get Base Type
    // An array section is considered a built-in type
    BaseElementTy =
        OSSArraySectionExpr::getBaseOriginalType(
                          E->getBase());
    if (BaseElementTy->isAnyPointerType()) {
      BaseElementTy = BaseElementTy->getPointeeType();
    } else if (BaseElementTy->isArrayType()) {
      BaseElementTy = BaseElementTy->getAsArrayTypeUnsafe()->getElementType();
    } else {
      llvm_unreachable("Unhandled Type");
    }
    BaseElementTy = GetInnermostElementType(BaseElementTy);

    // Get the inner expr
    const Expr *TmpE = E;
    // First come OSSArraySection
    while (const OSSArraySectionExpr *ASE = dyn_cast<OSSArraySectionExpr>(TmpE->IgnoreParenImpCasts())) {
      // Stop in the innermost ArrayToPointerDecay
      TmpE = ASE->getBase();
      // Add indexes
      llvm::Value *Idx, *IdxEnd;

      const Expr *LowerB = ASE->getLowerBound();
      if (LowerB)
        Idx = CGF.EmitScalarExpr(LowerB);
      else
        // OpenMP 5.0 2.1.5 When the lower-bound is absent it defaults to 0.
        Idx = llvm::ConstantInt::getSigned(OSSArgTy, 0);
      Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);

      const Expr *LengthUpper = ASE->getLengthUpper();
      bool ColonForm = ASE->isColonForm();
      if (LengthUpper &&
          (!OSSSyntax || (OSSSyntax && !ColonForm))) {
        // depend(in: array[ : length])
        // in(array[ ; length])
        IdxEnd = CGF.EmitScalarExpr(LengthUpper);
        IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);
        IdxEnd = CGF.Builder.CreateAdd(Idx, IdxEnd);
      } else if (LengthUpper
                 && (OSSSyntax && ColonForm)) {
        // in(array[ : upper])
        IdxEnd = CGF.EmitScalarExpr(LengthUpper);
        IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);
        IdxEnd = CGF.Builder.CreateAdd(llvm::ConstantInt::getSigned(OSSArgTy, 1), IdxEnd);
      } else if (ASE->getColonLoc().isInvalid()) {
        assert(!LengthUpper);
        // OSSArraySection without ':' are regular array subscripts
        IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
      } else {

        // OpenMP 5.0 2.1.5
        // depend(in: array[lower : ]) -> [lower, dimsize)
        // When the length is absent it defaults to ⌈(size - lowerbound)∕stride⌉,
        // where size is the size of the array dimension.
        //
        // OmpSs-2
        // in(array[lower ; ]) -> [lower, dimsize)
        // in(array[lower : ]) -> [lower, dimsize)
        QualType BaseOriginalTy =
          OSSArraySectionExpr::getBaseOriginalType(ASE->getBase());

        if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(BaseOriginalTy)) {
          uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
          IdxEnd = llvm::ConstantInt::getSigned(OSSArgTy, DimSize);
        } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(BaseOriginalTy)) {
          auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
          IdxEnd = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        } else {
          llvm_unreachable("Unhandled array type");
        }
      }
      IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);

      Starts.push_back(Idx);
      Ends.push_back(IdxEnd);
      // If we see a Pointer we must to add one dimension and done
      if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
        assert(LengthUpper && "Sema should have forbidden unspecified sizes in pointers");
        Dims.push_back(CGF.Builder.CreateSExt(CGF.EmitScalarExpr(LengthUpper), OSSArgTy));
        break;
      }
    }
    while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
      // Stop in the innermost ArrayToPointerDecay
      TmpE = ASE->getBase();
      // Add indexes
      llvm::Value *Idx = CGF.EmitScalarExpr(ASE->getIdx());
      Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);
      llvm::Value *IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
      Starts.push_back(Idx);
      Ends.push_back(IdxEnd);
      // If we see a Pointer we must to add one dimension and done
      if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
        break;
      }
    }

    Ptr = CGF.EmitScalarExpr(TmpE);
    if (const OSSArrayShapingExpr *OSA = dyn_cast<OSSArrayShapingExpr>(TmpE->IgnoreParenImpCasts())) {
      // We must to emit VLA args
      if (OSA->getType()->isVariablyModifiedType()) {
        CGF.EmitVariablyModifiedType(OSA->getType());
      }
    }
    FillDimsFromInnermostExpr(TmpE);
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    // Get Base Type
    BaseElementTy = GetInnermostElementType(E->getType());
    // Get the inner expr
    const Expr *TmpE = E;
    while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
      // Stop in the innermost ArrayToPointerDecay
      TmpE = ASE->getBase();
      // Add indexes
      llvm::Value *Idx = CGF.EmitScalarExpr(ASE->getIdx());
      Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);
      llvm::Value *IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
      Starts.push_back(Idx);
      Ends.push_back(IdxEnd);
      // If we see a Pointer we must to add one dimension and done
      if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
        break;
      }
    }

    Ptr = CGF.EmitScalarExpr(TmpE);
    if (const OSSArrayShapingExpr *OSA = dyn_cast<OSSArrayShapingExpr>(TmpE->IgnoreParenImpCasts())) {
      // We must to emit VLA args
      if (OSA->getType()->isVariablyModifiedType()) {
        CGF.EmitVariablyModifiedType(OSA->getType());
      }
    }
    FillDimsFromInnermostExpr(TmpE);
  }

  void VisitMemberExpr(const MemberExpr *E) {
    Ptr = CGF.EmitMemberExpr(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  void VisitUnaryDeref(const UnaryOperator *E) {
    Ptr = CGF.EmitUnaryOpLValue(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  ArrayRef<llvm::Value *> getStarts() const {
    return Starts;
  }

  ArrayRef<llvm::Value *> getEnds() const {
    return Ends;
  }

  ArrayRef<llvm::Value *> getDims() const {
    return Dims;
  }

  QualType getBaseElementTy() {
    return BaseElementTy;
  }

  llvm::Value *getPtr() {
    return Ptr;
  }

};
} // namespace

// Gathers VLA dims info for VLA.DIMS (if IsPtr) and CAPTURED bundles
static void GatherVLADims(CodeGenFunction &CGF, llvm::Value *V, QualType Q,
                          SmallVectorImpl<llvm::Value *> &DimsWithValue,
                          SmallVectorImpl<llvm::Value *> &CapturedList,
                          bool IsPtr) {
  assert(DimsWithValue.empty() && "DimsWithValue must be empty");
  // C long -> LLVM long
  llvm::Type *OSSArgTy = CGF.ConvertType(CGF.getContext().LongTy);

  if (!IsPtr)
    DimsWithValue.push_back(V);
  while (Q->isArrayType()) {
    if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(Q)) {
      auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
      llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
      if (!IsPtr)
        DimsWithValue.push_back(DimExpr);
      CapturedList.push_back(DimExpr);
      Q = BaseArrayTy->getElementType();
    } else if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(Q)) {
      uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
      llvm::Value *DimConstant = llvm::ConstantInt::getSigned(OSSArgTy, DimSize);
      if (!IsPtr)
        DimsWithValue.push_back(DimConstant);
      CapturedList.push_back(DimConstant);
      Q = BaseArrayTy->getElementType();
    } else {
      llvm_unreachable("Unhandled array type");
    }
  }
}

void CGOmpSsRuntime::EmitCopyCtorFunc(
    llvm::Value *DSAValue, const CXXConstructExpr *CtorE,
    const VarDecl *CopyD, const VarDecl *InitD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());
  // If we have already created the function we're done
  auto It = GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_copy), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(CopyD->getType());
  ImplicitParamDecl SrcArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

  Args.push_back(&SrcArg);
  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  GlobalDecl CtorGD(CtorD, Ctor_Complete);
  llvm::Value *CtorValue = CGM.getAddrOfCXXStructor(CtorGD);

  std::string Name = "oss_copy_ctor";
  Name += CtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue SrcLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&SrcArg),
                                             PQ->castAs<PointerType>());
  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *SrcBegin = SrcLV.getPointer(CGF);
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
                                                      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *SrcCur = CGF.Builder.CreatePHI(SrcBegin->getType(), 2,
                                             "arrayctor.src.cur");
  DstCur->addIncoming(DstBegin, EntryBB);
  SrcCur->addIncoming(SrcBegin, EntryBB);

  {
    CodeGenFunction::OSSPrivateScope InitScope(CGF);
    InitScope.addPrivate(InitD, [SrcCur, SrcLV]() -> Address {

      return Address(SrcCur, SrcLV.getAlignment());
    });
    (void)InitScope.Privatize();
    CGF.EmitExprAsInit(CtorE, CopyD,
                       CGF.MakeAddrLValue(DstCur, DstLV.getType(), DstLV.getAlignment()),
                       /*capturedByInit=*/false);
  }

  // Go to the next element. Move SrcBegin too
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  llvm::Value *SrcDest =
    CGF.Builder.CreateInBoundsGEP(SrcCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                                  "arrayctor.src.next");
  SrcCur->addIncoming(SrcDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_copy), ArrayRef<llvm::Value*>{DSAValue, Fn});

}

void CGOmpSsRuntime::EmitCtorFunc(
    llvm::Value *DSAValue, const VarDecl *CopyD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getInit());
  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());

  GlobalDecl CtorGD(CtorD, Ctor_Complete);
  llvm::Value *CtorValue = CGM.getAddrOfCXXStructor(CtorGD);

  auto It = GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_init), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(CopyD->getType());
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  std::string Name = "oss_ctor";
  Name += CtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
                                                      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  DstCur->addIncoming(DstBegin, EntryBB);

  CGF.EmitExprAsInit(CtorE, CopyD,
                     CGF.MakeAddrLValue(DstCur, DstLV.getType(), DstLV.getAlignment()),
                     /*capturedByInit=*/false);

  // Go to the next element
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_init), ArrayRef<llvm::Value*>{DSAValue, Fn});
}

void CGOmpSsRuntime::EmitDtorFunc(
    llvm::Value *DSAValue, const VarDecl *CopyD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  QualType Q = CopyD->getType();

  const RecordType *RT = Q->getAs<RecordType>();
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());

  if (RD->hasTrivialDestructor())
    return;

  const CXXDestructorDecl *DtorD = RD->getDestructor();

  GlobalDecl DtorGD(DtorD, Dtor_Complete);
  llvm::Value *DtorValue = CGM.getAddrOfCXXStructor(DtorGD);

  auto It = GenericCXXNonPodMethodDefs.find(DtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_deinit), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(Q);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  std::string Name = "oss_dtor";
  Name += DtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
                                                      "arraydtor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arraydtor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arraydtor.dst.cur");
  DstCur->addIncoming(DstBegin, EntryBB);

  CGF.EmitCXXDestructorCall(DtorD, Dtor_Complete,
                         /*ForVirtualBase=*/false, /*Delegating=*/false,
                         Address(DstCur, DstLV.getAlignment()), Q);

  // Go to the next element
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arraydtor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arraydtor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arraydtor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[DtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_deinit), ArrayRef<llvm::Value*>{DSAValue, Fn});
}

void CGOmpSsRuntime::EmitDSAShared(
  CodeGenFunction &CGF, const Expr *E,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
    llvm::Value *DSAValue;
    if ((VD->getType()->isReferenceType()
         || DRE->refersToEnclosingVariableOrCapture())
        && !getTaskCaptureAddr(VD).isValid()) {
      // 1. Record Ref Address to be reused in task body and other clasues
      // 2. Record Addr of lambda captured variables to be reused in task body and
      //    other clauses.
      LValue LV = CGF.EmitDeclRefLValue(DRE);
      CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
      DSAValue = LV.getPointer(CGF);
      TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSAValue);
    } else {
      DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
      TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSAValue);
    }
    QualType Q = VD->getType();
    // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
    bool IsPtr = Q->isPointerType();
    SmallVector<llvm::Value *, 4> DimsWithValue;
    while (Q->isPointerType()) {
      Q = Q->getPointeeType();
    }
    if (Q->isVariableArrayType())
      GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);

    if (!DimsWithValue.empty())
      TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  } else if (const CXXThisExpr *ThisE = dyn_cast<CXXThisExpr>(E)) {
    TaskInfo.emplace_back(
        getBundleStr(OSSB_shared),
        CGF.EmitScalarExpr(ThisE));
  } else {
    llvm_unreachable("Unhandled expression");
  }
}

void CGOmpSsRuntime::EmitDSAPrivate(
  CodeGenFunction &CGF, const OSSDSAPrivateDataTy &PDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(PDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *DSAValue;
  if ((VD->getType()->isReferenceType()
       || DRE->refersToEnclosingVariableOrCapture())
      && !getTaskCaptureAddr(VD).isValid()) {
    // 1. Record Ref Address to be reused in task body and other clasues
    // 2. Record Addr of lambda captured variables to be reused in task body and
    //    other clauses.
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
    DSAValue = LV.getPointer(CGF);
    TaskInfo.emplace_back(getBundleStr(OSSB_private), DSAValue);
  } else {
    DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
    TaskInfo.emplace_back(getBundleStr(OSSB_private), DSAValue);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType())
    GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  const DeclRefExpr *CopyE = cast<DeclRefExpr>(PDataTy.Copy);
  const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());

  if (!CopyD->getType().isPODType(CGF.getContext())) {
    EmitCtorFunc(DSAValue, CopyD, TaskInfo);
    EmitDtorFunc(DSAValue, CopyD, TaskInfo);
  }
}

void CGOmpSsRuntime::EmitDSAFirstprivate(
  CodeGenFunction &CGF, const OSSDSAFirstprivateDataTy &FpDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  const DeclRefExpr *DRE = cast<DeclRefExpr>(FpDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *DSAValue;
  if ((VD->getType()->isReferenceType()
       || DRE->refersToEnclosingVariableOrCapture())
      && !getTaskCaptureAddr(VD).isValid()) {
    // 1. Record Ref Address to be reused in task body and other clasues
    // 2. Record Addr of lambda captured variables to be reused in task body and
    //    other clauses.
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
    DSAValue = LV.getPointer(CGF);
    TaskInfo.emplace_back(getBundleStr(OSSB_firstprivate), DSAValue);
  } else {
    DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
    TaskInfo.emplace_back(getBundleStr(OSSB_firstprivate), DSAValue);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType())
    GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  // TODO: is this sufficient to skip COPY/DEINIT? (task functions)
  if (FpDataTy.Copy) {
    const DeclRefExpr *CopyE = cast<DeclRefExpr>(FpDataTy.Copy);
    const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());
    const DeclRefExpr *InitE = cast<DeclRefExpr>(FpDataTy.Init);
    const VarDecl *InitD = cast<VarDecl>(InitE->getDecl());

    if (!CopyD->getType().isPODType(CGF.getContext())) {
      const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getAnyInitializer());

      EmitCopyCtorFunc(DSAValue, CtorE, CopyD, InitD, TaskInfo);
      EmitDtorFunc(DSAValue, CopyD, TaskInfo);
    }
  }
}

static llvm::Function *createComputeDepFunction(CodeGenFunction &CGF,
                                                const llvm::MapVector<const VarDecl *, LValue> &ExprInvolvedVarList,
                                                const llvm::MapVector<const Expr *, llvm::Value *> &VLASizeInvolvedMap,
                                                const llvm::DenseMap<const VarDecl *, Address> &CaptureInvolvedMap,
                                                const Address CXXThisAddress,
                                                ArrayRef<QualType> RetTypes,
                                                CodeGenFunction &NewCGF,
                                                DeclRefExpr *&StructRetDRE) {
  ASTContext &C = CGF.CGM.getContext();
  SmallVector<llvm::Type *, 4> ArgsTypeList;
  SmallVector<llvm::Value *, 4> ArgsValueList;

  for (const auto &p : ExprInvolvedVarList) {
    llvm::Value *V = p.second.getPointer(CGF);
    ArgsTypeList.push_back(V->getType());
    ArgsValueList.push_back(V);
  }
  for (const auto &p : VLASizeInvolvedMap) {
    llvm::Value *V = p.second;
    ArgsTypeList.push_back(V->getType());
    ArgsValueList.push_back(V);
  }
  for (const auto &p : CaptureInvolvedMap) {
    Address Addr = p.second;
    llvm::Value *V = Addr.getPointer();
    ArgsTypeList.push_back(V->getType());
    ArgsValueList.push_back(V);
  }
  if (CXXThisAddress.isValid()) {
    llvm::Value *CXXThisValue = CXXThisAddress.getPointer();
    ArgsTypeList.push_back(CXXThisValue->getType());
    ArgsValueList.push_back(CXXThisValue);
  }

  RecordDecl *RD = RecordDecl::Create(C, TTK_Struct,
                                      C.getTranslationUnitDecl(),
                                      SourceLocation(), SourceLocation(),
                                      &C.Idents.get("_depend_unpack_t"));
  for (QualType RetQ : RetTypes) {
    RD->addDecl(FieldDecl::Create(C, RD, SourceLocation(), SourceLocation(),
                                  nullptr, RetQ, nullptr, nullptr, false,
                                  ICIS_NoInit));
  }
  RD->completeDefinition();

  QualType StructRetQ = C.getTagDeclType(RD);
  llvm::StructType *StructRetTy = cast<llvm::StructType>(CGF.CGM.getTypes().ConvertType(StructRetQ));

  auto *FuncType =
    llvm::FunctionType::get(StructRetTy, ArgsTypeList, /*IsVarArgs=*/ false);

  auto *FuncVar = llvm::Function::Create(
      FuncType, llvm::GlobalValue::InternalLinkage, CGF.CurFn->getAddressSpace(),
      "compute_dep", &CGF.CGM.getModule());


  // Set names for arguments.
  auto AI = FuncVar->arg_begin();
  for (unsigned i = 0, e = ArgsValueList.size(); i != e; ++i, ++AI)
    AI->setName(ArgsValueList[i]->getName());

  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create(CGF.CGM.getModule().getContext(), "entry", FuncVar);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *AllocaInsertPointer = new llvm::BitCastInst(Undef, CGF.Int32Ty, "allocapt", EntryBB);

  NewCGF.AllocaInsertPt = AllocaInsertPointer;
  NewCGF.Builder.SetInsertPoint(EntryBB);
  NewCGF.CurFuncDecl = nullptr;
  NewCGF.FnRetTy = StructRetQ;
  NewCGF.CurFn = FuncVar;
  NewCGF.CurFnInfo = CGF.CurFnInfo;
  Address StructRetAddr = NewCGF.CreateMemTemp(StructRetQ, "return.val");
  // Address StructRetAddr = Address(NewCGF.Builder.CreateAlloca(StructRetTy, nullptr, "return.val"), C.getTypeAlignInChars(StructRetQ));
  NewCGF.ReturnValue = StructRetAddr;

  return FuncVar;
}

void CGOmpSsRuntime::EmitDependencyList(
    CodeGenFunction &CGF, const OSSDepDataTy &Dep,
    SmallVectorImpl<llvm::Value *> &List) {

  OSSDependInfoGathering DependInfoGathering(CGF);
  DependInfoGathering.Visit(Dep.E);

  CodeGenFunction NewCGF(CGF.CGM);
  DeclRefExpr *StructRetDRE;

  llvm::Function *ComputeDepFun =
    createComputeDepFunction(
      CGF, DependInfoGathering.getInvolvedVarList(),
      DependInfoGathering.getVLASizeInvolvedMap(),
      DependInfoGathering.getCaptureInvolvedMap(),
      DependInfoGathering.getThisAddress(),
      DependInfoGathering.getRetTypes(),
      NewCGF, StructRetDRE);

  // TODO change name
  {
    CodeGenFunction::OSSPrivateScope InitScope(NewCGF);
    auto ArgI = ComputeDepFun->arg_begin();
    for (const auto &p : DependInfoGathering.getInvolvedVarList()) {
      const VarDecl *VD = p.first;
      LValue LV = p.second;
      InitScope.addPrivate(VD, [&ArgI, &LV]() -> Address {
        return Address(ArgI, LV.getAlignment());
      });

      ++ArgI;
    }
    for (const auto &p : DependInfoGathering.getVLASizeInvolvedMap()) {
      const Expr *VLASizeExpr = p.first;
      InitScope.addPrivateVLA(VLASizeExpr, [ArgI]() -> llvm::Value * {
        return ArgI;
      });

      ++ArgI;
    }
    CaptureMapTy CaptureReplacedMap;
    for (const auto &p : DependInfoGathering.getCaptureInvolvedMap()) {
      const VarDecl *VD = p.first;
      Address Addr = p.second;
      Address NewAddr = Address(ArgI, Addr.getAlignment());
      CaptureReplacedMap.try_emplace(VD, NewAddr);
      ++ArgI;
    }
    CaptureMapStack.push_back(CaptureReplacedMap);

    // Map 'this' to compute_dep param
    const Address CXXThisAddress = DependInfoGathering.getThisAddress();
    if (CXXThisAddress.isValid()) {
      InitScope.setThis(Address(ArgI, CXXThisAddress.getAlignment()));
      ++ArgI;
    }

    (void)InitScope.Privatize();

    // C long -> LLVM long
    llvm::Type *OSSArgTy = CGF.ConvertType(NewCGF.getContext().LongTy);

    OSSDependVisitor DepVisitor(NewCGF, Dep.OSSSyntax);
    DepVisitor.Visit(Dep.E);


    SmallVector<llvm::Value *, 4> Starts(
        DepVisitor.getStarts().begin(),
        DepVisitor.getStarts().end());

    SmallVector<llvm::Value *, 4> Ends(
        DepVisitor.getEnds().begin(),
        DepVisitor.getEnds().end());

    SmallVector<llvm::Value *, 4> Dims(
        DepVisitor.getDims().begin(),
        DepVisitor.getDims().end());
    QualType BaseElementTy = DepVisitor.getBaseElementTy();
    llvm::Value *Ptr = DepVisitor.getPtr();

    uint64_t BaseElementSize =
               NewCGF.CGM
                 .getDataLayout()
                 .getTypeSizeInBits(NewCGF
                                    .ConvertType(BaseElementTy))/8;

    List.push_back(Ptr);
    bool First = true;
    for (size_t i = Dims.size() - 1; i > 0; --i) {
      llvm::Value *Dim = Dims[i];
      llvm::Value *IdxStart;
      llvm::Value *IdxEnd;
      // In arrays we have to output all dimensions, but
      // the number of indices may be less than the number
      // of dimensions (array[1] -> int array[10][20])
      if (i < Starts.size()) {
        IdxStart = Starts[Starts.size() - i - 1];
        IdxEnd = Ends[Starts.size() - i - 1];
      } else {
        IdxStart = llvm::ConstantInt::getSigned(OSSArgTy, 0);
        IdxEnd = Dim;
      }
      if (First) {
        First = false;
        Dim = NewCGF.Builder.CreateMul(Dim,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
        IdxStart = NewCGF.Builder.CreateMul(IdxStart,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
        IdxEnd = NewCGF.Builder.CreateMul(IdxEnd,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
      }
      List.push_back(Dim);
      List.push_back(IdxStart);
      List.push_back(IdxEnd);
    }
    llvm::Value *Dim = Dims[0];
    llvm::Value *IdxStart;
    llvm::Value *IdxEnd;
    if (Starts.size() > 0) {
      IdxStart = Starts[Starts.size() - 1];
      IdxEnd = Ends[Starts.size() - 1];
    } else {
      IdxStart = llvm::ConstantInt::getSigned(OSSArgTy, 0);
      IdxEnd = Dim;
    }

    if (First) {
      First = false;
      Dim = NewCGF.Builder.CreateMul(Dim,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxStart = NewCGF.Builder.CreateMul(IdxStart,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxEnd = NewCGF.Builder.CreateMul(IdxEnd,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
    }
    List.push_back(Dim);
    List.push_back(IdxStart);
    List.push_back(IdxEnd);

    Address RetAddr = NewCGF.ReturnValue;
    for (size_t i = 0; i < List.size(); ++i) {
      NewCGF.Builder.CreateStore(List[i], NewCGF.Builder.CreateStructGEP(RetAddr, i));
    }

    NewCGF.Builder.CreateRet(NewCGF.Builder.CreateLoad(RetAddr));

    // After finishing the computedep function remove the alloca instruction
    llvm::Instruction *AllocaInsertPointer = NewCGF.AllocaInsertPt;
    NewCGF.AllocaInsertPt = nullptr;
    AllocaInsertPointer->eraseFromParent();
  }

  // Pop temporal empty refmap, we are not in compute_dep function anymore
  CaptureMapStack.pop_back();

  assert(List.size() == DependInfoGathering.getRetTypes().size());
  List.clear();
  // Fill the bundle

  List.push_back(DependInfoGathering.getBaseValue());
  List.push_back(ComputeDepFun);

  for (const auto &p : DependInfoGathering.getInvolvedVarList()) {
    LValue LV = p.second;
    List.push_back(LV.getPointer(CGF));
  }
  for (const auto &p : DependInfoGathering.getVLASizeInvolvedMap()) {
    llvm::Value *VLASizeValue = p.second;
    List.push_back(VLASizeValue);
  }
  for (const auto &p : DependInfoGathering.getCaptureInvolvedMap()) {
    Address Addr = p.second;
    llvm::Value *V = Addr.getPointer();
    List.push_back(V);
  }
  const Address CXXThisAddress = DependInfoGathering.getThisAddress();
  if (CXXThisAddress.isValid()) {
    List.push_back(CXXThisAddress.getPointer());
  }
}

void CGOmpSsRuntime::EmitDependency(
    std::string Name, CodeGenFunction &CGF, const OSSDepDataTy &Dep,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  SmallVector<llvm::Value*, 4> DepData;
  EmitDependencyList(CGF, Dep, DepData);
  TaskInfo.emplace_back(Name, llvm::makeArrayRef(DepData));
}

/// Check if the combiner is a call to UDR and if it is so return the
/// UDR decl used for reduction.
static const OSSDeclareReductionDecl *
getReductionDecl(const Expr *ReductionOp) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(ReductionOp))
    if (const auto *DRD = dyn_cast<OSSDeclareReductionDecl>(DRE->getDecl()))
      return DRD;
  return nullptr;
}

static llvm::Value *emitReduceInitFunction(CodeGenModule &CGM,
                                           const OSSReductionDataTy &Red) {
  ASTContext &C = CGM.getContext();

  const DeclRefExpr *RHSRef = cast<DeclRefExpr>(Red.RHS);
  const VarDecl *RHSVD = cast<VarDecl>(RHSRef->getDecl());

  const OSSDeclareReductionDecl *DRD =
      getReductionDecl(Red.ReductionOp);

  QualType Q = RHSVD->getType();
  QualType PQ = C.getPointerType(Q);

  FunctionArgList Args;
  ImplicitParamDecl PrivArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl OrigArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NBytesArg(C, C.getSizeType(), ImplicitParamDecl::Other);

  Args.emplace_back(&PrivArg);
  Args.emplace_back(&OrigArg);
  Args.emplace_back(&NBytesArg);

  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = "red_init";
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());

  LValue PrivLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&PrivArg),
                                              PQ->castAs<PointerType>());
  LValue OrigLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&OrigArg),
                                              PQ->castAs<PointerType>());
  llvm::Value *NBytesValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NBytesArg), /*Volatile=*/false,
                       NBytesArg.getType(), NBytesArg.getLocation());

  uint64_t BaseElementSize =
             CGF.CGM
               .getDataLayout()
               .getTypeSizeInBits(CGF
                                  .ConvertType(Q))/8;

  llvm::Value *NelemsValue = CGF.Builder.CreateExactUDiv(NBytesValue,
                                  llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()),
                                                         BaseElementSize));

  // Find the end of the array.
  llvm::Value *OrigBegin = OrigLV.getPointer(CGF);
  llvm::Value *PrivBegin = PrivLV.getPointer(CGF);
  llvm::Value *PrivEnd = CGF.Builder.CreateInBoundsGEP(PrivBegin, NelemsValue,
                                                      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *PrivCur = CGF.Builder.CreatePHI(PrivBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *OrigCur = CGF.Builder.CreatePHI(OrigBegin->getType(), 2,
                                             "arrayctor.src.cur");
  PrivCur->addIncoming(PrivBegin, EntryBB);
  OrigCur->addIncoming(OrigBegin, EntryBB);

  if (DRD && DRD->getInitializer()) {

    const auto *PrivVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getInitPriv())->getDecl());
    const auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getInitOrig())->getDecl());

    CodeGenFunction::OSSPrivateScope InitScope(CGF);
    InitScope.addPrivate(PrivVD, [PrivCur, PrivLV]() -> Address {
      return Address(PrivCur, PrivLV.getAlignment());
    });
    InitScope.addPrivate(OrigVD, [OrigCur, OrigLV]() -> Address {
      return Address(OrigCur, OrigLV.getAlignment());
    });
    (void)InitScope.Privatize();

    if (DRD->getInitializerKind() == OSSDeclareReductionDecl::CallInit) {
      // initializer(foo(&omp_priv, &omp_orig))
      CGF.EmitIgnoredExpr(DRD->getInitializer());
    } else {
      // initializer(omp_priv = ...)
      // initializer(omp_priv(...))
      CGF.EmitExprAsInit(PrivVD->getInit(), PrivVD,
                         CGF.MakeAddrLValue(PrivLV.getPointer(CGF), PrivLV.getType(), PrivLV.getAlignment()),
                         /*capturedByInit=*/false);
    }
  } else {
    assert(RHSVD->hasInit() && "RHSVD has no initializer");
    CGF.EmitExprAsInit(RHSVD->getInit(), RHSVD,
                       CGF.MakeAddrLValue(PrivLV.getPointer(CGF), PrivLV.getType(), PrivLV.getAlignment()),
                       /*capturedByInit=*/false);
  }

  // Go to the next element. Move OrigBegin too
  llvm::Value *PrivNext =
    CGF.Builder.CreateInBoundsGEP(PrivCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  PrivCur->addIncoming(PrivNext, CGF.Builder.GetInsertBlock());

  llvm::Value *OrigDest =
    CGF.Builder.CreateInBoundsGEP(OrigCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                                  "arrayctor.src.next");
  OrigCur->addIncoming(OrigDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(PrivNext, PrivEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();
  return Fn;
}

static llvm::Value *emitReduceCombFunction(CodeGenModule &CGM,
                                           const OSSReductionDataTy &Red) {
  ASTContext &C = CGM.getContext();
  const auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(Red.LHS)->getDecl());
  const auto *RHSVD = cast<VarDecl>(cast<DeclRefExpr>(Red.RHS)->getDecl());

  const OSSDeclareReductionDecl *DRD =
      getReductionDecl(Red.ReductionOp);

  QualType Q = LHSVD->getType();
  QualType PQ = C.getPointerType(Q);

  FunctionArgList Args;
  ImplicitParamDecl OutArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl InArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NBytesArg(C, C.getSizeType(), ImplicitParamDecl::Other);

  Args.emplace_back(&OutArg);
  Args.emplace_back(&InArg);
  Args.emplace_back(&NBytesArg);

  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = "red_comb";
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());

  LValue OutLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&OutArg),
                                              PQ->castAs<PointerType>());
  LValue InLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&InArg),
                                              PQ->castAs<PointerType>());
  llvm::Value *NBytesValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NBytesArg), /*Volatile=*/false,
                       NBytesArg.getType(), NBytesArg.getLocation());

  uint64_t BaseElementSize =
             CGF.CGM
               .getDataLayout()
               .getTypeSizeInBits(CGF
                                  .ConvertType(Q))/8;

  llvm::Value *NelemsValue = CGF.Builder.CreateExactUDiv(NBytesValue,
                                  llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()),
                                                         BaseElementSize));

  // Find the end of the array.
  llvm::Value *InBegin = InLV.getPointer(CGF);
  llvm::Value *OutBegin = OutLV.getPointer(CGF);
  llvm::Value *OutEnd = CGF.Builder.CreateInBoundsGEP(OutBegin, NelemsValue,
                                                      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *OutCur = CGF.Builder.CreatePHI(OutBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *InCur = CGF.Builder.CreatePHI(InBegin->getType(), 2,
                                             "arrayctor.src.cur");
  OutCur->addIncoming(OutBegin, EntryBB);
  InCur->addIncoming(InBegin, EntryBB);

  // Remap lhs and rhs variables to the addresses of the function arguments.
  if (DRD) {
    const auto *OutVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getCombinerOut())->getDecl());
    const auto *InVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getCombinerIn())->getDecl());
    CodeGenFunction::OSSPrivateScope CombScope(CGF);
    CombScope.addPrivate(OutVD, [OutCur, OutLV]() -> Address {
      return Address(OutCur, OutLV.getAlignment());
    });
    CombScope.addPrivate(InVD, [InCur, InLV]() -> Address {
      return Address(InCur, InLV.getAlignment());
    });
    (void)CombScope.Privatize();

    CGF.EmitIgnoredExpr(DRD->getCombiner());
  } else {
    // Emit the combiner body:
    // %2 = <ReductionOp>(<type> *%lhs, <type> *%rhs)
    // store <type> %2, <type>* %lhs
    CodeGenFunction::OSSPrivateScope CombScope(CGF);
    CombScope.addPrivate(LHSVD, [OutCur, OutLV]() -> Address {
      return Address(OutCur, OutLV.getAlignment());
    });
    CombScope.addPrivate(RHSVD, [InCur, InLV]() -> Address {
      return Address(InCur, InLV.getAlignment());
    });
    (void)CombScope.Privatize();

    CGF.EmitIgnoredExpr(Red.ReductionOp);
  }

  // Go to the next element. Move InBegin too
  llvm::Value *OutNext =
    CGF.Builder.CreateInBoundsGEP(OutCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  OutCur->addIncoming(OutNext, CGF.Builder.GetInsertBlock());

  llvm::Value *InDest =
    CGF.Builder.CreateInBoundsGEP(InCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                                  "arrayctor.src.next");
  InCur->addIncoming(InDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(OutNext, OutEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();
  return Fn;
}

static llvm::ConstantInt *reductionKindToNanos6Enum(CodeGenFunction &CGF, QualType Q, BinaryOperatorKind BOK) {
// enum ReductionType
// {
//   RED_TYPE_CHAR = 1000,
//   RED_TYPE_SIGNED_CHAR = 2000,
//   RED_TYPE_UNSIGNED_CHAR = 3000,
//   RED_TYPE_SHORT = 4000,
//   RED_TYPE_UNSIGNED_SHORT = 5000,
//   RED_TYPE_INT = 6000,
//   RED_TYPE_UNSIGNED_INT = 7000,
//   RED_TYPE_LONG = 8000,
//   RED_TYPE_UNSIGNED_LONG = 9000,
//   RED_TYPE_LONG_LONG = 10000,
//   RED_TYPE_UNSIGNED_LONG_LONG = 11000,
//   RED_TYPE_FLOAT = 12000,
//   RED_TYPE_DOUBLE = 13000,
//   RED_TYPE_LONG_DOUBLE = 14000,
//   RED_TYPE_COMPLEX_FLOAT = 15000,
//   RED_TYPE_COMPLEX_DOUBLE = 16000,
//   RED_TYPE_COMPLEX_LONG_DOUBLE = 17000,
//   RED_TYPE_BOOLEAN = 18000,
//   NUM_RED_TYPES = 19000
// };
// enum ReductionOperation
// {
//   RED_OP_ADDITION = 0,
//   RED_OP_PRODUCT = 1,
//   RED_OP_BITWISE_AND = 2,
//   RED_OP_BITWISE_OR = 3,
//   RED_OP_BITWISE_XOR = 4,
//   RED_OP_LOGICAL_AND = 5,
//   RED_OP_LOGICAL_OR = 6,
//   RED_OP_LOGICAL_XOR = 7,
//   RED_OP_LOGICAL_NXOR = 8,
//   RED_OP_MAXIMUM = 9,
//   RED_OP_MINIMUM = 10,
//   NUM_RED_OPS = 11
// };
//
// int Result = ReductionType + ReductionOperation
//
// In case of UDR return -1

  llvm::Type *RedOpTy = CGF.ConvertType(CGF.getContext().IntTy);

  if (BOK == BO_Comma)
    return cast<llvm::ConstantInt>(llvm::ConstantInt::getSigned(RedOpTy, -1));

  int ReductionType = -1;
  int ReductionOperation = -1;

  // Template instantiated types are not canonical
  // See SubstTemplateTypeParmType
  Q = Q.getCanonicalType();

  if (Q == CGF.getContext().CharTy)                   ReductionType = 1000;
  else if (Q == CGF.getContext().SignedCharTy)        ReductionType = 2000;
  else if (Q == CGF.getContext().UnsignedCharTy)      ReductionType = 3000;
  else if (Q == CGF.getContext().ShortTy)             ReductionType = 4000;
  else if (Q == CGF.getContext().UnsignedShortTy)     ReductionType = 5000;
  else if (Q == CGF.getContext().IntTy)               ReductionType = 6000;
  else if (Q == CGF.getContext().UnsignedIntTy)       ReductionType = 7000;
  else if (Q == CGF.getContext().LongTy)              ReductionType = 8000;
  else if (Q == CGF.getContext().UnsignedLongTy)      ReductionType = 9000;
  else if (Q == CGF.getContext().LongLongTy)          ReductionType = 10000;
  else if (Q == CGF.getContext().UnsignedLongLongTy)  ReductionType = 11000;
  else if (Q == CGF.getContext().FloatTy)             ReductionType = 12000;
  else if (Q == CGF.getContext().DoubleTy)            ReductionType = 13000;
  else if (Q == CGF.getContext().LongDoubleTy)        ReductionType = 14000;
  else if (Q == CGF.getContext().FloatComplexTy)      ReductionType = 15000;
  else if (Q == CGF.getContext().DoubleComplexTy)     ReductionType = 16000;
  else if (Q == CGF.getContext().LongDoubleComplexTy) ReductionType = 17000;
  else if (Q == CGF.getContext().BoolTy)              ReductionType = 18000;
  else llvm_unreachable("unhandled reduction type");

  if (BOK == BO_Add)           ReductionOperation = 0;
  else if (BOK == BO_Mul)      ReductionOperation = 1;
  else if (BOK == BO_And)      ReductionOperation = 2;
  else if (BOK == BO_Or)       ReductionOperation = 3;
  else if (BOK == BO_Xor)      ReductionOperation = 4;
  else if (BOK == BO_LAnd)     ReductionOperation = 5;
  else if (BOK == BO_LOr)      ReductionOperation = 6;
  // else if (BOK == BO_LXor)  ReductionOperation = 7;
  // else if (BOK == BO_LNXor) ReductionOperation = 8;
  else if (BOK == BO_GT)       ReductionOperation = 9;
  else if (BOK == BO_LT)       ReductionOperation = 10;
  else llvm_unreachable("unhandled reduction operation");

  return cast<llvm::ConstantInt>(llvm::ConstantInt::getSigned(RedOpTy, ReductionType + ReductionOperation));

}

void CGOmpSsRuntime::EmitReduction(
    std::string RedName, std::string RedInitName, std::string RedCombName,
    CodeGenFunction &CGF, const OSSReductionDataTy &Red,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  SmallVector<llvm::Value *, 4> List;

  llvm::ConstantInt *RedKind = reductionKindToNanos6Enum(CGF, Red.ReductionOp->getType(), Red.ReductionKind);
  llvm::Value *RedInit = nullptr;
  llvm::Value *RedComb = nullptr;
  if (!RedKind->isMinusOne()) {
    auto It = BuiltinRedMap.find(RedKind);
    if (It == BuiltinRedMap.end()) {
      RedInit = emitReduceInitFunction(CGF.CGM, Red);
      RedComb = emitReduceCombFunction(CGF.CGM, Red);
      BuiltinRedMap[RedKind] = {RedInit, RedComb};
    } else {
      RedInit = It->second.first;
      RedComb = It->second.second;
    }
  } else {
    const OSSDeclareReductionDecl *D = getReductionDecl(Red.ReductionOp);
    auto It = UDRMap.find(getReductionDecl(Red.ReductionOp));
    if (It == UDRMap.end()) {
      RedInit = emitReduceInitFunction(CGF.CGM, Red);
      RedComb = emitReduceCombFunction(CGF.CGM, Red);
      UDRMap[D] = {RedInit, RedComb};
    } else {
      RedInit = It->second.first;
      RedComb = It->second.second;
    }
  }

  EmitDependencyList(CGF, {/*OSSSyntax=*/true, Red.Ref}, List);
  // First operand has to be the DSA over the dependency is made
  llvm::Value *DepBaseDSA = List[0];

  List.insert(List.begin(), RedKind);

  TaskInfo.emplace_back(RedName, makeArrayRef(List));
  TaskInfo.emplace_back(RedInitName, ArrayRef<llvm::Value *>{DepBaseDSA, RedInit});
  TaskInfo.emplace_back(RedCombName, ArrayRef<llvm::Value *>{DepBaseDSA, RedComb});
}

void CGOmpSsRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                      SourceLocation Loc,
                                      const OSSTaskDataTy &Data) {
  if (Data.empty()) {
    // Regular taskwait
    llvm::Function *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);
    CGF.Builder.CreateCall(
        Callee, {},
        {
          llvm::OperandBundleDef(
            std::string(getBundleStr(OSSB_directive)),
            llvm::ConstantDataArray::getString(
              CGM.getLLVMContext(), getBundleStr(OSSB_taskwait)))
        });
  } else {
    // taskwait with deps -> task with deps if(0)
    llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
    llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
    SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
    TaskInfo.emplace_back(
        getBundleStr(OSSB_directive),
        llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

    // Add if(0) flag
    llvm::Type *Int1Ty = CGF.ConvertType(CGF.getContext().BoolTy);
    TaskInfo.emplace_back(getBundleStr(OSSB_if), llvm::ConstantInt::getSigned(Int1Ty, 0));

    // Push Task Stack
    TaskStack.push_back(TaskContext());
    CaptureMapStack.push_back(CaptureMapTy());

    InTaskEmission = true;
    EmitDirectiveData(CGF, Data, TaskInfo);
    InTaskEmission = false;

    llvm::Instruction *Result =
      CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));
    CGF.Builder.CreateCall(ExitCallee, Result);

    // Pop Task Stack
    TaskStack.pop_back();
    CaptureMapStack.pop_back();
  }
}

// We're in task body context once we set InsertPt
bool CGOmpSsRuntime::inTaskBody() {
  return !TaskStack.empty() && TaskStack.back().InsertPt;
}

llvm::AssertingVH<llvm::Instruction> CGOmpSsRuntime::getTaskInsertPt() {
  return TaskStack.back().InsertPt;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskTerminateHandler() {
  return TaskStack.back().TerminateHandler;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskTerminateLandingPad() {
  return TaskStack.back().TerminateLandingPad;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskUnreachableBlock() {
  return TaskStack.back().UnreachableBlock;
}

Address CGOmpSsRuntime::getTaskExceptionSlot() {
  return TaskStack.back().ExceptionSlot;
}

Address CGOmpSsRuntime::getTaskEHSelectorSlot() {
  return TaskStack.back().EHSelectorSlot;
}

Address CGOmpSsRuntime::getTaskNormalCleanupDestSlot() {
  return TaskStack.back().NormalCleanupDestSlot;
}

Address CGOmpSsRuntime::getTaskCaptureAddr(const VarDecl *VD) {
  for (auto ItMap = CaptureMapStack.rbegin();
       ItMap != CaptureMapStack.rend(); ++ItMap) {
    auto it = ItMap->find(VD);
    if (it != ItMap->end()) {
      return it->second;
    }
  }
  return Address::invalid();
}

void CGOmpSsRuntime::setTaskInsertPt(llvm::Instruction *I) {
  TaskStack.back().InsertPt = I;
}

void CGOmpSsRuntime::setTaskTerminateHandler(llvm::BasicBlock *BB) {
  TaskStack.back().TerminateHandler = BB;
}

void CGOmpSsRuntime::setTaskTerminateLandingPad(llvm::BasicBlock *BB) {
  TaskStack.back().TerminateLandingPad = BB;
}

void CGOmpSsRuntime::setTaskUnreachableBlock(llvm::BasicBlock *BB) {
  TaskStack.back().UnreachableBlock = BB;
}

void CGOmpSsRuntime::setTaskExceptionSlot(Address Addr) {
  TaskStack.back().ExceptionSlot = Addr;
}

void CGOmpSsRuntime::setTaskEHSelectorSlot(Address Addr) {
  TaskStack.back().EHSelectorSlot = Addr;
}

void CGOmpSsRuntime::setTaskNormalCleanupDestSlot(Address Addr) {
  TaskStack.back().NormalCleanupDestSlot = Addr;
}

// Borrowed brom CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

static void EmitLoopType(const OSSLoopDataTy &LoopData, CodeGenFunction &CGF,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  SmallVector<llvm::Value*, 4> List;
  // LT → <    → 0
  // LE → <=   → 1
  // GT → >    → 2
  // GE → >=   → 3
  int IsLessOp = !*LoopData.TestIsLessOp * 2;
  int IsStrictOp = !LoopData.TestIsStrictOp * 1;
  int LoopCmp = IsLessOp + IsStrictOp;
  List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopCmp));
  List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.IndVar->getType()->isSignedIntegerOrEnumerationType()));
  List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.LB->getType()->isSignedIntegerOrEnumerationType()));
  List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.UB->getType()->isSignedIntegerOrEnumerationType()));
  List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.Step->getType()->isSignedIntegerOrEnumerationType()));
  // TODO: missing step increment/decrement
  TaskInfo.emplace_back(getBundleStr(OSSB_loop_type), List);
}

void CGOmpSsRuntime::EmitDirectiveData(
    CodeGenFunction &CGF, const OSSTaskDataTy &Data,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
    const OSSLoopDataTy &LoopData) {

  SmallVector<llvm::Value*, 4> CapturedList;
  for (const Expr *E : Data.DSAs.Shareds) {
    EmitDSAShared(CGF, E, TaskInfo, CapturedList);
  }
  for (const OSSDSAPrivateDataTy &PDataTy : Data.DSAs.Privates) {
    EmitDSAPrivate(CGF, PDataTy, TaskInfo, CapturedList);
  }
  for (const OSSDSAFirstprivateDataTy &FpDataTy : Data.DSAs.Firstprivates) {
    EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList);
  }

  if (Data.Cost) {
    llvm::Value *V = CGF.EmitScalarExpr(Data.Cost);
    CapturedList.push_back(V);
    TaskInfo.emplace_back(getBundleStr(OSSB_cost), V);
  }
  if (Data.Priority) {
    llvm::Value *V = CGF.EmitScalarExpr(Data.Priority);
    CapturedList.push_back(V);
    TaskInfo.emplace_back(getBundleStr(OSSB_priority), V);
  }
  if (Data.Label) {
    llvm::Value *V = CGF.EmitScalarExpr(Data.Label);
    TaskInfo.emplace_back(getBundleStr(OSSB_label), V);
  }
  if (Data.Wait) {
    TaskInfo.emplace_back(
        getBundleStr(OSSB_wait),
        llvm::ConstantInt::getTrue(CGM.getLLVMContext()));
  }

  if (!LoopData.empty()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_indvar), CGF.EmitLValue(LoopData.IndVar).getPointer(CGF));
    llvm::Value *LB = CGF.EmitScalarExpr(LoopData.LB);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_lowerbound), LB);
    CapturedList.push_back(LB);
    llvm::Value *UB = CGF.EmitScalarExpr(LoopData.UB);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_upperbound), UB);
    CapturedList.push_back(UB);
    llvm::Value *Step = CGF.EmitScalarExpr(LoopData.Step);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_step), Step);
    CapturedList.push_back(Step);
    if (LoopData.Chunksize) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Chunksize);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_chunksize), V);
    }
    if (LoopData.Grainsize) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Grainsize);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_grainsize), V);
    }
    EmitLoopType(LoopData, CGF, TaskInfo);
  }

  if (!CapturedList.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_captured), CapturedList);

  for (const OSSDepDataTy &Dep : Data.Deps.Ins) {
    EmitDependency(getBundleStr(OSSB_in), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Outs) {
    EmitDependency(getBundleStr(OSSB_out), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Inouts) {
    EmitDependency(getBundleStr(OSSB_inout), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Concurrents) {
    EmitDependency(getBundleStr(OSSB_concurrent), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Commutatives) {
    EmitDependency(getBundleStr(OSSB_commutative), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakIns) {
    EmitDependency(getBundleStr(OSSB_weakin), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakOuts) {
    EmitDependency(getBundleStr(OSSB_weakout), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakInouts) {
    EmitDependency(getBundleStr(OSSB_weakinout), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakConcurrents) {
    EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakCommutatives) {
    EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, Dep, TaskInfo);
  }
  for (const OSSReductionDataTy &Red : Data.Reductions.RedList) {
    EmitReduction(getBundleStr(OSSB_reduction),
                  getBundleStr(OSSB_redinit),
                  getBundleStr(OSSB_redcomb),
                  CGF, Red, TaskInfo);
  }
  for (const OSSReductionDataTy &Red : Data.Reductions.WeakRedList) {
    EmitReduction(getBundleStr(OSSB_weakreduction),
                  getBundleStr(OSSB_redinit),
                  getBundleStr(OSSB_redcomb),
                  CGF, Red, TaskInfo);
  }

  if (Data.If)
    TaskInfo.emplace_back(getBundleStr(OSSB_if), CGF.EvaluateExprAsBool(Data.If));
  if (Data.Final)
    TaskInfo.emplace_back(getBundleStr(OSSB_final), CGF.EvaluateExprAsBool(Data.Final));
}

RValue CGOmpSsRuntime::emitTaskFunction(CodeGenFunction &CGF,
                                        const FunctionDecl *FD,
                                        const CallExpr *CE,
                                        ReturnValueSlot ReturnValue) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGM.getContext();

  SourceLocation Loc = CE->getBeginLoc();

  SmallVector<Expr *, 4> ParmCopies;
  SmallVector<Expr *, 4> FirstprivateCopies;
  SmallVector<Expr *, 4> SharedCopies;

  CodeGenFunction::OSSPrivateScope InitScope(CGF);

  InTaskEmission = true;
  CaptureMapStack.push_back(CaptureMapTy());

  auto ArgI = CE->arg_begin();
  auto ParI = FD->param_begin(); 
  while (ArgI != CE->arg_end()) {

    // QualType ParQ = (*ParI)->getType();
    QualType ParQ = (*ParI)->getType().getUnqualifiedType();
    if (ParQ->isArrayType())
      ParQ = Ctx.getBaseElementType(ParQ).getCanonicalType();

    // The a new VarDecl like ParamArgDecl, but in context of function call
    auto *ParmDecl =
      VarDecl::Create(Ctx,
                      const_cast<DeclContext *>(cast<DeclContext>(CGF.CurCodeDecl)),
                      Loc,
                      Loc,
                      &Ctx.Idents.get("call_arg"),
                      ParQ,
                      Ctx.getTrivialTypeSourceInfo(ParQ, Loc),
                      SC_Auto);
    ParmDecl->setImplicit();
    ParmDecl->setReferenced();
    ParmDecl->markUsed(Ctx);
    ParmDecl->setInitStyle(VarDecl::CInit);
    ParmDecl->setInit(const_cast<Expr *>(*ArgI));

    CGF.EmitVarDecl(*ParmDecl);

    Expr *ParmRef = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(), ParmDecl,
        /*RefersToEnclosingVariableOrCapture=*/false, Loc, ParQ.getNonReferenceType(), VK_LValue);

    if (!(*ParI)->getType()->isReferenceType()) {
      ParmRef =
        ImplicitCastExpr::Create(Ctx, ParmRef->getType(), CK_LValueToRValue,
                                 ParmRef, /*BasePath=*/nullptr,
                                 VK_RValue);
      FirstprivateCopies.push_back(ParmRef);
    } else {
      // We want to pass references as shared so task can modify the original value
      SharedCopies.push_back(ParmRef);
    }
    ParmCopies.push_back(ParmRef);

    LValue ParmLV = CGF.EmitLValue(ParmRef);

    InitScope.addPrivate(*ParI, [&CGF, &ParmLV]() -> Address { return ParmLV.getAddress(CGF); });
    // We do need to do this every time because the next param may use the previous one
    (void)InitScope.Privatize();

    ++ArgI;
    ++ParI;
  }
  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

  TaskStack.push_back(TaskContext());

  bool IsMethodCall = false;
  if (const auto *CXXE = dyn_cast<CXXMemberCallExpr>(CE)) {
    IsMethodCall = true;
    const Expr *callee = CXXE->getCallee()->IgnoreParens();
    const MemberExpr *ME = cast<MemberExpr>(callee);
    const Expr *Base = ME->getBase();
    LValue This = CGF.EmitLValue(Base);
    TaskInfo.emplace_back(getBundleStr(OSSB_firstprivate), This.getPointer(CGF));
  }

  // NOTE: this should do only one iteration
  for (const auto *Attr : FD->specific_attrs<OSSTaskDeclAttr>()) {
    SmallVector<llvm::Value*, 4> CapturedList;
    for (const Expr *E : SharedCopies) {
      EmitDSAShared(CGF, E, TaskInfo, CapturedList);
    }
    for (const Expr *E : FirstprivateCopies) {
      OSSDSAFirstprivateDataTy FpDataTy;
      // Ignore ImplicitCast built for the new function call
      FpDataTy.Ref = E->IgnoreImpCasts();
      FpDataTy.Copy = FpDataTy.Init = nullptr;
      EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList);
    }
    if (const Expr *E = Attr->getIfExpr()) {
      TaskInfo.emplace_back(getBundleStr(OSSB_if), CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getFinalExpr()) {
      TaskInfo.emplace_back(getBundleStr(OSSB_final), CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getCostExpr()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_cost), V);
    }
    if (const Expr *E = Attr->getPriorityExpr()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_priority), V);
    }
    if (const Expr *E = Attr->getLabelExpr()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      TaskInfo.emplace_back(getBundleStr(OSSB_label), V);
    }
    // in()
    for (const Expr *E : Attr->ins()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_in), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->outs()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_out), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->inouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_inout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->concurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_concurrent), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->commutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_commutative), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakin), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakinout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, Dep, TaskInfo);
    }
    // depend(in :)
    for (const Expr *E : Attr->depIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_in), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_out), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_inout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_concurrent), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_commutative), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakin), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakinout), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, Dep, TaskInfo);
    }
    if (!CapturedList.empty())
      TaskInfo.emplace_back(getBundleStr(OSSB_captured), CapturedList);
  }
  InTaskEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();

  // From EmitCallExpr
  RValue RV;
  if (IsMethodCall) {
    const Expr *callee = cast<CXXMemberCallExpr>(CE)->getCallee()->IgnoreParens();
    const MemberExpr *ME = cast<MemberExpr>(callee);

    CXXMemberCallExpr *NewCXXE = CXXMemberCallExpr::Create(
        Ctx, const_cast<MemberExpr *>(ME), ParmCopies, Ctx.VoidTy, VK_RValue, SourceLocation());

    RV = CGF.EmitCXXMemberCallExpr(NewCXXE, ReturnValue);
  } else {
    // Regular function call
    CGCallee callee = CGF.EmitCallee(CE->getCallee());

    CallExpr *NewCE = CallExpr::Create(Ctx, const_cast<Expr *>(CE->getCallee()), ParmCopies,
                                       Ctx.VoidTy, VK_RValue, SourceLocation());

    RV = CGF.EmitCall(CE->getCallee()->getType(), callee, NewCE, ReturnValue);
  }

  CGF.EHStack.popTerminate();

  // TODO: do we need this? we're pushing a terminate...
  // EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
  EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
  EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

  return RV;
}

void CGOmpSsRuntime::emitTaskCall(CodeGenFunction &CGF,
                                  const OSSExecutableDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data) {

  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

  // Push Task Stack
  TaskStack.push_back(TaskContext());
  CaptureMapStack.push_back(CaptureMapTy());

  InTaskEmission = true;
  EmitDirectiveData(CGF, Data, TaskInfo);
  InTaskEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  CGF.EmitStmt(D.getAssociatedStmt());
  CGF.EHStack.popTerminate();

  // TODO: do we need this? we're pushing a terminate...
  // EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
  EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
  EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

}

void CGOmpSsRuntime::emitLoopCall(CodeGenFunction &CGF,
                                  const OSSLoopDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data,
                                  const OSSLoopDataTy &LoopData) {

  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;

  OmpSsBundleKind LoopDirectiveBundleKind = OSSB_unknown;
  if (isa<OSSTaskForDirective>(D)) LoopDirectiveBundleKind = OSSB_task_for;
  else if (isa<OSSTaskLoopDirective>(D)) LoopDirectiveBundleKind = OSSB_taskloop;
  else if (isa<OSSTaskLoopForDirective>(D)) LoopDirectiveBundleKind = OSSB_taskloop_for;
  else llvm_unreachable("Unexpected loop directive in codegen");

  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(LoopDirectiveBundleKind)));

  CodeGenFunction::LexicalScope ForScope(CGF, cast<ForStmt>(D.getAssociatedStmt())->getSourceRange());

  // Emit for-init before task entry
  CGF.EmitStmt(cast<ForStmt>(D.getAssociatedStmt())->getInit());

  // Push Task Stack
  TaskStack.push_back(TaskContext());
  CaptureMapStack.push_back(CaptureMapTy());

  InTaskEmission = true;
  EmitDirectiveData(CGF, Data, TaskInfo, LoopData);
  InTaskEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  CGF.EmitStmt(cast<ForStmt>(D.getAssociatedStmt())->getBody());
  CGF.EHStack.popTerminate();

  // TODO: do we need this? we're pushing a terminate...
  // EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
  EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
  EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

}
